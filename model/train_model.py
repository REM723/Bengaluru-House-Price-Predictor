"""
Improved BHP training script.

Key insight: 240 sparse one-hot location columns cripple tree models
(they need hundreds of splits to learn what LR learns in one coefficient).
Fix: target encoding collapses location -> one smoothed mean-price number.
Trees then handle it naturally and outperform LR.

Saves the best model + artifacts to server/artifacts/ and model/.
"""

import warnings
warnings.filterwarnings('ignore')

import os, pickle, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '..', 'data')
ARTIFACTS  = os.path.join(SCRIPT_DIR, '..', 'server', 'artifacts')


# ── 1. DATA PIPELINE ────────────────────────────────────────────────────────

def load_and_clean():
    df = pd.read_csv(os.path.join(DATA_DIR, 'Bengaluru_House_Data.csv'))
    df = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
    df = df.dropna().copy()

    df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

    def sqft_to_num(x):
        p = x.split('-')
        if len(p) == 2:
            try: return (float(p[0]) + float(p[1])) / 2
            except: return None
        try: return float(x)
        except: return None

    df['total_sqft'] = df['total_sqft'].apply(sqft_to_num)
    df = df[df['total_sqft'].notnull()]

    df['price_per_sqft'] = df['price'] * 100_000 / df['total_sqft']
    df['location'] = df['location'].str.strip()

    # collapse rare locations (<=10 samples) -> 'other'
    counts = df['location'].value_counts()
    df['location'] = df['location'].apply(lambda x: 'other' if counts[x] <= 10 else x)

    # business-logic outlier: sqft/bhk >= 300
    df = df[~(df['total_sqft'] / df['bhk'] < 300)]

    # price_per_sqft within +-1 std per location
    parts = []
    for _, sub in df.groupby('location'):
        m, s = sub['price_per_sqft'].mean(), sub['price_per_sqft'].std()
        parts.append(sub[(sub['price_per_sqft'] > m - s) & (sub['price_per_sqft'] <= m + s)])
    df = pd.concat(parts, ignore_index=True)

    # BHK outliers: a 3BHK shouldn't be cheaper per sqft than a 2BHK in the same area
    exclude = []
    for _, loc_df in df.groupby('location'):
        stats = {bhk: {'mean': g['price_per_sqft'].mean(), 'count': len(g)}
                 for bhk, g in loc_df.groupby('bhk')}
        for bhk, g in loc_df.groupby('bhk'):
            prev = stats.get(bhk - 1)
            if prev and prev['count'] > 5:
                exclude.extend(g[g['price_per_sqft'] < prev['mean']].index.tolist())
    df = df.drop(exclude)

    # implausible bathrooms
    df = df[df['bath'] < df['bhk'] + 2]
    return df.drop(['size', 'price_per_sqft'], axis='columns').reset_index(drop=True)


# ── 2. TARGET ENCODING ───────────────────────────────────────────────────────

def build_location_encoding(df_train, smoothing=10):
    """
    Smoothed target (mean price) encoding per location.
    smoothed = (n * loc_mean + k * global_mean) / (n + k)
    Rare locations are pulled toward the global mean.
    """
    global_mean = df_train['price'].mean()
    stats = df_train.groupby('location')['price'].agg(['mean', 'count'])
    stats['smoothed'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / \
                        (stats['count'] + smoothing)
    return stats['smoothed'].to_dict(), global_mean


def apply_target_encoding(df, enc_map, global_mean):
    df = df.copy()
    df['location_enc'] = df['location'].map(enc_map).fillna(global_mean)
    return df.drop('location', axis='columns')


def add_features(df):
    df = df.copy()
    df['sqft_per_bhk']   = df['total_sqft'] / df['bhk']
    df['bath_bhk_ratio'] = df['bath'] / df['bhk']
    return df


# ── 3. CROSS-VALIDATED BENCHMARK ────────────────────────────────────────────

def cv_with_target_enc(model_fn, df_full, n_splits=5, use_log=False):
    """
    Proper CV that recomputes target encoding inside each fold to avoid leakage.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []
    for tr_idx, val_idx in kf.split(df_full):
        df_tr  = df_full.iloc[tr_idx]
        df_val = df_full.iloc[val_idx]

        enc_map, g_mean = build_location_encoding(df_tr)
        df_tr  = apply_target_encoding(df_tr,  enc_map, g_mean)
        df_val = apply_target_encoding(df_val, enc_map, g_mean)
        df_tr  = add_features(df_tr)
        df_val = add_features(df_val)

        X_tr, y_tr   = df_tr.drop('price', axis=1),  df_tr['price']
        X_val, y_val = df_val.drop('price', axis=1), df_val['price']

        if use_log:
            m = model_fn()
            m.fit(X_tr, np.log1p(y_tr))
            pred = np.expm1(m.predict(X_val))
        else:
            m = model_fn()
            m.fit(X_tr, y_tr)
            pred = m.predict(X_val)

        scores.append(r2_score(y_val, pred))

    return np.array(scores)


# ── 4. MAIN ──────────────────────────────────────────────────────────────────

print("-- Loading data --")
df = load_and_clean()
print(f"   {len(df)} rows after cleaning")

# location names list (for the API dropdown)
location_names = sorted([l for l in df['location'].unique() if l != 'other'])

# ---- hold-out test set (same random_state as notebook for fair comparison) ----
df_train, df_test = train_test_split(df, test_size=0.2, random_state=10)

enc_map, g_mean = build_location_encoding(df_train)

df_tr_enc   = add_features(apply_target_encoding(df_train, enc_map, g_mean))
df_te_enc   = add_features(apply_target_encoding(df_test,  enc_map, g_mean))
X_tr, y_tr  = df_tr_enc.drop('price', axis=1), df_tr_enc['price']
X_te, y_te  = df_te_enc.drop('price', axis=1), df_te_enc['price']

print("\n-- Benchmark (5-fold CV, target encoding recomputed per fold) --")

candidates = [
    ("LinearRegression",          lambda: LinearRegression(),                           False),
    ("Ridge",                     lambda: RidgeCV(alphas=[0.1, 1, 10, 100]),            False),
    ("Ridge + log(price)",        lambda: RidgeCV(alphas=[0.1, 1, 10, 100]),            True),
    ("RandomForest",              lambda: RandomForestRegressor(
                                      n_estimators=400, max_depth=15,
                                      min_samples_leaf=3, max_features='sqrt',
                                      random_state=42, n_jobs=-1),                       False),
    ("GradientBoosting",          lambda: GradientBoostingRegressor(
                                      n_estimators=500, max_depth=5,
                                      learning_rate=0.06, min_samples_leaf=4,
                                      subsample=0.85, random_state=42),                  False),
    ("GradientBoosting+log",      lambda: GradientBoostingRegressor(
                                      n_estimators=500, max_depth=5,
                                      learning_rate=0.06, min_samples_leaf=4,
                                      subsample=0.85, random_state=42),                  True),
]

results = {}
for name, model_fn, use_log in candidates:
    scores = cv_with_target_enc(model_fn, df, use_log=use_log)
    # also compute hold-out test score
    m = model_fn()
    if use_log:
        m.fit(X_tr, np.log1p(y_tr))
        test_score = r2_score(y_te, np.expm1(m.predict(X_te)))
    else:
        m.fit(X_tr, y_tr)
        test_score = r2_score(y_te, m.predict(X_te))
    print(f"  {name:<30}  cv={scores.mean():.4f} +/- {scores.std():.4f}  test={test_score:.4f}")
    results[name] = (model_fn, scores.mean(), use_log)


# ── 5. TRAIN FINAL BEST MODEL ON FULL DATA ───────────────────────────────────

best_name, (best_fn, best_cv, best_log) = max(results.items(), key=lambda x: x[1][1])
print(f"\n-- Best: {best_name}  CV={best_cv:.4f} --")

# recompute encoding on full dataset for the production model
final_enc_map, final_g_mean = build_location_encoding(df)
df_full_enc = add_features(apply_target_encoding(df, final_enc_map, final_g_mean))
X_full, y_full = df_full_enc.drop('price', axis=1), df_full_enc['price']

final_model = best_fn()
if best_log:
    final_model.fit(X_full, np.log1p(y_full))
else:
    final_model.fit(X_full, y_full)

feature_names = [col.lower() for col in X_full.columns]
print(f"   Features ({len(feature_names)}): {feature_names}")


# ── 6. SAVE ARTIFACTS ────────────────────────────────────────────────────────

with open(os.path.join(ARTIFACTS, 'banglore_home_prices_model.pickle'), 'wb') as f:
    pickle.dump(final_model, f)

with open(os.path.join(ARTIFACTS, 'columns.json'), 'w') as f:
    json.dump({'data_columns': feature_names, 'location_names': location_names}, f)

with open(os.path.join(ARTIFACTS, 'model_meta.json'), 'w') as f:
    json.dump({
        'model_type':         best_name,
        'uses_log_transform': best_log,
        'cv_score':           round(best_cv, 4),
        'location_encodings': {k: round(v, 4) for k, v in final_enc_map.items()},
        'global_mean':        round(final_g_mean, 4),
    }, f, indent=2)

print(f"   Saved to server/artifacts/ and model/")
print(f"   Log transform: {best_log}")
