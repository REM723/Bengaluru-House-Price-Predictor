import pickle
import json
import numpy as np
import os

__model          = None
__location_enc   = None   # dict: location_name -> smoothed mean price
__global_mean    = None   # fallback for unknown locations
__location_names = None   # sorted list for the dropdown
__uses_log       = False  # whether model was trained on log(price)

_artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')


def get_estimated_price(location, sqft, bhk, bath):
    loc_enc = __location_enc.get(location.lower(), __global_mean)

    x = np.array([[
        sqft,
        bath,
        bhk,
        loc_enc,
        sqft / bhk,
        bath / bhk,
    ]])

    pred = __model.predict(x)[0]
    if __uses_log:
        pred = np.expm1(pred)
    return round(float(pred), 2)


def load_saved_artifacts():
    global __model, __location_enc, __global_mean, __location_names, __uses_log

    print("loading saved artifacts...start")

    with open(os.path.join(_artifacts_dir, 'columns.json'), 'r') as f:
        cols = json.load(f)
        __location_names = cols.get('location_names', [])

    with open(os.path.join(_artifacts_dir, 'model_meta.json'), 'r') as f:
        meta = json.load(f)
        __uses_log     = meta.get('uses_log_transform', False)
        __global_mean  = meta.get('global_mean', 0)
        raw_enc        = meta.get('location_encodings', {})
        __location_enc = {k.lower(): v for k, v in raw_enc.items()}

    with open(os.path.join(_artifacts_dir, 'banglore_home_prices_model.pickle'), 'rb') as f:
        __model = pickle.load(f)

    print("loading saved artifacts...done")


def get_location_names():
    return __location_names


if __name__ == '__main__':
    load_saved_artifacts()
    print("Locations:", len(get_location_names()))
    print("Indira Nagar 1000sqft 2BHK 2bath ->", get_estimated_price('indira nagar', 1000, 2, 2))
    print("1st Phase JP Nagar 1000sqft 3BHK 3bath ->", get_estimated_price('1st phase jp nagar', 1000, 3, 3))
    print("Whitefield 1500sqft 3BHK 2bath ->", get_estimated_price('whitefield', 1500, 3, 2))
