from flask import Flask, request, jsonify
import util

app = Flask(__name__)


def _add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


@app.after_request
def after_request(response):
    return _add_cors(response)


@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({'locations': util.get_location_names()})
    return response


@app.route('/predict_home_price', methods=['POST', 'OPTIONS'])
def predict_home_price():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        # Accept both JSON body and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        total_sqft = float(data['total_sqft'])
        location = data['location']
        bhk = int(data['bhk'])
        bath = int(data['bath'])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400

    estimated_price = util.get_estimated_price(location, total_sqft, bhk, bath)
    return jsonify({'estimated_price': estimated_price})


if __name__ == '__main__':
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    app.run(debug=True, port=5000)
