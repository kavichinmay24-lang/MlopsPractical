
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the trained model
MODEL_PATH = Path(__file__).parent / "model.pkl"
model = joblib.load(MODEL_PATH)

class_names = ["setosa", "versicolor", "virginica"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Flask ML API is running",
        "endpoint": "/predict",
        "method": "POST",
        "features": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or "features" not in data:
        return jsonify({
            "error": "Send JSON containing a features array."
        }), 400

    try:
        features = np.asarray(data["features"], dtype=float)

        if features.shape != (4,) or not np.isfinite(features).all():
            return jsonify({
                "error": "Provide exactly four finite numeric features."
            }), 400

        prediction = int(model.predict([features])[0])

        return jsonify({
            "prediction": prediction,
            "class": class_names[prediction]
        })

    except (TypeError, ValueError):
        return jsonify({
            "error": "Features must contain four numeric values."
        }), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
