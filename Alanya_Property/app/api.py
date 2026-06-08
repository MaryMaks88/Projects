import os
import joblib
import flask
import numpy as np

def load_model(model_path=""):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        raise FileNotFoundError(f"Model not found on {model_path}")

model = load_model("model/cat_model.joblib")
app = flask.Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = flask.request.get_json(force=True)

        area = float(data["area_sq_m"])
        rooms = int(data["count_rooms"])
        district = str(data["district"])
        distance = float(data["distance_to_sea_m"])
        housing_class = str(data["class"])

        avg_room_size = area / rooms
        area_distance_ratio = area / (distance + 1)

        features = [
            area,
            district,
            distance,
            housing_class,
            rooms,
            avg_room_size,
            area_distance_ratio
        ]

        pred_log = model.predict([features])

        real_price = np.expm1(pred_log[0])

        return flask.jsonify({
            "status": "success",
            "predicted_price_eur": round(real_price, 2),
            "average_deviation_percent": 14.36
        })

    except Exception as e:
        return flask.jsonify({
            "status": "error",
            "message": f"Issue during prediction: {str(e)}"
            }), 400


if __name__ == "__main__":
    app.run(debug=True)



