from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "flight_price_sample.csv"
MODEL_PATH = BASE_DIR / "flight_price_model.joblib"

CITY_COORDS = {
    "Delhi": (28.5562, 77.1000),
    "Mumbai": (19.0896, 72.8656),
    "Bengaluru": (13.1986, 77.7066),
    "Hyderabad": (17.2403, 78.4294),
    "Chennai": (12.9941, 80.1709),
    "Kolkata": (22.6547, 88.4467),
    "Jaipur": (26.8242, 75.8122),
    "Lucknow": (26.7606, 80.8893),
    "Ahmedabad": (23.0772, 72.6347),
    "Pune": (18.5822, 73.9197),
}

FEATURES = [
    "airline",
    "source",
    "destination",
    "travel_class",
    "month",
    "day_of_week",
    "departure_hour",
    "total_stops",
    "booking_days_before",
    "distance_km",
    "duration_mins",
]
TARGET = "price_inr"
CATEGORICAL_FEATURES = ["airline", "source", "destination", "travel_class"]


def haversine_km(source: str, destination: str) -> int:
    lat1, lon1 = CITY_COORDS[source]
    lat2, lon2 = CITY_COORDS[destination]
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return int(round(2 * radius * math.asin(math.sqrt(a))))


def build_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df_model = raw_df.copy()
    df_model["journey_date"] = pd.to_datetime(df_model["journey_date"], errors="coerce")
    df_model["month"] = df_model["journey_date"].dt.month
    df_model["day_of_week"] = df_model["journey_date"].dt.dayofweek
    df_model["departure_hour"] = pd.to_datetime(
        df_model["departure_time"], format="%H:%M", errors="coerce"
    ).dt.hour
    df_model["distance_km"] = df_model.apply(
        lambda row: haversine_km(row["source"], row["destination"]), axis=1
    )
    df_model["duration_mins"] = df_model.apply(
        lambda row: int(round(42 + row["distance_km"] / 11.5 + int(row["total_stops"]) * 65)),
        axis=1,
    )
    return df_model.dropna(subset=FEATURES + [TARGET]).copy()


def train_model() -> Pipeline:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    raw_df = pd.read_csv(DATA_PATH)
    prepared_df = build_dataframe(raw_df)
    X = prepared_df[FEATURES]
    y = prepared_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", [c for c in FEATURES if c not in CATEGORICAL_FEATURES]),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=220,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE (INR)": mean_absolute_error(y_test, y_pred),
        "RMSE (INR)": float((mean_squared_error(y_test, y_pred)) ** 0.5),
        "R² score": r2_score(y_test, y_pred),
    }
    print(f"Model trained: {metrics}")
    joblib.dump(model, MODEL_PATH)
    return model


def load_model() -> Pipeline:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_model()


MODEL = load_model()
app = Flask("flight_cost_prediction_api")
CORS(app, resources={r"/*": {"origins": "*"}})


def build_request_features(payload: dict) -> pd.DataFrame:
    journey_date = pd.to_datetime(payload["journey_date"], errors="raise")
    departure_time = str(payload["departure_time"])
    distance = haversine_km(payload["source"], payload["destination"])
    duration = int(round(42 + distance / 11.5 + int(payload["total_stops"]) * 65))
    return pd.DataFrame(
        [{
            "airline": payload["airline"],
            "source": payload["source"],
            "destination": payload["destination"],
            "travel_class": payload["travel_class"],
            "month": int(journey_date.month),
            "day_of_week": int(journey_date.dayofweek),
            "departure_hour": int(departure_time.split(":")[0]),
            "total_stops": int(payload["total_stops"]),
            "booking_days_before": int(payload["booking_days_before"]),
            "distance_km": distance,
            "duration_mins": duration,
        }]
    )


@app.get("/health")
def health() -> tuple[dict, int]:
    return jsonify({"status": "ok", "message": "Flight prediction API is running"}), 200


@app.post("/predict")
def predict() -> tuple[dict, int]:
    payload = request.get_json(silent=True) or {}
    required = [
        "airline",
        "source",
        "destination",
        "journey_date",
        "departure_time",
        "total_stops",
        "travel_class",
        "booking_days_before",
    ]
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        return jsonify({"error": "Missing required fields: " + ", ".join(missing)}), 400

    try:
        airline = str(payload["airline"])
        source = str(payload["source"])
        destination = str(payload["destination"])
        travel_class = str(payload["travel_class"])
        stops = int(payload["total_stops"])
        booking_days = int(payload["booking_days_before"])
        journey_date = pd.to_datetime(payload["journey_date"], errors="raise")
        time_text = str(payload["departure_time"])
        pd.to_datetime(time_text, format="%H:%M", errors="raise")

        if source not in CITY_COORDS or destination not in CITY_COORDS:
            raise ValueError("Choose a supported source and destination city.")
        if source == destination:
            raise ValueError("Source and destination must be different.")
        if airline not in set(pd.read_csv(DATA_PATH)["airline"].dropna().unique()):
            raise ValueError("Choose a supported airline.")
        if travel_class not in {"Economy", "Business"}:
            raise ValueError("Choose Economy or Business class.")
        if stops not in (0, 1, 2):
            raise ValueError("Stops must be 0, 1, or 2.")
        if not 0 <= booking_days <= 90:
            raise ValueError("Advance booking must be between 0 and 90 days.")
        if journey_date.date() < pd.Timestamp.today().date():
            raise ValueError("Travel date must be today or a future date.")

        features = build_request_features(payload)
        predicted_price = max(0.0, float(MODEL.predict(features)[0]))
        return jsonify(
            {
                "predicted_price_inr": round(predicted_price),
                "route_distance_km": int(features.iloc[0]["distance_km"]),
                "estimated_duration_minutes": int(features.iloc[0]["duration_mins"]),
                "model": "Random Forest Regressor",
                "disclaimer": "Educational estimate from synthetic sample data; not a live fare quotation.",
            }
        ), 200
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
