"""Model training and prediction helpers for the smart-city carbon-risk app."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = [
    "district_id",
    "temperature_c",
    "humidity_pct",
    "traffic_congestion_index",
    "grid_energy_demand_mw",
    "public_transit_delay_mins",
    "industrial_zone_activity",
]


def generate_sensor_data(n_samples=5000):
    """Create the synthetic sensor readings used to train the demo model."""
    np.random.seed(42)
    data = {
        "district_id": np.random.randint(1, 6, n_samples),
        "temperature_c": np.random.uniform(15, 42, n_samples),
        "humidity_pct": np.random.uniform(30, 90, n_samples),
        "traffic_congestion_index": np.random.uniform(0, 100, n_samples),
        "grid_energy_demand_mw": np.random.uniform(200, 800, n_samples),
        "public_transit_delay_mins": np.random.uniform(0, 45, n_samples),
        "industrial_zone_activity": np.random.uniform(10, 100, n_samples),
    }
    frame = pd.DataFrame(data)
    frame["carbon_emission_risk"] = frame.apply(assign_risk, axis=1)
    return frame


def assign_risk(row):
    score = (
        row["traffic_congestion_index"] * 0.4
        + row["grid_energy_demand_mw"] * 0.05
        + row["industrial_zone_activity"] * 0.3
    )
    if score > 70 or (
        row["temperature_c"] > 38 and row["grid_energy_demand_mw"] > 650
    ):
        return "High Risk"
    if score > 45:
        return "Medium Risk"
    return "Low Risk"


def train_model():
    """Train the classifier and return it with evaluation data."""
    frame = generate_sensor_data()
    features = frame[FEATURE_COLUMNS]
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["carbon_emission_risk"])
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "model": model,
        "encoder": encoder,
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=range(len(encoder.classes_))
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=range(len(encoder.classes_)),
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
        "class_names": list(encoder.classes_),
        "sample_count": len(frame),
    }


def predict_risk(trained, sensor_reading):
    """Predict a risk label and per-class confidence for one sensor reading."""
    readings = pd.DataFrame([sensor_reading], columns=FEATURE_COLUMNS)
    probabilities = trained["model"].predict_proba(readings)[0]
    predicted_index = int(np.argmax(probabilities))
    return {
        "risk": trained["encoder"].inverse_transform([predicted_index])[0],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": dict(zip(trained["class_names"], probabilities)),
    }


def main():
    trained = train_model()
    print("Model training complete!")
    print(f"Test accuracy: {trained['accuracy']:.2%}")
    print("\n=== Classification Report ===")
    print(pd.DataFrame(trained["classification_report"]).T)
    live_sensor_reading = {
        "district_id": 3,
        "temperature_c": 40.2,
        "humidity_pct": 45.0,
        "traffic_congestion_index": 88.5,
        "grid_energy_demand_mw": 720.0,
        "public_transit_delay_mins": 30.0,
        "industrial_zone_activity": 40.0,
    }
    result = predict_risk(trained, live_sensor_reading)
    print(f"\nLive Sensor Carbon Risk Prediction: {result['risk']}")
    print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
