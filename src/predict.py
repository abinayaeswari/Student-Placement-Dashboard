"""
predict.py
Loads the trained models saved by src/model.py and predicts placement for one student.
No Streamlit code here, so it can be tested on its own.
"""
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "placement_model.joblib"

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
# Maps the saved-metrics keys (written by model.py) to the model keys
METRIC_KEYS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


class ModelNotFoundError(Exception):
    pass


def load_bundle(path=MODEL_PATH):
    """Load the saved model bundle. Raises ModelNotFoundError if it is missing."""
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            f"Trained model not found at '{path}'. Run `python src/model.py` once to create it."
        )
    return joblib.load(path)


def validate_inputs(values: dict, bundle: dict) -> list:
    """Return a list of human-readable problems (empty list = inputs are fine)."""
    problems = []
    for col in bundle["feature_columns"]:
        if col not in values or values[col] is None or (
            isinstance(values[col], float) and pd.isna(values[col])
        ):
            problems.append(f"Missing value for {col}")
    for col in bundle["cat_cols"]:
        if col in values and values[col] not in list(bundle["encoders"][col].classes_):
            problems.append(f"Invalid value '{values[col]}' for {col}")
    for col in bundle["num_cols"]:
        if col in values:
            try:
                float(values[col])
            except (TypeError, ValueError):
                problems.append(f"{col} must be a number")
    return problems


def predict_student(values: dict, bundle: dict, model_key: str = "logistic_regression") -> dict:
    """
    Predict placement for one student.
    Returns {"label": "Placed"/"Not Placed", "prob_placed": float, "model": str}.
    """
    problems = validate_inputs(values, bundle)
    if problems:
        raise ValueError("; ".join(problems))
    if model_key not in MODEL_LABELS or model_key not in bundle:
        raise ValueError(f"Unknown model '{model_key}'")

    row = {c: values[c] for c in bundle["feature_columns"]}
    X = pd.DataFrame([row], columns=bundle["feature_columns"])
    for c in bundle["cat_cols"]:
        X[c] = bundle["encoders"][c].transform(X[c])
    for c in bundle["num_cols"]:
        X[c] = X[c].astype(float)

    model = bundle[model_key]
    if model_key == "logistic_regression":
        X_in = bundle["scaler"].transform(X)
    else:
        X_in = X
    prob_placed = float(model.predict_proba(X_in)[0][1])
    label = "Placed" if prob_placed >= 0.5 else "Not Placed"
    return {"label": label, "prob_placed": prob_placed, "model": MODEL_LABELS[model_key]}


def feature_importance(bundle: dict) -> pd.Series:
    """Real feature importances from the trained Random Forest."""
    rf = bundle["random_forest"]
    return pd.Series(rf.feature_importances_, index=bundle["feature_columns"]).sort_values(ascending=False)
