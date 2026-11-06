"""Prediction helpers (no Streamlit code here, so they are easy to test)."""
import joblib
import numpy as np
import pandas as pd

PROFILES = ["Dataset median (default)", "Typical patient with the disease", "Typical healthy person"]


def load_bundle(key):
    return joblib.load(f"models/{key}_model.joblib")


def profile_defaults(bundle, profile):
    """Default input values for the form, taken from the training data."""
    idx = PROFILES.index(profile)
    num = {c: [m["median"], m["median_disease"], m["median_healthy"]][idx] for c, m in bundle["meta"]["numeric"].items()}
    cat = {c: [m["mode"], m["mode_disease"], m["mode_healthy"]][idx] for c, m in bundle["meta"]["categorical"].items()}
    return {**num, **cat}


def predict(bundle, values: dict):
    row = pd.DataFrame([values])[bundle["columns"]]
    return float(bundle["pipeline"].predict_proba(row)[0, 1]), row


def risk_level(p):
    return ("Low", "green") if p < 0.30 else (("Moderate", "orange") if p < 0.60 else ("High", "red"))


def compare_table(bundle, values: dict, importance: pd.DataFrame, top=6):
    """How the person's values compare with the typical healthy / diseased person (top features only)."""
    rows = []
    for f in importance.feature:
        if f in bundle["meta"]["numeric"]:
            m = bundle["meta"]["numeric"][f]
            span = m["median_disease"] - m["median_healthy"]
            pos = 0.5 if span == 0 else (values[f] - m["median_healthy"]) / span
            rows.append({"feature": f, "your_value": values[f], "healthy_median": m["median_healthy"],
                         "disease_median": m["median_disease"], "position": float(np.clip(pos, -0.5, 1.5))})
        if len(rows) == top:
            break
    return pd.DataFrame(rows)
