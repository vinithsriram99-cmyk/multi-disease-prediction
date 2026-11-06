"""Pipeline builder shared by training and the app: impute -> (log) -> scale / one-hot -> model."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def build_pipeline(model, num_cols, cat_cols, log_cols=()):
    log_cols = [c for c in log_cols if c in num_cols]
    plain = [c for c in num_cols if c not in log_cols]
    parts = []
    if plain:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), plain))
    if log_cols:
        parts.append(("log", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("log", FunctionTransformer(np.log1p)),
                                       ("scale", StandardScaler())]), log_cols))
    if cat_cols:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))
    return Pipeline([("prep", ColumnTransformer(parts)), ("model", model)])
