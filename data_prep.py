"""Loading + cleaning for the three disease datasets (one function per disease).
Missing values are NOT filled here - imputation happens inside the model pipeline so it is
learned only from training data (no data leakage)."""
import numpy as np
import pandas as pd

DATA = {"kidney": "data/kidney_disease.csv", "liver": "data/indian_liver_patient.csv",
        "parkinsons": "data/parkinsons.csv"}
NAMES = {"kidney": "Chronic Kidney Disease", "liver": "Liver Disease", "parkinsons": "Parkinson's Disease"}


def load_kidney():
    df = pd.read_csv(DATA["kidney"])
    rep = {"rows_raw": len(df), "missing_cells_raw": int(df.isna().sum().sum())}
    df = df.drop(columns="id")
    for c in df.select_dtypes(exclude="number").columns:              # stray spaces / tabs / '?' markers
        df[c] = df[c].astype(str).str.strip().str.lower().replace({"?": np.nan, "nan": np.nan, "": np.nan})
    for c in ["pcv", "wc", "rc"]:                                     # numbers stored as text
        df[c] = pd.to_numeric(df[c], errors="coerce")
    y = (df.pop("classification") == "ckd").astype(int)               # 1 = chronic kidney disease
    rep.update(rows_clean=len(df), positive_rate_pct=round(100 * y.mean(), 1),
               missing_cells_after_cleaning=int(df.isna().sum().sum()),
               note="Missing values are imputed inside the model pipeline (median / most frequent).")
    return df, y, np.arange(len(df)), rep


def load_liver():
    df = pd.read_csv(DATA["liver"])
    rep = {"rows_raw": len(df), "missing_cells_raw": int(df.isna().sum().sum())}
    rep["duplicate_rows_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
    y = (df.pop("Dataset") == 1).astype(int)                          # source: 1 = liver patient, 2 = not
    rep.update(rows_clean=len(df), positive_rate_pct=round(100 * y.mean(), 1),
               note="Skewed enzyme columns are log-transformed inside the pipeline.")
    return df, y, np.arange(len(df)), rep


def load_parkinsons():
    df = pd.read_csv(DATA["parkinsons"])
    rep = {"rows_raw": len(df), "missing_cells_raw": int(df.isna().sum().sum())}
    groups = df["name"].str.extract(r"(S\d+)")[0].astype("category").cat.codes.to_numpy()   # patient id
    y = df.pop("status").astype(int)                                  # 1 = Parkinson's
    df = df.drop(columns="name")
    rep.update(rows_clean=len(df), patients=int(len(set(groups))), positive_rate_pct=round(100 * y.mean(), 1),
               note="195 voice recordings come from only 32 people (6-7 each). Cross-validation keeps all "
                    "recordings of one person together; a random split would leak and look too good.")
    return df, y, groups, rep


LOADERS = {"kidney": load_kidney, "liver": load_liver, "parkinsons": load_parkinsons}
LOG_COLS = {"liver": ["Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
                      "Alamine_Aminotransferase", "Aspartate_Aminotransferase"]}
