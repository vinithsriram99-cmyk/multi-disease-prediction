"""Step 1: clean each dataset and save the cleaned CSVs + a cleaning report.  Usage: python3 preprocess.py"""
import json
from data_prep import LOADERS

report = {}
for key, loader in LOADERS.items():
    X, y, groups, rep = loader()
    out = X.copy(); out["target"] = y.values
    out.to_csv(f"data/clean_{key}.csv", index=False)
    report[key] = rep
    print(f"\n== {key}: {out.shape}")
    for k, v in rep.items():
        print(f"  {k}: {v}")
json.dump(report, open("reports/cleaning_report.json", "w"), indent=2)
