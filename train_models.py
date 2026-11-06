"""Step 2: train, tune and evaluate models for every disease.  Usage: python3 train_models.py

Evaluation = NESTED cross-validation: the outer 5 folds give honest scores, the inner folds pick hyper-parameters.
Parkinson's uses group-aware folds so recordings of one person never appear in both train and test."""
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from data_prep import LOADERS, LOG_COLS, NAMES
from modeling import build_pipeline

SEED = 42
GRIDS = {
    "Logistic Regression": (LogisticRegression(max_iter=5000, class_weight="balanced"),
                            {"model__C": [0.01, 0.1, 1, 10]}),
    "Random Forest": (RandomForestClassifier(random_state=SEED, class_weight="balanced_subsample", n_jobs=-1),
                      {"model__n_estimators": [200], "model__max_depth": [4, None],
                       "model__min_samples_leaf": [1, 3]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED),
                          {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
                           "model__max_depth": [2, 3]}),
}


def scores(y, prob, t=0.5):
    p = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return {"accuracy": round(accuracy_score(y, p), 4), "precision": round(precision_score(y, p, zero_division=0), 4),
            "recall": round(recall_score(y, p), 4), "f1": round(f1_score(y, p), 4),
            "roc_auc": round(roc_auc_score(y, prob), 4),
            "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp),
                                 "false_negative": int(fn), "true_positive": int(tp)}}


summary = {}
for key, loader in LOADERS.items():
    X, y, groups, rep = loader()
    y = y.reset_index(drop=True); X = X.reset_index(drop=True)
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    log_cols = LOG_COLS.get(key, [])
    outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))
    print(f"\n######## {NAMES[key]}: {X.shape[0]} rows, {X.shape[1]} features, {y.mean():.0%} positive")
    results, oof_store, imp_store = {}, {}, {}
    for name, (model, grid) in GRIDS.items():
        t0 = time.time()
        oof, imps = np.zeros(len(y)), []
        for tr, te in outer:
            inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(X.iloc[tr], y.iloc[tr], groups[tr]))
            gs = GridSearchCV(build_pipeline(clone(model), num_cols, cat_cols, log_cols), grid,
                              scoring="roc_auc", cv=inner, n_jobs=-1).fit(X.iloc[tr], y.iloc[tr])
            oof[te] = gs.predict_proba(X.iloc[te])[:, 1]
            pi = permutation_importance(gs.best_estimator_, X.iloc[te], y.iloc[te], scoring="roc_auc",
                                        n_repeats=5, random_state=SEED)
            imps.append(pi.importances_mean)
        fold_auc = [roc_auc_score(y.iloc[te], oof[te]) for _, te in outer]
        results[name] = {**scores(y, oof), "cv_auc_mean": round(float(np.mean(fold_auc)), 4),
                         "cv_auc_std": round(float(np.std(fold_auc)), 4), "train_seconds": round(time.time() - t0, 1)}
        oof_store[name], imp_store[name] = oof, np.mean(imps, axis=0)
        print(f"  {name:20s} AUC {results[name]['roc_auc']:.3f} (+/-{results[name]['cv_auc_std']:.3f})  "
              f"acc {results[name]['accuracy']:.3f}  prec {results[name]['precision']:.3f}  "
              f"rec {results[name]['recall']:.3f}  f1 {results[name]['f1']:.3f}  [{results[name]['train_seconds']}s]")

    best = max(results, key=lambda n: (results[n]["roc_auc"], results[n]["f1"]))
    print("  -> best:", best)
    # final model: tune on all data (group-aware folds), refit on all data, ready for the app
    model, grid = GRIDS[best]
    final_cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))
    gs = GridSearchCV(build_pipeline(clone(model), num_cols, cat_cols, log_cols), grid, scoring="roc_auc",
                      cv=final_cv, n_jobs=-1).fit(X, y)
    extra = {}
    if key == "parkinsons":     # show the leakage effect: random folds (same person in train AND test)
        naive = cross_val_predict(clone(gs.best_estimator_), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                  method="predict_proba")[:, 1]
        extra["random_split_auc_leaky"] = round(roc_auc_score(y, naive), 4)
        extra["random_split_accuracy_leaky"] = round(accuracy_score(y, naive >= 0.5), 4)
        print("  leaky random-split AUC would be", extra["random_split_auc_leaky"], "vs honest", results[best]["roc_auc"])

    fpr, tpr, _ = roc_curve(y, oof_store[best])
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(f"models/{key}_roc.csv", index=False)
    (pd.DataFrame({"feature": X.columns, "importance": imp_store[best].round(4)})
     .sort_values("importance", ascending=False).to_csv(f"models/{key}_importance.csv", index=False))
    pd.DataFrame(results).T.drop(columns="confusion_matrix").to_csv(f"models/{key}_model_comparison.csv")
    meta = {"numeric": {c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median()),
                            "median_disease": float(X.loc[y == 1, c].median()),
                            "median_healthy": float(X.loc[y == 0, c].median())} for c in num_cols},
            "categorical": {c: {"options": sorted(X[c].dropna().unique().tolist()),
                                "mode": X[c].mode()[0], "mode_disease": X.loc[y == 1, c].mode()[0],
                                "mode_healthy": X.loc[y == 0, c].mode()[0]} for c in cat_cols}}
    joblib.dump({"pipeline": gs.best_estimator_, "model_name": best, "columns": X.columns.tolist(), "meta": meta,
                 "best_params": {k: str(v) for k, v in gs.best_params_.items()}}, f"models/{key}_model.joblib")
    summary[key] = {"disease": NAMES[key], "best_model": best, "best_params": {k: str(v) for k, v in gs.best_params_.items()},
                    "rows": int(len(y)), "features": int(X.shape[1]), "positive_rate_pct": round(100 * y.mean(), 1),
                    "nested_cv_metrics_of_best": results[best], "all_models": results, **extra,
                    "grids_searched": {n: {k: [str(v) for v in vs] for k, vs in g[1].items()} for n, g in GRIDS.items()}}
json.dump(summary, open("models/evaluation_results.json", "w"), indent=2)
print("\nSaved models/ (joblib models, metrics, ROC, importance).")
