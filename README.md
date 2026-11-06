# Multiple Disease Prediction System

Streamlit app that predicts the likelihood of **chronic kidney disease, liver disease and Parkinson's disease**
from medical test values, using one tuned machine-learning model per disease.

## Project structure
| File / folder | Purpose |
|---|---|
| `data/` | The three raw datasets (kidney_disease, indian_liver_patient, parkinsons) |
| `data_prep.py` | Loading + cleaning for each dataset |
| `modeling.py` | Pipeline: impute -> log/scale/one-hot -> model |
| `preprocess.py` | Step 1: cleans the data, saves `data/clean_*.csv` and `reports/cleaning_report.json` |
| `train_models.py` | Step 2: trains Logistic Regression, Random Forest, Gradient Boosting; nested cross-validation; saves `models/` |
| `predict_utils.py`, `labels.py` | Prediction helpers and friendly field names |
| `app.py` | Step 3: Streamlit app (prediction pages, performance, data insights) |
| `models/` | Generated: trained models (`*.joblib`), metrics, ROC data, feature importance, `evaluation_results.json` |

## Run
```bash
python3 -m pip install -r requirements.txt
python3 preprocess.py
python3 train_models.py        # about 1-3 minutes
python3 -m streamlit run app.py
```

## Methodology
- **Preprocessing:** text cleaned (spaces, tabs, `?`), numbers stored as text converted, 13 duplicate liver rows removed.
  Missing values are imputed *inside* the pipeline (median / most frequent), learned from training folds only (no leakage).
  Numeric features scaled (StandardScaler); categorical features one-hot encoded; skewed liver enzymes log-transformed.
- **Models:** Logistic Regression, Random Forest and Gradient Boosting (scikit-learn; used instead of XGBoost to avoid extra installs).
  Class weights handle imbalance.
- **Validation:** nested 5-fold cross-validation (outer folds = honest scores, inner folds = hyper-parameter search by ROC-AUC).
  Parkinson's uses group-aware folds because its 195 recordings come from only 32 people.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix (regression metrics MAE/RMSE do not apply to classification).

## Results (nested cross-validation)
| Disease | Best model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Kidney | Random Forest | 99.3% | 99.2% | 99.6% | 99.4% | 1.000 |
| Liver | Logistic Regression | 65.8% | 88.1% | 60.1% | 71.5% | 0.752 |
| Parkinson's | Logistic Regression | 74.9% | 88.9% | 76.2% | 82.1% | 0.844 |

(Re-running training may change the last digits slightly.)

## Limitations
- Small public datasets (195-583 records); not validated clinically. **Not a medical diagnosis.**
- The liver dataset is genuinely hard to separate (AUC about 0.75).
- Kidney results are near-perfect because lab values directly reflect the disease; real-world performance would be lower.
- Parkinson's results come from only 32 people.

## Privacy
Inputs are used in memory only to compute a prediction; nothing is stored or logged.
