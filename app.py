"""Multiple Disease Prediction - Streamlit app.   Run:  streamlit run app.py
Inputs are used only to make a prediction and are never stored."""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data_prep import NAMES
from labels import LABELS, PARKINSONS_HELP
from predict_utils import PROFILES, compare_table, load_bundle, predict, profile_defaults, risk_level

st.set_page_config(page_title="Multiple Disease Prediction", page_icon="🩺", layout="wide")
KEYS = ["kidney", "liver", "parkinsons"]
DISCLAIMER = "⚠️ Educational project. Predictions come from small public datasets and are **not a medical diagnosis**. Always consult a doctor."


@st.cache_resource
def bundle(key):
    return load_bundle(key)


@st.cache_data
def results():
    return json.load(open("models/evaluation_results.json"))


@st.cache_data
def csv(path):
    return pd.read_csv(path)


try:
    RES = results()
except FileNotFoundError:
    st.error("Models not found. Run `python3 preprocess.py` and `python3 train_models.py` first.")
    st.stop()

st.sidebar.title("🩺 Disease Prediction")
page = st.sidebar.radio("Menu", ["Home", "Kidney disease", "Liver disease", "Parkinson's disease",
                                 "Model performance", "Data insights", "About & privacy"])
st.sidebar.caption("Inputs are processed in memory only and are not stored.")


def gauge(p):
    level, color = risk_level(p)
    fig = go.Figure(go.Indicator(mode="gauge+number", value=round(100 * p, 1), number={"suffix": "%"},
                                 title={"text": f"Predicted probability - {level} risk"},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                                        "steps": [{"range": [0, 30], "color": "#e8f5e9"},
                                                  {"range": [30, 60], "color": "#fff3e0"},
                                                  {"range": [60, 100], "color": "#ffebee"}]}))
    fig.update_layout(height=300, margin=dict(t=60, b=10))
    return fig


def predict_page(key):
    b, name = bundle(key), NAMES[key]
    st.title(f"{name} prediction")
    st.caption(f"Model: {b['model_name']} (trained on {RES[key]['rows']} records) | " + DISCLAIMER)
    if key == "parkinsons":
        st.info(PARKINSONS_HELP)
    if key == "liver":
        st.warning("This dataset is hard to separate: the model is only moderately accurate "
                   f"(ROC-AUC {RES[key]['nested_cv_metrics_of_best']['roc_auc']:.2f}). Treat the result as a rough indicator.")
    profile = st.radio("Start from", PROFILES, horizontal=True, key=f"{key}_profile")
    start = profile_defaults(b, profile)
    labels, values = LABELS[key], {}
    with st.form(f"{key}_form"):
        cols = st.columns(3)
        for i, c in enumerate(b["columns"]):
            with cols[i % 3]:
                label = labels.get(c, c)
                if c in b["meta"]["categorical"]:
                    opts = b["meta"]["categorical"][c]["options"]
                    values[c] = st.selectbox(label, opts, index=opts.index(start[c]), key=f"{key}_{c}_{profile}")
                else:
                    m = b["meta"]["numeric"][c]
                    span = (m["max"] - m["min"]) or 1.0
                    values[c] = st.number_input(label, min_value=float(min(0.0, m["min"])), max_value=float(m["max"] + span),
                                                value=float(start[c]), step=float(span / 100), format="%.6g",
                                                key=f"{key}_{c}_{profile}")
        go_btn = st.form_submit_button("Predict", type="primary")
    if go_btn:
        p, _ = predict(b, values)
        left, right = st.columns([1, 1])
        left.plotly_chart(gauge(p), use_container_width=True)
        level, _c = risk_level(p)
        right.metric("Risk level", level)
        right.write(f"The model estimates a **{100 * p:.1f}%** probability of {name.lower()} for these values.")
        imp = csv(f"models/{key}_importance.csv")
        cmp = compare_table(b, values, imp)
        if len(cmp):
            st.subheader("Why? How your values compare (most influential measurements)")
            fig = px.bar(cmp, x="position", y="feature", orientation="h",
                         labels={"position": "0 = typical healthy value, 1 = typical patient value"},
                         color="position", color_continuous_scale=["green", "gold", "red"], range_color=[-0.5, 1.5])
            fig.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cmp.drop(columns="position").round(4), hide_index=True, use_container_width=True)
        if level != "Low":
            st.warning("Elevated risk: please consult a doctor for proper tests. This tool cannot diagnose.")


if page == "Home":
    st.title("🩺 Multiple Disease Prediction System")
    st.write("Predicts the likelihood of **kidney disease, liver disease and Parkinson's disease** from test results, "
             "using one machine-learning model per disease. Choose a disease in the menu, enter the values, and press Predict.")
    st.caption(DISCLAIMER)
    cols = st.columns(3)
    for col, k in zip(cols, KEYS):
        m = RES[k]["nested_cv_metrics_of_best"]
        col.subheader(NAMES[k])
        col.metric("ROC-AUC (cross-validated)", f"{m['roc_auc']:.3f}")
        col.write(f"Model: **{RES[k]['best_model']}**  \nAccuracy {m['accuracy']:.1%} | Precision {m['precision']:.1%}  \n"
                  f"Recall {m['recall']:.1%} | F1 {m['f1']:.1%}")
elif page == "Kidney disease":
    predict_page("kidney")
elif page == "Liver disease":
    predict_page("liver")
elif page == "Parkinson's disease":
    predict_page("parkinsons")

elif page == "Model performance":
    st.title("📊 Model performance")
    st.caption("All numbers come from nested 5-fold cross-validation (honest estimates on unseen data).")
    key = st.selectbox("Disease", KEYS, format_func=lambda k: NAMES[k])
    r = RES[key]
    st.write(f"Best model: **{r['best_model']}**, best settings: `{r['best_params']}`")
    comp = csv(f"models/{key}_model_comparison.csv").rename(columns={"Unnamed: 0": "model"})
    st.dataframe(comp, hide_index=True, use_container_width=True)
    a, b2 = st.columns(2)
    cm = r["nested_cv_metrics_of_best"]["confusion_matrix"]
    mat = [[cm["true_negative"], cm["false_positive"]], [cm["false_negative"], cm["true_positive"]]]
    a.plotly_chart(px.imshow(mat, text_auto=True, x=["Predicted healthy", "Predicted disease"], y=["Actually healthy", "Actually disease"],
                             color_continuous_scale="Blues", title="Confusion matrix"), use_container_width=True)
    roc = csv(f"models/{key}_roc.csv")
    fig = go.Figure([go.Scatter(x=roc.fpr, y=roc.tpr, name="Model"), go.Scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(dash="dash"))])
    fig.update_layout(title=f"ROC curve (AUC {r['nested_cv_metrics_of_best']['roc_auc']:.3f})", xaxis_title="False positive rate", yaxis_title="True positive rate")
    b2.plotly_chart(fig, use_container_width=True)
    imp = csv(f"models/{key}_importance.csv").head(10)
    st.plotly_chart(px.bar(imp[::-1], x="importance", y="feature", orientation="h",
                           title="Most important features (drop in AUC when the feature is shuffled)"), use_container_width=True)
    if key == "parkinsons":
        st.info(f"195 recordings come from only 32 people. A normal random split would let the model see the same person in training and testing "
                f"and report AUC {r['random_split_auc_leaky']:.3f} (too optimistic). We keep each person's recordings together, "
                f"giving the honest AUC {r['nested_cv_metrics_of_best']['roc_auc']:.3f}.")
    if key == "kidney":
        st.info("Kidney disease is detected almost perfectly because blood and urine tests (hemoglobin, creatinine, specific gravity) directly reflect the disease. "
                "With only 400 patients, real-world accuracy would likely be lower.")

elif page == "Data insights":
    st.title("🔍 Data insights")
    key = st.selectbox("Disease", KEYS, format_func=lambda k: NAMES[k])
    try:
        df = csv(f"data/clean_{key}.csv")
    except FileNotFoundError:
        st.error("Run `python3 preprocess.py` first."); st.stop()
    df["status"] = df["target"].map({1: "Disease", 0: "Healthy"})
    st.plotly_chart(px.pie(df, names="status", title="Class balance"), use_container_width=True)
    feat = st.selectbox("Feature", [c for c in df.columns if c not in ("target", "status")])
    if df[feat].dtype == "object" or str(df[feat].dtype).startswith("str"):
        st.plotly_chart(px.histogram(df, x=feat, color="status", barmode="group"), use_container_width=True)
    else:
        st.plotly_chart(px.histogram(df, x=feat, color="status", barmode="overlay", opacity=0.65, nbins=30), use_container_width=True)
    num = df.select_dtypes("number")
    st.plotly_chart(px.imshow(num.corr().round(2), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation heatmap"),
                    use_container_width=True)
    st.write("Missing values per column:", df.isna().sum()[df.isna().sum() > 0].to_dict() or "none")

else:
    st.title("About & privacy")
    st.markdown(f"""
**Data:** UCI Chronic Kidney Disease (400 patients), Indian Liver Patient Dataset (583 patients), Parkinson's voice dataset (195 recordings, 32 people).

**Pipeline:** clean data -> impute missing values -> scale / encode -> train Logistic Regression, Random Forest and Gradient Boosting
-> tune with nested cross-validation -> keep the best model per disease.

**Privacy:** values you type are used only to compute the prediction in memory. Nothing is saved, logged or sent anywhere.
For real deployments you would add login, encryption and compliance checks.

{DISCLAIMER}
""")
