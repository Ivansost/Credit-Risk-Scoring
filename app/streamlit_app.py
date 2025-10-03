# streamlit_report_app.py — Screenshot-ready, Streamlit-only (Altair) report
# Run: streamlit run streamlit_report_app.py
# Deps: pip install streamlit pandas numpy scikit-learn joblib altair

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

# -----------------------------
# Page + Sidebar
# -----------------------------
st.set_page_config(page_title="Credit Risk — Screenshot Report", page_icon="📊", layout="wide")

ROOT = os.getcwd()
CSV_PATH_DEFAULT = os.path.join(ROOT, "data", "processed", "credit_clean.csv")
ARTIFACT_PATH_DEFAULT = os.path.join(ROOT, "data", "model_rf.joblib")

st.sidebar.header("Configuration")
csv_path = st.sidebar.text_input("Processed CSV path", value=CSV_PATH_DEFAULT)
artifact_path = st.sidebar.text_input("Model artifact (.joblib, optional)", value=ARTIFACT_PATH_DEFAULT)
twd_to_cad = st.sidebar.number_input("TWD → CAD rate", min_value=0.001, max_value=1.0, value=0.04, step=0.001)
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
top_k = st.sidebar.number_input("Top N rows for 'big spenders' tables", min_value=3, max_value=50, value=10, step=1)

# -----------------------------
# Maps for human-readable labels
# -----------------------------
EDU_MAP = {1:"Graduate school", 2:"University", 3:"High school", 4:"Others", 0:"Unknown", 5:"Unknown", 6:"Unknown"}
EDU_ORDER = ["Graduate school", "University", "High school", "Others", "Unknown"]
MAR_MAP = {1:"Married", 2:"Single", 3:"Others", 0:"Unknown"}
MAR_ORDER = ["Married", "Single", "Others", "Unknown"]
SEX_MAP = {1:"Male", 2:"Female"}
SEX_ORDER = ["Male","Female"]

def to_label(series, which):
    if which == "EDUCATION": return series.map(EDU_MAP).fillna("Unknown")
    if which == "MARRIAGE":  return series.map(MAR_MAP).fillna("Unknown")
    if which == "SEX":       return series.map(SEX_MAP).fillna("Unknown")
    return series

def find_target_col(df: pd.DataFrame):
    if "DEFAULT_FLAG" in df.columns: return "DEFAULT_FLAG"
    if "DEFAULT" in df.columns: return "DEFAULT"
    for c in df.columns:
        if c.strip().lower() == "default payment next month":
            return c
    return None

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_resource
def load_artifact(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)  # expected keys: model, features, (optional) threshold

# -----------------------------
# Load data / model
# -----------------------------
df = load_csv(csv_path) if os.path.exists(csv_path) else None
artifact = load_artifact(artifact_path)

st.title("📊 Credit Risk — Screenshot-ready Report")
st.caption("All visuals are pure Streamlit/Altair for easy screenshots in your README.")

if df is None:
    st.error("CSV not found. Set a valid path in the sidebar.")
    st.stop()

target_col = find_target_col(df)
if target_col is None:
    st.error("Could not identify the default flag column. Expected one of: DEFAULT_FLAG, DEFAULT, 'default payment next month'.")
    st.stop()

# Derive Canadian currency columns for readability
if "LIMIT_BAL" in df.columns:
    df["Credit Limit (Canadian)"] = df["LIMIT_BAL"] * twd_to_cad
if "BILL_AMT1" in df.columns:
    df["Latest Bill (Canadian)"] = df["BILL_AMT1"] * twd_to_cad

# Utilization (fallback if not provided)
if "UTIL_RATIO" not in df.columns and {"LIMIT_BAL", "BILL_AMT1"} <= set(df.columns):
    denom = df["LIMIT_BAL"].replace(0, np.nan)
    df["UTIL_RATIO"] = (df["BILL_AMT1"] / denom).fillna(0)

# Human-readable categorical columns
if "EDUCATION" in df.columns:
    df["Education"] = to_label(df["EDUCATION"], "EDUCATION")
if "MARRIAGE" in df.columns:
    df["Marital Status"] = to_label(df["MARRIAGE"], "MARRIAGE")
if "SEX" in df.columns:
    df["Sex"] = to_label(df["SEX"], "SEX")

# -----------------------------
# Tabs
# -----------------------------
tab_findings, tab_who, tab_model, tab_explain = st.tabs([
    "Findings (Clean & Simple)",
    "Who Defaults / Has Higher Limits / Spends Most",
    "Model Performance (Optional)",
    "Explain (Random Forest)"
])
# -----------------------------
# Findings (clean, no ambiguous categories)
# -----------------------------
with tab_findings:
    st.subheader("Dataset Snapshot")
    st.write(f"Rows: **{len(df)}**  •  Columns: **{len(df.columns)}**")
    st.dataframe(df.head(8))

    col1, col2 = st.columns(2)

    # Average Credit Limit (Canadian) by Education
    with col1:
        if {"Education", "Credit Limit (Canadian)"} <= set(df.columns):
            sub = df[~df["Education"].isin(["Others", "Unknown"])]

            grp = (
                sub.groupby("Education")["Credit Limit (Canadian)"]
                .mean()
                .reindex([e for e in EDU_ORDER if e not in ["Others", "Unknown"]])
                .reset_index()
                .rename(columns={"Credit Limit (Canadian)":"Average Credit Limit (Canadian)"})
            )
            st.markdown("### Average Credit Limit (Canadian) by Education")
            st.dataframe(grp)

            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

            chart = (
                alt.Chart(grp, title="Average Credit Limit (Canadian)")
                .mark_bar()
                .encode(
                    x=alt.X("Education:N", sort=[e for e in EDU_ORDER if e not in ["Others", "Unknown"]]),
                    y="Average Credit Limit (Canadian):Q"
                )
                .properties(height=400)
            )
            st.altair_chart(chart, use_container_width=True)

    # Default Rate by Marital Status
    with col2:
        if {"Marital Status", target_col} <= set(df.columns):
            sub = df[~df["Marital Status"].isin(["Others", "Unknown"])]

            rates = (
                sub.groupby("Marital Status")[target_col]
                .mean()
                .reindex([m for m in MAR_ORDER if m not in ["Others", "Unknown"]])
                .reset_index()
                .rename(columns={target_col:"Default Rate"})
            )
            st.markdown("### Default Rate by Marital Status")
            st.dataframe(rates)

            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

            chart = (
                alt.Chart(rates, title="Default Rate by Marital Status")
                .mark_bar()
                .encode(
                    x=alt.X("Marital Status:N", sort=[m for m in MAR_ORDER if m not in ["Others", "Unknown"]]),
                    y=alt.Y("Default Rate:Q")
                )
                .properties(height=400)
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    # Utilization Ratio Distribution stays unchanged
    if "UTIL_RATIO" in df.columns:
        st.markdown("### Utilization Ratio Distribution")
        # bin edges with Altair (approx)
        util_df = pd.DataFrame({"Utilization Ratio": df["UTIL_RATIO"].clip(upper=3.0)})
        hist = alt.Chart(util_df, title="Utilization Ratio Distribution (capped at 3.0)")\
            .mark_bar()\
            .encode(
                x=alt.X("Utilization Ratio:Q", bin=alt.Bin(maxbins=40)),
                y="count()"
            )
        st.altair_chart(hist, use_container_width=True)

# -----------------------------
# Who defaults, who has higher limits, who spends most
# -----------------------------
with tab_who:
    st.subheader("Who is more likely to default?")
    colA, colB, colC = st.columns(3)

    # Default rate by Education
    with colA:
        if {"Education", target_col} <= set(df.columns):
            t = (
                df.groupby("Education", dropna=False)[target_col]
                .mean()
                .reindex(EDU_ORDER)
                .reset_index()
                .rename(columns={target_col:"Default Rate"})
            )
            st.markdown("#### By Education")
            st.dataframe(t)
            st.altair_chart(
                alt.Chart(t, title="Default Rate by Education")
                .mark_bar()
                .encode(x=alt.X("Education:N", sort=EDU_ORDER), y="Default Rate:Q"),
                use_container_width=True
            )

    # Default rate by Sex
    with colB:
        if {"Sex", target_col} <= set(df.columns):
            t = (
                df.groupby("Sex", dropna=False)[target_col]
                .mean()
                .reindex(SEX_ORDER)
                .reset_index()
                .rename(columns={target_col:"Default Rate"})
            )
            st.markdown("#### By Sex")
            st.dataframe(t)
            st.altair_chart(
                alt.Chart(t, title="Default Rate by Sex")
                .mark_bar()
                .encode(x=alt.X("Sex:N", sort=SEX_ORDER), y="Default Rate:Q"),
                use_container_width=True
            )

    # Default rate by Age bucket (if AGE present)
    with colC:
        if "AGE" in df.columns:
            age_bins = [18,25,35,45,55,65,100]
            age_labels = ["18-24","25-34","35-44","45-54","55-64","65+"]
            df["_AGE_BIN_"] = pd.cut(df["AGE"], bins=age_bins, labels=age_labels, right=False)
            t = (
                df.groupby("_AGE_BIN_", dropna=False)[target_col]
                .mean()
                .reset_index()
                .rename(columns={target_col:"Default Rate", "_AGE_BIN_":"Age Group"})
            )
            st.markdown("#### By Age Group")
            st.dataframe(t)
            st.altair_chart(
                alt.Chart(t, title="Default Rate by Age Group")
                .mark_bar()
                .encode(x=alt.X("Age Group:N", sort=age_labels), y="Default Rate:Q"),
                use_container_width=True
            )

    st.markdown("---")
    st.subheader("Who has higher credit limits (Canadian)?")
    col1, col2 = st.columns(2)

    # Credit limit by Education
    with col1:
        if {"Education", "Credit Limit (Canadian)"} <= set(df.columns):
            t = (
                df.groupby("Education", dropna=False)["Credit Limit (Canadian)"]
                .mean()
                .reindex(EDU_ORDER)
                .reset_index()
                .rename(columns={"Credit Limit (Canadian)":"Average Credit Limit (Canadian)"})
            )
            st.markdown("#### By Education")
            st.dataframe(t)
            st.altair_chart(
                alt.Chart(t, title="Average Credit Limit (Canadian) by Education")
                .mark_bar()
                .encode(x=alt.X("Education:N", sort=EDU_ORDER), y="Average Credit Limit (Canadian):Q"),
                use_container_width=True
            )

    # Credit limit by Marital Status
    with col2:
        if {"Marital Status", "Credit Limit (Canadian)"} <= set(df.columns):
            t = (
                df.groupby("Marital Status", dropna=False)["Credit Limit (Canadian)"]
                .mean()
                .reindex(MAR_ORDER)
                .reset_index()
                .rename(columns={"Credit Limit (Canadian)":"Average Credit Limit (Canadian)"})
            )
            st.markdown("#### By Marital Status")
            st.dataframe(t)
            st.altair_chart(
                alt.Chart(t, title="Average Credit Limit (Canadian) by Marital Status")
                .mark_bar()
                .encode(x=alt.X("Marital Status:N", sort=MAR_ORDER), y="Average Credit Limit (Canadian):Q"),
                use_container_width=True
            )

    st.markdown("---")
    st.subheader("Who spends the most?")
    col3, col4 = st.columns(2)

    # Latest Bill (Canadian) — top spenders
    with col3:
        if "Latest Bill (Canadian)" in df.columns:
            cols = ["ID","Latest Bill (Canadian)","Credit Limit (Canadian)","UTIL_RATIO"]
            show = [c for c in cols if c in df.columns]
            top_spend = df.sort_values("Latest Bill (Canadian)", ascending=False)[show].head(top_k)
            st.markdown(f"#### Top {top_k} by Latest Bill (Canadian)")
            st.dataframe(top_spend)

    # Utilization ratio — top users of their limit
    with col4:
        if "UTIL_RATIO" in df.columns:
            cols = ["ID","UTIL_RATIO","Latest Bill (Canadian)","Credit Limit (Canadian)"]
            show = [c for c in cols if c in df.columns]
            top_util = df.sort_values("UTIL_RATIO", ascending=False)[show].head(top_k)
            st.markdown(f"#### Top {top_k} by Utilization Ratio")
            st.dataframe(top_util)

# -----------------------------
# Model Performance (optional)
# -----------------------------
with tab_model:
    st.subheader("Model Performance (if artifact loaded)")
    if artifact is None:
        st.info("No model artifact found. Provide a .joblib with keys: 'model', 'features' (optional 'threshold').")
    else:
        model = artifact.get("model")
        feats = artifact.get("features", [])
        if not feats:
            st.warning("Artifact has no 'features' list. Trying to score with available numeric columns.")
            feats = [c for c in df.columns if c not in ["Education","Marital Status","Sex"] and pd.api.types.is_numeric_dtype(df[c])]
        missing = [c for c in feats if c not in df.columns]
        if missing:
            st.error(f"Dataset is missing required feature columns: {missing}")
        else:
            X = df[feats]
            y = df[target_col].astype(int).values
            if hasattr(model, "predict_proba"):
                pd_prob = model.predict_proba(X)[:,1]
            else:
                # fallback to decision_function scaled to [0,1]
                raw = model.decision_function(X)
                pd_prob = (raw - raw.min())/(raw.max()-raw.min() + 1e-9)

            # ROC
            fpr, tpr, _ = roc_curve(y, pd_prob)
            roc_auc = auc(fpr, tpr)
            roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
            st.markdown("### ROC Curve")
            roc_chart = alt.Chart(roc_df).mark_line().encode(x="FPR:Q", y="TPR:Q").properties(title=f"ROC Curve (AUC={roc_auc:.3f})")
            st.altair_chart(roc_chart, use_container_width=True)

            # PR
            prec, rec, _ = precision_recall_curve(y, pd_prob)
            ap = average_precision_score(y, pd_prob)
            pr_df = pd.DataFrame({"Recall": rec, "Precision": prec})
            st.markdown("### Precision–Recall Curve")
            pr_chart = alt.Chart(pr_df).mark_line().encode(x="Recall:Q", y="Precision:Q").properties(title=f"Precision–Recall (AP={ap:.3f})")
            st.altair_chart(pr_chart, use_container_width=True)

            # Confusion Matrix at threshold (as a simple table)
            st.markdown(f"### Confusion Matrix @ threshold = {threshold:.2f}")
            y_pred = (pd_prob >= threshold).astype(int)
            cm = confusion_matrix(y, y_pred, labels=[0,1])
            cm_df = pd.DataFrame(cm, index=["No Default (True)","Default (True)"], columns=["No Default (Pred)","Default (Pred)"])
            st.dataframe(cm_df)

# -----------------------------
# Explainability
# -----------------------------
with tab_explain:
    st.subheader("Feature Importance (Random Forest)")
    if artifact is None:
        st.info("Load a Random Forest artifact with 'feature_importances_' to see this.")
    else:
        model = artifact.get("model")
        feats = artifact.get("features", [])
        if hasattr(model, "feature_importances_") and feats:
            imps = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False).reset_index()
            imps.columns = ["Feature","Importance"]
            st.dataframe(imps)
            fi_chart = alt.Chart(imps, title="Feature Importance (Random Forest)")\
                .mark_bar()\
                .encode(y=alt.Y("Feature:N", sort="-x"), x="Importance:Q")
            st.altair_chart(fi_chart, use_container_width=True)
        else:
            st.warning("Model does not expose feature_importances_ or features are missing.")

st.markdown("---")
st.caption("Tip: Use sidebar controls to tweak currency rate and table sizes. All charts/tables are designed for clean screenshots.")
