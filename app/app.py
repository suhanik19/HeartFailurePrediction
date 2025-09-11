import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.train_model import (
    split_data, train_logistic_regression, train_random_forest, train_xgboost
)
from src.evaluate_model import evaluate_model
from src.explainability import explain_model


# === Load and Train Models ===
@st.cache_resource
def load_models():
    df = pd.read_csv("data/heart_failure_clinical_records.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    logreg = train_logistic_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    return df, X_train, X_test, y_train, y_test, logreg, rf, xgb


# === Streamlit App ===
st.title("Heart Failure Prediction App")
st.write("Enter patient information to predict risk of death from heart failure.")

# load models and the data
df, X_train, X_test, y_train, y_test, logreg, rf, xgb = load_models()

# ask user for their input
st.sidebar.header("Patient Data Input")

def user_input_features():
    data = {}
    for col in df.drop(columns=["DEATH_EVENT"]).columns:
        if df[col].nunique() < 10 and df[col].dtype in [int, float]:
            data[col] = st.sidebar.selectbox(col, sorted(df[col].unique()))
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            default_val = float(df[col].median())
            data[col] = st.sidebar.slider(col, min_val, max_val, default_val)
    return pd.DataFrame([data])

input_df = user_input_features()

# model choices
model_choice = st.radio("Choose Model", ["Logistic Regression", "Random Forest", "XGBoost"])

if model_choice == "Logistic Regression":
    model = logreg
elif model_choice == "Random Forest":
    model = rf
else:
    model = xgb

# prediction
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

st.subheader("Prediction Result")
if prediction == 1:
    st.error(f"⚠️ High Risk of Heart Failure (Probability: {proba:.2f})" if proba else "⚠️ High Risk")
else:
    st.success(f"✅ Low Risk of Heart Failure (Probability: {proba:.2f})" if proba else "✅ Low Risk")

# SHAP explanation
if st.button("Explain Prediction"):
    st.subheader("🔍 Model Explainability with SHAP")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(input_df)

    # create plot
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches='tight', dpi=100)
