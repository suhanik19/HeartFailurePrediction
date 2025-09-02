import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sys, os

# Fix imports so app can find src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train_model import (
    split_data, train_logistic_regression, train_random_forest, train_neural_net
)
from src.evaluate_model import evaluate_model
from src.explainability import explain_model
from src.data_processing import preprocess_data


# load and train models
@st.cache_resource
def load_models():
    df = pd.read_csv("data/heart_failure_clinical_records.csv")
    df_processed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_processed, target_col="HeartDisease")

    logreg = train_logistic_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)
    nn = train_neural_net(X_train, y_train)

    return df, df_processed, X_train, X_test, y_train, y_test, logreg, rf, nn


# streamlit app
st.title("Heart Disease Prediction App")
st.write("Enter patient information to predict risk of heart disease.")
df, df_processed, X_train, X_test, y_train, y_test, logreg, rf, nn = load_models()


# sidebar - user inputs
st.sidebar.header("Patient Data Input")

def user_input_features():
    data = {}
    for col in df.drop(columns=["HeartDisease"]).columns:
        if df[col].dtype == "object":  # categorical
            data[col] = st.sidebar.selectbox(col, sorted(df[col].unique()))
        else:  # numeric
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            default_val = float(df[col].median())
            data[col] = st.sidebar.slider(col, min_val, max_val, default_val)
    return pd.DataFrame([data])

input_df = user_input_features()

input_processed = preprocess_data(input_df)

missing_cols = set(X_train.columns) - set(input_processed.columns)
for col in missing_cols:
    input_processed[col] = 0
input_processed = input_processed[X_train.columns]


# model choice
model_choice = st.radio("Choose Model", ["Logistic Regression", "Random Forest", "Neural Net"])

if model_choice == "Logistic Regression":
    model = logreg
elif model_choice == "Neural Net":
    model = nn
else:
    model = rf

if model_choice == "Random Forest":
    if st.button("Explain Prediction"):
        st.subheader("🔍 Model Explainability with SHAP")
        # SHAP explanation
        input_processed = preprocess_data(input_df)

        missing_cols = set(X_train.columns) - set(input_processed.columns)
        for col in missing_cols:
            input_processed[col] = 0
        input_processed = input_processed[X_train.columns]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_processed)

        if isinstance(shap_values, list):
            shap_values_to_use = shap_values[1][0, :]
            base_value = explainer.expected_value[1]
        else:
            if shap_values.ndim == 3:
                shap_values_to_use = shap_values[0, :, 1]
                base_value = explainer.expected_value[1]
            else:
                shap_values_to_use = shap_values[0, :]
                base_value = explainer.expected_value

        single_explanation = shap.Explanation(
            values=shap_values_to_use,
            base_values=base_value,
            data=input_processed.iloc[0].values,
            feature_names=input_processed.columns,
        )

        # plot waterfall
        shap.plots.waterfall(single_explanation, max_display=10)
        st.pyplot(bbox_inches="tight", dpi=100)

        # plot global bar chart
        st.subheader("Global Feature Importance")
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_train, plot_type="bar", max_display=10, show=False)
        else:
            if shap_values.ndim == 3:
                shap.summary_plot(shap_values[:, :, 1], X_train, plot_type="bar", max_display=10, show=False)
            else:
                shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=10, show=False)
        st.pyplot(bbox_inches="tight", dpi=100)
    else:
        st.info("ℹ️ Model explainability with SHAP is currently only supported for Random Forest.")


# prediction
prediction = model.predict(input_processed)[0]
proba = model.predict_proba(input_processed)[0][1] if hasattr(model, "predict_proba") else None

st.subheader("Prediction Result")
if prediction == 1:
    st.error(f"⚠️ High Risk of Heart Disease (Probability: {proba:.2f})" if proba is not None else "⚠️ High Risk")
else:
    st.success(f"✅ Low Risk of Heart Disease (Probability: {proba:.2f})" if proba is not None else "✅ Low Risk")



