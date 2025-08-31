import pandas as pd
from src.train_model import (
    split_data, train_logistic_regression, train_random_forest, train_xgboost
)
from src.evaluate_model import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve
)
from src.explainability import explain_model

# === Load Data ===
df = pd.read_csv("data/heart_failure_clinical_records.csv")

# === Split Data ===
X_train, X_test, y_train, y_test = split_data(df)

# === Train Models ===
logreg = train_logistic_regression(X_train, y_train)
rf = train_random_forest(X_train, y_train)
xgb = train_xgboost(X_train, y_train)

# === Evaluate ===
evaluate_model(logreg, X_test, y_test, name="Logistic Regression")
plot_confusion_matrix(logreg, X_test, y_test, name="Logistic Regression")
plot_roc_curve(logreg, X_test, y_test, name="Logistic Regression")

evaluate_model(rf, X_test, y_test, name="Random Forest")
plot_confusion_matrix(rf, X_test, y_test, name="Random Forest")
plot_roc_curve(rf, X_test, y_test, name="Random Forest")

evaluate_model(xgb, X_test, y_test, name="XGBoost")
plot_confusion_matrix(xgb, X_test, y_test, name="XGBoost")
plot_roc_curve(xgb, X_test, y_test, name="XGBoost")

# === Explainability ===
print("\nSHAP Feature Importance (XGBoost):")
shap_values = explain_model(xgb, X_train)
