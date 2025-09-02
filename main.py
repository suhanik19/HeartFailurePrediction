import pandas as pd
import numpy as np
import shap

from src.data_processing import preprocess_data
from src.train_model import (
    split_data, train_logistic_regression, train_random_forest, train_neural_net
)
from src.evaluate_model import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve
)
from src.explainability import explain_model

# Load Data
df = pd.read_csv("data/heart_failure_clinical_records.csv")

# Preprocess Data
df = preprocess_data(df)

# Split Data 
X_train, X_test, y_train, y_test = split_data(df)

# Train Models 
logreg = train_logistic_regression(X_train, y_train)
rf = train_random_forest(X_train, y_train)
nn = train_neural_net(X_train, y_train)

# Evaluate 
evaluate_model(logreg, X_test, y_test, name="Logistic Regression")
plot_confusion_matrix(logreg, X_test, y_test, name="Logistic Regression")
plot_roc_curve(logreg, X_test, y_test, name="Logistic Regression")

evaluate_model(rf, X_test, y_test, name="Random Forest")
plot_confusion_matrix(rf, X_test, y_test, name="Random Forest")
plot_roc_curve(rf, X_test, y_test, name="Random Forest")

evaluate_model(nn, X_test, y_test, name="Neural Net")
plot_confusion_matrix(nn, X_test, y_test, name="Neural Net")
plot_roc_curve(nn, X_test, y_test, name="Neural Net")



# Explainability for Random Forest
print("\nSHAP Feature Importance (Random Forest):")
shap_values = explain_model(rf, X_train, X_test, sample_index=0)

random_indices = np.random.choice(X_test.index, size=5, replace=False)

for i, idx in enumerate(random_indices, start=1):
    print(f"\n Waterfall plot for random patient {i} (index {idx}):")
    
    # Get the SHAP explanation for this patient
    single_explanation = shap_values[X_test.index.get_loc(idx)]
    
    if len(single_explanation.values.shape) > 1:
        single_explanation.values = single_explanation.values[1]
        single_explanation.base_values = single_explanation.base_values[1]
        single_explanation.data = single_explanation.data
        single_explanation.feature_names = shap_values.feature_names
    
    # Plot waterfall
    shap.plots.waterfall(single_explanation, max_display=10)
