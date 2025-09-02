import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train, X_test=None, sample_index=0, class_index=1, max_display=10):
    """
    Generate SHAP explanations for a trained model.

    Args:
        model: Trained model (e.g., RandomForestClassifier)
        X_train: Training features (pd.DataFrame)
        X_test: Test features (pd.DataFrame), optional
        sample_index: Which sample from X_test to explain in detail
        class_index: Which class to explain (0 = negative, 1 = positive for binary classification)
        max_display: Number of features to display in plots
    """
    
    data_for_explainer = X_train if X_test is None else X_test
    data_for_explainer = data_for_explainer.astype("float64")

    print(" Generating SHAP values...")
    explainer = shap.Explainer(model, data_for_explainer)
    shap_values = explainer(data_for_explainer)

    # global feature importance bar
    print("\n Global Feature Importance:")
    shap.summary_plot(shap_values, data_for_explainer, plot_type="bar", max_display=max_display)
    plt.show()

    # detailed impact (scatter summary)
    print("\n Detailed Feature Impact:")
    shap.summary_plot(shap_values, data_for_explainer, max_display=max_display)
    plt.show()

    if X_test is not None:
        print(f"\n Explanation for sample index {sample_index}, class {class_index}:")

        single_explanation = shap_values[sample_index]

        if len(single_explanation.values.shape) > 1:
            single_explanation.values = single_explanation.values[class_index]
            single_explanation.base_values = single_explanation.base_values[class_index]
            single_explanation.data = single_explanation.data  # keep feature values
            single_explanation.feature_names = shap_values.feature_names

        print("SHAP is using these features:", shap_values.feature_names)
        
        shap.plots.waterfall(single_explanation, max_display=max_display)
        plt.show()

    return shap_values
