import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train, max_display=10):
    """
    Explain model predictions using SHAP values.
    Args:
        model: Trained model
        X_train: Training data (pd.DataFrame)
        max_display: Number of features to display in summary
    """
    # Create SHAP explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # Summary plot
    shap.summary_plot(shap_values, X_train, max_display=max_display, plot_type="bar")
    plt.show()

    return shap_values
