import shap
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explain_model_with_shap(model, X: pd.DataFrame, feature_names: list = None):
    """
    Explains a model's predictions using SHAP (SHapley Additive exPlanations).

    Args:
        model: The trained model to be explained.
        X (pd.DataFrame): The input features used for explanation.
        feature_names (list, optional): A list of feature names. If None, uses X.columns.

    Returns:
        tuple: A tuple containing the SHAP explainer and SHAP values.
    """
    try:
        logger.info("Starting model explanation with SHAP...")

        if feature_names:
            X.columns = feature_names

        # SHAP is optimized for tree-based models like RandomForest
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X)

        shap_values = explainer.shap_values(X)

        logger.info("SHAP explanation completed.")
        return explainer, shap_values

    except Exception as e:
        logger.error(f"An error occurred during SHAP explanation: {e}")
        raise

def plot_shap_summary(shap_values, X: pd.DataFrame, plot_type: str = "dot"):
    """
    Generates a SHAP summary plot.

    Args:
        shap_values: The SHAP values obtained from the explainer.
        X (pd.DataFrame): The input features.
        plot_type (str, optional): The type of SHAP summary plot. Defaults to "dot".
    """
    try:
        logger.info(f"Generating SHAP summary plot of type: {plot_type}")
        shap.summary_plot(shap_values, X, plot_type=plot_type, show=False)

    except Exception as e:
        logger.error(f"An error occurred while generating the SHAP plot: {e}")
        raise
