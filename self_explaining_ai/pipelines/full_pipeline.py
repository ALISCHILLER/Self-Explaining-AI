from self_explaining_ai.data_ingestion import load_data
from self_explaining_ai.profiling import profiling_report
from self_explaining_ai.features import feature_builder
from self_explaining_ai.models import predictor, anomaly_detector
from self_explaining_ai.explainability import shap_explainer, causal_analysis
from self_explaining_ai.nlg import generate_report
from self_explaining_ai.retraining import drift_monitor
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_full_pipeline(
    data_path: str,
    numerical_features: list,
    categorical_features: list,
    target: str,
    model_type: str = 'classification',
    output_dir: str = 'output'
):
    """
    Runs the full data science pipeline from data ingestion to report generation.

    Args:
        data_path (str): Path to the input data file (CSV).
        numerical_features (list): List of numerical feature column names.
        categorical_features (list): List of categorical feature column names.
        target (str): The name of the target variable.
        model_type (str, optional): Type of model ('classification' or 'regression'). Defaults to 'classification'.
        output_dir (str, optional): Directory to save outputs. Defaults to 'output'.
    """
    try:
        logger.info("Starting the full data science pipeline...")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load Data
        df = load_data.load_from_csv(data_path)

        # 2. Generate Profiling Report
        profiling_report_path = os.path.join(output_dir, "data_profile.html")
        profiling_report.generate_profiling_report(df, "Sales Data Analysis", profiling_report_path)

        # 3. Feature Engineering
        feature_pipeline = feature_builder.create_feature_pipeline(numerical_features, categorical_features)
        X = df.drop(target, axis=1)
        y = df[target]
        X_processed = feature_pipeline.fit_transform(X)
        X_processed = pd.DataFrame(X_processed, columns=feature_pipeline.get_feature_names_out())

        # 4. Train Model
        if model_type == 'classification':
            model, X_test, y_test = predictor.train_classification_model(X_processed, y)
        elif model_type == 'regression':
            model, X_test, y_test = predictor.train_regression_model(X_processed, y)
        else:
            raise ValueError("Invalid model type specified.")

        # 5. Explain Model
        explainer, shap_values = shap_explainer.explain_model_with_shap(model, X_test)
        shap_summary_plot_path = os.path.join(output_dir, "shap_summary.png")
        shap_explainer.plot_shap_summary(shap_values, X_test)
        # Add saving plot logic if needed, e.g., using matplotlib.pyplot.savefig

        # 6. Generate NLG Report
        # This is a simplified example. A real report would be more detailed.
        report_data = {
            "model_type": model_type,
            "num_features": len(numerical_features) + len(categorical_features),
            "sample_size": len(df),
            "target_variable": target
        }
        # Assume a template file 'report_template.txt' exists in a 'templates' directory
        # report_str = generate_report.generate_report_from_template("report_template.txt", report_data)
        # report_path = os.path.join(output_dir, "final_report.txt")
        # generate_report.save_report_to_file(report_str, report_path)

        logger.info("Full pipeline completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred in the full pipeline: {e}")
        raise

if __name__ == '__main__':
    # Example usage (requires a sample CSV file)
    # Create a dummy CSV for testing
    data = {
        'numeric_feat1': [1, 2, 3, 4, 5],
        'numeric_feat2': [10, 20, 25, 30, 35],
        'categorical_feat': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 0]
    }
    dummy_df = pd.DataFrame(data)
    dummy_csv_path = "dummy_data.csv"
    dummy_df.to_csv(dummy_csv_path, index=False)

    run_full_pipeline(
        data_path=dummy_csv_path,
        numerical_features=['numeric_feat1', 'numeric_feat2'],
        categorical_features=['categorical_feat'],
        target='target',
        model_type='classification'
    )
