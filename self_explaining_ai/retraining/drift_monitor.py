from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column_mapping: dict = None
) -> dict:
    """
    Checks for data drift between a reference and current dataset.

    Args:
        reference_df (pd.DataFrame): The reference (e.g., training) dataset.
        current_df (pd.DataFrame): The current (e.g., production) dataset.
        column_mapping (dict, optional): A mapping of columns for Evidently.

    Returns:
        dict: A dictionary containing the drift report.
    """
    try:
        logger.info("Checking for data drift...")

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

        drift_report = report.as_dict()

        if drift_report['metrics'][0]['result']['dataset_drift']:
            logger.warning("Data drift detected!")
        else:
            logger.info("No data drift detected.")

        return drift_report

    except Exception as e:
        logger.error(f"An error occurred during data drift check: {e}")
        raise

def check_target_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column_mapping: dict = None
) -> dict:
    """
    Checks for target drift and model performance for classification tasks.

    Args:
        reference_df (pd.DataFrame): The reference dataset with target and prediction.
        current_df (pd.DataFrame): The current dataset with target and prediction.
        column_mapping (dict, optional): A mapping of columns for Evidently.

    Returns:
        dict: A dictionary containing the target drift report.
    """
    try:
        logger.info("Checking for target drift (classification)...")

        report = Report(metrics=[TargetDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

        return report.as_dict()

    except Exception as e:
        logger.error(f"An error occurred during target drift check: {e}")
        raise

def check_regression_performance(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column_mapping: dict = None
) -> dict:
    """
    Checks for model performance changes in regression tasks.

    Args:
        reference_df (pd.DataFrame): The reference dataset with target and prediction.
        current_df (pd.DataFrame): The current dataset with target and prediction.
        column_mapping (dict, optional): A mapping of columns for Evidently.

    Returns:
        dict: A dictionary containing the regression performance report.
    """
    try:
        logger.info("Checking for regression model performance changes...")

        report = Report(metrics=[RegressionPreset()])
        report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

        return report.as_dict()

    except Exception as e:
        logger.error(f"An error occurred during regression performance check: {e}")
        raise
