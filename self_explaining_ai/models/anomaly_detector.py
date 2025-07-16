from sklearn.ensemble import IsolationForest
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_anomaly_detector(X: pd.DataFrame, contamination: float = 'auto', random_state: int = 42, **kwargs):
    """
    Trains an anomaly detection model using Isolation Forest.

    Args:
        X (pd.DataFrame): The input features.
        contamination (float or 'auto', optional): The proportion of outliers in the data set.
                                                   Defaults to 'auto'.
        random_state (int, optional): The seed used by the random number generator.
                                      Defaults to 42.
        **kwargs: Additional arguments for the IsolationForest model.

    Returns:
        object: The trained IsolationForest model.
    """
    try:
        logger.info("Starting anomaly detection model training...")

        model = IsolationForest(contamination=contamination, random_state=random_state, **kwargs)
        model.fit(X)

        logger.info("Anomaly detection model training completed.")
        return model

    except Exception as e:
        logger.error(f"An error occurred during anomaly detection model training: {e}")
        raise

def predict_anomalies(model, X: pd.DataFrame) -> pd.Series:
    """
    Predicts anomalies using a trained Isolation Forest model.

    Args:
        model: The trained IsolationForest model.
        X (pd.DataFrame): The input features for prediction.

    Returns:
        pd.Series: A pandas Series with predictions (-1 for anomalies, 1 for inliers).
    """
    try:
        logger.info("Predicting anomalies...")
        predictions = model.predict(X)
        logger.info("Anomaly prediction completed.")
        return pd.Series(predictions, index=X.index)

    except Exception as e:
        logger.error(f"An error occurred during anomaly prediction: {e}")
        raise
