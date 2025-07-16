from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_classification_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Trains and evaluates a classification model with hyperparameter tuning.

    Args:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): The seed used by the random number generator. Defaults to 42.

    Returns:
        tuple: A tuple containing the trained model, test features, and test target.
    """
    try:
        logger.info("Starting classification model training...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }

        rfc = RandomForestClassifier(random_state=random_state)
        grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Classification model trained. Best parameters: {grid_search.best_params_}")
        logger.info(f"Test set accuracy: {accuracy:.4f}")

        return best_model, X_test, y_test

    except Exception as e:
        logger.error(f"An error occurred during classification model training: {e}")
        raise

def train_regression_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Trains and evaluates a regression model with hyperparameter tuning.

    Args:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): The seed used by the random number generator. Defaults to 42.

    Returns:
        tuple: A tuple containing the trained model, test features, and test target.
    """
    try:
        logger.info("Starting regression model training...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }

        rfr = RandomForestRegressor(random_state=random_state)
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        logger.info(f"Regression model trained. Best parameters: {grid_search.best_params_}")
        logger.info(f"Test set Mean Squared Error: {mse:.4f}")

        return best_model, X_test, y_test

    except Exception as e:
        logger.error(f"An error occurred during regression model training: {e}")
        raise

def train_clustering_model(X: pd.DataFrame, algorithm: str = 'kmeans', n_clusters: int = 3, **kwargs):
    """
    Trains a clustering model.

    Args:
        X (pd.DataFrame): The input features.
        algorithm (str, optional): The clustering algorithm to use ('kmeans', 'dbscan', 'hdbscan').
                                  Defaults to 'kmeans'.
        n_clusters (int, optional): The number of clusters for KMeans. Defaults to 3.
        **kwargs: Additional arguments for the clustering algorithms.

    Returns:
        object: The trained clustering model.
    """
    try:
        logger.info(f"Starting clustering model training with {algorithm}...")

        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, **kwargs)
        elif algorithm == 'dbscan':
            model = DBSCAN(**kwargs)
        elif algorithm == 'hdbscan':
            model = HDBSCAN(**kwargs)
        else:
            raise ValueError("Unsupported clustering algorithm. Choose from 'kmeans', 'dbscan', 'hdbscan'.")

        model.fit(X)

        if hasattr(model, 'labels_'):
            silhouette = silhouette_score(X, model.labels_)
            logger.info(f"Clustering model trained. Silhouette Score: {silhouette:.4f}")

        logger.info("Clustering model training completed.")
        return model

    except Exception as e:
        logger.error(f"An error occurred during clustering model training: {e}")
        raise
