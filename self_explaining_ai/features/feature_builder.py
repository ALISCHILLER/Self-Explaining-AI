from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_feature_pipeline(
    numerical_features: list,
    categorical_features: list,
    ordinal_features: list = None,
    imputation_strategy: str = 'median'
) -> Pipeline:
    """
    Creates a feature engineering pipeline for preprocessing numerical,
    categorical, and optional ordinal features.

    Args:
        numerical_features (list): A list of column names for numerical features.
        categorical_features (list): A list of column names for categorical features.
        ordinal_features (list, optional): A list of column names for ordinal features. Defaults to None.
        imputation_strategy (str, optional): The imputation strategy for missing values.
                                             Can be 'mean', 'median', or 'most_frequent'.
                                             Defaults to 'median'.

    Returns:
        Pipeline: A scikit-learn pipeline for feature processing.
    """
    try:
        logger.info("Creating feature engineering pipeline...")

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=imputation_strategy)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        transformers = [
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]

        if ordinal_features:
            ordinal_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder())
            ])
            transformers.append(('ord', ordinal_transformer, ordinal_features))

        preprocessor = ColumnTransformer(transformers=transformers)

        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        logger.info("Feature engineering pipeline created successfully.")
        return pipeline

    except Exception as e:
        logger.error(f"An error occurred while creating the feature pipeline: {e}")
        raise
