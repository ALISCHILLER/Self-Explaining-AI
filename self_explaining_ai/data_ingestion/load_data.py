import pandas as pd
from sqlalchemy import create_engine
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_from_csv(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        logger.info(f"Loading data from CSV file: {file_path}")
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data from CSV: {e}")
        raise

def load_from_excel(file_path: str) -> pd.DataFrame:
    """
    Loads data from an Excel file.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        logger.info(f"Loading data from Excel file: {file_path}")
        return pd.read_excel(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data from Excel: {e}")
        raise

def load_from_db(connection_string: str, query: str) -> pd.DataFrame:
    """
    Loads data from a database.

    Args:
        connection_string (str): The database connection string.
        query (str): The SQL query to execute.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        logger.info(f"Loading data from database with connection string: {connection_string}")
        engine = create_engine(connection_string)
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.error(f"An error occurred while loading data from database: {e}")
        raise

def load_from_api(url: str) -> pd.DataFrame:
    """
    Loads data from an API.

    Args:
        url (str): The API endpoint URL.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        logger.info(f"Loading data from API: {url}")
        response = requests.get(url)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while loading data from API: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
