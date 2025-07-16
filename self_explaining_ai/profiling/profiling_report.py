import pandas as pd
from ydata_profiling import ProfileReport
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_profiling_report(df: pd.DataFrame, title: str, output_file: str):
    """
    Generates a data profiling report and saves it to an HTML file.

    Args:
        df (pd.DataFrame): The DataFrame to be profiled.
        title (str): The title of the report.
        output_file (str): The path to the output HTML file.
    """
    try:
        logger.info(f"Generating data profiling report with title: {title}")
        profile = ProfileReport(df, title=title, explorative=True)
        profile.to_file(output_file)
        logger.info(f"Profiling report saved to: {output_file}")
    except Exception as e:
        logger.error(f"An error occurred while generating the profiling report: {e}")
        raise
