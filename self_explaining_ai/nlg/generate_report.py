from jinja2 import Environment, FileSystemLoader
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_report_from_template(template_name: str, data: dict, template_dir: str = 'templates') -> str:
    """
    Generates a report from a Jinja2 template file.

    Args:
        template_name (str): The name of the template file.
        data (dict): A dictionary of data to be rendered in the template.
        template_dir (str, optional): The directory where templates are stored.
                                      Defaults to 'templates'.

    Returns:
        str: The rendered report as a string.
    """
    try:
        logger.info(f"Generating report from template: {template_name}")

        # Set up the Jinja2 environment
        env = Environment(loader=FileSystemLoader(searchpath=template_dir))
        template = env.get_template(template_name)

        # Render the template with the provided data
        report = template.render(data)

        logger.info("Report generated successfully.")
        return report

    except Exception as e:
        logger.error(f"An error occurred while generating the report: {e}")
        raise

def save_report_to_file(report: str, output_path: str):
    """
    Saves a report to a file.

    Args:
        report (str): The report content to be saved.
        output_path (str): The path to the output file.
    """
    try:
        logger.info(f"Saving report to: {output_path}")
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info("Report saved successfully.")

    except Exception as e:
        logger.error(f"An error occurred while saving the report: {e}")
        raise
