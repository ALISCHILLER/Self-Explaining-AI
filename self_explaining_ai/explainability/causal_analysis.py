import dowhy
from dowhy import CausalModel
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def perform_causal_analysis(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: list,
    method_name: str = "backdoor.propensity_score_matching"
):
    """
    Performs causal analysis to estimate the effect of a treatment on an outcome.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for causal analysis.
        treatment (str): The name of the treatment variable.
        outcome (str): The name of the outcome variable.
        common_causes (list): A list of names of common cause variables.
        method_name (str, optional): The causal estimation method.
                                     Defaults to "backdoor.propensity_score_matching".

    Returns:
        CausalEstimate: An object containing the causal estimate and other details.
    """
    try:
        logger.info("Starting causal analysis...")
        logger.info(f"Treatment: {treatment}, Outcome: {outcome}")

        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes
        )

        # 1. Identify the causal effect
        identified_estimand = model.identify_effect()
        logger.info(f"Causal estimand identified: {identified_estimand}")

        # 2. Estimate the causal effect
        causal_estimate = model.estimate_effect(
            identified_estimand,
            method_name=method_name
        )
        logger.info(f"Causal estimate calculated: {causal_estimate.value}")

        # 3. Refute the estimate (optional but recommended)
        refutation = model.refute_estimate(
            identified_estimand,
            causal_estimate,
            method_name="random_common_cause"
        )
        logger.info(f"Refutation result: {refutation}")

        logger.info("Causal analysis completed.")
        return causal_estimate

    except Exception as e:
        logger.error(f"An error occurred during causal analysis: {e}")
        raise
