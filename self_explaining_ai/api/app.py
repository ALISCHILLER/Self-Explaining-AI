from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import logging
from self_explaining_ai.pipelines import full_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Self-Explaining AI API",
    description="An API for running automated data analysis and getting explanations.",
    version="1.0.0"
)

class PipelineRequest(BaseModel):
    data_path: str
    numerical_features: list
    categorical_features: list
    target: str
    model_type: str = 'classification'

def run_pipeline_background(request: PipelineRequest):
    """Function to run the pipeline in the background."""
    try:
        logger.info("Starting background pipeline task.")
        full_pipeline.run_full_pipeline(
            data_path=request.data_path,
            numerical_features=request.numerical_features,
            categorical_features=request.categorical_features,
            target=request.target,
            model_type=request.model_type
        )
        logger.info("Background pipeline task finished.")
    except Exception as e:
        logger.error(f"Error in background pipeline task: {e}")

@app.post("/run-pipeline/", status_code=202)
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Triggers a full data science pipeline run.
    """
    try:
        logger.info(f"Received pipeline request for data: {request.data_path}")
        background_tasks.add_task(run_pipeline_background, request)
        return {"message": "Pipeline run has been started in the background."}
    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the Self-Explaining AI API!"}

def run_api(host: str = "0.0.0.0", port: int = 8000):
    """
    Runs the FastAPI application.
    """
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_api()
