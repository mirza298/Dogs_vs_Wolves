from DogVsWolves import logger
from DogVsWolves.pipeline.step_01_data_ingestion import DataIngestionTrainingPipeline


STEP_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>> step {STEP_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> step {STEP_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e