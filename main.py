from DogVsWolves import logger
from DogVsWolves.pipeline.step_01_data_ingestion import DataIngestionTrainingPipeline
from DogVsWolves.pipeline.step_02_train_model import ModelTrainingPipeline


STEP_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>> step {STEP_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> step {STEP_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STEP_NAME = "Model Training"

try:
    logger.info(f">>>>>> step {STEP_NAME} started <<<<<<")
    model_training = ModelTrainingPipeline()
    model_training.main()
    logger.info(f">>>>>> step {STEP_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e