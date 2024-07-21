from DogVsWolves.config.configuration import ConfigurationManager
from DogVsWolves.pipeline.external_data_evaluation import ExternalDataEvaluation
from DogVsWolves import logger

try:
    logger.info(f">>>>>> Evaluation started <<<<<<")
    config = ConfigurationManager()
    evaluation_config = config.get_evaluation_config()

    model_eval = ExternalDataEvaluation(evaluation_config)
    model_eval.evaluate()

    logger.info(f"\n>>>>>> Evaluation completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e