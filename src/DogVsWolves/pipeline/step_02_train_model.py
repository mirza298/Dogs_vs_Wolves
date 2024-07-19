from DogVsWolves.config.configuration import ConfigurationManager
from DogVsWolves.components.model_manager import *
from DogVsWolves import logger
import time

STEP_NAME = "Data Ingestion"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_config = config.get_model_config()

        model = ConvolutionalNeuralNetwork(model_config)
        print("Model arhitecture: ", model)

        prepare_data = PrepareData(model_config)
        train_loader, validation_loader, test_loader = prepare_data.split()

        init_time = time.time()
        train_procedure = ModelTrainer(model_config, model)
        train_results = train_procedure.train_model(train_loader, validation_loader)
        test_results = train_procedure.evaluate_model(test_loader)
        train_procedure.plot_results(train_results, init_time, model_config.trained_model_path)
        train_procedure.save_model(init_time)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> step {STEP_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> step {STEP_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e