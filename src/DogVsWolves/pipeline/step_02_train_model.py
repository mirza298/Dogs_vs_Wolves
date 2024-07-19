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

        init_time = time.time()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ConvolutionalNeuralNetwork(model_config, init_time)
        model = model.to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=model_config.params_learning_rate)
        print("Model arhitecture: ", model)

        prepare_data = PrepareData(model_config)
        train_loader, validation_loader, test_loader = prepare_data.split()

        model.fit(train_loader, validation_loader, device, loss_function, optimizer)
        test_loss, test_acc  = model.evaluate_model(test_loader, device, loss_function)
        print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f} ")
        model.plot_results()
        model.save_model()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> step {STEP_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> step {STEP_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e