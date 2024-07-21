from DogVsWolves.config.configuration import ConfigurationManager
from DogVsWolves.components.model_manager import *
from DogVsWolves import logger
from DogVsWolves.utils.common import json, dataclass_to_dict, create_directories

STEP_NAME = "Data Ingestion"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_config = config.get_train_validation_test_config()

        # Prepare data
        prepare_data = PrepareData(model_config)
        train_loader, validation_loader, test_loader = prepare_data.split()

        # Initalize model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ConvolutionalNeuralNetwork()
        model = model.to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=model_config.params_learning_rate)
        print("Model arhitecture: ", model)

        # Train model
        model.fit(model_config.params_epochs, train_loader, validation_loader, device, loss_function, optimizer)

        # Evaluate model on test data
        test_loss, test_acc, y_pred, y_true  = model.evaluate_model(test_loader, device, loss_function)
        print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f} ")

        # Plot train results and Confusion matrix for test results
        model_dir = str(model_config.trained_model_path) + f"_{int(model.init_time)}"
        create_directories([model_dir])
        model.plot_results(model_dir)
        plot_confusion_matrix(y_true, y_pred, prepare_data.data.image_labels_mapping, model_dir)

        # Save model and config
        model.save_model(model_dir, model_config.trained_model_inference_path)
        with open(f'{model_dir}/model_config.json', 'w') as f:
            json.dump(json.dumps(dataclass_to_dict(model_config)), f)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> step {STEP_NAME} started <<<<<<")
        model_training = ModelTrainingPipeline()
        model_training.main()
        logger.info(f">>>>>> step {STEP_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e