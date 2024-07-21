import os
from DogVsWolves.constants import *
from DogVsWolves.utils.common import read_yaml, create_directories
from DogVsWolves.entity.config_entity import (DataIngestionConfig, TrainValidationTestConfig, EvaluationConfig)


class ConfigurationManager:
    def __init__(self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=Path(config.unzip_dir) 
        )

        return data_ingestion_config


    def get_train_validation_test_config(self) -> TrainValidationTestConfig:
        config = self.config.model
        data = os.path.join(self.config.data_ingestion.unzip_dir, "data")

        create_directories([config.root_dir])

        train_validation_test_config = TrainValidationTestConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            trained_model_inference_path=Path(config.trained_model_inference_path),
            data=Path(data),
            params_train_size= self.params.TRAIN_SIZE,
            params_validation_size=self.params.VALIDATION_SIZE,
            params_test_size=self.params.TEST_SIZE,
            params_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_epochs=self.params.EPOCHS,
            params_learning_rate=self.params.LEARNING_RATE
        )

        return train_validation_test_config
    

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation

        evaluation_config = EvaluationConfig(
            data_dir=Path(config.data_dir),
            trained_model_inference_path=Path(config.trained_model_inference_path),
            params_image_size=self.params.IMAGE_SIZE
        )

        return evaluation_config