import os
from DogVsWolves.constants import *
from DogVsWolves.utils.common import read_yaml, create_directories
from DogVsWolves.entity.config_entity import (DataIngestionConfig, ModelConfig)


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


    def get_model_config(self) -> ModelConfig:
        config = self.config.model
        data = os.path.join(self.config.data_ingestion.unzip_dir, "data")

        create_directories([config.root_dir])

        model_config = ModelConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            data=Path(data),
            params_train_size= self.params.TRAIN_SIZE,
            params_validation_size=self.params.VALIDATION_SIZE,
            params_test_size=self.params.TEST_SIZE,
            params_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_epochs=self.params.EPOCHS,
            params_learning_rate=self.params.LEARNING_RATE,
            params_channels=self.params.CHANNELS,
            params_kernel_size_cl=self.params.KERNEL_SIZE_CL,
            params_optimizer=self.params.OPTIMIZER
        )

        return model_config