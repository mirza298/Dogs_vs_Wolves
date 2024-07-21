from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class TrainValidationTestConfig:
    root_dir: Path
    trained_model_path: Path
    trained_model_inference_path: Path
    data: Path
    params_train_size: float
    params_validation_size: float
    params_test_size: float
    params_augmentation: bool
    params_image_size: int
    params_batch_size: int
    params_epochs: int
    params_learning_rate: float
    params_tolerance: int
    params_min_delta: float


@dataclass(frozen=True)
class EvaluationConfig:
    data_dir: Path
    trained_model_inference_path: Path
    params_image_size: int