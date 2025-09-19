from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: List[int]
    params_learning_rate: float
    params_classes: int
    params_freeze_base: bool
    params_weights: str
    params_log_model_summary: bool 

    
@dataclass
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_learning_rate: float
    params_optimizer: str
    params_loss_fn: str
    params_device: str
    params_patience: int   #NEW
    num_classes: int
    checkpoint_interval: int  
    params_val_split: float = 0.2  

@dataclass
class EvaluationConfig:
    model_path: Path
    training_data: Path
    image_size: list
    batch_size: int
    device: str
    num_classes: int
    metrics_file: Path
