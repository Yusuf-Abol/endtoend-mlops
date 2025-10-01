import os
from ccclassifier.constants import *
from ccclassifier.utils.common import read_yaml, create_directories, save_json
from ccclassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model   # from config.yaml
        base_model = self.params.base_model       # from params.yaml
        training = self.params.training           # from params.yaml

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_classes=base_model.num_classes,
            params_image_size=base_model.image_size,
            params_learning_rate=training.learning_rate,
            params_freeze_base=base_model.freeze_base,
            params_weights=base_model.weights,
            params_log_model_summary=config.log_model_summary
        )
        return prepare_base_model_config

    
    
    def get_training_config(self) -> TrainingConfig:
        training_config_yaml = self.config.training      # from config.yaml
        base_model_yaml = self.config.base_model         # from config.yaml
        training_params = self.params.training           # from params.yaml

        training_data = os.path.join(
            self.config.data_ingestion.unzip_dir,
            "Covid19-dataset"
        )

        create_directories([Path(training_config_yaml.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training_config_yaml.root_dir),
            trained_model_path=Path(training_config_yaml.trained_model_path),
            updated_base_model_path=Path(self.config.prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),

            # hyperparameters → from params.yaml
            params_epochs=training_params.epochs,
            params_batch_size=training_params.batch_size,
            params_is_augmentation=getattr(training_params, "augmentation", False),
            params_image_size=self.params.base_model.image_size,
            params_learning_rate=training_params.learning_rate,
            params_optimizer=training_params.optimizer,
            params_loss_fn=training_params.loss_fn,
            params_device=training_params.device,
            params_val_split=getattr(training_params, "val_split", 0.2),
            params_patience=getattr(training_params, "patience", 5),
            checkpoint_interval=getattr(training_params, "checkpoint_interval", 5),

            # num_classes from params.yaml.base_model
            num_classes=self.params.base_model.num_classes
        )

        return training_config


    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = self.config.evaluation
        training = self.config.training
        base_model = self.config.base_model

        return EvaluationConfig(
            model_path=Path(training.trained_model_path),
            training_data=Path(self.config.data_ingestion.unzip_dir) / "Covid19-dataset",
            image_size=base_model.image_size,
            batch_size=training.batch_size,
            device=training.device,
            num_classes=base_model.num_classes,
            metrics_file=Path(eval_config.metrics_file)
        )
