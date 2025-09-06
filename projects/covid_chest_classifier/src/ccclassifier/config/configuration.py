from ccclassifier.constants import *
from ccclassifier.utils.common import read_yaml, create_directories
from ccclassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig)
                                                #TrainingConfig,
                                                #EvaluationConfig)

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
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.base_model.image_size,
            params_learning_rate=self.params.training.learning_rate,
            params_weights=self.params.base_model.weights,
            params_classes=self.params.base_model.num_classes,
            params_freeze_base=self.params.base_model.freeze_base,
        )

        return prepare_base_model_config