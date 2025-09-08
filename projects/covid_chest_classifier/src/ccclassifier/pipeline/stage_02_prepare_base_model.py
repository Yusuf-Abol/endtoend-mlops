from ccclassifier.config.configuration import ConfigurationManager
from ccclassifier.components.prepare_base_model import PrepareBaseModel
from ccclassifier import logger

STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()

        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)

        # Step 1: Load base model
        model = prepare_base_model.get_base_model()

        # Step 2: (optional) log summary
        prepare_base_model.log_model_summary(model)

        # Step 3: Update full model
        prepare_base_model.update_base_model()

        logger.info("Base model prepared and updated successfully.")


if __name__ == '__main__':
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
