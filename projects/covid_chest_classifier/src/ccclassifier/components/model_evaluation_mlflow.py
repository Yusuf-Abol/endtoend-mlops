import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import mlflow
import mlflow.pytorch
from pathlib import Path
from urllib.parse import urlparse
from ccclassifier.utils.common import read_yaml, create_directories, save_json

from ccclassifier.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _valid_generator(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        full_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=transform
        )
        
        train_size = int(0.7 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        self.valid_generator = DataLoader(
            val_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )

    @staticmethod
    def load_model(path: Path) -> nn.Module:
        # Load state dict and create model architecture
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        
        # You need to recreate your model architecture here
        # Example for ResNet18:
        from torchvision.models import resnet18
        model = resnet18(num_classes=3)  # Adjust num_classes as needed
        model.load_state_dict(state_dict)
        
        return model
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self.model.to(self.device)
        self._valid_generator()
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.valid_generator:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(self.valid_generator)
        accuracy = correct / total
        self.score = [avg_loss, accuracy]
        self.save_score()
    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="ResNet18Model")
            else:
                mlflow.pytorch.log_model(self.model, "model")

