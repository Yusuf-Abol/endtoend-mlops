import yaml
import urllib.request as request
from zipfile import ZipFile
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

from ccclassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def get_base_model(self):
        """Load pretrained ResNet backbone"""
        weights = None
        if self.config.params_weights:  # e.g. "IMAGENET1K_V1"
            weights = getattr(models, "ResNet18_Weights").IMAGENET1K_V1

        self.model = models.resnet18(weights=weights)

        # Save backbone weights (state_dict only)
        self.save_model(self.config.base_model_path, self.model.state_dict())
        return self.model

    def log_model_summary(self, model):
        """Logs the model architecture summary if enabled in config"""
        if self.config.params_log_model_summary:
            summary(
                model,
                input_size=(1, 3, *self.config.params_image_size),  # batch_size=1
                col_names=["input_size", "output_size", "num_params", "trainable"],
            )

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, learning_rate):
        """Freeze layers + replace classifier"""
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, classes)

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        return model, optimizer, loss_fn

    def update_base_model(self):
        """Prepare full model for training"""
        self.model, self.optimizer, self.loss_fn = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=self.config.params_freeze_base,
            learning_rate=self.config.params_learning_rate
        )

        # Save updated model weights (state_dict)
        self.save_model(self.config.updated_base_model_path, self.model.state_dict())

    @staticmethod
    def save_model(path: Path, obj):
        """Save model weights or optimizer state"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, path)
