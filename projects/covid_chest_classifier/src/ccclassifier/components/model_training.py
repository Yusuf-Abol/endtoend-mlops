import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
from torch.utils.data import random_split

from ccclassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config.params_device)

        # --- Build base ResNet18 ---
        self.model = models.resnet18(weights=None)   # fresh model
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.config.num_classes)

        # --- Load saved weights from stage 02 ---
        state_dict = torch.load(
            self.config.updated_base_model_path,
            map_location=self.device,
            weights_only=True   # fix warning
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)

        # --- Loss function ---
        if self.config.params_loss_fn == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config.params_loss_fn}")

        # --- Optimizer ---
        if self.config.params_optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.params_learning_rate
            )
        elif self.config.params_optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.params_learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.params_optimizer}")

        # --- Data transforms ---
        train_transforms = transforms.Compose([
            transforms.Resize((self.config.params_image_size[0], self.config.params_image_size[1])),
            transforms.RandomHorizontalFlip() if self.config.params_is_augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((self.config.params_image_size[0], self.config.params_image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # --- Datasets & loaders ---
        full_train_dataset = datasets.ImageFolder(
            root=str(self.config.training_data / "train"),
            transform=train_transforms
        )

        # Split into train/val
        val_size = int(0.2 * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

        # Final test set (untouched)
        test_dataset = datasets.ImageFolder(
            root=str(self.config.training_data / "test"),
            transform=val_transforms
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )

    # Helper methods
    def get_base_model(self):
        return self.model

    def train_valid_generator(self):
        return self.train_loader, self.val_loader
    
    start_epoch = 0


    # Training loop
    def train(self):
        best_val_loss = float("inf")
        early_stop_counter = 0
        start_epoch = 0   # >> always define a default

        # --- Resume from latest checkpoint if available ---
        import glob
        checkpoints = glob.glob(str(self.config.root_dir / "checkpoint_epoch_*.pth"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)  # pick latest
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float("inf"))  # resume best loss
            early_stop_counter = checkpoint.get('early_stop_counter', 0)   # resume patience
            print(f">> Resumed training from {latest_checkpoint} (epoch {start_epoch})")

        # --- Training loop ---
        for epoch in range(start_epoch, self.config.params_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.params_epochs}")
            print("-" * 30)

            # --- Training phase ---
            self.model.train()
            train_loss, correct, total = 0, 0, 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss /= total
            train_acc = 100. * correct / total

            # --- Validation phase ---
            self.model.eval()
            val_loss, correct, total = 0, 0, 0

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_loss /= total
            val_acc = 100. * correct / total

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # --- Save best model ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.trained_model_path)
                print(f">>> Saved best model weights to {self.config.trained_model_path}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"No improvement. Early stop counter: {early_stop_counter}/{self.config.params_patience}")

                if early_stop_counter >= self.config.params_patience:
                    print(">> Early stopping triggered.")
                    break

            # --- Save checkpoint every N epochs ---
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                checkpoint_path = self.config.root_dir / f"checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                    'best_val_loss': best_val_loss,        #  save best loss
                    'early_stop_counter': early_stop_counter,  # save patience counter
                }, checkpoint_path)
                print(f">> Checkpoint saved at {checkpoint_path}")

