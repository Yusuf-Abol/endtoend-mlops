import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

# 1. Start MLflow run
mlflow.set_experiment("pytorch_image_classification")

with mlflow.start_run():
    # --- Data ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_data = datasets.ImageFolder("data/train", transform=transform)
    test_data  = datasets.ImageFolder("data/test", transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

    # --- Model ---
    model = models.resnet18(pretrained=False, num_classes=len(train_data.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Log hyperparams
    mlflow.log_param("model", "resnet18")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("lr", 0.001)
    mlflow.log_param("batch_size", 32)

    # --- Training Loop ---
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Log metrics
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

    # --- Evaluation ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    mlflow.log_metric("test_accuracy", accuracy)

    # --- Save model ---
    mlflow.pytorch.log_model(model, "model")
