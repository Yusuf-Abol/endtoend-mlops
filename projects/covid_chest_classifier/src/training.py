from collections import Counter

import pandas as pd
import torch
from tqdm.notebook import tqdm
import mlflow
import mlflow.pytorch
from src.utils import early_stopping, checkpointing

def class_counts(dataset):
    c = Counter(x[1] for x in tqdm(dataset))
    try:
        class_to_index = dataset.class_to_idx
    except AttributeError:
        class_to_index = dataset.dataset.class_to_idx
    return pd.Series({cat: c[idx] for cat, idx in class_to_index.items()})


# ------------------ Your helper functions ------------------
def train_epoch(model, optimizer, loss_fn, data_loader, device="cpu"):
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(data_loader.dataset)

def score(model, data_loader, loss_fn, device="cpu"):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(torch.argmax(outputs, dim=1) == targets).item()

    n_samples = len(data_loader.dataset)
    avg_loss = total_loss / n_samples
    accuracy = total_correct / n_samples
    return avg_loss, accuracy

# ------------------ Train function ------------------
def train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=23,
    device="cpu",
    scheduler=None,
    checkpoint_path=None,
    early_stopping_fn=None,
):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    learning_rates = []
    best_val_loss = float("inf")
    early_stop_counter = 0
    last_epoch = 0

    # MLflow experiment
    with mlflow.start_run(run_name="image_classification_run"):

        # Log hyperparameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        mlflow.log_param("loss_fn", loss_fn.__class__.__name__)
        mlflow.log_param("scheduler", scheduler.__class__.__name__ if scheduler else "None")

        # Initial evaluation before training
        train_loss, train_acc = score(model, train_loader, loss_fn, device)
        val_loss, val_acc = score(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        mlflow.log_metric("train_loss", train_loss, step=0)
        mlflow.log_metric("train_acc", train_acc, step=0)
        mlflow.log_metric("val_loss", val_loss, step=0)
        mlflow.log_metric("val_acc", val_acc, step=0)

        print("Initial evaluation before training:")
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc*100:.2f}%")
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc*100:.2f}%")

        # Training loop
        for epoch in range(1, epochs + 1):
            last_epoch = epoch
            print(f"\nEpoch {epoch}/{epochs}")

            train_epoch(model, optimizer, loss_fn, train_loader, device)
            train_loss, train_acc = score(model, train_loader, loss_fn, device)
            val_loss, val_acc = score(model, val_loader, loss_fn, device)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            # Learning rate
            lr = optimizer.param_groups[0]["lr"]
            learning_rates.append(lr)
            mlflow.log_metric("learning_rate", lr, step=epoch)
            if scheduler:
                scheduler.step()

            print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc*100:.2f}%")
            print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc*100:.2f}%")
            print(f"Learning rate: {lr:.6f}")

            # Checkpointing
            if checkpoint_path:
                checkpointing(val_loss, best_val_loss, model, optimizer, checkpoint_path)

            # Early stopping
            if early_stopping_fn:
                early_stop_counter, stop = early_stopping_fn(val_loss, best_val_loss, early_stop_counter)
                if stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # Save final model to MLflow
        mlflow.pytorch.log_model(model, "model")

    return learning_rates, train_losses, val_losses, train_accuracies, val_accuracies, last_epoch
