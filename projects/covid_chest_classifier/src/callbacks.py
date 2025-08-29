import torch

def early_stopping(validation_loss, best_val_loss, counter, patience=5):
    """
    Simple early stopping: stop if validation_loss doesn't improve for `patience` epochs.
    """
    stop = False
    if validation_loss < best_val_loss:
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        stop = True

    return counter, stop


def checkpointing(validation_loss, best_val_loss, model, optimizer, save_path):
    """
    Save model checkpoint if validation_loss improves.
    """
    if validation_loss < best_val_loss:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": validation_loss,
            },
            save_path,
        )
        print(f"Checkpoint saved with validation loss {validation_loss:.4f}")