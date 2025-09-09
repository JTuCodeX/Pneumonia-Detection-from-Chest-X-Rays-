import matplotlib.pyplot as plt

def plot_training(history_ce, history_focal):
    """
    Compare CrossEntropy vs FocalLoss histories
    history_ce and history_focal are dicts with keys:
    ['train_loss', 'train_acc', 'val_loss', 'val_acc']
    """
    epochs = range(1, len(history_ce["train_loss"]) + 1)

    plt.figure(figsize=(14, 6))

    # --- Loss plot ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_ce["train_loss"], "b-o", label="CE Train Loss")
    plt.plot(epochs, history_ce["val_loss"], "b--", label="CE Val Loss")
    plt.plot(epochs, history_focal["train_loss"], "r-o", label="Focal Train Loss")
    plt.plot(epochs, history_focal["val_loss"], "r--", label="Focal Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    # --- Accuracy plot ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_ce["train_acc"], "b-o", label="CE Train Acc")
    plt.plot(epochs, history_ce["val_acc"], "b--", label="CE Val Acc")
    plt.plot(epochs, history_focal["train_acc"], "r-o", label="Focal Train Acc")
    plt.plot(epochs, history_focal["val_acc"], "r--", label="Focal Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
