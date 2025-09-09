import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from data_loader import train_loader, val_loader, train_dataset
from visualize import plot_training

# --- Device configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Load pretrained ResNet18 ---
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all layers (optional: fine-tuning later)
for param in model.parameters():
    param.requires_grad = False

# Replace final fully connected layer for 2 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

# --- Loss & Optimizer ---
# You can switch between CrossEntropyLoss and FocalLoss here
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# --- Focal Loss implementation ---
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Validation ---
def validate():
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    return avg_loss, acc

# --- Training Loop ---
def train_model(num_epochs=10, loss_method=''):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        # Validation
        val_loss, val_acc = validate()

        # Save history
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Save trained model
    if loss_method == 'CE':
        torch.save(model.state_dict(), "pneumonia_resnet18_CE.pth")
        print("Model saved to pneumonia_resnet18_CE.pth")

    else:
        torch.save(model.state_dict(), "pneumonia_resnet18_FOCAL.pth")
        print("Model saved to pneumonia_resnet18_FOCAL.pth")


    return history

# --- Run training ---
if __name__ == "__main__":
    print("Classes:", train_dataset.classes)  # ['NORMAL', 'PNEUMONIA']
    criterion = nn.CrossEntropyLoss()
    history_ce = train_model(num_epochs=10, loss_method="CE")

    # Train with FocalLoss
    criterion = FocalLoss(alpha=1, gamma=2)
    
    
    history_focal = train_model(num_epochs=10, loss_method="FOCAL")

    
    plot_training(history_ce, history_focal)
