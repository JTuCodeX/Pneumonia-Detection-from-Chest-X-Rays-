import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils   # <-- added utils here
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Define dataset path (update if different)
DATA_DIR = "./chest_xray"   # assumes dataset is one level up from src/

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # resize images for CNN input
    transforms.ToTensor(),           # convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])  # normalize
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Load datasets (train, val, test)
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)



# Verify dataset sizes
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# --- Visualization helper ---
def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = std * img + mean
    img = img.clip(0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()

# Display a batch of training images
if __name__ == "__main__":
    class_names = train_dataset.classes
    print("Classes:", class_names)

    images, labels = next(iter(train_loader))
    out = utils.make_grid(images[:12])  # show first 8 images
    imshow(out, title=[class_names[x] for x in labels[:12]])


