# Pneumonia Detection from Chest X-Rays 
A deep learning project for classifying chest X-ray images as NORMAL or PNEUMONIA using transfer learning with ResNet18 in PyTorch. The pipeline covers dataset preprocessing (resize, normalize, augmentation), model training with CrossEntropy and Focal Loss, evaluation with comprehensive metrics (accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC), and visualizations including confusion matrix, ROC and PR curves. Includes full documentation for reproducibility and learning.


This project implements a **deep learning pipeline** for classifying chest X-ray images into **Normal** or **Pneumonia**.  
It uses **PyTorch** with a fine-tuned **ResNet18** model, trained on the widely used **Chest X-Ray dataset**.

---

## 📂 Repository Contents

- `data_loader.py` → Data preprocessing, augmentation, and PyTorch dataloaders  
- `model.py` → Model definition, training loop, loss functions (CrossEntropy, Focal Loss)  
- `evaluate.py` → Model evaluation, metrics, confusion matrix, ROC/PR curves  
- `outputs/` → Saved model checkpoints and plots (to be generated after training)  

---

## 📊 Dataset

The project uses the **Chest X-Ray dataset** (Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)).

- **Classes:** `NORMAL` and `PNEUMONIA`  
- **Preprocessing:**  
  - Resize images to **224×224**  
  - Normalize with ImageNet mean & std  
  - Data augmentation (random flips, rotations) for training  

---

## ⚙️ Installation & Setup

```bash
# Clone repo
git clone https://github.com/yourusername/pneumonia-xray-classification.git
cd pneumonia-xray-classification

# Create environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## 🚀 Training the Model

Default behaviour (current code):

Run the main training script:

```bash
python model.py 
```
The script will first train using CrossEntropyLoss and save:

pneumonia_resnet18_CE.pth

It will then switch to FocalLoss and train again, saving:

pneumonia_resnet18_FOCAL.pth

Training progress (loss/accuracy) is printed per epoch and the script will call plot_training(...) to visualize the histories.

If you prefer to run only one loss (optional)

The repository ships with the sequential approach. To train only one loss without editing the core loop, you can:

Temporarily comment out one of the train_model(...) calls at the bottom of model.py,

OR add a small CLI wrapper (example below) to control whether to run CE, FOCAL, or BOTH.

Example: small argparse wrapper to add to the bottom of model.py (optional)

```bash
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["CE","FOCAL","BOTH"], default="BOTH")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    print("Classes:", train_dataset.classes)

    if args.loss in ("CE","BOTH"):
        criterion = nn.CrossEntropyLoss()
        history_ce = train_model(num_epochs=args.epochs, loss_method="CE")

    if args.loss in ("FOCAL","BOTH"):
        criterion = FocalLoss(alpha=1, gamma=2)
        history_focal = train_model(num_epochs=args.epochs, loss_method="FOCAL")

    # Adjust plotting depending on what's available
    try:
        plot_training(history_ce if 'history_ce' in locals() else None,
                      history_focal if 'history_focal' in locals() else None)
    except Exception:
        pass

```

## 🧪 Evaluation

Evaluate a trained checkpoint using evaluate.py:

```bash
python evaluate.py --weights pneumonia_resnet18_CE.pth
```

or

```bash
python evaluate.py --weights pneumonia_resnet18_FOCAL.pth
```

evaluate.py will compute and print:

Accuracy, Precision, Recall (Sensitivity), F1 Score, Specificity

ROC-AUC, PR-AUC

Confusion matrix and a per-class classification report



## 📜 License

This project is licensed under the MIT License.
Feel free to use and modify for research or educational purposes.


## 🙌 Acknowledgements

Dataset: Chest X-Ray Images (Pneumonia)

Model: ResNet18, torchvision

Framework: PyTorch
