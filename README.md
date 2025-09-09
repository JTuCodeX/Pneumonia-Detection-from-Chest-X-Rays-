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

- Train with CrossEntropy loss:
```bash
python model.py --loss CE
```

- Train with Focal loss:
```bash
python model.py --loss FOCAL
```

-Models will be saved as:

- pneumonia_resnet18_CE.pth

- pneumonia_resnet18_FOCAL.pth


## 📜 License

This project is licensed under the MIT License.
Feel free to use and modify for research or educational purposes.


## 🙌 Acknowledgements

Dataset: Chest X-Ray Images (Pneumonia)

Model: ResNet18, torchvision

Framework: PyTorch
