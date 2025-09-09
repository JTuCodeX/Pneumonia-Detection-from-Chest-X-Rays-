import argparse, os, torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
)
import numpy as np
import matplotlib.pyplot as plt

from data_loader import test_loader, val_transform  # uses your existing transforms/loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # index 0 -> NORMAL, 1 -> PNEUMONIA

def load_model(weights_path: str):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # replace head
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

@torch.no_grad()
def predict_logits(model, loader):
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

def evaluate(weights_path: str, plot_curves: bool = True):
    print(f"Loading model from: {weights_path}")
    model = load_model(weights_path)

    logits, labels_t = predict_logits(model, test_loader)
    probs = torch.softmax(logits, dim=1).numpy()
    y_true = labels_t.numpy()
    y_pred = probs.argmax(axis=1)

    # Basic scores
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    # Per-class report (nice for portfolio)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-9)  # recall for Pneumonia
    specificity = tn / (tn + fp + 1e-9)  # recall for Normal

    # AUCs (use probability of class 1 = Pneumonia)
    y_scores = probs[:, 1]
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        roc_auc = float("nan")

    # PR-AUC is informative for imbalance
    pr_precision, pr_recall, pr_thresh = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(pr_recall, pr_precision)

    print("\n=== Test Set Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Pneumonia): {precision:.4f}")
    print(f"Recall/Sensitivity (Pneumonia): {recall:.4f}")
    print(f"F1 (Pneumonia): {f1:.4f}")
    print(f"Specificity (Normal): {specificity:.4f}")
    print(f"ROC-AUC (Pneumonia): {roc_auc:.4f}")
    print(f"PR-AUC  (Pneumonia): {pr_auc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(report)

    if plot_curves:
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC Curve (Pneumonia)")
        plt.legend()

        # PR
        plt.subplot(1,2,2)
        plt.plot(pr_recall, pr_precision, label=f"PR-AUC = {pr_auc:.3f}")
        plt.xlabel("Recall (Sensitivity)")
        plt.ylabel("Precision (PPV)")
        plt.title("Precision–Recall Curve (Pneumonia)")
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--no-plots", action="store_true", help="Disable ROC/PR plots")
    args = ap.parse_args()
    evaluate(args.weights, plot_curves=not args.no_plots)
