"""
Evaluation module for Crop Disease Detection.

Loads the best checkpoint, runs inference on the test set, and produces:
  • Overall accuracy
  • Per-class precision / recall / F1 (sklearn classification report)
  • Confusion matrix plot

Usage:
    python -m evaluation.evaluate --data_dir data/PlantVillage --model models/best_model.pth
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import create_dataloaders
from models.model import CropDiseaseModel


# ──────────────────────────────────────────────────────────────────────────────
# Confusion matrix plot
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list,
    save_path: str,
) -> None:
    """Generate and save a heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[EVAL] Confusion matrix saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    data_dir: str,
    model_path: str,
    batch_size: int = 32,
    num_workers: int = 2,
    backbone: str = "efficientnet",
    save_dir: str = "evaluation",
) -> dict:
    """Evaluate the model on the test split.

    Returns
    -------
    dict with keys: accuracy, classification_report (str), confusion_matrix (ndarray).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    _, _, test_loader, class_names = create_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers,
    )
    num_classes = len(class_names)

    # ── Model ────────────────────────────────────────────────────────────
    model = CropDiseaseModel(
        num_classes=num_classes, backbone=backbone, pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── Inference ────────────────────────────────────────────────────────
    all_preds = []
    all_labels = []

    for images, labels in tqdm(test_loader, desc="[EVAL] Testing"):
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    # ── Metrics ──────────────────────────────────────────────────────────
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4,
    )

    print(f"\n{'='*60}")
    print(f"  TEST ACCURACY: {accuracy:.4f}")
    print(f"{'='*60}")
    print(report)

    # ── Save confusion matrix ────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        os.path.join(save_dir, "confusion_matrix.png"),
    )

    # Save classification report as text
    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    print(f"[EVAL] Classification report saved → {report_path}")

    return {
        "accuracy": accuracy,
        "classification_report": report,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate crop disease classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--backbone", type=str, default="efficientnet", choices=["efficientnet", "resnet50"])
    parser.add_argument("--save_dir", type=str, default="evaluation")
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        data_dir=args.data_dir,
        model_path=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        backbone=args.backbone,
        save_dir=args.save_dir,
    )
