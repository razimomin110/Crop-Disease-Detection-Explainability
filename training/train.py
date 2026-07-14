"""
Training pipeline for Crop Disease Detection.

Implements two-phase transfer learning with early stopping:
  Phase 1 — backbone frozen, train classifier head.
  Phase 2 — unfreeze last layers, fine-tune at lower LR.

Usage:
    python -m training.train --data_dir data/PlantVillage --epochs 20 --batch_size 32
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import create_dataloaders
from models.model import CropDiseaseModel


# ──────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch, return (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation, return (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Plot training curves
# ──────────────────────────────────────────────────────────────────────────────
def save_training_curves(history: dict, save_dir: str) -> None:
    """Save loss and accuracy curves to disk."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[TRAIN] Training curves saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────────────────────
def train(
    data_dir: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    backbone: str = "efficientnet",
    patience: int = 5,
    num_workers: int = 2,
    save_dir: str = "models",
) -> None:
    """Full two-phase training pipeline."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, _, class_names = create_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers,
    )
    num_classes = len(class_names)

    # ── Model ────────────────────────────────────────────────────────────
    model = CropDiseaseModel(
        num_classes=num_classes, backbone=backbone, pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    os.makedirs(save_dir, exist_ok=True)

    # Save class names for inference
    with open(os.path.join(save_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    total_epochs_run = 0

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 1 — Frozen backbone: train classifier head only
    # ═══════════════════════════════════════════════════════════════════
    phase1_epochs = epochs // 2
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Frozen backbone  ({phase1_epochs} epochs)")
    print(f"{'='*60}")

    model.freeze_backbone()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
    )
    early_stop = EarlyStopping(patience=patience)

    for epoch in range(1, phase1_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        total_epochs_run += 1

        print(
            f"  [P1 {epoch:02d}/{phase1_epochs}] "
            f"loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({elapsed:.1f}s)"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

        if early_stop.step(val_loss):
            print("  [P1] Early stopping triggered.")
            break

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 2 — Fine-tune: unfreeze last layers
    # ═══════════════════════════════════════════════════════════════════
    phase2_epochs = epochs - phase1_epochs
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Fine-tuning backbone  ({phase2_epochs} epochs)")
    print(f"{'='*60}")

    model.unfreeze_backbone()
    # Lower learning rate for fine-tuning to avoid catastrophic forgetting
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 0.1,
    )
    early_stop = EarlyStopping(patience=patience)

    for epoch in range(1, phase2_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        total_epochs_run += 1

        print(
            f"  [P2 {epoch:02d}/{phase2_epochs}] "
            f"loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({elapsed:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

        if early_stop.step(val_loss):
            print("  [P2] Early stopping triggered.")
            break

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n[TRAIN] Finished — {total_epochs_run} epochs, best val_acc={best_val_acc:.4f}")
    print(f"[TRAIN] Model saved → {os.path.join(save_dir, 'best_model.pth')}")
    save_training_curves(history, save_dir)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train crop disease classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs (split across phases)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--backbone", type=str, default="efficientnet", choices=["efficientnet", "resnet50"])
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--save_dir", type=str, default="models", help="Where to save checkpoints")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        patience=args.patience,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
    )
