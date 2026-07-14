"""
Inference module for Crop Disease Detection.

Loads a trained model and runs single-image prediction with Grad-CAM
overlay and severity estimation.

Usage:
    python -m inference.predict --image path/to/leaf.jpg --model models/best_model.pth
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import IMAGENET_MEAN, IMAGENET_STD, TARGET_CLASSES
from explainability.gradcam import GradCAM
from explainability.severity import estimate_severity
from models.model import CropDiseaseModel


# ──────────────────────────────────────────────────────────────────────────────
# Pre-processing for a single image
# ──────────────────────────────────────────────────────────────────────────────
_inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_image(image_path: str):
    """Load and pre-process an image from disk.

    Returns
    -------
    tensor : torch.Tensor
        Pre-processed tensor of shape (1, 3, 224, 224).
    original : np.ndarray
        Original image as RGB uint8 array.
    """
    pil_image = Image.open(image_path).convert("RGB")
    original = np.array(pil_image)
    tensor = _inference_transform(pil_image).unsqueeze(0)
    return tensor, original


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────
def predict_image(
    model: CropDiseaseModel,
    image_path: str,
    class_names: list,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run prediction on a single image.

    Returns
    -------
    dict with keys:
        predicted_class, confidence, severity, heatmap, overlay, original
    """
    tensor, original = load_image(image_path)
    tensor = tensor.to(device)

    # ── Grad-CAM ─────────────────────────────────────────────────────────
    target_layer = model.get_target_layer()
    grad_cam = GradCAM(model, target_layer)

    heatmap = grad_cam.generate_heatmap(tensor)
    overlay = GradCAM.overlay_heatmap(original, heatmap)

    # ── Prediction (forward pass already done inside Grad-CAM) ───────────
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = probs.max(1)

    predicted_class = class_names[pred_idx.item()]
    confidence_pct = confidence.item() * 100.0

    # ── Severity ─────────────────────────────────────────────────────────
    # For "healthy" classes, severity is always 0
    if "healthy" in predicted_class.lower():
        severity = {
            "severity_pct": 0.0,
            "severity_label": "Healthy",
            "affected_pixels": 0,
            "total_pixels": heatmap.size,
        }
    else:
        severity = estimate_severity(heatmap)

    grad_cam.remove_hooks()

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence_pct, 2),
        "severity": severity,
        "heatmap": heatmap,
        "overlay": overlay,
        "original": original,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Save visualisation
# ──────────────────────────────────────────────────────────────────────────────
def save_prediction_figure(result: dict, save_path: str) -> None:
    """Save a side-by-side figure: original | Grad-CAM overlay."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(result["original"])
    ax1.set_title("Original")
    ax1.axis("off")

    ax2.imshow(result["overlay"])
    ax2.set_title("Grad-CAM Overlay")
    ax2.axis("off")

    severity = result["severity"]
    fig.suptitle(
        f"Prediction: {result['predicted_class']}\n"
        f"Confidence: {result['confidence']:.1f}%  |  "
        f"Severity: {severity['severity_pct']:.1f}% ({severity['severity_label']})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFER] Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Predict crop disease from a leaf image")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Model checkpoint")
    parser.add_argument("--backbone", type=str, default="efficientnet", choices=["efficientnet", "resnet50"])
    parser.add_argument("--output", type=str, default="inference/gradcam_output.png", help="Output image path")
    parser.add_argument("--class_names_file", type=str, default="models/class_names.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names
    if os.path.exists(args.class_names_file):
        with open(args.class_names_file, "r") as f:
            class_names = json.load(f)
    else:
        class_names = TARGET_CLASSES

    # Load model
    model = CropDiseaseModel(
        num_classes=len(class_names), backbone=args.backbone, pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    # Predict
    result = predict_image(model, args.image, class_names, device)

    # Print results
    print(f"\n{'='*50}")
    print(f"  Disease   : {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.1f}%")
    print(f"  Severity  : {result['severity']['severity_pct']:.1f}% "
          f"({result['severity']['severity_label']})")
    print(f"{'='*50}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_prediction_figure(result, args.output)
