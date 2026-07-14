"""
Model architecture for Crop Disease Detection.

Provides a wrapper around EfficientNet-B0 (default) or ResNet50 with a custom
classifier head, plus helpers to freeze / unfreeze the backbone for two-phase
transfer learning.
"""

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False

from torchvision import models


class CropDiseaseModel(nn.Module):
    """Transfer-learning model for plant disease classification.

    Parameters
    ----------
    num_classes : int
        Number of disease classes.
    backbone : str
        ``"efficientnet"`` (default, uses timm EfficientNet-B0) or
        ``"resnet50"`` (torchvision fallback).
    pretrained : bool
        Whether to load ImageNet pre-trained weights.
    """

    def __init__(
        self,
        num_classes: int = 10,
        backbone: str = "efficientnet",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone

        if backbone == "efficientnet" and _HAS_TIMM:
            self.model = timm.create_model(
                "efficientnet_b0", pretrained=pretrained, num_classes=0
            )
            in_features = self.model.num_features  # 1280 for B0
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes),
            )
        else:
            # Fallback: torchvision ResNet50
            if backbone == "efficientnet" and not _HAS_TIMM:
                print("[WARNING] timm not installed — falling back to ResNet50")
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
            in_features = self.model.fc.in_features  # 2048
            self.model.fc = nn.Identity()  # remove original head
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes),
            )

    # ── Forward pass ─────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)  # (B, in_features)
        return self.classifier(features)

    # ── Freeze / Unfreeze helpers for two-phase training ─────────────────
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Phase 1)."""
        for param in self.model.parameters():
            param.requires_grad = False
        # Keep the classifier trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("[MODEL] Backbone frozen — only classifier head is trainable.")

    def unfreeze_backbone(self, unfreeze_all: bool = False) -> None:
        """Unfreeze backbone layers for fine-tuning (Phase 2).

        By default, unfreezes the last ~30% of parameters. Set
        ``unfreeze_all=True`` to unfreeze everything.
        """
        params = list(self.model.parameters())
        if unfreeze_all:
            for p in params:
                p.requires_grad = True
            print("[MODEL] Entire backbone unfrozen.")
        else:
            # Unfreeze the last 30 % of backbone layers
            cutoff = int(len(params) * 0.7)
            for p in params[:cutoff]:
                p.requires_grad = False
            for p in params[cutoff:]:
                p.requires_grad = True
            print(
                f"[MODEL] Unfroze last {len(params) - cutoff}/{len(params)} "
                "backbone params for fine-tuning."
            )

    # ── Utility ──────────────────────────────────────────────────────────
    def get_target_layer(self):
        """Return the last convolutional layer for Grad-CAM."""
        if self.backbone_name == "efficientnet" and _HAS_TIMM:
            # timm EfficientNet: last conv block
            return self.model.conv_head
        else:
            # ResNet50: layer4
            return self.model.layer4[-1]

    def count_parameters(self) -> dict:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
