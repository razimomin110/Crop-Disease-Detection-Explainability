"""
Dataset module for Crop Disease Detection.

Handles image loading, augmentation pipelines, train/val/test splitting,
and DataLoader creation for the PlantVillage dataset subset.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# ──────────────────────────────────────────────────────────────────────────────
# Target classes (10-class subset of PlantVillage)
# ──────────────────────────────────────────────────────────────────────────────
TARGET_CLASSES: List[str] = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Apple___Apple_scab",
    "Apple___healthy",
]

# ImageNet normalisation statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation / pre-processing transforms
# ──────────────────────────────────────────────────────────────────────────────
def get_transforms(mode: str = "train") -> transforms.Compose:
    """Return augmentation pipeline for the given mode.

    Train transforms simulate real-world smartphone capture conditions:
      • RandomResizedCrop  — varying framing / zoom
      • RandomHorizontalFlip — orientation invariance
      • RandomRotation      — slight tilt of the phone
      • ColorJitter          — lighting & white-balance differences
      • GaussianBlur         — focus / motion blur
    """
    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            # Simulate smartphone lighting variation & background noise
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────
class PlantDiseaseDataset(Dataset):
    """PyTorch Dataset for the PlantVillage image folder structure.

    Expected layout:
        data_dir/
            ClassName1/
                img001.jpg
                ...
            ClassName2/
                ...
    """

    def __init__(
        self,
        data_dir: str,
        class_names: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.class_names = class_names or TARGET_CLASSES
        self.class_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(self.class_names)
        }

        # Collect (image_path, label_idx) pairs
        self.samples: List[Tuple[str, int]] = []
        for cls_name in self.class_names:
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"[WARNING] Class directory not found, skipping: {cls_dir}")
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls_name])
                    )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {data_dir}. "
                "Ensure class folders exist with image files."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────
def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train / val / test DataLoaders with a 70 / 15 / 15 split.

    Parameters
    ----------
    data_dir : str
        Root directory containing class sub-folders.
    batch_size : int
        Batch size for all loaders.
    num_workers : int
        Number of parallel data-loading workers.
    train_ratio, val_ratio : float
        Proportions for the split (test_ratio = 1 - train - val).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_loader, val_loader, test_loader, class_names
    """
    # Build full dataset (no transform yet — we'll assign per-split)
    full_dataset = PlantDiseaseDataset(data_dir, transform=None)
    class_names = full_dataset.class_names
    n = len(full_dataset)

    # Shuffled indices
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create subset datasets with appropriate transforms
    train_ds = _TransformSubset(full_dataset, train_indices, get_transforms("train"))
    val_ds = _TransformSubset(full_dataset, val_indices, get_transforms("val"))
    test_ds = _TransformSubset(full_dataset, test_indices, get_transforms("test"))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[DATA] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, class_names


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper: apply per-split transforms to a Subset
# ──────────────────────────────────────────────────────────────────────────────
class _TransformSubset(Dataset):
    """Wraps a base dataset + index list + a specific transform."""

    def __init__(self, base_dataset: PlantDiseaseDataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.base.samples[self.indices[idx]]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
