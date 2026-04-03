"""
Unified crack segmentation dataset: CRACK500 + DeepCrack collections.

Loads multiple crack datasets with different formats into a single
training/validation/test pipeline with consistent augmentation.
"""
from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader, Dataset


DATASET_MEAN = (0.485, 0.456, 0.406)
DATASET_STD = (0.229, 0.224, 0.225)

# Default dataset roots
CRACK500_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/crack500/CRACK500")
DEEPCRACK_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/deepcrack")


def _read_image(path: Path) -> np.ndarray:
    """Read image as RGB uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    """Read mask as binary float32 (0 or 1)."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        from PIL import Image
        mask = np.array(Image.open(path).convert("L"))
    return (mask > 0).astype(np.float32)


def _build_transform(patch_size: int, augment: bool) -> A.Compose:
    transforms = []
    if augment:
        # Use a larger crop region then resize to patch_size for multi-scale training
        crop_size = int(patch_size * 1.25)  # 640 for patch_size=512
        transforms.extend([
            A.PadIfNeeded(min_height=crop_size, min_width=crop_size,
                          border_mode=cv2.BORDER_REFLECT_101, fill=0, fill_mask=0, p=1.0),
            A.RandomCrop(crop_size, crop_size),
            # Multi-scale resize: randomly scale 0.75x-1.25x then crop to patch_size
            A.RandomScale(scale_limit=(-0.25, 0.25), p=0.5),
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_REFLECT_101, fill=0, fill_mask=0, p=1.0),
            A.RandomCrop(patch_size, patch_size),
            # Geometric augmentation
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-15, 15),
                     border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            # Elastic deformation — crucial for thin crack sensitivity
            A.ElasticTransform(alpha=80, sigma=9, p=0.3),
            # Color augmentation
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.25),
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.04)),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MedianBlur(blur_limit=3),
            ], p=0.2),
            # Simulate different lighting/weather conditions
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(16, 48),
                            hole_width_range=(16, 48), fill=0, p=0.15),
        ])
    else:
        transforms.append(
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_REFLECT_101, fill=0, fill_mask=0, p=1.0),
        )

    transforms.extend([
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2(),
    ])
    return A.Compose(transforms)


# ---------------------------------------------------------------------------
# Per-source sample parsers
# ---------------------------------------------------------------------------

def _parse_crack500(root: Path, split: str) -> list[tuple[Path, Path]]:
    """Parse CRACK500 split file (train/val/test).

    Handles nested directory structure: traincrop/traincrop/xxx.jpg
    The txt files reference 'traincrop/xxx.jpg' but files live in 'traincrop/traincrop/xxx.jpg'.
    """
    split_file = root / f"{split}.txt"
    if not split_file.exists():
        return []
    samples = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            img_path = root / parts[0]
            mask_path = root / parts[1]
            # Try nested path if direct path doesn't exist
            if not img_path.exists():
                subdir = parts[0].split("/")[0]  # e.g. "traincrop"
                fname = "/".join(parts[0].split("/")[1:])
                img_path = root / subdir / subdir / fname
                mask_path = root / subdir / subdir / "/".join(parts[1].split("/")[1:])
            if img_path.exists() and mask_path.exists() and mask_path.stat().st_size > 0:
                samples.append((img_path, mask_path))
    return samples


def _parse_deepcrack_dir(img_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    """Match images to masks by stem name across different extensions."""
    if not img_dir.exists() or not gt_dir.exists():
        return []
    gt_map = {p.stem: p for p in gt_dir.iterdir() if p.is_file()}
    samples = []
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.stem in gt_map:
            samples.append((img_path, gt_map[img_path.stem]))
    return samples


def _parse_deepcrack_augmented(root: Path) -> list[tuple[Path, Path]]:
    """Parse CrackTree260_augmented (images/ + masks/ with matching names)."""
    img_dir = root / "images"
    mask_dir = root / "masks"
    if not img_dir.exists() or not mask_dir.exists():
        return []
    mask_map = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}
    samples = []
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.stem in mask_map:
            samples.append((img_path, mask_map[img_path.stem]))
    return samples


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class CrackSegDataset(Dataset):
    """Generic crack segmentation dataset from a list of (image, mask) pairs."""

    def __init__(self, samples: list[tuple[Path, Path]], patch_size: int, augment: bool):
        self.samples = samples
        self.transform = _build_transform(patch_size, augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        image = _read_image(img_path)
        mask = _read_mask(mask_path)
        # Resize mask to match image if dimensions differ (e.g. Stone331)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].float().unsqueeze(0)


class SampledDataset(Dataset):
    """Wraps a dataset and samples a fixed number of items per epoch."""

    def __init__(self, dataset: Dataset, max_samples: int):
        self.dataset = dataset
        self.max_samples = min(max_samples, len(dataset))
        self._indices = list(range(len(dataset)))
        self.resample()

    def resample(self) -> None:
        random.shuffle(self._indices)
        self._epoch_indices = self._indices[:self.max_samples]

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[self._epoch_indices[idx]]


# ---------------------------------------------------------------------------
# Build dataloaders
# ---------------------------------------------------------------------------

def _collect_deepcrack_samples(root: Path) -> list[tuple[Path, Path]]:
    """Collect all non-augmented DeepCrack samples."""
    samples = []
    for name, img_sub, gt_sub in [
        ("CrackLS315", "CrackLS315_img", "CrackLS315_gt"),
        ("CrackTree260", "CrackTree260_img", "CrackTree260_gt/gt"),
        ("CRKWH100", "CRKWH100_img", "CRKWH100_gt"),
        ("Stone331", "Stone331_img", "Stone331_gt"),
    ]:
        d = root / name
        if d.exists():
            samples.extend(_parse_deepcrack_dir(d / img_sub, d / gt_sub))
    return samples


def _split_samples(
    samples: list[tuple[Path, Path]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Split samples into train/val/test."""
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n_train = int(len(samples) * train_ratio)
    n_val = int(len(samples) * val_ratio)
    train = [samples[i] for i in indices[:n_train]]
    val = [samples[i] for i in indices[n_train:n_train + n_val]]
    test = [samples[i] for i in indices[n_train + n_val:]]
    return train, val, test


def build_dataloaders(
    crack500_root: str | Path = CRACK500_ROOT,
    deepcrack_root: str | Path = DEEPCRACK_ROOT,
    patch_size: int = 512,
    batch_size: int = 6,
    num_workers: int = 8,
    augmented_max_samples: int = 3000,
    crack500_only: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders.

    Args:
        crack500_only: If True, only use CRACK500 dataset (no DeepCrack).
    """
    crack500_root = Path(crack500_root)
    deepcrack_root = Path(deepcrack_root)

    # CRACK500: use existing splits
    c500_train = _parse_crack500(crack500_root, "train")
    c500_val = _parse_crack500(crack500_root, "val")
    c500_test = _parse_crack500(crack500_root, "test")

    train_samples = c500_train
    val_samples = c500_val
    test_samples = c500_test

    print(f"[dataset] CRACK500:  train={len(c500_train)} val={len(c500_val)} test={len(c500_test)}")

    if not crack500_only:
        dc_all = _collect_deepcrack_samples(deepcrack_root)
        dc_train, dc_val, dc_test = _split_samples(dc_all)
        aug_all = _parse_deepcrack_augmented(deepcrack_root / "CrackTree260_augmented")
        print(f"[dataset] DeepCrack: train={len(dc_train)} val={len(dc_val)} test={len(dc_test)} (from {len(dc_all)} total)")
        print(f"[dataset] Augmented: {len(aug_all)} total, sampling {min(augmented_max_samples, len(aug_all))} per epoch")
        train_samples = c500_train + dc_train
        val_samples = c500_val + dc_val
        test_samples = c500_test + dc_test

    # Build datasets
    train_ds = CrackSegDataset(train_samples, patch_size, augment=True)

    if not crack500_only:
        aug_all = _parse_deepcrack_augmented(deepcrack_root / "CrackTree260_augmented")
        if aug_all:
            train_aug = SampledDataset(
                CrackSegDataset(aug_all, patch_size, augment=True),
                max_samples=augmented_max_samples,
            )
            train_ds = ConcatDataset([train_ds, train_aug])

    val_ds = CrackSegDataset(val_samples, patch_size, augment=False)
    test_ds = CrackSegDataset(test_samples, patch_size, augment=False)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
