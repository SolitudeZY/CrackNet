"""
CRACK500 数据集加载器。

功能:
  - CRACK500Dataset: 通过 split file (train.txt/val.txt/test.txt) 加载图像+掩码
  - 面向细裂缝分割的训练/验证增强管线
"""
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet 标准归一化
DATASET_MEAN = [0.485, 0.456, 0.406]
DATASET_STD = [0.229, 0.224, 0.225]


def build_train_augmentation(img_size: int = 512) -> A.Compose:
    """
    训练增强管线。

    细裂缝任务对边界和宽度很敏感，这里避免使用会明显抹粗/扭曲裂缝的强形变。
    """
    return A.Compose([
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_REFLECT_101,
            fill=0,
            fill_mask=0,
            p=1.0,
        ),
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.08, rotate_limit=10,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8, p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.03)),
            A.GaussianBlur(blur_limit=(3, 3)),
        ], p=0.15),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def build_val_augmentation(img_size: int = 512) -> A.Compose:
    """验证/测试: 仅 resize + normalize。"""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


class CRACK500Dataset(Dataset):
    """
    CRACK500 裂缝分割数据集。

    通过 split file (train.txt / val.txt / test.txt) 加载图像-掩码对。
    每行格式: '<image_relpath> <mask_relpath>'
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        augmentation: Optional[A.Compose] = None,
        img_size: int = 512,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.augmentation = augmentation
        self.img_size = img_size

        split_file = self.dataset_dir / f"{split}.txt"
        self.samples = self._parse_split_file(split_file)

        if not self.samples:
            raise FileNotFoundError(f"在 {split_file} 中未找到有效样本")
        print(f"  Dataset [CRACK500-{split}]: {len(self.samples)} 张")

    def _parse_split_file(self, split_file: Path) -> list[tuple[Path, Path]]:
        """解析 split file，过滤无效/损坏样本。"""
        samples = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                img_path = self.dataset_dir / parts[0]
                mask_path = self.dataset_dir / parts[1]
                if not img_path.exists() or not mask_path.exists():
                    continue
                if mask_path.stat().st_size == 0:
                    continue
                samples.append((img_path, mask_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取图片: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            from PIL import Image
            pil_mask = Image.open(str(mask_path)).convert("L")
            mask = np.array(pil_mask)

        # 二值化掩码
        mask = (mask > 0).astype(np.uint8) * 255

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.augmentation:
            aug = self.augmentation(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            image = image.astype(np.float32) / 255.0
            for c in range(3):
                image[:, :, c] = (image[:, :, c] - DATASET_MEAN[c]) / DATASET_STD[c]

        # HWC → CHW
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # 掩码: 归一化到 [0, 1]
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0
        mask_t = torch.from_numpy((mask_f > 0.5).astype(np.float32)).unsqueeze(0)

        return image_t, mask_t


if __name__ == "__main__":
    dataset_dir = Path("../dataset/CRACK500")
    if dataset_dir.exists():
        for split in ("train", "val", "test"):
            split_file = dataset_dir / f"{split}.txt"
            if split_file.exists():
                ds = CRACK500Dataset(dataset_dir, split=split)
                print(f"  {split}: {len(ds)} 张")
