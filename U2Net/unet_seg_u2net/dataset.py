"""
裂缝分割数据集加载器。

支持:
  - CrackTree260_augmented (DeepCrack, 35100 张, images/ + masks/ 分离)
  - CRACK500 (1896 训练, jpg+png 同目录配对)
  - DeepCrack 外部测试集 (CRKWH100, CrackLS315, Stone331)

CPU 加速:
  - cv2.setNumThreads(0): 避免 OpenCV 内部线程与 DataLoader worker 冲突
  - 图像解码后直接 resize, 减少大尺寸数据在内存中的停留
"""
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# 关键 CPU 加速: 避免 OpenCV 线程与 DataLoader multiprocess 冲突
cv2.setNumThreads(0)
os.environ["OMP_NUM_THREADS"] = "1"

# ImageNet 标准归一化 (跨域通用, 与 v3 一致)
DATASET_MEAN = [0.485, 0.456, 0.406]
DATASET_STD = [0.229, 0.224, 0.225]


def build_train_augmentation(img_size: int = 320) -> A.Compose:
    """
    训练增强管线。
    离线增强已覆盖几何变换, 在线只做轻量像素级增强。
    """
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.15),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def build_val_augmentation(img_size: int = 320) -> A.Compose:
    """验证/测试: 仅 resize + normalize。"""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def split_by_original_image(
    images_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """
    按原始图像 ID 划分 train/val，避免数据泄露。

    文件命名: {SOURCE_ID}_r{角度}_f{翻转}_{裁剪位置}.jpg
    """
    images_dir = Path(images_dir)
    all_images = sorted(images_dir.glob("*.jpg"))
    if not all_images:
        all_images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))

    pattern = re.compile(r"^(.+)_r\d{2}_f[hvn]_[a-z]{2}$")
    groups: Dict[str, List[Path]] = defaultdict(list)

    for img_path in all_images:
        match = pattern.match(img_path.stem)
        source_id = match.group(1) if match else img_path.stem
        groups[source_id].append(img_path)

    source_ids = sorted(groups.keys())
    n_val = max(1, int(len(source_ids) * val_ratio))

    rng = random.Random(seed)
    rng.shuffle(source_ids)

    val_ids = set(source_ids[:n_val])

    train_files, val_files = [], []
    for sid in sorted(groups.keys()):
        target = val_files if sid in val_ids else train_files
        target.extend(sorted(groups[sid]))

    return train_files, val_files


class CrackDataset(Dataset):
    """
    通用裂缝分割数据集。

    支持三种目录布局:
      1. 分离目录: images_dir(jpg) + masks_dir(png/bmp) — DeepCrack 风格
      2. 同目录配对: images_dir 内 jpg+png 同名配对 — CRACK500 风格
         (masks_dir=None 时, 仅扫描 .jpg 文件, .png 自动作为掩码)
      3. file_list: 直接传入文件路径列表
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Optional[Path] = None,
        augmentation: Optional[A.Compose] = None,
        img_size: int = 320,
        file_list: Optional[List[Path]] = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.augmentation = augmentation
        self.img_size = img_size

        if file_list is not None:
            self.images = sorted(file_list)
        elif self.masks_dir and self.masks_dir != self.images_dir:
            # 分离目录: 扫描所有图像格式
            suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".JPG"}
            self.images = sorted(p for p in self.images_dir.iterdir() if p.suffix in suffixes)
        else:
            # 同目录 (CRACK500) 或无 masks_dir: 仅扫描 jpg 避免把 png 掩码当图片
            self.images = sorted(self.images_dir.glob("*.jpg"))
            if not self.images:
                self.images = sorted(p for p in self.images_dir.iterdir()
                                     if p.suffix.lower() in (".jpg", ".jpeg"))

        if not self.images:
            raise FileNotFoundError(f"在 {self.images_dir} 中未找到任何图片")

        self._verify_pairs()
        print(f"  Dataset [{self.images_dir.name}]: {len(self.images)} 张")

    def _find_mask(self, img_path: Path) -> Optional[Path]:
        """查找对应的掩码文件。"""
        stem = img_path.stem
        if self.masks_dir:
            for ext in (".png", ".bmp", ".jpg", ".tif"):
                p = self.masks_dir / (stem + ext)
                if p.exists():
                    return p
        else:
            # 同目录: jpg→png 配对
            p = img_path.with_suffix(".png")
            if p.exists() and p != img_path:
                return p
        return None

    def _verify_pairs(self):
        missing = [p.name for p in self.images[:5] if self._find_mask(p) is None]
        if missing:
            raise FileNotFoundError(f"以下图片找不到对应掩码: {missing}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # 读取图像 — 直接用 IMREAD_COLOR 避免额外转换
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取图片: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码
        mask_path = self._find_mask(img_path)
        if mask_path is None:
            raise RuntimeError(f"找不到掩码: {img_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"无法读取掩码: {mask_path}")

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 二值化掩码
        mask = ((mask > 127).astype(np.uint8)) * 255

        if self.augmentation:
            aug = self.augmentation(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            image = image.astype(np.float32) / 255.0
            for c in range(3):
                image[:, :, c] = (image[:, :, c] - DATASET_MEAN[c]) / DATASET_STD[c]

        # HWC → CHW, float32
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # 强制将 mask 规整到 0 和 1 (不论输入是 0-255 还是其他)
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f = mask_f / 255.0
        mask_t = torch.from_numpy((mask_f > 0.5).astype(np.float32)).unsqueeze(0)

        return image_t, mask_t


if __name__ == "__main__":
    from pathlib import Path

    # 测试 DeepCrack 划分
    aug_dir = Path("/home/fs-ai/YOLO-crack-detection/dataset/deepcrack/CrackTree260_augmented/images")
    if aug_dir.exists():
        train_files, val_files = split_by_original_image(aug_dir)
        print(f"DeepCrack: 训练={len(train_files)} 验证={len(val_files)} 总={len(train_files)+len(val_files)}")

    # 测试 CRACK500 (jpg+png 同目录)
    c500_dir = Path("/home/fs-ai/YOLO-crack-detection/dataset/crack500/CRACK500/traincrop/traincrop")
    if c500_dir.exists():
        ds = CrackDataset(images_dir=c500_dir, augmentation=build_val_augmentation(320))
        img, msk = ds[0]
        print(f"CRACK500: img={img.shape}, mask={msk.shape}, mask_range=[{msk.min():.0f},{msk.max():.0f}]")
