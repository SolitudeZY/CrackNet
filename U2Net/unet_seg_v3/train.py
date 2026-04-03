"""
U-Net 裂缝语义分割训练脚本 (v3)。
v3 改进:
  - 使用 albumentations 替代手写增强，加入弹性变形、噪声、模糊、CLAHE 等
  - 多数据集联合训练: CRACK500 + DeepCrack CrackTree260
  - 通用 CrackDataset 支持多种图片/掩码格式
用法: python train.py
"""
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from tqdm import tqdm

from U_Net_MobileNetV2_model import U_Net_MobileNetV2


# ======================== 评估指标计算 ========================
def calculate_metrics(preds, targets, threshold=0.5):
    """
    计算语义分割的标准指标: Precision, Recall, IoU (Jaccard), 和 Dice (F1-Score)
    preds: (N, 1, H, W) sigmoid 概率值 [0, 1]
    targets: (N, 1, H, W) 真实标签 [0, 1]
    """
    pred_mask = (preds > threshold).float()
    targets_flat = targets.view(-1)
    preds_flat = pred_mask.view(-1)

    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1 - targets_flat)).sum().item()
    fn = ((1 - preds_flat) * targets_flat).sum().item()

    return {
        "precision": tp / (tp + fp + 1e-6),
        "recall": tp / (tp + fn + 1e-6),
        "iou": tp / (tp + fp + fn + 1e-6),
        "dice": (2 * tp) / (2 * tp + fp + fn + 1e-6),
    }


# ======================== 自动检测图像尺寸 ========================
def detect_dataset_img_size(images_dir: Path) -> int:
    """随机读取几张图片，自动推断应该使用的 IMG_SIZE"""
    print(f"正在自动检测数据集图片尺寸...")
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not image_files:
        print("未找到图片，默认使用 640")
        return 640

    sizes = []
    for img_path in image_files[:5]:
        with Image.open(img_path) as img:
            sizes.append(max(img.size))

    detected_size = max(set(sizes), key=sizes.count)
    aligned_size = ((detected_size + 31) // 32) * 32
    print(f"检测到最大边长约 {detected_size}，对齐到32的倍数后，IMG_SIZE 自动设置为: {aligned_size}")
    return aligned_size


# ======================== 配置 ========================
CRACK500_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/crack500/CRACK500")
DEEPCRACK_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/deepcrack")

# 创建以时间戳命名的独立运行目录
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path(__file__).resolve().parent / "runs" / "train" / f"exp_{current_time}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = RUN_DIR / "weights"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 自动检测图片尺寸 (以 CRACK500 为基准)
IMG_SIZE = detect_dataset_img_size(CRACK500_ROOT / "traincrop" / "traincrop")

# 动态加载均值和标准差
STATS_FILE = Path(__file__).resolve().parent / "dataset_stats.json"
if STATS_FILE.exists():
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)
        DATASET_MEAN = stats["mean"]
        DATASET_STD = stats["std"]
        dataset_name = stats.get("dataset_name", "Unknown")
    print(f"加载自定义数据集 ({dataset_name}) 统计信息: Mean={DATASET_MEAN}, Std={DATASET_STD}")
else:
    print("未找到 dataset_stats.json，回退到 ImageNet 默认值")
    DATASET_MEAN = [0.485, 0.456, 0.406]
    DATASET_STD = [0.229, 0.224, 0.225]

CONFIG = {
    "epochs": 300,
    "batch_size": 8,
    "encoder_lr": 1e-4,
    "decoder_lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup_epochs": 3,
    "patience": 70,
    "num_workers": 4,
    "seed": 42,
    "grad_accumulation": 2,
    "attention_type": "CBAM",
    "deep_supervision": True,
    "resume": False,
    "resume_ckpt": "/home/fs-ai/YOLO-crack-detection/scripts/unet_seg_v3/runs/train/exp_20260325_112514/weights/last.pt",
}


# ======================== Albumentations 数据增强 ========================
def build_train_augmentation(img_size: int) -> A.Compose:
    """
    构建训练时的数据增强管线 (v3 升级版)。
    使用 albumentations 库，原生支持 image+mask 联合变换。

    相比 v2 新增:
      - 弹性变形 (ElasticTransform): 模拟裂缝形态变化
      - 网格畸变 (GridDistortion): 模拟镜头畸变
      - 高斯噪声 (GaussNoise): 模拟边缘设备采图噪声
      - 高斯模糊 / 运动模糊: 模拟对焦不准和运动拍摄
      - CLAHE: 自适应直方图均衡化，增强低对比度裂缝
      - 随机缩放裁剪 (RandomResizedCrop): 多尺度学习
      - CoarseDropout: 随机遮挡，类似 Cutout 正则化
    """
    return A.Compose([
        # --- 几何变换 (同时作用于 image 和 mask) ---
        # 随机缩放裁剪: 0.25x~1.0x 面积比例裁剪后 resize，增强多尺度鲁棒性
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.25, 1.0),
            ratio=(0.75, 1.33),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=0.5,
        ),
        # 未被 RandomResizedCrop 触发时，保底 resize
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-30, 30),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        # 弹性变形: 裂缝分割中收益最高的单一增强
        A.ElasticTransform(
            alpha=80,
            sigma=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3,
        ),
        # 网格畸变: 模拟镜头畸变
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),

        # --- 像素级变换 (仅作用于 image) ---
        # 亮度/对比度
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5,
        ),
        # 色相/饱和度/明度
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.4,
        ),
        # CLAHE: 自适应直方图均衡化，增强低对比度裂缝可见度
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        # 高斯噪声: 模拟边缘设备传感器噪声
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        # 模糊 (三选一)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        # CoarseDropout: 类似 Cutout，随机遮挡矩形区域，增强鲁棒性
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(int(img_size * 0.05), int(img_size * 0.1)),
            hole_width_range=(int(img_size * 0.05), int(img_size * 0.1)),
            fill=0,
            p=0.2,
        ),
        # 归一化 (albumentations 内置，直接输出 float32 numpy)
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def build_val_augmentation(img_size: int) -> A.Compose:
    """验证/测试时仅做 resize + normalize。"""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


# ======================== 通用 Dataset ========================
class CrackDataset(Dataset):
    """
    通用裂缝分割数据集，支持多种格式:
      - CRACK500: 图片和同名 .png 掩码在同一目录
      - DeepCrack: 图片和掩码分别在不同目录，掩码为 .bmp 格式
      - YOLO txt 标注 (兼容旧版)
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Optional[Path] = None,
        labels_dir: Optional[Path] = None,
        augmentation: Optional[A.Compose] = None,
        img_size: int = 640,
        img_suffixes: Optional[List[str]] = None,
    ):
        """
        Args:
            images_dir: 图片目录
            masks_dir: 掩码目录 (DeepCrack 模式: 与图片分离的掩码目录)
                       若为 None，则在 images_dir 中寻找同名掩码 (CRACK500 模式)
            labels_dir: YOLO txt 标签目录 (兼容旧版)
            augmentation: albumentations 增强管线
            img_size: 输出图片尺寸
            img_suffixes: 图片文件后缀白名单，如 [".jpg"]。
                          若为 None 则自动推断:
                            - 有 masks_dir 时收集所有常见图片格式
                            - 无 masks_dir 时仅收集 .jpg (避免和同目录的 .png 掩码冲突)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.augmentation = augmentation
        self.img_size = img_size

        # 自动推断图片后缀
        if img_suffixes is not None:
            suffixes = img_suffixes
        elif self.masks_dir:
            # DeepCrack 模式: 图片和掩码分离，可以安全收集所有格式
            suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".JPG"]
        else:
            # CRACK500 模式: 图片和掩码在同一目录 (xxx.jpg=图片, xxx.png=掩码)
            # 只收集 .jpg 避免把 .png 掩码当作图片
            suffixes = [".jpg", ".JPG", ".jpeg"]

        self.images = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix in suffixes
        ])

        if len(self.images) == 0:
            raise FileNotFoundError(f"在 {self.images_dir} 中未找到任何图片")

        # 验证配对
        self._verify_pairs()
        print(f"  数据集加载完成: {self.images_dir.name} -> {len(self.images)} 张图片")

    def _find_mask(self, img_path: Path) -> Optional[Path]:
        """查找图片对应的掩码文件，支持多种格式和目录布局。"""
        stem = img_path.stem

        if self.masks_dir:
            # DeepCrack 模式: 在单独的掩码目录中查找
            for ext in [".bmp", ".png", ".jpg", ".tif"]:
                mask_path = self.masks_dir / (stem + ext)
                if mask_path.exists():
                    return mask_path
        else:
            # CRACK500 模式: 同目录下同名 .png
            mask_path = img_path.with_suffix(".png")
            if mask_path.exists() and mask_path != img_path:
                return mask_path

        return None

    def _verify_pairs(self):
        """验证前几对 image-mask 是否存在。"""
        missing = []
        for img_path in self.images[:5]:
            mask_path = self._find_mask(img_path)
            if mask_path is None and self.labels_dir is None:
                missing.append(img_path.name)

        if missing:
            raise FileNotFoundError(
                f"以下图片找不到对应掩码: {missing}\n"
                f"图片目录: {self.images_dir}\n"
                f"掩码目录: {self.masks_dir}"
            )

    def __len__(self):
        return len(self.images)

    def _parse_yolo_label(self, line: str, img_w: int, img_h: int):
        """解析一行 YOLO 标注，支持多边形和边界框。"""
        parts = line.strip().split()
        if len(parts) == 5:
            _, x_c, y_c, w, h = map(float, parts)
            x1 = (x_c - w / 2) * img_w
            y1 = (y_c - h / 2) * img_h
            x2 = (x_c + w / 2) * img_w
            y2 = (y_c + h / 2) * img_h
            return "bbox", [x1, y1, x2, y2]
        else:
            coords = [float(v) for v in parts[1:]]
            points = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_w
                y = coords[i + 1] * img_h
                points.append((x, y))
            return "polygon", points

    def _generate_mask_from_label(self, label_path: Path, img_w: int, img_h: int) -> np.ndarray:
        """从 YOLO txt 标签生成二值掩码 (numpy array)。"""
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    label_type, coords = self._parse_yolo_label(line, img_w, img_h)
                    if label_type == "polygon" and len(coords) >= 3:
                        pts = np.array(coords, dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    elif label_type == "bbox":
                        x1, y1, x2, y2 = [int(c) for c in coords]
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # 读取图片 (统一转 RGB, numpy HWC uint8)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取图片: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码
        mask_path = self._find_mask(img_path)
        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"无法读取掩码: {mask_path}")
            # 处理图片与掩码尺寸不一致 (如 Stone331: img 1024x1024, mask 512x512)
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        elif self.labels_dir:
            label_path = self.labels_dir / (img_path.stem + ".txt")
            mask = self._generate_mask_from_label(label_path, image.shape[1], image.shape[0])
        else:
            raise RuntimeError(f"找不到掩码: {img_path}")

        # 确保掩码是二值的 (0 或 255)
        mask = ((mask > 127).astype(np.uint8)) * 255

        # 应用增强 (albumentations 同时变换 image 和 mask)
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented["image"]  # float32 HWC (已 normalize)
            mask = augmented["mask"]    # uint8 HW
        else:
            # 无增强时手动 resize + normalize
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            image = image.astype(np.float32) / 255.0
            for c in range(3):
                image[:, :, c] = (image[:, :, c] - DATASET_MEAN[c]) / DATASET_STD[c]

        # 转 tensor: image (3, H, W), mask (1, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()  # HWC -> CHW

        mask_float = mask.astype(np.float32)
        if mask_float.max() > 1.0:
            mask_float = mask_float / 255.0
        mask_float = (mask_float > 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_float).unsqueeze(0)  # (1, H, W)

        return image_tensor, mask_tensor


# ======================== Loss ========================
def boundary_loss_gpu(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    基于 GPU 的近似 Boundary Loss，使用形态学池化代替 CPU 上的距离变换，大幅加速训练。
    """
    target_pad = F.pad(target, (1, 1, 1, 1), mode='reflect')

    # 膨胀 (Max Pooling)
    dilation = F.max_pool2d(target_pad, kernel_size=3, stride=1)
    # 腐蚀 (1 - Max Pooling on inverse)
    erosion = 1 - F.max_pool2d(1 - target_pad, kernel_size=3, stride=1)

    boundary = dilation - erosion

    bce_boundary = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    loss = (bce_boundary * boundary).sum() / (boundary.sum() + 1e-5)

    return loss


def tversky_focal_boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    复合损失: Focal Tversky + OHEM Focal BCE + Boundary Loss
    """
    prob = torch.sigmoid(logits)

    # 1. Tversky Index
    tp = (prob * target).sum(dim=(2, 3))
    fp = (prob * (1 - target)).sum(dim=(2, 3))
    fn = ((1 - prob) * target).sum(dim=(2, 3))

    tversky_index = (tp + 1e-5) / (tp + alpha * fp + beta * fn + 1e-5)

    # 2. Focal Tversky Loss
    focal_tversky = (1 - tversky_index) ** (1 / gamma)

    # 3. OHEM Focal BCE Loss
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    p_t = prob * target + (1 - prob) * (1 - target)
    focal_bce = (1 - p_t) ** gamma * bce

    focal_bce_flat = focal_bce.view(-1)
    k = int(0.20 * focal_bce_flat.numel())
    topk_loss, _ = torch.topk(focal_bce_flat, k)

    # 4. Boundary Loss
    b_loss = boundary_loss_gpu(logits, target)

    return focal_tversky.mean() + topk_loss.mean() + 0.1 * b_loss


# ======================== Metrics ========================
@torch.no_grad()
def compute_iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """计算 batch 平均 IoU。"""
    pred = (torch.sigmoid(logits) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# ======================== Training ========================
def train_one_epoch(model, loader, optimizer, scaler, device, accumulation_steps=2, deep_supervision=False, alpha=0.5, beta=0.5):
    model.train()
    total_loss = 0.0

    optimizer.zero_grad()
    for i, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)

        with torch.amp.autocast("cuda"):
            if deep_supervision:
                out_main, out_d2, out_d3 = model(images)
                loss_main = tversky_focal_boundary_loss(out_main, masks, alpha=alpha, beta=beta)
                loss_d2 = tversky_focal_boundary_loss(out_d2, masks, alpha=alpha, beta=beta)
                loss_d3 = tversky_focal_boundary_loss(out_d3, masks, alpha=alpha, beta=beta)
                loss = loss_main + 0.4 * loss_d2 + 0.2 * loss_d3
            else:
                logits = model(images)
                loss = tversky_focal_boundary_loss(logits, masks, alpha=alpha, beta=beta)

            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_metrics = {"precision": [], "recall": [], "iou": [], "dice": []}

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = tversky_focal_boundary_loss(logits, masks)

        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        batch_metrics = calculate_metrics(probs, masks)
        for k in all_metrics:
            all_metrics[k].append(batch_metrics[k])

    avg_loss = total_loss / len(loader.dataset)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    return avg_loss, avg_metrics


# ======================== 多数据集构建 ========================
def build_datasets(cfg: dict) -> Tuple[Dataset, Dataset]:
    """
    构建训练集和验证集。
    训练集: CRACK500 traincrop + DeepCrack CrackTree260 (联合训练)
    验证集: CRACK500 valcrop
    """
    train_aug = build_train_augmentation(IMG_SIZE)
    val_aug = build_val_augmentation(IMG_SIZE)

    # ---- 训练集 ----
    train_datasets = []

    # 1. CRACK500 traincrop (主数据集)
    crack500_train_dir = CRACK500_ROOT / "traincrop" / "traincrop"
    if crack500_train_dir.exists():
        print(f"\n[训练集] 加载 CRACK500 traincrop:")
        ds = CrackDataset(
            images_dir=crack500_train_dir,
            augmentation=train_aug,
            img_size=IMG_SIZE,
        )
        train_datasets.append(ds)

    # 2. DeepCrack CrackTree260 (补充训练数据)
    cracktree_img_dir = DEEPCRACK_ROOT / "CrackTree260" / "CrackTree260_img"
    cracktree_gt_dir = DEEPCRACK_ROOT / "CrackTree260" / "CrackTree260_gt" / "gt"
    if cracktree_img_dir.exists() and cracktree_gt_dir.exists():
        print(f"\n[训练集] 加载 DeepCrack CrackTree260:")
        ds = CrackDataset(
            images_dir=cracktree_img_dir,
            masks_dir=cracktree_gt_dir,
            augmentation=train_aug,
            img_size=IMG_SIZE,
        )
        train_datasets.append(ds)
    else:
        print(f"\n[警告] DeepCrack CrackTree260 未找到，跳过。请确认已解压到: {cracktree_img_dir}")

    if not train_datasets:
        raise FileNotFoundError("没有找到任何训练数据集！")

    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    print(f"\n[训练集] 总计: {len(train_ds)} 张图片")

    # ---- 验证集 ----
    print(f"\n[验证集] 加载 CRACK500 valcrop:")
    val_ds = CrackDataset(
        images_dir=CRACK500_ROOT / "valcrop" / "valcrop",
        augmentation=val_aug,
        img_size=IMG_SIZE,
    )

    return train_ds, val_ds


def main():
    cfg = CONFIG
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据集 (v3: 多数据集联合训练)
    train_ds, val_ds = build_datasets(cfg)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    # 模型
    model = U_Net_MobileNetV2(
        pretrained=True,
        attention_type=cfg["attention_type"],
        deep_supervision=cfg["deep_supervision"],
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,} "
          f"(Attention: {cfg['attention_type']}, DeepSup: {cfg['deep_supervision']})")

    # 差分学习率
    optimizer = torch.optim.AdamW([
        {"params": model.encoder_params(), "lr": cfg["encoder_lr"]},
        {"params": model.decoder_params(), "lr": cfg["decoder_lr"]},
    ], weight_decay=cfg["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] - cfg["warmup_epochs"], eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 1
    best_iou = 0.0
    best_dice = 0.0
    best_recall = 0.0
    best_precision = 0.0
    patience_counter = 0

    # 断点续训
    if cfg.get("resume", False) and cfg.get("resume_ckpt", ""):
        ckpt_path = Path(cfg["resume_ckpt"])
        if ckpt_path.exists():
            print(f"Resuming training from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1

            if "val_metrics" in ckpt:
                best_iou = ckpt["val_metrics"].get("iou", 0.0)
                best_dice = ckpt["val_metrics"].get("dice", 0.0)
                best_recall = ckpt["val_metrics"].get("recall", 0.0)
                best_precision = ckpt["val_metrics"].get("precision", 0.0)

            print(f"Successfully resumed from epoch {start_epoch - 1}. Current Best IoU: {best_iou:.4f}")
        else:
            print(f"Warning: Resume checkpoint {ckpt_path} not found. Starting from scratch.")

    log_path = RUN_DIR / "training_log.csv"

    if start_epoch == 1:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "precision", "recall", "iou", "dice", "lr", "alpha", "beta"])

    # 初始化动态 Loss 权重
    current_alpha = 0.5
    current_beta = 0.5

    print(f"\n{'Epoch':>5} | {'TrainLoss':>10} | {'ValLoss':>8} | {'Prec':>6} | {'Recall':>6} | {'IoU':>6} | {'Dice':>6} | {'LR':>8}")
    print("-" * 80)

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        # Warmup
        if epoch <= cfg["warmup_epochs"]:
            warmup_factor = epoch / cfg["warmup_epochs"]
            optimizer.param_groups[0]["lr"] = cfg["encoder_lr"] * warmup_factor
            optimizer.param_groups[1]["lr"] = cfg["decoder_lr"] * warmup_factor

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            accumulation_steps=cfg.get("grad_accumulation", 1),
            deep_supervision=cfg.get("deep_supervision", False),
            alpha=current_alpha,
            beta=current_beta,
        )
        val_loss, val_metrics = validate(model, val_loader, device)

        # 动态 Loss 权重调整
        p = val_metrics['precision']
        r = val_metrics['recall']
        diff = r - p
        adjustment = diff * 0.1
        current_alpha = min(max(current_alpha + adjustment + 0.02, 0.2), 0.8)
        current_beta = 1.0 - current_alpha

        if epoch > cfg["warmup_epochs"]:
            scheduler.step()

        current_lr = optimizer.param_groups[1]["lr"]
        elapsed = time.time() - t0

        print(
            f"{epoch:>5} |train_loss:{train_loss:>6.4f} |val_loss:{val_loss:>6.4f} | "
            f"precision:{val_metrics['precision']:>6.4f} |recall:{val_metrics['recall']:>6.4f} | "
            f"iou:{val_metrics['iou']:>6.4f} |dice:{val_metrics['dice']:>6.4f} | "
            f"{current_lr:>8.6f} ({elapsed:.0f}s) [a={current_alpha:.2f}]"
        )

        # 日志 (v3: 额外记录 alpha/beta)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{val_metrics['precision']:.6f}", f"{val_metrics['recall']:.6f}",
                f"{val_metrics['iou']:.6f}", f"{val_metrics['dice']:.6f}",
                f"{current_lr:.8f}", f"{current_alpha:.4f}", f"{current_beta:.4f}",
            ])

        # Checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }, SAVE_DIR / "last.pt")

        is_best_recall = val_metrics['recall'] > best_recall
        is_best_precision = val_metrics['precision'] > best_precision
        is_best_iou = val_metrics['iou'] > best_iou
        is_best_dice = val_metrics['dice'] > best_dice

        if is_best_recall:
            best_recall = val_metrics['recall']
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, SAVE_DIR / "best_recall.pt")
            print(f"        -> best Recall saved: {best_recall:.4f}")

        if is_best_precision:
            best_precision = val_metrics['precision']
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, SAVE_DIR / "best_precision.pt")
            print(f"        -> best Precision saved: {best_precision:.4f}")

        if is_best_iou:
            best_iou = val_metrics['iou']
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, SAVE_DIR / "best_iou.pt")
            print(f"        -> best IoU saved: {best_iou:.4f}")

        if is_best_dice:
            best_dice = val_metrics['dice']
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, SAVE_DIR / "best_dice.pt")
            print(f"        -> best Dice saved: {best_dice:.4f}")

        if is_best_recall or is_best_precision or is_best_iou or is_best_dice:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (patience={cfg['patience']})")
                break

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}, Best val Dice: {best_dice:.4f}, "
          f"Best val Recall: {best_recall:.4f}, Best val Precision: {best_precision:.4f}")
    print(f"Results saved to: {RUN_DIR}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
