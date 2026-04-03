"""
U2NETP 裂缝分割测试/评估脚本。

功能:
  - 加载 PyTorch checkpoint 评估模型
  - 支持多个测试集 (验证划分, CRKWH100, CrackLS315, Stone331)
  - 滑动窗口推理 (处理任意尺寸图像)
  - 可选 TTA (水平翻转)
  - 可选结果可视化保存

用法:
  python test.py --weights runs/.../weights/best_iou.pt
  python test.py --weights best_iou.pt --save-vis --tta
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import U2NETP
from dataset import (
    CrackDataset,
    build_val_augmentation,
    split_by_original_image,
    DATASET_MEAN,
    DATASET_STD,
)

DEEPCRACK_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/deepcrack")
IMG_SIZE = 320


# ======================== 滑动窗口推理 ========================
def sliding_window_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    window_size: int = 320,
    stride: int = 213,
    tta: bool = False,
) -> np.ndarray:
    """
    滑动窗口推理，处理任意尺寸图像。

    Args:
        model: 模型 (eval mode)
        image: RGB 图像, HWC, uint8
        device: 设备
        window_size: 窗口大小
        stride: 步幅 (默认 1/3 重叠)
        tta: 是否使用水平翻转 TTA
    Returns:
        prob_map: 概率图, HW, float32 [0, 1]
    """
    h, w = image.shape[:2]

    # 小图直接推理
    if h <= window_size and w <= window_size:
        return _infer_single(model, image, device, tta)

    # 滑动窗口
    count_map = np.zeros((h, w), dtype=np.float32)
    prob_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = min(y, h - window_size)
            x1 = min(x, w - window_size)
            y2 = y1 + window_size
            x2 = x1 + window_size

            patch = image[y1:y2, x1:x2]
            pred = _infer_single(model, patch, device, tta)

            prob_map[y1:y2, x1:x2] += pred
            count_map[y1:y2, x1:x2] += 1.0

    count_map = np.maximum(count_map, 1.0)
    return prob_map / count_map


def _infer_single(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tta: bool = False,
) -> np.ndarray:
    """推理单张图像，返回概率图。"""
    h, w = image.shape[:2]

    # 预处理
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    for c in range(3):
        img[:, :, c] = (img[:, :, c] - DATASET_MEAN[c]) / DATASET_STD[c]
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # 推理
    with torch.amp.autocast("cuda"):
        outputs = model(tensor)
        if isinstance(outputs, (tuple, list)):
            d0 = outputs[0]
        else:
            d0 = outputs
    prob = torch.sigmoid(d0)

    # TTA: 水平翻转
    if tta:
        tensor_flip = torch.flip(tensor, dims=[3])
        with torch.amp.autocast("cuda"):
            outputs_flip = model(tensor_flip)
            if isinstance(outputs_flip, (tuple, list)):
                d0_flip = outputs_flip[0]
            else:
                d0_flip = outputs_flip
        prob_flip = torch.sigmoid(d0_flip)
        prob_flip = torch.flip(prob_flip, dims=[3])
        prob = (prob + prob_flip) / 2.0

    # 还原到原始尺寸
    prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
    return prob.squeeze().cpu().numpy()


# ======================== 指标计算 ========================
def calculate_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """计算二值分割指标。"""
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = (target > 0.5).astype(np.float32)

    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()
    tn = ((1 - pred_bin) * (1 - target_bin)).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    # 处理全背景图片(tn>0, tp=0, fn=0)且模型完全预测为背景(fp=0)的情况
    if tp == 0 and fp == 0 and fn == 0:
        iou = 1.0
        dice = 1.0
    else:
        iou = tp / (tp + fp + fn + 1e-6)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
    }


# ======================== 可视化 ========================
def save_visualization(
    image: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    save_path: Path,
    threshold: float = 0.5,
):
    """保存原图 + 预测 + GT 的对比可视化。"""
    h, w = image.shape[:2]

    # 预测叠加 (红色)
    overlay = image.copy()
    pred_bin = (pred > threshold).astype(np.uint8)
    overlay[pred_bin > 0] = [255, 0, 0]
    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    # GT 叠加 (绿色)
    overlay_gt = image.copy()
    mask_bin = (mask > 127).astype(np.uint8)
    overlay_gt[mask_bin > 0] = [0, 255, 0]
    blended_gt = cv2.addWeighted(image, 0.6, overlay_gt, 0.4, 0)

    # 概率热力图
    heatmap = (pred * 255).clip(0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 拼接: [原图 | GT叠加 | 预测叠加 | 热力图]
    canvas = np.concatenate([
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(blended_gt, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
        heatmap_color,
    ], axis=1)

    cv2.imwrite(str(save_path), canvas)


# ======================== 数据集评估 ========================
@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    name: str,
    images_dir: Path,
    masks_dir: Path,
    device: torch.device,
    tta: bool = False,
    save_vis_dir: Optional[Path] = None,
    file_list: Optional[List[Path]] = None,
) -> Dict[str, float]:
    """在单个数据集上评估。"""
    model.eval()

    if file_list is not None:
        img_files = sorted(file_list)
    else:
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".JPG"}
        img_files = sorted(p for p in images_dir.iterdir() if p.suffix in suffixes)

    all_metrics = {"precision": [], "recall": [], "iou": [], "dice": []}

    for img_path in tqdm(img_files, desc=f"  {name}", ncols=80):
        # 读取图像
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码
        stem = img_path.stem
        mask = None
        for ext in (".png", ".bmp", ".jpg", ".tif"):
            mask_path = masks_dir / (stem + ext)
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                break
        if mask is None:
            continue

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 推理
        prob = sliding_window_inference(model, image, device, window_size=IMG_SIZE, tta=tta)

        # 指标
        mask_bin = (mask > 127).astype(np.float32)
        m = calculate_metrics(prob, mask_bin)
        for k in all_metrics:
            all_metrics[k].append(m[k])

        # 可视化
        if save_vis_dir:
            save_visualization(image, prob, mask, save_vis_dir / f"{stem}.jpg")

    avg = {k: np.mean(v) if v else 0.0 for k, v in all_metrics.items()}
    print(f"  [{name:>12s}] P={avg['precision']:.4f}  R={avg['recall']:.4f}  "
          f"IoU={avg['iou']:.4f}  Dice={avg['dice']:.4f}  ({len(img_files)} 张)")
    return avg


# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(description="U2NETP 裂缝分割测试")
    parser.add_argument("--weights", type=str, required=True, help="PyTorch checkpoint 路径")
    parser.add_argument("--tta", action="store_true", help="使用水平翻转 TTA")
    parser.add_argument("--save-vis", action="store_true", help="保存可视化结果")
    parser.add_argument("--threshold", type=float, default=0.5, help="二值化阈值")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    print(f"加载 checkpoint: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model = U2NETP(3, 1).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if "epoch" in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")
    if "val_metrics" in ckpt:
        m = ckpt["val_metrics"]
        print(f"  Val metrics: IoU={m.get('iou', 0):.4f} Dice={m.get('dice', 0):.4f}")

    # 可视化目录
    vis_dir = None
    if args.save_vis:
        vis_dir = Path(args.weights).parent.parent / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"  可视化保存到: {vis_dir}")

    # 评估数据集
    results = {}

    # 1. 内部验证集
    aug_img_dir = DEEPCRACK_ROOT / "CrackTree260_augmented" / "images"
    aug_mask_dir = DEEPCRACK_ROOT / "CrackTree260_augmented" / "masks"
    if aug_img_dir.exists():
        _, val_files = split_by_original_image(aug_img_dir, val_ratio=0.1, seed=42)
        sub_vis = vis_dir / "CT260_val" if vis_dir else None
        if sub_vis:
            sub_vis.mkdir(parents=True, exist_ok=True)
        # 只评估部分验证样本 (3510 张太多)
        eval_files = val_files[:200]
        results["CT260_val"] = evaluate_dataset(
            model, "CT260_val", aug_img_dir, aug_mask_dir, device,
            tta=args.tta, save_vis_dir=sub_vis, file_list=eval_files,
        )

    # 2. 外部测试集
    test_datasets = [
        ("CRKWH100", DEEPCRACK_ROOT / "CRKWH100" / "CRKWH100_img",
         DEEPCRACK_ROOT / "CRKWH100" / "CRKWH100_gt"),
        ("CrackLS315", DEEPCRACK_ROOT / "CrackLS315" / "CrackLS315_img",
         DEEPCRACK_ROOT / "CrackLS315" / "CrackLS315_gt"),
        ("Stone331", DEEPCRACK_ROOT / "Stone331" / "Stone331_img",
         DEEPCRACK_ROOT / "Stone331" / "Stone331_gt"),
    ]

    print("\n" + "=" * 70)
    print("多数据集测试结果")
    print("=" * 70)

    for name, img_dir, mask_dir in test_datasets:
        if not img_dir.exists():
            print(f"  [{name:>12s}] 跳过 (目录不存在)")
            continue
        sub_vis = vis_dir / name if vis_dir else None
        if sub_vis:
            sub_vis.mkdir(parents=True, exist_ok=True)
        results[name] = evaluate_dataset(
            model, name, img_dir, mask_dir, device,
            tta=args.tta, save_vis_dir=sub_vis,
        )

    # 汇总
    ext_results = {k: v for k, v in results.items() if k != "CT260_val"}
    if ext_results:
        avg_all = {}
        for k in ("precision", "recall", "iou", "dice"):
            avg_all[k] = np.mean([r[k] for r in ext_results.values()])
        print("-" * 70)
        print(f"  [{'AVERAGE':>12s}] P={avg_all['precision']:.4f}  R={avg_all['recall']:.4f}  "
              f"IoU={avg_all['iou']:.4f}  Dice={avg_all['dice']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
