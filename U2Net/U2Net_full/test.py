"""
U2Net 测试与推理脚本，默认面向 CRACK500。

示例:
  python test.py --weights runs/train/exp_xxx/weights/best_iou.pt
  python test.py --weights runs/train/exp_xxx/weights/best_iou.pt --tta --save-vis
  python test.py --weights runs/train/exp_xxx/weights/best_iou.pt --image /path/to/demo.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import DATASET_MEAN, DATASET_STD
from model import build_model

CRACK500_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/crack500/CRACK500")
DEFAULT_IMG_SIZE = 512


def calculate_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = (target > 0.5).astype(np.float32)
    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()
    return {
        "precision": float(tp / (tp + fp + 1e-6)),
        "recall": float(tp / (tp + fn + 1e-6)),
        "iou": float(tp / (tp + fp + fn + 1e-6)),
        "dice": float((2 * tp) / (2 * tp + fp + fn + 1e-6)),
    }


def summarize_metrics(metric_list: list[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {"precision": 0.0, "recall": 0.0, "iou": 0.0, "dice": 0.0}
    keys = metric_list[0].keys()
    return {key: float(np.mean([m[key] for m in metric_list])) for key in keys}


def infer_patch(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    img_size: int,
    tta: bool,
) -> np.ndarray:
    h, w = image.shape[:2]
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32) / 255.0
    for c in range(3):
        resized[:, :, c] = (resized[:, :, c] - DATASET_MEAN[c]) / DATASET_STD[c]

    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    amp_enabled = device.type == "cuda"

    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
        pred = torch.sigmoid(model(tensor)[0])

    if tta:
        tensor_flip = torch.flip(tensor, dims=[3])
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            pred_flip = torch.sigmoid(model(tensor_flip)[0])
        pred = (pred + torch.flip(pred_flip, dims=[3])) / 2.0

    pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
    return pred.squeeze().detach().cpu().numpy()


def sliding_window_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    img_size: int,
    stride: Optional[int] = None,
    tta: bool = False,
) -> np.ndarray:
    h, w = image.shape[:2]
    window = img_size
    stride = stride or max(1, img_size * 2 // 3)

    if h <= window and w <= window:
        return infer_patch(model, image, device, img_size, tta)

    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(1, h - window + 1), stride))
    xs = list(range(0, max(1, w - window + 1), stride))
    if ys[-1] != h - window:
        ys.append(max(0, h - window))
    if xs[-1] != w - window:
        xs.append(max(0, w - window))

    for y in ys:
        for x in xs:
            patch = image[y:y + window, x:x + window]
            pred = infer_patch(model, patch, device, img_size, tta)
            ph, pw = pred.shape
            prob_map[y:y + ph, x:x + pw] += pred
            count_map[y:y + ph, x:x + pw] += 1.0

    return prob_map / np.maximum(count_map, 1.0)


def save_visualization(image_rgb: np.ndarray, prob: np.ndarray, save_prefix: Path, threshold: float) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask_bin = (prob > threshold).astype(np.uint8) * 255
    overlay = image_bgr.copy()
    overlay[mask_bin > 0] = [0, 0, 255]
    blended = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)
    heatmap = cv2.applyColorMap((prob * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    result = np.concatenate([image_bgr, blended, heatmap], axis=1)

    cv2.imwrite(str(save_prefix.with_name(f"{save_prefix.name}_mask.png")), mask_bin)
    cv2.imwrite(str(save_prefix.with_name(f"{save_prefix.name}_overlay.jpg")), blended)
    cv2.imwrite(str(save_prefix.with_name(f"{save_prefix.name}_heatmap.jpg")), heatmap)
    cv2.imwrite(str(save_prefix.with_name(f"{save_prefix.name}_result.jpg")), result)


def load_split_samples(dataset_root: Path, split: str) -> list[tuple[Path, Path]]:
    split_file = dataset_root / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file 不存在: {split_file}")

    samples = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            img_path = dataset_root / parts[0]
            mask_path = dataset_root / parts[1]
            if img_path.exists() and mask_path.exists():
                samples.append((img_path, mask_path))
    return samples


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    dataset_root: Path,
    split: str,
    device: torch.device,
    img_size: int,
    tta: bool,
) -> list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]]:
    cached = []
    samples = load_split_samples(dataset_root, split)
    for img_path, mask_path in tqdm(samples, desc=f"{split:>5s}", ncols=88):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != image_rgb.shape[:2]:
            mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        prob = sliding_window_inference(model, image_rgb, device, img_size=img_size, tta=tta)
        cached.append((img_path, image_rgb, prob, (mask > 0).astype(np.float32)))
    return cached


def search_best_threshold(
    cached_predictions: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]],
    metric_name: str = "iou",
    threshold_min: float = 0.4,
    threshold_max: float = 0.9,
    threshold_step: float = 0.02,
) -> tuple[float, Dict[str, float]]:
    thresholds = np.arange(threshold_min, threshold_max + 1e-8, threshold_step)
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "iou": 0.0, "dice": 0.0}
    best_score = -1.0

    for threshold in thresholds:
        metric_list = [calculate_metrics(prob, mask, float(threshold)) for _, _, prob, mask in cached_predictions]
        avg_metrics = summarize_metrics(metric_list)
        score = avg_metrics[metric_name]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = avg_metrics

    return best_threshold, best_metrics


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    dataset_root: Path,
    split: str,
    device: torch.device,
    img_size: int,
    threshold: float,
    tta: bool,
    save_vis_dir: Optional[Path],
) -> Dict[str, float]:
    cached_predictions = collect_predictions(model, dataset_root, split, device, img_size, tta)
    metric_list = [calculate_metrics(prob, mask, threshold) for _, _, prob, mask in cached_predictions]
    avg = summarize_metrics(metric_list)
    print(
        f"[{split}] P={avg['precision']:.4f} R={avg['recall']:.4f} "
        f"IoU={avg['iou']:.4f} Dice={avg['dice']:.4f} ({len(cached_predictions)} 张)"
    )

    if save_vis_dir is not None:
        for img_path, image_rgb, prob, _ in cached_predictions:
            save_visualization(image_rgb, prob, save_vis_dir / img_path.stem, threshold)
    return avg


@torch.no_grad()
def infer_single_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    img_size: int,
    threshold: float,
    tta: bool,
    output_dir: Optional[Path],
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图像: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prob = sliding_window_inference(model, image_rgb, device, img_size=img_size, tta=tta)

    save_dir = output_dir or image_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    save_visualization(image_rgb, prob, save_dir / image_path.stem, threshold)

    mask_pixels = int((prob > threshold).sum())
    total_pixels = int(prob.size)
    print(f"Saved to: {save_dir}")
    print(f"Crack pixels: {mask_pixels} / {total_pixels} ({mask_pixels / max(total_pixels, 1) * 100:.2f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test full U2NET on CRACK500")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default=str(CRACK500_ROOT))
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--model", type=str, default="", choices=["", "u2net", "u2netp"])
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument(
        "--auto-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在 val 集上搜索最佳阈值后再评估 test",
    )
    parser.add_argument("--threshold-metric", type=str, default="iou", choices=["iou", "dice"], help="阈值搜索优化目标")
    parser.add_argument("--threshold-min", type=float, default=0.4)
    parser.add_argument("--threshold-max", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_name = args.model or ckpt.get("model_name") or ckpt.get("config", {}).get("model", "u2net")
    img_size = int(ckpt.get("img_size", ckpt.get("config", {}).get("img_size", args.img_size)))

    model = build_model(model_name, in_ch=3, out_ch=1).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {args.weights}")
    print(f"Input size: {img_size}")

    if args.image:
        infer_single_image(
            model=model,
            image_path=Path(args.image).expanduser().resolve(),
            device=device,
            img_size=img_size,
            threshold=args.threshold,
            tta=args.tta,
            output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
        )
        return

    vis_dir = None
    if args.save_vis:
        vis_dir = Path(args.weights).resolve().parent.parent / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    eval_threshold = args.threshold

    val_cached = collect_predictions(model, dataset_root, "val", device, img_size, args.tta)
    if args.auto_threshold:
        eval_threshold, best_val_metrics = search_best_threshold(
            val_cached,
            metric_name=args.threshold_metric,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
        print(
            f"Auto threshold from val: {eval_threshold:.2f} "
            f"({args.threshold_metric}={best_val_metrics[args.threshold_metric]:.4f})"
        )
        results["val"] = best_val_metrics
    else:
        results["val"] = summarize_metrics([calculate_metrics(prob, mask, eval_threshold) for _, _, prob, mask in val_cached])

    print(
        f"[val] P={results['val']['precision']:.4f} R={results['val']['recall']:.4f} "
        f"IoU={results['val']['iou']:.4f} Dice={results['val']['dice']:.4f} ({len(val_cached)} 张)"
    )
    if vis_dir is not None:
        split_vis_dir = vis_dir / "val"
        split_vis_dir.mkdir(parents=True, exist_ok=True)
        for img_path, image_rgb, prob, _ in val_cached:
            save_visualization(image_rgb, prob, split_vis_dir / img_path.stem, eval_threshold)

    split_vis_dir = None
    if vis_dir is not None:
        split_vis_dir = vis_dir / "test"
        split_vis_dir.mkdir(parents=True, exist_ok=True)
    results["test"] = evaluate_split(
        model=model,
        dataset_root=dataset_root,
        split="test",
        device=device,
        img_size=img_size,
        threshold=eval_threshold,
        tta=args.tta,
        save_vis_dir=split_vis_dir,
    )

    print("\nSummary")
    print(f"threshold: {eval_threshold:.2f}")
    for split, metrics in results.items():
        print(
            f"{split:>5s}: P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f}"
        )


if __name__ == "__main__":
    main()
