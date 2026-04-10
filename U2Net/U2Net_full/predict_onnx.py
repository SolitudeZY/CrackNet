"""
ONNX inference script for U2Net.

CLI is aligned with test.py as closely as possible.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from dataset import DATASET_MEAN, DATASET_STD

CRACK500_ROOT = Path("/home/fs-ai/unet-crack/dataset/CRACK500")
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


def create_session(weights_path: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed. Please install it first.") from exc

    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(weights_path, providers=providers)


def infer_patch(
    session,
    image: np.ndarray,
    img_size: int,
    tta: bool,
) -> np.ndarray:
    h, w = image.shape[:2]
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32) / 255.0
    for c in range(3):
        resized[:, :, c] = (resized[:, :, c] - DATASET_MEAN[c]) / DATASET_STD[c]
    tensor = resized.transpose(2, 0, 1)[None, ...].astype(np.float32)

    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: tensor})[0]

    if tta:
        tensor_flip = tensor[:, :, :, ::-1].copy()
        pred_flip = session.run(None, {input_name: tensor_flip})[0]
        pred = (pred + pred_flip[:, :, :, ::-1]) / 2.0

    pred = pred.squeeze()
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    return pred


def sliding_window_inference(
    session,
    image: np.ndarray,
    img_size: int,
    stride: Optional[int] = None,
    tta: bool = False,
) -> np.ndarray:
    h, w = image.shape[:2]
    window = img_size
    stride = stride or max(1, img_size * 2 // 3)

    if h <= window and w <= window:
        return infer_patch(session, image, img_size, tta)

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
            pred = infer_patch(session, patch, img_size, tta)
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


def collect_predictions(
    session,
    dataset_root: Path,
    split: str,
    img_size: int,
    tta: bool,
) -> list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]]:
    cached = []
    samples = load_split_samples(dataset_root, split)
    for img_path, mask_path in samples:
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != image_rgb.shape[:2]:
            mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        prob = sliding_window_inference(session, image_rgb, img_size=img_size, tta=tta)
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


def evaluate_split(
    session,
    dataset_root: Path,
    split: str,
    img_size: int,
    threshold: float,
    tta: bool,
    save_vis_dir: Optional[Path],
) -> Dict[str, float]:
    cached_predictions = collect_predictions(session, dataset_root, split, img_size, tta)
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


def infer_single_image(
    session,
    image_path: Path,
    img_size: int,
    threshold: float,
    tta: bool,
    output_dir: Optional[Path],
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图像: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prob = sliding_window_inference(session, image_rgb, img_size=img_size, tta=tta)

    save_dir = output_dir or image_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    save_visualization(image_rgb, prob, save_dir / image_path.stem, threshold)

    mask_pixels = int((prob > threshold).sum())
    total_pixels = int(prob.size)
    print(f"Saved to: {save_dir}")
    print(f"Crack pixels: {mask_pixels} / {total_pixels} ({mask_pixels / max(total_pixels, 1) * 100:.2f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with U2Net ONNX model")
    parser.add_argument("--weights", type=str, required=True, help="Path to .onnx model")
    parser.add_argument("--dataset-root", type=str, default=str(CRACK500_ROOT))
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
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
    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {weights_path}")

    session = create_session(str(weights_path))
    print(f"ONNX model: {weights_path}")
    print(f"Providers: {session.get_providers()}")
    print(f"Input size: {args.img_size}")

    if args.image:
        infer_single_image(
            session=session,
            image_path=Path(args.image).expanduser().resolve(),
            img_size=args.img_size,
            threshold=args.threshold,
            tta=args.tta,
            output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
        )
        return

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    vis_dir = None
    if args.save_vis:
        vis_dir = weights_path.parent / "visualizations_onnx"
        vis_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    eval_threshold = args.threshold

    val_cached = collect_predictions(session, dataset_root, "val", args.img_size, args.tta)
    if args.auto_threshold:
        eval_threshold, best_val_metrics = search_best_threshold(
            val_cached,
            metric_name=args.threshold_metric,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
        print(f"Auto threshold from val: {eval_threshold:.2f} ({args.threshold_metric}={best_val_metrics[args.threshold_metric]:.4f})")
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
        session=session,
        dataset_root=dataset_root,
        split="test",
        img_size=args.img_size,
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
