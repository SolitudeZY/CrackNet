"""
Testing and inference script for DeepLabV3+ crack segmentation.

Supports:
- Dataset evaluation (CRACK500 val/test with auto-threshold)
- Single image / directory inference
- TTA (test-time augmentation)
- Morphological post-processing for thinner, cleaner masks
- EMA model loading

单图：
  python /home/fs-ai/CrackNet/script/test.py \
    --weights /home/fs-ai/CrackNet/script/runs/20260401_151044/weights/best_iou.pt \
    --image /home/fs-ai/Pictures/crack_2.png \
    --refine \
    --threshold 0.5

  更强一点去 blob：
  python /home/fs-ai/CrackNet/script/test.py \
    --weights /home/fs-ai/CrackNet/script/runs/20260401_151044/weights/best_iou.pt \
    --image <你的原图路径> \
    --refine \
    --threshold 0.55 \
    --max-blob-area 200 \
    --min-blob-aspect-ratio 2.8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from model import build_model
from sliding_window import sliding_window_predict
from metrics import compute_metrics, average_metric_list
from dataset import DATASET_MEAN, DATASET_STD, CRACK500_ROOT


DEFAULT_DATASET_ROOT = CRACK500_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test CrackNet on CRACK500 or run camera-style inference")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--auto-threshold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threshold-metric", type=str, default="iou", choices=["iou", "dice", "precision"])
    parser.add_argument("--threshold-min", type=float, default=0.3)
    parser.add_argument("--threshold-max", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--input-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    # Post-processing options
    parser.add_argument("--refine", action="store_true",
                        help="Apply morphological refinement to thin coarse masks")
    parser.add_argument("--refine-kernel", type=int, default=1,
                        help="Kernel size for optional morphological cleanup before refinement")
    parser.add_argument("--min-component-area", type=int, default=4,
                        help="Remove connected components smaller than this area during refinement")
    parser.add_argument("--max-blob-area", type=int, default=400,
                        help="For larger components, require a crack-like aspect ratio or remove them")
    parser.add_argument("--min-blob-aspect-ratio", type=float, default=2.2,
                        help="Minimum aspect ratio for large refined components to be kept")
    parser.add_argument("--use-ema", action="store_true",
                        help="Load EMA weights if available in checkpoint")
    return parser.parse_args()


def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for c in range(3):
        image[:, :, c] = (image[:, :, c] - DATASET_MEAN[c]) / DATASET_STD[c]
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()


def filter_connected_components(
    mask_binary: np.ndarray,
    min_component_area: int,
    max_blob_area: int,
    min_blob_aspect_ratio: float,
) -> np.ndarray:
    """Remove only obviously spurious dense blobs while preserving elongated crack components."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    filtered = np.zeros_like(mask_binary)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        short_side = max(min(width, height), 1)
        long_side = max(width, height)
        aspect_ratio = long_side / short_side
        fill_ratio = area / max(width * height, 1)

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_mask = (labels[y:y + height, x:x + width] == label).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = float(cv2.contourArea(contours[0])) if contours else 0.0
        perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0
        circularity = 0.0
        if perimeter > 1e-6:
            circularity = 4.0 * np.pi * contour_area / (perimeter * perimeter)

        touches_border = (
            x == 0
            or y == 0
            or x + width >= mask_binary.shape[1]
            or y + height >= mask_binary.shape[0]
        )
        is_dense_blob = fill_ratio > 0.55
        is_roundish = aspect_ratio < min_blob_aspect_ratio
        is_circular = circularity > 0.22

        if area >= max_blob_area and is_dense_blob and is_roundish and is_circular and not touches_border:
            continue

        filtered[labels == label] = 255

    return filtered


def refine_mask(
    mask_binary: np.ndarray,
    kernel_size: int = 1,
    min_component_area: int = 4,
    max_blob_area: int = 400,
    min_blob_aspect_ratio: float = 2.2,
) -> np.ndarray:
    """Refine crack masks with optional cleanup, conservative component filtering, thinning, and slight dilation."""
    cleaned = mask_binary.copy()
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    cleaned = filter_connected_components(
        cleaned,
        min_component_area=min_component_area,
        max_blob_area=max_blob_area,
        min_blob_aspect_ratio=min_blob_aspect_ratio,
    )

    skeleton = cv2.ximgproc.thinning(cleaned, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    refined = cv2.dilate(skeleton, dilate_kernel, iterations=1)
    return refined


def refine_mask_fallback(
    mask_binary: np.ndarray,
    kernel_size: int = 1,
    min_component_area: int = 4,
    max_blob_area: int = 400,
    min_blob_aspect_ratio: float = 2.2,
) -> np.ndarray:
    """Fallback refinement without ximgproc, while keeping component filtering conservative."""
    cleaned = mask_binary.copy()
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    cleaned = filter_connected_components(
        cleaned,
        min_component_area=min_component_area,
        max_blob_area=max_blob_area,
        min_blob_aspect_ratio=min_blob_aspect_ratio,
    )
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thinned = cv2.erode(cleaned, thin_kernel, iterations=1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    refined = cv2.dilate(thinned, dilate_kernel, iterations=1)
    return refined


def apply_refinement(
    mask_binary: np.ndarray,
    kernel_size: int = 1,
    min_component_area: int = 4,
    max_blob_area: int = 400,
    min_blob_aspect_ratio: float = 2.2,
) -> np.ndarray:
    """Apply crack-specific refinement with fallback if ximgproc is not available."""
    try:
        return refine_mask(
            mask_binary,
            kernel_size=kernel_size,
            min_component_area=min_component_area,
            max_blob_area=max_blob_area,
            min_blob_aspect_ratio=min_blob_aspect_ratio,
        )
    except (AttributeError, cv2.error):
        return refine_mask_fallback(
            mask_binary,
            kernel_size=kernel_size,
            min_component_area=min_component_area,
            max_blob_area=max_blob_area,
            min_blob_aspect_ratio=min_blob_aspect_ratio,
        )


def run_inference(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    device: torch.device,
    patch_size: int,
    stride: int,
    tta: bool,
) -> np.ndarray:
    image_t = preprocess_image(image_bgr)
    prob = sliding_window_predict(
        model=model,
        image=image_t,
        patch_size=patch_size,
        stride=stride,
        device=device,
        use_amp=device.type == "cuda",
        tta=tta,
    )
    return prob.squeeze().cpu().numpy()


def save_prediction(
    save_prefix: Path,
    image_bgr: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    refine: bool = False,
    refine_kernel: int = 1,
    min_component_area: int = 4,
    max_blob_area: int = 400,
    min_blob_aspect_ratio: float = 2.2,
) -> np.ndarray:
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    mask = (prob > threshold).astype(np.uint8) * 255

    if refine:
        mask = apply_refinement(
            mask,
            kernel_size=refine_kernel,
            min_component_area=min_component_area,
            max_blob_area=max_blob_area,
            min_blob_aspect_ratio=min_blob_aspect_ratio,
        )

    overlay = image_bgr.copy()
    overlay[mask > 0] = [0, 0, 255]
    blended = cv2.addWeighted(image_bgr, 0.65, overlay, 0.35, 0)
    heatmap = cv2.applyColorMap((prob * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    result = np.concatenate([image_bgr, blended, heatmap], axis=1)

    cv2.imwrite(str(save_prefix.with_name(f"{save_prefix.name}_mask.png")), mask)
    cv2.imwrite(str(save_prefix.with_name(f"{save_prefix.name}_overlay.jpg")), blended)
    cv2.imwrite(str(save_prefix.with_name(f"{save_prefix.name}_heatmap.jpg")), heatmap)
    cv2.imwrite(str(save_prefix.with_name(f"{save_prefix.name}_result.jpg")), result)
    return mask


def load_split_samples(dataset_root: Path, split: str) -> list[tuple[Path, Path]]:
    split_file = dataset_root / f"{split}.txt"
    samples = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            img_path = dataset_root / parts[0]
            mask_path = dataset_root / parts[1]
            # Handle nested directory (e.g. traincrop/traincrop/xxx.jpg)
            if not img_path.exists():
                subdir = parts[0].split("/")[0]
                fname = "/".join(parts[0].split("/")[1:])
                img_path = dataset_root / subdir / subdir / fname
                mask_path = dataset_root / subdir / subdir / "/".join(parts[1].split("/")[1:])
            if img_path.exists() and mask_path.exists():
                samples.append((img_path, mask_path))
    return samples


@torch.no_grad()
def collect_split_predictions(
    model: torch.nn.Module,
    dataset_root: Path,
    split: str,
    device: torch.device,
    patch_size: int,
    stride: int,
    tta: bool,
) -> list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]]:
    cached = []
    for img_path, mask_path in load_split_samples(dataset_root, split):
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != image_bgr.shape[:2]:
            mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        prob = run_inference(model, image_bgr, device, patch_size, stride, tta)
        cached.append((img_path, image_bgr, prob, (mask > 0).astype(np.float32)))
    return cached


def search_best_threshold(
    cached_predictions: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]],
    metric_name: str,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[float, dict[str, float]]:
    thresholds = np.arange(threshold_min, threshold_max + 1e-8, threshold_step)
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "iou": 0.0, "dice": 0.0}
    best_score = -1.0
    for threshold in thresholds:
        metric_list = [compute_metrics(prob, mask, float(threshold)) for _, _, prob, mask in cached_predictions]
        avg_metrics = average_metric_list(metric_list)
        score = avg_metrics[metric_name]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = avg_metrics
    return best_threshold, best_metrics


def evaluate_cached(
    cached_predictions: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]],
    threshold: float,
    save_dir: Path | None = None,
    refine: bool = False,
    refine_kernel: int = 1,
    min_component_area: int = 4,
    max_blob_area: int = 400,
    min_blob_aspect_ratio: float = 2.2,
) -> dict[str, float]:
    metrics = []
    for img_path, image_bgr, prob, mask in cached_predictions:
        metrics.append(compute_metrics(prob, mask, threshold))
        if save_dir is not None:
            save_prediction(
                save_dir / img_path.stem,
                image_bgr,
                prob,
                threshold,
                refine=refine,
                refine_kernel=refine_kernel,
                min_component_area=min_component_area,
                max_blob_area=max_blob_area,
                min_blob_aspect_ratio=min_blob_aspect_ratio,
            )
    return average_metric_list(metrics)


def infer_path(
    model: torch.nn.Module,
    path: Path,
    output_dir: Path,
    device: torch.device,
    patch_size: int,
    stride: int,
    tta: bool,
    threshold: float,
    refine: bool = False,
    refine_kernel: int = 1,
    min_component_area: int = 4,
    max_blob_area: int = 400,
    min_blob_aspect_ratio: float = 2.2,
) -> None:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    prob = run_inference(model, image_bgr, device, patch_size, stride, tta)
    refined_mask = save_prediction(
        output_dir / path.stem,
        image_bgr,
        prob,
        threshold,
        refine=refine,
        refine_kernel=refine_kernel,
        min_component_area=min_component_area,
        max_blob_area=max_blob_area,
        min_blob_aspect_ratio=min_blob_aspect_ratio,
    )
    mask_pixels = int((refined_mask > 0).sum())
    total_pixels = int(prob.size)
    print(f"{path.name}: crack_pixels={mask_pixels}/{total_pixels} ({mask_pixels / max(total_pixels, 1) * 100:.2f}%)")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    patch_size = int(ckpt.get("patch_size", cfg.get("patch_size", args.patch_size)))
    stride = int(ckpt.get("stride", cfg.get("stride", args.stride)))

    model = build_model(pretrained=False).to(device)

    # Load EMA weights if requested and available
    if args.use_ema and "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    threshold = float(ckpt.get("val_metrics", {}).get("threshold", 0.5)) if args.threshold < 0 else args.threshold

    print(f"Device: {device}")
    print(f"Checkpoint: {args.weights}")
    print(f"Threshold: {threshold:.2f}{' (from checkpoint)' if args.threshold < 0 else ''}")
    print(f"Patch size: {patch_size}")
    print(f"Stride: {stride}")
    if args.refine:
        print(
            "Refinement: ON "
            f"(kernel={args.refine_kernel}, min_area={args.min_component_area}, "
            f"max_blob_area={args.max_blob_area}, min_blob_aspect_ratio={args.min_blob_aspect_ratio})"
        )

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(args.weights).resolve().parent.parent / "predictions"

    if args.image:
        infer_path(
            model,
            Path(args.image).expanduser().resolve(),
            output_dir,
            device,
            patch_size,
            stride,
            args.tta,
            threshold,
            refine=args.refine,
            refine_kernel=args.refine_kernel,
            min_component_area=args.min_component_area,
            max_blob_area=args.max_blob_area,
            min_blob_aspect_ratio=args.min_blob_aspect_ratio,
        )
        return

    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
        for image_path in sorted(input_dir.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                infer_path(
                    model,
                    image_path,
                    output_dir,
                    device,
                    patch_size,
                    stride,
                    args.tta,
                    threshold,
                    refine=args.refine,
                    refine_kernel=args.refine_kernel,
                    min_component_area=args.min_component_area,
                    max_blob_area=args.max_blob_area,
                    min_blob_aspect_ratio=args.min_blob_aspect_ratio,
                )
        return

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    val_cached = collect_split_predictions(model, dataset_root, "val", device, patch_size, stride, args.tta)
    if args.threshold < 0 and not args.auto_threshold:
        val_metrics = evaluate_cached(val_cached, threshold)
    elif args.auto_threshold:
        threshold, val_metrics = search_best_threshold(
            val_cached,
            metric_name=args.threshold_metric,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
        print(f"Auto threshold selected from val: {threshold:.2f} ({args.threshold_metric}={val_metrics[args.threshold_metric]:.4f})")
    else:
        val_metrics = evaluate_cached(val_cached, threshold)

    vis_root = output_dir if args.save_vis else None
    if vis_root is not None:
        (vis_root / "val").mkdir(parents=True, exist_ok=True)
    val_metrics = evaluate_cached(
        val_cached,
        threshold,
        vis_root / "val" if vis_root is not None else None,
        refine=args.refine,
        refine_kernel=args.refine_kernel,
        min_component_area=args.min_component_area,
        max_blob_area=args.max_blob_area,
        min_blob_aspect_ratio=args.min_blob_aspect_ratio,
    )
    print(
        f"[val]  P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} "
        f"IoU={val_metrics['iou']:.4f} Dice={val_metrics['dice']:.4f}"
    )

    test_cached = collect_split_predictions(model, dataset_root, "test", device, patch_size, stride, args.tta)
    if vis_root is not None:
        (vis_root / "test").mkdir(parents=True, exist_ok=True)
    test_metrics = evaluate_cached(
        test_cached,
        threshold,
        vis_root / "test" if vis_root is not None else None,
        refine=args.refine,
        refine_kernel=args.refine_kernel,
        min_component_area=args.min_component_area,
        max_blob_area=args.max_blob_area,
        min_blob_aspect_ratio=args.min_blob_aspect_ratio,
    )
    print(
        f"[test] P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} "
        f"IoU={test_metrics['iou']:.4f} Dice={test_metrics['dice']:.4f}"
    )

    print("\nSummary")
    print(f"threshold: {threshold:.2f}")
    print(f"  val: P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} IoU={val_metrics['iou']:.4f} Dice={val_metrics['dice']:.4f}")
    print(f" test: P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} IoU={test_metrics['iou']:.4f} Dice={test_metrics['dice']:.4f}")


if __name__ == "__main__":
    main()
