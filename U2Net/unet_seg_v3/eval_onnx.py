"""
ONNX 模型数据集评估脚本: 支持 YOLOv8-seg 和 U-Net 两种模型格式。

自动检测模型类型，在指定数据集文件夹上计算 Precision / Recall / IoU / Dice。

用法:
  # U-Net 模型评估
  python eval_onnx.py --onnx model.onnx --dataset /path/to/dataset --type unet

  # YOLOv8-seg 模型评估
  python eval_onnx.py --onnx yolo.onnx --dataset /path/to/dataset --type yolo

  # 自动检测模型类型
  python eval_onnx.py --onnx model.onnx --dataset /path/to/dataset

  # 指定掩码目录 (DeepCrack 格式)
  python eval_onnx.py --onnx model.onnx --images /path/to/images --masks /path/to/masks

数据集目录格式支持:
  1. CRACK500: 图片 .jpg 和掩码 .png 在同一目录
  2. DeepCrack: --images 和 --masks 分别指定
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


# ======================== 指标计算 ========================
def calculate_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """计算单张图片的 Precision, Recall, IoU, Dice。pred/target: 二值 {0,1}"""
    pred_flat = pred.astype(bool).ravel()
    target_flat = target.astype(bool).ravel()

    tp = (pred_flat & target_flat).sum()
    fp = (pred_flat & ~target_flat).sum()
    fn = (~pred_flat & target_flat).sum()

    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    return {"precision": precision, "recall": recall, "iou": iou, "dice": dice}


# ======================== 模型类型检测 ========================
def detect_model_type(session: ort.InferenceSession) -> str:
    """根据输出 shape 自动判断模型类型。"""
    outputs = session.get_outputs()
    out0_shape = outputs[0].shape

    # YOLOv8-seg: output0=[1,37,8400], output1=[1,32,160,160]
    if len(outputs) >= 2 and len(out0_shape) == 3:
        return "yolo"
    # U-Net: output=[B,1,H,W]
    if len(out0_shape) == 4:
        return "unet"

    raise ValueError(f"无法识别模型类型，输出 shape: {[o.shape for o in outputs]}")


# ======================== U-Net 推理 ========================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.clip(-x, -50, 50)))


def predict_unet(
    session: ort.InferenceSession,
    image: np.ndarray,
    img_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    U-Net 滑动窗口推理。
    image: BGR numpy (H, W, 3)
    返回: 二值掩码 (H, W) uint8 {0, 255}
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    orig_h, orig_w = img_rgb.shape[:2]

    # Pad if needed
    h, w = orig_h, orig_w
    if h < img_size or w < img_size:
        pad_h = max(0, img_size - h)
        pad_w = max(0, img_size - w)
        img_rgb = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        h, w = img_rgb.shape[:2]

    # HWC → NCHW + normalize
    img_chw = img_rgb.transpose(2, 0, 1)  # (3, H, W)
    img_norm = (img_chw - mean.reshape(3, 1, 1)) / std.reshape(3, 1, 1)
    img_norm = img_norm[np.newaxis]  # (1, 3, H, W)

    overlap = 1 / 3
    stride = int(img_size * (1 - overlap))
    n_rows = max(1, (h - img_size + stride - 1) // stride + 1)
    n_cols = max(1, (w - img_size + stride - 1) // stride + 1)

    pred_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    input_name = session.get_inputs()[0].name

    for row in range(n_rows):
        for col in range(n_cols):
            y1 = min(row * stride, h - img_size)
            x1 = min(col * stride, w - img_size)
            tile = img_norm[:, :, y1:y1 + img_size, x1:x1 + img_size].astype(np.float32)

            logits = session.run(None, {input_name: tile})[0]
            prob = sigmoid(logits).squeeze()  # (img_size, img_size)

            pred_map[y1:y1 + img_size, x1:x1 + img_size] += prob
            count_map[y1:y1 + img_size, x1:x1 + img_size] += 1.0

    pred_map /= np.maximum(count_map, 1.0)
    pred_map = pred_map[:orig_h, :orig_w]

    return ((pred_map > threshold) * 255).astype(np.uint8)


# ======================== YOLOv8-seg 推理 ========================
def predict_yolo(
    session: ort.InferenceSession,
    image: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """
    YOLOv8-seg 推理。
    image: BGR numpy (H, W, 3)
    返回: 合并后的二值分割掩码 (H, W) uint8 {0, 255}
    """
    orig_h, orig_w = image.shape[:2]

    # 获取模型输入尺寸
    inp_shape = session.get_inputs()[0].shape
    inp_h = inp_shape[2] if isinstance(inp_shape[2], int) else 640
    inp_w = inp_shape[3] if isinstance(inp_shape[3], int) else 640

    # Letterbox resize (保持宽高比)
    scale = min(inp_h / orig_h, inp_w / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    pad_h, pad_w = (inp_h - new_h) // 2, (inp_w - new_w) // 2

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((inp_h, inp_w, 3), 114, dtype=np.uint8)
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    # 预处理: BGR → RGB, HWC → NCHW, normalize to [0,1]
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})

    # output0: [1, 4+nc+nm, num_dets] → nm=32 mask coefficients
    # output1: [1, 32, mask_h, mask_w] → prototype masks
    det_output = outputs[0]  # (1, 37, 8400)
    proto = outputs[1]       # (1, 32, 160, 160)

    # 解析检测: [1, 37, 8400] → 转置为 [8400, 37]
    det = det_output[0].T  # (8400, 37)

    # 前 4 列: cx, cy, w, h
    boxes = det[:, :4]
    # 接下来 nc 列: class scores (37-4-32=1 class)
    nc = det.shape[1] - 4 - 32
    class_scores = det[:, 4:4 + nc]
    # 最后 32 列: mask coefficients
    mask_coefs = det[:, 4 + nc:]

    # 过滤低置信度
    if nc == 1:
        scores = class_scores[:, 0]
    else:
        scores = class_scores.max(axis=1)

    valid = scores > conf_threshold
    if not valid.any():
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    boxes = boxes[valid]
    scores = scores[valid]
    mask_coefs = mask_coefs[valid]

    # cx,cy,w,h → x1,y1,x2,y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # NMS
    indices = _nms(xyxy, scores, iou_threshold)
    if len(indices) == 0:
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    xyxy = xyxy[indices]
    mask_coefs = mask_coefs[indices]

    # 生成实例掩码: mask_coefs @ proto → per-instance masks
    proto_squeezed = proto[0]  # (32, mask_h, mask_w)
    mask_h, mask_w = proto_squeezed.shape[1], proto_squeezed.shape[2]

    # (N, 32) @ (32, mask_h*mask_w) → (N, mask_h*mask_w)
    inst_masks = sigmoid(mask_coefs @ proto_squeezed.reshape(32, -1))
    inst_masks = inst_masks.reshape(-1, mask_h, mask_w)  # (N, mask_h, mask_w)

    # 合并所有实例掩码为语义掩码
    merged = np.zeros((mask_h, mask_w), dtype=np.float32)
    for i in range(inst_masks.shape[0]):
        # 用 bbox 裁剪掩码 (缩放到 mask 坐标系)
        sx, sy = mask_w / inp_w, mask_h / inp_h
        bx1 = max(0, int(xyxy[i, 0] * sx))
        by1 = max(0, int(xyxy[i, 1] * sy))
        bx2 = min(mask_w, int(xyxy[i, 2] * sx))
        by2 = min(mask_h, int(xyxy[i, 3] * sy))

        crop_mask = np.zeros_like(inst_masks[i])
        crop_mask[by1:by2, bx1:bx2] = inst_masks[i, by1:by2, bx1:bx2]
        merged = np.maximum(merged, crop_mask)

    # resize 回原图尺寸 (先去 letterbox padding 再 resize)
    merged_full = cv2.resize(merged, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)
    # 去 padding
    merged_crop = merged_full[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
    # resize 到原始尺寸
    merged_orig = cv2.resize(merged_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return ((merged_orig > 0.5) * 255).astype(np.uint8)


def _nms(boxes, scores, iou_threshold):
    """简单的 NMS 实现。"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


# ======================== 数据集收集 ========================
def collect_pairs(
    dataset_dir: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    masks_dir: Optional[Path] = None,
) -> List[Tuple[Path, Path]]:
    """
    收集 (图片, 掩码) 对。
    支持:
      1. --dataset: CRACK500 格式 (同目录, .jpg + .png)
      2. --images + --masks: DeepCrack 格式 (分离目录, 多种后缀)
    """
    pairs = []
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG"}
    mask_exts = {".bmp", ".png", ".jpg", ".tif"}

    if images_dir and masks_dir:
        # DeepCrack 模式
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix not in img_exts:
                continue
            for ext in mask_exts:
                mask_path = masks_dir / (img_path.stem + ext)
                if mask_path.exists():
                    pairs.append((img_path, mask_path))
                    break
    elif dataset_dir:
        # CRACK500 模式: .jpg = image, .png = mask
        for img_path in sorted(dataset_dir.glob("*.jpg")):
            mask_path = img_path.with_suffix(".png")
            if mask_path.exists():
                pairs.append((img_path, mask_path))
        # 也支持同目录下的其他组合
        if not pairs:
            for img_path in sorted(dataset_dir.iterdir()):
                if img_path.suffix in img_exts:
                    for ext in mask_exts:
                        mask_path = dataset_dir / (img_path.stem + ext)
                        if mask_path.exists() and mask_path != img_path:
                            pairs.append((img_path, mask_path))
                            break

    return pairs


# ======================== 主评估函数 ========================
def evaluate(
    onnx_path: str,
    model_type: str,
    pairs: List[Tuple[Path, Path]],
    img_size: int = 640,
    threshold: float = 0.5,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """在数据集上评估 ONNX 模型。"""

    # 加载模型
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    active = session.get_providers()[0]
    print(f"  Provider: {active}")

    # 自动检测模型类型
    if model_type == "auto":
        model_type = detect_model_type(session)
    print(f"  Model type: {model_type}")

    all_metrics = {"precision": [], "recall": [], "iou": [], "dice": []}
    total_time = 0.0

    for i, (img_path, mask_path) in enumerate(pairs):
        # 读取
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or gt_mask is None:
            continue

        # GT mask → 二值
        if gt_mask.shape[:2] != image.shape[:2]:
            gt_mask = cv2.resize(gt_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_binary = (gt_mask > 127).astype(np.uint8)

        # 推理
        t0 = time.time()
        if model_type == "unet":
            pred_mask = predict_unet(session, image, img_size, mean, std, threshold)
        elif model_type == "yolo":
            pred_mask = predict_yolo(session, image)
        else:
            raise ValueError(f"未知模型类型: {model_type}")
        total_time += time.time() - t0

        # 计算指标
        pred_binary = (pred_mask > 127).astype(np.uint8)
        m = calculate_metrics(pred_binary, gt_binary)
        for k in all_metrics:
            all_metrics[k].append(m[k])

        if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
            avg_iou = np.mean(all_metrics["iou"])
            print(f"    [{i + 1}/{len(pairs)}] running IoU={avg_iou:.4f} ({total_time/(i+1)*1000:.0f}ms/img)")

    avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    avg["avg_time_ms"] = total_time / len(pairs) * 1000 if pairs else 0
    avg["num_images"] = len(pairs)
    return avg


def main():
    parser = argparse.ArgumentParser(description="ONNX 模型数据集评估 (支持 YOLO / U-Net)")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("--type", type=str, default="auto", choices=["auto", "yolo", "unet"],
                        help="模型类型 (默认自动检测)")
    # 数据集指定方式 (二选一)
    parser.add_argument("--dataset", type=str, default=None,
                        help="数据集目录 (CRACK500 格式: .jpg + .png 同目录)")
    parser.add_argument("--images", type=str, default=None,
                        help="图片目录 (DeepCrack 格式)")
    parser.add_argument("--masks", type=str, default=None,
                        help="掩码目录 (DeepCrack 格式)")
    # 推理参数
    parser.add_argument("--img-size", type=int, default=None,
                        help="U-Net 滑动窗口大小 (默认: 从模型输入推断)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="U-Net 二值化阈值 (默认 0.5)")
    parser.add_argument("--norm", type=str, default="imagenet",
                        choices=["imagenet", "crack500"],
                        help="归一化方式 (默认 imagenet)")
    # 输出
    parser.add_argument("--save", type=str, default=None,
                        help="保存结果到 JSON 文件")
    args = parser.parse_args()

    # 归一化参数
    if args.norm == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1)
    else:
        mean = np.array([0.4942, 0.4959, 0.4968], dtype=np.float32).reshape(3, 1)
        std = np.array([0.1746, 0.1726, 0.1690], dtype=np.float32).reshape(3, 1)

    # 推断 img_size
    if args.img_size is None:
        session_tmp = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
        inp_shape = session_tmp.get_inputs()[0].shape
        if isinstance(inp_shape[2], int):
            args.img_size = inp_shape[2]
        else:
            args.img_size = 640
        print(f"推断 img_size={args.img_size}")

    # 收集数据
    if args.images and args.masks:
        pairs = collect_pairs(images_dir=Path(args.images), masks_dir=Path(args.masks))
        ds_name = Path(args.images).parent.name
    elif args.dataset:
        pairs = collect_pairs(dataset_dir=Path(args.dataset))
        ds_name = Path(args.dataset).name
    else:
        parser.error("请指定 --dataset 或 --images + --masks")
        return

    if not pairs:
        print("错误: 未找到任何图片-掩码配对")
        return

    print(f"\n{'='*60}")
    print(f"模型: {Path(args.onnx).name}")
    print(f"数据集: {ds_name} ({len(pairs)} 张)")
    print(f"{'='*60}")

    result = evaluate(
        onnx_path=args.onnx,
        model_type=args.type,
        pairs=pairs,
        img_size=args.img_size,
        threshold=args.threshold,
        mean=mean,
        std=std,
    )

    print(f"\n{'─'*60}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  IoU:       {result['iou']:.4f}")
    print(f"  Dice/F1:   {result['dice']:.4f}")
    print(f"  Avg Time:  {result['avg_time_ms']:.1f} ms/img")
    print(f"{'─'*60}")

    if args.save:
        result["model"] = str(args.onnx)
        result["dataset"] = ds_name
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存: {args.save}")

    return result


if __name__ == "__main__":
    main()
