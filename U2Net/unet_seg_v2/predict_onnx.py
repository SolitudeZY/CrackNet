import argparse
import json
import math
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# ======================== 配置 ========================
# 动态加载均值和标准差 (与训练时保持一致)
STATS_FILE = Path(__file__).resolve().parent / "dataset_stats.json"

THRESHOLD = 0.5
TILE_SIZE = 640        # 滑动窗口大小
OVERLAP = 1/3          # 窗口重叠比例

if STATS_FILE.exists():
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)
        IMAGENET_MEAN = np.array(stats["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        IMAGENET_STD = np.array(stats["std"], dtype=np.float32).reshape(1, 3, 1, 1)
else:
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# ======================== 辅助函数 ========================
def preprocess_image(image: Image.Image) -> np.ndarray:
    """将 PIL Image 转换为模型所需的归一化 numpy array (1, C, H, W)"""
    img_np = np.array(image).astype(np.float32) / 255.0
    # HWC to CHW
    img_np = np.transpose(img_np, (2, 0, 1))
    # Add batch dimension
    img_np = np.expand_dims(img_np, axis=0)
    # Normalize
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    return img_np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def overlay_mask(image: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.5) -> Image.Image:
    """将二值掩码叠加到原图上"""
    img_np = np.array(image).copy()
    
    # 确保 mask 形状和图像一致
    if mask.shape[:2] != (img_np.shape[0], img_np.shape[1]):
        mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    binary_mask = mask > 0
    img_np[binary_mask] = img_np[binary_mask] * (1 - alpha) + np.array(color) * alpha
    return Image.fromarray(img_np.astype(np.uint8))

# ======================== 推理核心 ========================
def predict_onnx_sliding_window(ort_session, image: Image.Image) -> np.ndarray:
    """使用 ONNX Runtime 进行滑动窗口推理"""
    w, h = image.size
    orig_w, orig_h = w, h
    
    # Padding
    if w < TILE_SIZE or h < TILE_SIZE:
        pad_w = max(0, TILE_SIZE - w)
        pad_h = max(0, TILE_SIZE - h)
        # PIL padding is (left, top, right, bottom)
        from PIL import ImageOps
        image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
        w, h = image.size

    # 预处理全图
    full_tensor = preprocess_image(image) # (1, 3, H, W)
    
    stride = int(TILE_SIZE * (1 - OVERLAP))
    n_rows = math.ceil((h - TILE_SIZE) / stride) + 1
    n_cols = math.ceil((w - TILE_SIZE) / stride) + 1
    
    pred_map = np.zeros((1, h, w), dtype=np.float32)
    count_map = np.zeros((1, h, w), dtype=np.float32)
    
    input_name = ort_session.get_inputs()[0].name
    
    # 开始滑动窗口
    t0 = time.time()
    for row in range(n_rows):
        for col in range(n_cols):
            y1 = min(row * stride, h - TILE_SIZE)
            x1 = min(col * stride, w - TILE_SIZE)
            y2 = y1 + TILE_SIZE
            x2 = x1 + TILE_SIZE
            
            # 裁剪小块 (1, 3, 640, 640)
            tile = full_tensor[:, :, y1:y2, x1:x2]
            
            # ONNX 推理
            logits = ort_session.run(None, {input_name: tile})[0] # 输出是列表，取第一个
            prob = sigmoid(logits).squeeze(0) # (1, 640, 640)
            
            pred_map[:, y1:y2, x1:x2] += prob
            count_map[:, y1:y2, x1:x2] += 1.0
            
    infer_time = time.time() - t0
    pred_map /= count_map
    
    # 裁剪回原始尺寸
    pred_map = np.squeeze(pred_map, axis=0) # (H, W)
    if pred_map.shape[0] > orig_h or pred_map.shape[1] > orig_w:
        pred_map = pred_map[:orig_h, :orig_w]
        
    pred_mask = (pred_map > THRESHOLD).astype(np.uint8) * 255
    return pred_mask, infer_time

def main():
    parser = argparse.ArgumentParser(description="Test ONNX model inference")
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model file")
    parser.add_argument("--image", type=str, required=True, help="Path to the test image")
    args = parser.parse_args()

    # 1. 加载 ONNX 模型
    print(f"Loading ONNX model: {args.onnx}")
    # 优先尝试使用 CUDA (GPU)，如果没有则回退到 CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        ort_session = ort.InferenceSession(args.onnx, providers=providers)
        active_provider = ort_session.get_providers()[0]
        print(f"✅ ONNX Runtime initialized with provider: {active_provider}")
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        return

    # 2. 读取图片
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return
        
    image = Image.open(img_path).convert("RGB")
    print(f"Processing image: {img_path.name} (Size: {image.size})")

    # 3. 推理
    pred_mask, infer_time = predict_onnx_sliding_window(ort_session, image)
    print(f"⚡ Inference completed in {infer_time:.4f} seconds.")

    # 4. 保存结果
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent / "runs" / "onnx_test" / f"exp_{current_time}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mask_out = output_dir / f"{img_path.stem}_mask.png"
    Image.fromarray(pred_mask).save(mask_out)
    
    vis_out = output_dir / f"{img_path.stem}_overlay.jpg"
    vis = overlay_mask(image, pred_mask, color=(0, 255, 0)) # ONNX预测结果用绿色显示
    vis.save(vis_out)
    
    print(f"✅ Results saved to {output_dir}")

if __name__ == "__main__":
    main()
