import os
import ctypes

# 方法1：设置环境变量允许重复加载 OpenMP 库
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 方法2：在导入 torch 之前，强制使用 ctypes 提前加载 libgomp.so（GNU OpenMP）和 libiomp5.so（Intel OpenMP）
# 这样可以避免 torch 和 numpy/cv2 在底层抢占多线程资源
try:
    ctypes.cdll.LoadLibrary("libgomp.so.1")
except Exception:
    pass

import json
import math
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt

# 导入模型
from U_Net_MobileNetV2_model import U_Net_MobileNetV2

# ======================== 配置 ========================
DATASET_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/crack500/CRACK500")
STATS_FILE = Path(__file__).resolve().parent / "dataset_stats.json"

TILE_SIZE = 640
OVERLAP = 1/3
THRESHOLD = 0.5

# 自动寻找最新的训练权重 (如果没有指定)
def get_latest_checkpoint():
    runs_dir = Path(__file__).resolve().parent / "runs" / "train"
    if not runs_dir.exists():
        return None
    # 找所有 exp_xxx 文件夹
    exps = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")])
    if not exps:
        return None
    latest_exp = exps[-1]
    best_pt = latest_exp / "weights" / "best_dice.pt"
    return best_pt if best_pt.exists() else None

# 加载统计信息
if STATS_FILE.exists():
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)
        IMAGENET_MEAN = stats["mean"]
        IMAGENET_STD = stats["std"]
else:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

# ======================== 模型加载与推理 ========================
def load_model(checkpoint_path: Path, device: torch.device):
    model = U_Net_MobileNetV2(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def predict_sliding_window(model, image, device):
    w, h = image.size
    orig_w, orig_h = w, h
    
    if w < TILE_SIZE or h < TILE_SIZE:
        pad_w = max(0, TILE_SIZE - w)
        pad_h = max(0, TILE_SIZE - h)
        # 注意这里的 padding 顺序是 (left, top, right, bottom)
        # 我们默认 pad 在 right 和 bottom，这样左上角对齐，就不会发生偏移
        image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
        w, h = image.size

    full_tensor = TF.to_tensor(image)
    full_tensor = TF.normalize(full_tensor, IMAGENET_MEAN, IMAGENET_STD)
    
    stride = int(TILE_SIZE * (1 - OVERLAP))
    n_rows = math.ceil((h - TILE_SIZE) / stride) + 1
    n_cols = math.ceil((w - TILE_SIZE) / stride) + 1
    
    pred_map = torch.zeros((1, h, w), device=device)
    count_map = torch.zeros((1, h, w), device=device)
    
    for row in range(n_rows):
        for col in range(n_cols):
            y1 = min(row * stride, h - TILE_SIZE)
            x1 = min(col * stride, w - TILE_SIZE)
            y2 = y1 + TILE_SIZE
            x2 = x1 + TILE_SIZE
            
            tile = full_tensor[:, y1:y2, x1:x2].unsqueeze(0).to(device)
            
            with torch.no_grad(), torch.amp.autocast("cuda"):
                logits = model(tile)
                prob = torch.sigmoid(logits).squeeze(0)
                
            pred_map[:, y1:y2, x1:x2] += prob
            count_map[:, y1:y2, x1:x2] += 1.0
            
    pred_map /= count_map
    
    # 转换为 numpy 数组之前，先根据原图尺寸进行裁剪
    pred_map_np = pred_map.squeeze(0).cpu().numpy()
    if pred_map_np.shape[0] > orig_h or pred_map_np.shape[1] > orig_w:
        # 因为我们是在 right 和 bottom 进行的 padding，所以裁剪的时候只保留前 orig_h 行和前 orig_w 列即可
        pred_map_np = pred_map_np[:orig_h, :orig_w]
        
    pred_mask = (pred_map_np > THRESHOLD).astype(np.uint8) * 255
    
    return pred_mask

    # ======================== 指标计算 ========================
def calculate_per_image_metrics(pred_mask, gt_mask):
    # 强制形状对齐（防御性编程）
    if pred_mask.shape != gt_mask.shape:
        import cv2
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    pred_bin = (pred_mask > 0).astype(float)
    gt_bin = (gt_mask > 0).astype(float)
    
    tp = (pred_bin * gt_bin).sum()
    fp = (pred_bin * (1 - gt_bin)).sum()
    fn = ((1 - pred_bin) * gt_bin).sum()
    tn = ((1 - pred_bin) * (1 - gt_bin)).sum()
    
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "fp_pixels": int(fp),
        "fn_pixels": int(fn),
        "tp_pixels": int(tp)
    }

def visualize_errors(image, pred_mask, gt_mask, save_path):
    # 确保掩码形状与图像一致
    if pred_mask.shape[:2] != (image.size[1], image.size[0]):
        import cv2
        pred_mask = cv2.resize(pred_mask, (image.size[0], image.size[1]), interpolation=cv2.INTER_NEAREST)
    if gt_mask.shape[:2] != (image.size[1], image.size[0]):
        import cv2
        gt_mask = cv2.resize(gt_mask, (image.size[0], image.size[1]), interpolation=cv2.INTER_NEAREST)

    pred_bin = pred_mask > 0
    gt_bin = gt_mask > 0
    
    tp_mask = pred_bin & gt_bin
    
    fp_mask = pred_bin & ~gt_bin # 误检 (红色)
    fn_mask = ~pred_bin & gt_bin # 漏检 (蓝色)
    
    overlay = np.array(image).copy()
    
    # 绿色表示正确预测
    overlay[tp_mask] = [0, 255, 0]
    # 红色表示误检 (把背景当裂缝)
    overlay[fp_mask] = [255, 0, 0]
    # 蓝色表示漏检 (没检测出裂缝)
    overlay[fn_mask] = [0, 0, 255]
    
    Image.fromarray(overlay).save(save_path)

# 统计文件夹中的图片数量
def count_images_in_folder(folder_path, recursive=False, extensions=None):
    """
    统计文件夹中的图片文件数量
    
    参数:
        folder_path: 要统计的文件夹路径
        recursive: 是否递归统计子文件夹中的图片，默认为False
        extensions: 图片文件扩展名列表，默认为常见图片格式
    
    返回:
        图片文件数量
    """
    # 默认支持的图片扩展名
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', 
                      '.tiff', '.tif', '.webp', '.svg', '.ico'}
    
    # 确保扩展名都是小写
    extensions = {ext.lower() for ext in extensions}
    
    image_count = 0
    
    try:
        if recursive:
            # 递归统计所有子文件夹
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    # 获取文件扩展名并转换为小写
                    ext = os.path.splitext(file)[1].lower()
                    if ext in extensions:
                        image_count += 1
        else:
            # 只统计当前文件夹
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    ext = os.path.splitext(item)[1].lower()
                    if ext in extensions:
                        image_count += 1
        
        return image_count
        
    except FileNotFoundError:
        print(f"错误：找不到文件夹 '{folder_path}'")
        return 0
    except PermissionError:
        print(f"错误：没有权限访问文件夹 '{folder_path}'")
        return 0
    except Exception as e:
        print(f"发生错误：{e}")
        return 0
# ======================== 主逻辑 ========================
def run_diagnostic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = get_latest_checkpoint()
    if not ckpt_path:
        print("Error: No checkpoint found in runs/train/.")
        return
        
    print(f"Loading model from {ckpt_path}...")
    model = load_model(ckpt_path, device)
    
    test_images_dir = DATASET_ROOT / "testcrop" / "testcrop"
    test_masks_dir = DATASET_ROOT / "testcrop" / "testcrop"
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent / "runs" / "diagnostic" / f"exp_{current_time}"
    output_dir.mkdir(parents=True, exist_ok=True)
    error_vis_dir = output_dir / "error_analysis"
    error_vis_dir.mkdir(exist_ok=True)
    
    image_files = sorted(list(test_images_dir.glob("*.jpg")))
    
    results = []
    
    print("Running diagnostic tests...")
    for img_path in tqdm(image_files):
        image = Image.open(img_path).convert("RGB")
        pred_mask = predict_sliding_window(model, image, device)
        
        gt_path = test_masks_dir / (img_path.stem + ".png")
        if not gt_path.exists():
            continue
            
        gt_mask = np.array(Image.open(gt_path).convert("L"))
        
        # 计算单张图片指标
        metrics = calculate_per_image_metrics(pred_mask, gt_mask)
        metrics["filename"] = img_path.name
        results.append(metrics)
        
        # 为问题严重的图片生成可视化分析图
        # 如果 IoU 极低，或者是极端漏检/误检情况，保存图片
        if metrics["iou"] < 0.3 or metrics["fp_pixels"] > 5000 or metrics["fn_pixels"] > 5000:
            vis_path = error_vis_dir / f"{img_path.stem}_diag.jpg"
            visualize_errors(image, pred_mask, gt_mask, vis_path)
            
    # 统计分析
    print("\n" + "="*50)
    print("DIAGNOSTIC REPORT")
    print("="*50)
    
    mean_iou = np.mean([r["iou"] for r in results])
    mean_dice = np.mean([r["dice"] for r in results])
    print(f"Overall Mean IoU:  {mean_iou:.4f}")
    print(f"Overall Mean Dice: {mean_dice:.4f}")
    print(f"因指标过低而保存的图片数量：{count_images_in_folder(folder_path=error_vis_dir)}")
    # 排序找出最严重的漏检 (False Negatives)
    worst_fn = sorted(results, key=lambda x: x["fn_pixels"], reverse=True)[:10]
    print("\n--- Top 10 Images with Most False Negatives (Missed Cracks) ---")
    for r in worst_fn:
        print(f"{r['filename']}: FN pixels={r['fn_pixels']}, Recall={r['recall']:.4f}")
        
    # 排序找出最严重的误检 (False Positives)
    worst_fp = sorted(results, key=lambda x: x["fp_pixels"], reverse=True)[:10]
    print("\n--- Top 10 Images with Most False Positives (Background as Cracks) ---")
    for r in worst_fp:
        print(f"{r['filename']}: FP pixels={r['fp_pixels']}, Precision={r['precision']:.4f}")
        
    # 保存 JSON 报告
    report_path = output_dir / "diagnostic_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed report saved to {report_path}")
    print(f"Error visualizations saved to {error_vis_dir}")

if __name__ == "__main__":
    run_diagnostic()
