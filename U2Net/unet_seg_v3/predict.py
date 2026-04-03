"""
U-Net 裂缝分割推理、可视化与裂缝宽度测量。

用法:
    python predict.py --test                              # 测试集批量评估
    python predict.py --image path/to/img.jpg             # 单张推理+可视化
    python predict.py --image path/to/img.jpg --measure   # 推理+测宽
"""
import math
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from tqdm import tqdm

# from model import CrackUNet
from U_Net_MobileNetV2_model import U_Net_MobileNetV2

# ======================== 配置 ========================
DATASET_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/crack500/CRACK500")

# 动态加载均值和标准差
STATS_FILE = Path(__file__).resolve().parent / "dataset_stats.json"

THRESHOLD = 0.5
TILE_SIZE = 640        # 滑动窗口大小（修改为640，与训练数据保持一致）
OVERLAP = 1/3          # 窗口重叠比例（推荐 1/3 ~ 1/2）

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

if STATS_FILE.exists():
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)
        IMAGENET_MEAN = stats["mean"]
        IMAGENET_STD = stats["std"]
        dataset_name = stats.get("dataset_name", "Unknown")
    print(f"✅ predict: 加载自定义数据集 ({dataset_name}) 统计信息")
else:
    print("⚠️ predict: 未找到 dataset_stats.json，回退到 ImageNet 默认值")
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

# ======================== 推理 ========================
def auto_detect_model_config(state_dict):
    """
    通过扫描 state_dict 的键名，自动推断训练时使用的 Attention 类型和是否开启了深度监督。
    """
    attention_type = 'None'
    deep_supervision = False
    
    for key in state_dict.keys():
        # 检测深度监督
        if 'ds_head' in key:
            deep_supervision = True
            
        # 检测 Attention 类型
        if 'attention.W_g' in key or 'attention.psi' in key or 'ag.W_g' in key or 'ag.psi' in key:
            attention_type = 'AG'
        elif 'attention.ca.fc1' in key or 'attention.sa.conv1' in key:
            attention_type = 'CBAM'
        elif 'attention.conv.weight' in key:
            attention_type = 'ECA'
            
    return attention_type, deep_supervision

def load_model(checkpoint_path: Path, device: torch.device) -> U_Net_MobileNetV2:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # 移除 'module.' 前缀（如果存在）
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # 兼容老版本的 'ag.' 命名，将其映射为新版本的 'attention.'
        new_state_dict = {}
        for k, v in state_dict.items():
            if '.ag.' in k:
                new_k = k.replace('.ag.', '.attention.')
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
                
        # 自动探测模型结构配置
        attention_type, deep_supervision = auto_detect_model_config(new_state_dict)
        print(f"Auto-detected model config: Attention='{attention_type}', DeepSupervision={deep_supervision}")
        
        # 动态初始化正确结构的模型
        model = U_Net_MobileNetV2(
            pretrained=False, 
            attention_type=attention_type, 
            deep_supervision=deep_supervision
        )
        
        model.load_state_dict(new_state_dict)
        print(f"✅ 模型加载成功: {checkpoint_path}")
        if "val_metrics" in ckpt:
            metrics = ckpt["val_metrics"]
            print(f"   -> IoU: {metrics.get('iou', 'N/A'):.4f}, Dice: {metrics.get('dice', 'N/A'):.4f}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

    model.to(device).eval()
    return model


def morphological_post_processing(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    形态学后处理：过滤掉小的孤立噪点（误报），从而提升 Precision。
    """
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    clean_mask = np.zeros_like(mask)
    # 标签 0 是背景，从 1 开始遍历
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # 如果连通域的面积大于阈值，则保留（裂缝通常是连续的细长条，面积不会太小）
        if area >= min_area:
            clean_mask[labels == i] = 255
            
    return clean_mask

def predict_image(model, image: Image.Image, device: torch.device, use_tta: bool = True) -> np.ndarray:
    """
    滑动窗口推理全图（支持 TTA 测试时增强）。
    返回 numpy 二值掩码 (0/255)。
    """
    model.eval()
    w, h = image.size
    orig_w, orig_h = w, h
    
    # Padding
    if w < TILE_SIZE or h < TILE_SIZE:
        pad_w = max(0, TILE_SIZE - w)
        pad_h = max(0, TILE_SIZE - h)
        image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
        w, h = image.size

    full_tensor = TF.to_tensor(image)
    full_tensor = TF.normalize(full_tensor, IMAGENET_MEAN, IMAGENET_STD)

    stride = int(TILE_SIZE * (1 - OVERLAP))
    n_rows = math.ceil((h - TILE_SIZE) / stride) + 1
    n_cols = math.ceil((w - TILE_SIZE) / stride) + 1

    def _infer_tensor(tensor_input):
        """对单个全图张量进行滑动窗口推理，返回概率图"""
        pred_map = torch.zeros((1, h, w), device=device)
        count_map = torch.zeros((1, h, w), device=device)
        
        for row in range(n_rows):
            for col in range(n_cols):
                y1 = min(row * stride, h - TILE_SIZE)
                x1 = min(col * stride, w - TILE_SIZE)
                y2 = y1 + TILE_SIZE
                x2 = x1 + TILE_SIZE

                tile = tensor_input[:, y1:y2, x1:x2].unsqueeze(0).to(device)

                with torch.no_grad(), torch.amp.autocast("cuda"):
                    logits = model(tile)
                    if isinstance(logits, tuple): # 兼容深度监督输出
                        logits = logits[0]
                    prob = torch.sigmoid(logits).squeeze(0)

                pred_map[:, y1:y2, x1:x2] += prob
                count_map[:, y1:y2, x1:x2] += 1.0
                
        return pred_map / count_map

    # 1. 原始推理
    final_prob = _infer_tensor(full_tensor)
    
    # 2. TTA (Test-Time Augmentation) 水平翻转增强
    if use_tta:
        tensor_flipped = torch.flip(full_tensor, dims=[2]) # 水平翻转 (W 维度)
        prob_flipped = _infer_tensor(tensor_flipped)
        # 将结果翻转回来并与原图结果平均
        prob_flipped_back = torch.flip(prob_flipped, dims=[2])
        final_prob = (final_prob + prob_flipped_back) / 2.0

    # 3. 阈值化
    pred_mask = (final_prob.squeeze(0).cpu().numpy() > THRESHOLD).astype(np.uint8) * 255
    
    # 4. 裁剪回原始尺寸
    if pred_mask.shape[0] > orig_h or pred_mask.shape[1] > orig_w:
        pred_mask = pred_mask[:orig_h, :orig_w]
        
    # 5. 形态学后处理 (去除孤立噪点)
    pred_mask = morphological_post_processing(pred_mask, min_area=150)

    return pred_mask


# ======================== 可视化 ========================
def overlay_mask(image: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.4) -> Image.Image:
    """将红色半透明 mask 叠加到原图上。"""
    overlay = Image.new("RGB", image.size, color)
    mask_img = Image.fromarray(mask).convert("L")
    # alpha blend: 仅在 mask 区域叠加颜色
    blended = Image.composite(overlay, image.convert("RGB"), mask_img)
    # 整体 alpha 混合，让 mask 区域半透明
    result = Image.blend(image.convert("RGB"), blended, alpha)
    return result


# ======================== 裂缝宽度测量 ========================
def skeletonize(mask: np.ndarray) -> np.ndarray:
    """形态学细化，纯 numpy 实现。输入/输出: bool array。"""
    from scipy.ndimage import binary_erosion, binary_dilation

    # Zhang-Suen thinning 的简化版：迭代腐蚀直到骨架稳定
    img = mask.copy().astype(bool)
    skel = np.zeros_like(img, dtype=bool)
    element = np.ones((3, 3), dtype=bool)

    while img.any():
        eroded = binary_erosion(img, structure=element)
        opened = binary_dilation(eroded, structure=element)
        skel |= (img & ~opened)
        img = eroded

    return skel


def measure_crack_width(mask: np.ndarray) -> dict:
    """
    测量裂缝宽度（像素）。
    方法: 距离变换 + 骨架化。骨架上每点的距离值 × 2 = 局部宽度。
    """
    from scipy.ndimage import distance_transform_edt

    binary = (mask > 0).astype(bool)
    if not binary.any():
        return {"max_width": 0, "mean_width": 0, "median_width": 0}

    # 距离变换: 每个前景像素到最近背景像素的距离
    dist = distance_transform_edt(binary)

    # 骨架化
    skeleton = skeletonize(binary)

    if not skeleton.any():
        # 裂缝太小无法骨架化，用 dist 最大值估计
        max_dist = dist.max()
        return {"max_width": float(max_dist * 2), "mean_width": float(max_dist * 2), "median_width": float(max_dist * 2)}

    # 骨架上的距离值 = 到边界的距离 ≈ 半宽度
    widths = dist[skeleton] * 2.0

    return {
        "max_width": float(widths.max()),
        "mean_width": float(widths.mean()),
        "median_width": float(np.median(widths)),
    }


# ======================== 命令行入口 ========================
def run_test(model, device):
    """测试集批量评估。"""
    test_images_dir = DATASET_ROOT / "testcrop" / "testcrop"
    test_masks_dir = DATASET_ROOT / "testcrop" / "testcrop"
    
    current_time = datetime.now().strftime("%Y%m%d_%H%d_%M%S")
    output_dir = Path(__file__).resolve().parent / "runs" / "val" / f"exp_{current_time}"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(test_images_dir.glob("*.jpg"))
    ious = []

    for img_path in tqdm(image_files, desc="Testing"):
        image = Image.open(img_path).convert("RGB")
        pred_mask = predict_image(model, image, device)

        # 计算 IoU（如果有 GT mask）
        gt_path = test_masks_dir / (img_path.stem + ".png")
        if gt_path.exists():
            gt = np.array(Image.open(gt_path).convert("L"))
            pred_bin = pred_mask > 0
            gt_bin = gt > 0
            intersection = (pred_bin & gt_bin).sum()
            union = (pred_bin | gt_bin).sum()
            iou = intersection / (union + 1e-6)
            ious.append(iou)

        # 保存可视化
        vis = overlay_mask(image, pred_mask)
        vis.save(output_dir / img_path.name)

    if ious:
        ious = np.array(ious)
        print(f"\n--- Test Results ({len(ious)} images) ---")
        print(f"  Mean IoU:   {ious.mean():.4f}")
        print(f"  Median IoU: {np.median(ious):.4f}")
        print(f"  Min IoU:    {ious.min():.4f} (worst)")
        print(f"  Max IoU:    {ious.max():.4f} (best)")
    print(f"Predictions saved to {output_dir}")


def run_single(model, device, image_path: str, do_measure: bool):
    """单张图片推理。"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent / "runs" / "detect" / f"exp_{current_time}"
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    pred_mask = predict_image(model, image, device)

    # 保存 mask
    name = Path(image_path).stem
    mask_out = output_dir / f"{name}_mask.png"
    Image.fromarray(pred_mask).save(mask_out)
    print(f"Mask saved: {mask_out}")

    # 保存可视化
    vis = overlay_mask(image, pred_mask)
    vis_out = output_dir / f"{name}_overlay.jpg"
    vis.save(vis_out)
    print(f"Overlay saved: {vis_out}")

    # 测量宽度
    if do_measure:
        widths = measure_crack_width(pred_mask)
        print(f"\n--- Crack Width (pixels) ---")
        print(f"  Max:    {widths['max_width']:.1f} px")
        print(f"  Mean:   {widths['mean_width']:.1f} px")
        print(f"  Median: {widths['median_width']:.1f} px")


def main():
    parser = argparse.ArgumentParser(description="U-Net crack segmentation inference")
    parser.add_argument("--test", action="store_true", help="Evaluate on test set")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--measure", action="store_true", help="Measure crack width")
    parser.add_argument("--checkpoint", type=str, default="", help="Model checkpoint path (defaults to latest in runs/train/)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = args.checkpoint
    if not ckpt_path:
        ckpt_path = get_latest_checkpoint()
        if not ckpt_path:
            print("Error: No checkpoint found in runs/train/ and none specified.")
            return
        print(f"Auto-selected latest checkpoint: {ckpt_path}")
        
    model = load_model(Path(ckpt_path), device)

    if args.test:
        run_test(model, device)
    elif args.image:
        run_single(model, device, args.image, args.measure)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
