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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from tqdm import tqdm

# from model import CrackUNet
from U_Net_MobileNetV2_model import U_Net_MobileNetV2

# ======================== 配置 ========================
DATASET_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/crack500/CRACK500")
# 默认使用 IoU 最高的最佳模型
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "best_dice.pt"

OUTPUT_DIR = Path(__file__).resolve().parent / "predictions"

# 动态加载均值和标准差
STATS_FILE = Path(__file__).resolve().parent / "dataset_stats.json"

THRESHOLD = 0.5
TILE_SIZE = 640        # 滑动窗口大小（修改为640，与训练数据保持一致）
OVERLAP = 1/3          # 窗口重叠比例（推荐 1/3 ~ 1/2）

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
def load_model(checkpoint_path: Path, device: torch.device) -> U_Net_MobileNetV2:
    # 确保模型定义与训练时一致
    # CrackUNet(pretrained=bool) 
    # model = CrackUNet(pretrained=False)
    model = U_Net_MobileNetV2(pretrained=False) # 将pretrained设为False是使用我们训练的模型参数而非预训练模型
    # 兼容性加载：处理可能的 DataParallel 包装或键名不匹配
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # 移除 'module.' 前缀（如果存在）
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"✅ 模型加载成功: {checkpoint_path}")
        if "val_metrics" in ckpt:
            metrics = ckpt["val_metrics"]
            print(f"   -> IoU: {metrics.get('iou', 'N/A'):.4f}, Dice: {metrics.get('dice', 'N/A'):.4f}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

    model.to(device).eval()
    return model


def predict_sliding_window(model: torch.nn.Module, image: Image.Image, device: torch.device) -> np.ndarray:
    """
    使用滑动窗口策略对任意分辨率图像进行推理。
    1. 将大图切分为多个小块（Tile）。
    2. 对每个 Tile 进行预测。
    3. 将预测结果拼接回原图尺寸，重叠区域取平均值以消除拼接缝。
    """
    w, h = image.size
    
    # 如果图像小于窗口大小，进行 padding
    if w < TILE_SIZE or h < TILE_SIZE:
        pad_w = max(0, TILE_SIZE - w)
        pad_h = max(0, TILE_SIZE - h)
        image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
        w, h = image.size  # 更新尺寸

    # 预处理全图
    full_tensor = TF.to_tensor(image)
    full_tensor = TF.normalize(full_tensor, IMAGENET_MEAN, IMAGENET_STD)
    
    # 计算步长（Stride）
    stride = int(TILE_SIZE * (1 - OVERLAP))
    
    # 计算行数和列数
    n_rows = math.ceil((h - TILE_SIZE) / stride) + 1
    n_cols = math.ceil((w - TILE_SIZE) / stride) + 1
    
    # 初始化概率图和计数图（用于重叠区域平均）
    prob_map = torch.zeros((h, w), device=device)
    count_map = torch.zeros((h, w), device=device)
    
    # 生成高斯权重窗口（让中心区域权重更高，边缘更低，进一步消除拼接缝）
    # 这里简单使用全 1 权重，如果接缝明显可改为高斯窗
    window_weight = torch.ones((TILE_SIZE, TILE_SIZE), device=device)

    with torch.no_grad():
        for row in range(n_rows):
            for col in range(n_cols):
                # 计算当前窗口坐标
                y1 = int(row * stride)
                x1 = int(col * stride)
                
                # 边界处理：如果是最后一行/列，向左/上偏移以保证窗口完整
                if y1 + TILE_SIZE > h: y1 = h - TILE_SIZE
                if x1 + TILE_SIZE > w: x1 = w - TILE_SIZE
                
                y2 = y1 + TILE_SIZE
                x2 = x1 + TILE_SIZE
                
                # 提取 Tile
                tile = full_tensor[:, y1:y2, x1:x2].unsqueeze(0).to(device)
                
                # 推理
                logits = model(tile)
                prob = torch.sigmoid(logits).squeeze(0).squeeze(0)  # (H, W)
                
                # 累加预测结果
                prob_map[y1:y2, x1:x2] += prob * window_weight
                count_map[y1:y2, x1:x2] += window_weight

    # 计算平均概率
    avg_prob = prob_map / count_map
    
    # 裁剪回原始尺寸（如果之前做了 padding）
    original_w, original_h = image.size
    # 注意：这里的 image 变量已经是 padding 后的了，我们需要传入原始尺寸参数或者记录原始尺寸
    # 为了简单，函数开头直接记录 w, h 即可。但这里为了严谨：
    # 如果函数开头做了 padding，此时 w, h 是 padding 后的。
    # 真正的原始尺寸需要调用者保证，或者我们在 padding 前记录。
    # 实际上，上面的代码中 w, h 已经被更新为 padding 后的尺寸。
    # 所以我们需要一种方式知道原始尺寸。
    # 修正：直接返回 padding 后的结果，由调用者裁剪？
    # 或者：在函数开头记录 original_size
    
    # 转换为 numpy mask
    mask = (avg_prob > THRESHOLD).cpu().numpy().astype(np.uint8) * 255
    
    return mask

def predict_image(model: U_Net_MobileNetV2, image: Image.Image, device: torch.device) -> np.ndarray:
    """对单张 PIL Image 推理，自动处理任意分辨率。"""
    w, h = image.size
    
    # 如果图像尺寸接近训练尺寸（416x416），直接推理
    if abs(w - TILE_SIZE) < 32 and abs(h - TILE_SIZE) < 32:
        image = image.resize((TILE_SIZE, TILE_SIZE), Image.BILINEAR)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        mask = (prob > THRESHOLD).astype(np.uint8) * 255
        return mask
    
    # 否则使用滑动窗口预测
    full_mask = predict_sliding_window(model, image, device)
    
    # 如果 predict_sliding_window 中做了 padding，这里裁剪回原始尺寸
    if full_mask.shape != (h, w):
        full_mask = full_mask[:h, :w]
        
    return full_mask


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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        vis.save(OUTPUT_DIR / img_path.name)

    if ious:
        ious = np.array(ious)
        print(f"\n--- Test Results ({len(ious)} images) ---")
        print(f"  Mean IoU:   {ious.mean():.4f}")
        print(f"  Median IoU: {np.median(ious):.4f}")
        print(f"  Min IoU:    {ious.min():.4f} (worst)")
        print(f"  Max IoU:    {ious.max():.4f} (best)")
    print(f"Predictions saved to {OUTPUT_DIR}")


def run_single(model, device, image_path: str, do_measure: bool):
    """单张图片推理。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    pred_mask = predict_image(model, image, device)

    # 保存 mask
    name = Path(image_path).stem
    mask_out = OUTPUT_DIR / f"{name}_mask.png"
    Image.fromarray(pred_mask).save(mask_out)
    print(f"Mask saved: {mask_out}")

    # 保存可视化
    vis = overlay_mask(image, pred_mask)
    vis_out = OUTPUT_DIR / f"{name}_overlay.jpg"
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
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT), help="Model checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)

    if args.test:
        run_test(model, device)
    elif args.image:
        run_single(model, device, args.image, args.measure)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
