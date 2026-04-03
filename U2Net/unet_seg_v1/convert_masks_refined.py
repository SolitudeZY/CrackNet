"""
使用 Segment Anything Model (SAM) 自动细化 YOLO 的粗糙多边形标注，生成高精度的二值裂缝掩码。
利用 5060Ti 的强大算力，使用 ViT-H (Huge) 模型进行像素级分割。
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

# --- 配置 ---
DATASET_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/youluoyuan-crack/youluoyuan")
# 使用最大的 ViT-H 模型以获得最佳精度
SAM_CHECKPOINT = "/home/fs-ai/YOLO-crack-detection/scripts/unet_seg/checkpoints/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLITS = ["train", "val", "test"]

# 裂缝细化参数
CONFIDENCE_THRESHOLD = 0.5  # SAM 预测掩码的置信度阈值

def load_sam_model():
    """加载 SAM 模型到 GPU"""
    print(f"正在加载 SAM 模型 ({SAM_MODEL_TYPE})... 设备: {DEVICE}")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    return predictor

def parse_yolo_polygon(line: str, img_w: int, img_h: int) -> np.ndarray:
    """
    解析 YOLO 多边形标注，返回点坐标数组。
    返回: np.ndarray shape (N, 2)
    """
    parts = line.strip().split()
    coords = [float(v) for v in parts[1:]]
    points = []
    for i in range(0, len(coords), 2):
        x = int(coords[i] * img_w)
        y = int(coords[i + 1] * img_h)
        points.append([x, y])
    return np.array(points)

def get_bbox_from_polygon(polygon: np.ndarray) -> np.ndarray:
    """从多边形点计算边界框 [x_min, y_min, x_max, y_max]"""
    x_min = np.min(polygon[:, 0])
    y_min = np.min(polygon[:, 1])
    x_max = np.max(polygon[:, 0])
    y_max = np.max(polygon[:, 1])
    return np.array([x_min, y_min, x_max, y_max])

def process_single_image(predictor, image_path: Path, label_path: Path, output_path: Path):
    """处理单张图片：读取图片和标注 -> SAM 细化 -> 保存掩码"""
    # 1. 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    # 2. 设置 SAM 图像编码
    predictor.set_image(image)

    # 3. 读取并解析所有标注的多边形
    polygons = []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                poly = parse_yolo_polygon(line, img_w, img_h)
                if len(poly) >= 3:
                    polygons.append(poly)

    # 4. 创建最终掩码画布 (黑色背景)
    final_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # 5. 对每个多边形标注进行细化
    for poly in polygons:
        # 策略 A: 使用多边形的包围框作为 SAM 的提示 (Box Prompt)
        # 这是最稳健的方法，让 SAM 在框内寻找裂缝
        bbox = get_bbox_from_polygon(poly)
        
        # 策略 B: 同时使用多边形中心点作为前景点提示 (可选，增强准确性)
        # center_x = int(np.mean(poly[:, 0]))
        # center_y = int(np.mean(poly[:, 1]))
        # input_point = np.array([[center_x, center_y]])
        # input_label = np.array([1]) # 1 表示前景

        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :], # SAM 期望 box 形状为 (1, 4)
            multimask_output=False # 我们只需要一个最好的掩码
        )
        
        # 获取预测结果
        best_mask = masks[0]
        score = scores[0]

        # 如果 SAM 找到了高置信度的物体，就使用它
        if score > CONFIDENCE_THRESHOLD:
            # 将 True/False 掩码转换为 255/0
            binary_mask = (best_mask * 255).astype(np.uint8)
            
            # 关键步骤：限制范围
            # SAM 可能会分割出框外的东西，我们利用原始多边形做一个粗略的过滤
            # 或者直接相信 SAM 的结果（通常效果更好）
            # 这里我们选择直接叠加 SAM 的结果
            final_mask = cv2.bitwise_or(final_mask, binary_mask)
        else:
            # 如果 SAM 没信心（比如裂缝太细看不清），回退到使用原始多边形填充
            # 创建一个临时的多边形掩码
            poly_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [poly], 255)
            final_mask = cv2.bitwise_or(final_mask, poly_mask)

    # 6. 保存结果
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), final_mask)

def convert_split_refined(predictor, split: str):
    """转换一个数据集分割"""
    images_dir = DATASET_ROOT / split / "images"
    labels_dir = DATASET_ROOT / split / "labels"
    # 这里我们直接输出到 masks 目录（不再叫 masks_refined），因为这是一个新数据集
    masks_dir = DATASET_ROOT / split / "masks" 
    
    if not images_dir.exists():
        print(f"跳过 {split}: 找不到目录 {images_dir}")
        return

    print(f"\n正在处理 {split} 集... 输出至: {masks_dir}")
    
    # 获取所有图片文件
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    for img_path in tqdm(image_files):
        # 对应的标签文件路径
        label_path = labels_dir / (img_path.stem + ".txt")
        # 输出掩码路径
        output_path = masks_dir / (img_path.stem + ".png")
        
        process_single_image(predictor, img_path, label_path, output_path)

if __name__ == "__main__":
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"错误: 找不到 SAM 模型权重: {SAM_CHECKPOINT}")
        print("请先下载 sam_vit_h_4b8939.pth")
        exit(1)

    # 1. 初始化模型
    predictor = load_sam_model()
    
    # 2. 处理所有分割
    for split in SPLITS:
        convert_split_refined(predictor, split)
        
    print("\n🎉 所有掩码细化完成！请检查 dataset/ultralytics/masks_refined 目录。")
