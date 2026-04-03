"""
将 YOLO 多边形标注转换为二值 mask PNG 图像。
输入: dataset/ultralytics/labels/{train,val,test}/*.txt (YOLO polygon 格式)
输出: dataset/ultralytics/masks/{train,val,test}/*.png  (单通道, 0=背景, 255=裂缝)
"""
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

# 数据集根目录
DATASET_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/ultralytics")
IMG_SIZE = 416
SPLITS = ["train", "val", "test"]


def parse_yolo_polygon(line: str, img_w: int, img_h: int) -> list[tuple[float, float]]:
    """解析一行 YOLO 多边形标注，返回像素坐标列表。
    格式: class_id x1 y1 x2 y2 ... xn yn (归一化 0-1)
    """
    parts = line.strip().split()
    # 跳过 class_id (parts[0])
    coords = [float(v) for v in parts[1:]]
    points = []
    for i in range(0, len(coords), 2):
        x = coords[i] * img_w
        y = coords[i + 1] * img_h
        points.append((x, y))
    return points


def label_to_mask(label_path: Path, img_w: int = IMG_SIZE, img_h: int = IMG_SIZE) -> Image.Image:
    """读取一个 YOLO label 文件，生成对应的二值 mask 图像。"""
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            polygon = parse_yolo_polygon(line, img_w, img_h)
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=255)

    return mask


def convert_split(split: str) -> int:
    """转换一个数据集分割的所有标注，返回处理数量。"""
    labels_dir = DATASET_ROOT / "labels" / split
    masks_dir = DATASET_ROOT / "masks" / split
    masks_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))
    count = 0

    for label_path in tqdm(label_files, desc=f"[{split}]"):
        mask = label_to_mask(label_path)
        # 输出文件名与 label 同名，扩展名改为 .png
        out_path = masks_dir / (label_path.stem + ".png")
        mask.save(out_path)
        count += 1

    return count


def verify(n_samples: int = 5):
    """随机抽查几张 mask，验证格式正确。"""
    print("\n--- 验证 ---")
    for split in SPLITS:
        masks_dir = DATASET_ROOT / "masks" / split
        mask_files = list(masks_dir.glob("*.png"))
        samples = random.sample(mask_files, min(n_samples, len(mask_files)))
        for mf in samples:
            img = Image.open(mf)
            assert img.size == (IMG_SIZE, IMG_SIZE), f"尺寸错误: {mf} -> {img.size}"
            assert img.mode == "L", f"模式错误: {mf} -> {img.mode}"
            pixels = set(img.getdata())
            assert pixels <= {0, 255}, f"像素值异常: {mf} -> {pixels}"
        print(f"  [{split}] {len(mask_files)} 张 mask, 抽查 {len(samples)} 张全部通过")


if __name__ == "__main__":
    total = 0
    for split in SPLITS:
        n = convert_split(split)
        total += n
        print(f"  {split}: {n} 张 mask 已生成")

    print(f"\n总计: {total} 张 mask")
    verify()
    print("\n完成!")
