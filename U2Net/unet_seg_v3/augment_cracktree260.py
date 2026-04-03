"""
DeepCrack 离线数据增强脚本。

完全复现论文描述的增强流程:
  1. 旋转: 9 个角度 (0°, 10°, 20°, ..., 80°)
  2. 翻转: 3 种 (无翻转 / 水平翻转 / 垂直翻转)
  3. 裁剪: 5 个 512×512 子图 (左上、右上、左下、右下、中心)

  260 × 9 × 3 × 5 = 35,100 张

输出目录: dataset/deepcrack/CrackTree260_augmented/
  ├── images/   (35100 张 .jpg)
  └── masks/    (35100 张 .png)

用法: python augment_cracktree260.py
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ======================== 配置 ========================
SRC_IMG_DIR = Path("/home/fs-ai/YOLO-crack-detection/dataset/deepcrack/CrackTree260/CrackTree260_img")
SRC_GT_DIR = Path("/home/fs-ai/YOLO-crack-detection/dataset/deepcrack/CrackTree260/CrackTree260_gt/gt")

OUT_DIR = Path("/home/fs-ai/YOLO-crack-detection/dataset/deepcrack/CrackTree260_augmented")
OUT_IMG_DIR = OUT_DIR / "images"
OUT_MASK_DIR = OUT_DIR / "masks"

CROP_SIZE = 512
ROTATION_ANGLES = list(range(0, 90, 10))  # 0, 10, 20, ..., 80 (9个)
FLIP_MODES = [None, "horizontal", "vertical"]  # 3种


def rotate_image(image: np.ndarray, angle: float, is_mask: bool = False) -> np.ndarray:
    """
    以图片中心为原点旋转，自动扩展画布以避免裁剪。
    mask 使用最近邻插值保持二值，image 使用双线性插值。
    """
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # 计算旋转后需要的画布大小
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # 调整平移使旋转后图片居中
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    # 图片用 REFLECT 填充边界 (避免黑边污染训练数据)
    # mask 用 0 填充 (旋转后边界外无裂缝)
    border_mode = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT_101
    border_value = 0

    return cv2.warpAffine(image, M, (new_w, new_h), flags=interp,
                          borderMode=border_mode, borderValue=border_value)


def flip_image(image: np.ndarray, mode: str) -> np.ndarray:
    """翻转图片。mode: 'horizontal' or 'vertical'."""
    if mode == "horizontal":
        return cv2.flip(image, 1)
    elif mode == "vertical":
        return cv2.flip(image, 0)
    return image


def crop_five(image: np.ndarray, crop_size: int):
    """
    裁剪 5 个子图: 左上、右上、左下、右下、中心。
    如果图片比 crop_size 小，先 pad。
    返回 [(crop, name), ...]
    """
    h, w = image.shape[:2]

    # 如果图片比 crop_size 小，pad 到 crop_size
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        if image.ndim == 3:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        h, w = image.shape[:2]

    crops = []
    # 左上
    crops.append((image[0:crop_size, 0:crop_size], "tl"))
    # 右上
    crops.append((image[0:crop_size, w - crop_size:w], "tr"))
    # 左下
    crops.append((image[h - crop_size:h, 0:crop_size], "bl"))
    # 右下
    crops.append((image[h - crop_size:h, w - crop_size:w], "br"))
    # 中心
    cy, cx = h // 2, w // 2
    y1 = cy - crop_size // 2
    x1 = cx - crop_size // 2
    crops.append((image[y1:y1 + crop_size, x1:x1 + crop_size], "ct"))

    return crops


def main():
    # 收集所有图片
    img_files = sorted([
        f for f in SRC_IMG_DIR.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    ])
    print(f"源图片: {len(img_files)} 张")
    print(f"旋转角度: {ROTATION_ANGLES} ({len(ROTATION_ANGLES)} 个)")
    print(f"翻转模式: {FLIP_MODES} ({len(FLIP_MODES)} 种)")
    print(f"裁剪: 5 个 {CROP_SIZE}×{CROP_SIZE} 子图")

    expected = len(img_files) * len(ROTATION_ANGLES) * len(FLIP_MODES) * 5
    print(f"预期输出: {len(img_files)} × {len(ROTATION_ANGLES)} × {len(FLIP_MODES)} × 5 = {expected}")

    # 创建输出目录
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in tqdm(img_files, desc="处理图片"):
        stem = img_path.stem

        # 找对应 mask
        mask_path = None
        for ext in (".bmp", ".png", ".jpg"):
            p = SRC_GT_DIR / (stem + ext)
            if p.exists():
                mask_path = p
                break
        if mask_path is None:
            print(f"警告: {stem} 没有找到 GT mask，跳过")
            continue

        # 读取
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"警告: 无法读取 {stem}，跳过")
            continue

        # 确保 mask 二值化
        mask = ((mask > 127).astype(np.uint8)) * 255

        # 对齐尺寸
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        for angle in ROTATION_ANGLES:
            # 旋转
            if angle == 0:
                rot_img = image
                rot_mask = mask
            else:
                rot_img = rotate_image(image, angle, is_mask=False)
                rot_mask = rotate_image(mask, angle, is_mask=True)

            for flip_mode in FLIP_MODES:
                # 翻转
                if flip_mode is None:
                    fl_img = rot_img
                    fl_mask = rot_mask
                    flip_tag = "n"
                else:
                    fl_img = flip_image(rot_img, flip_mode)
                    fl_mask = flip_image(rot_mask, flip_mode)
                    flip_tag = "h" if flip_mode == "horizontal" else "v"

                # 裁剪 5 个子图
                img_crops = crop_five(fl_img, CROP_SIZE)
                mask_crops = crop_five(fl_mask, CROP_SIZE)

                for (img_crop, pos_name), (mask_crop, _) in zip(img_crops, mask_crops):
                    # 再次确保 mask 二值
                    mask_crop = ((mask_crop > 127).astype(np.uint8)) * 255

                    # 文件名: {原始名}_r{角度}_f{翻转}_{位置}.jpg/png
                    out_name = f"{stem}_r{angle:02d}_f{flip_tag}_{pos_name}"
                    cv2.imwrite(str(OUT_IMG_DIR / f"{out_name}.jpg"), img_crop)
                    cv2.imwrite(str(OUT_MASK_DIR / f"{out_name}.png"), mask_crop)
                    count += 1

    print(f"\n完成! 共生成 {count} 对图片+掩码")
    print(f"  图片: {OUT_IMG_DIR}")
    print(f"  掩码: {OUT_MASK_DIR}")

    # 验证
    n_imgs = len(list(OUT_IMG_DIR.glob("*.jpg")))
    n_masks = len(list(OUT_MASK_DIR.glob("*.png")))
    print(f"\n验证: images={n_imgs}, masks={n_masks}, 预期={expected}")
    assert n_imgs == expected, f"图片数量不对: {n_imgs} != {expected}"
    assert n_masks == expected, f"掩码数量不对: {n_masks} != {expected}"
    print("数量验证通过!")


if __name__ == "__main__":
    main()
