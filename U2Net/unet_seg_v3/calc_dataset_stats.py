import os
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def calculate_mean_std(image_dirs):
    """
    计算数据集的 RGB 均值和标准差。
    使用增量计算方法，避免将所有图片加载到内存中导致 OOM。
    """
    print(f"开始计算数据集均值和标准差...")
    
    pixel_num = 0 # 像素总数
    channel_sum = np.zeros(3) # 通道像素值之和
    channel_sq_sum = np.zeros(3) # 通道像素平方值之和
    
    # 收集所有图片路径 (只收集原图 .jpg，避免计算掩码)
    image_paths = []
    for d in image_dirs:
        dir_path = Path(d)
        if dir_path.exists():
            image_paths.extend(list(dir_path.glob("*.jpg")))
            
    if not image_paths:
        raise ValueError("未找到任何图片文件！")
        
    print(f"共找到 {len(image_paths)} 张图片。")
    
    for path in tqdm(image_paths, desc="处理图片"):
        img = cv2.imread(str(path))
        if img is None:
            continue
            
        # OpenCV 默认读取是 BGR，转换为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 将像素值归一化到 [0, 1] 区间（与 torchvision.transforms.ToTensor() 行为一致）
        img = img.astype(np.float32) / 255.0
        
        # 展平为 (N, 3) 形状
        pixels = img.reshape(-1, 3)
        
        # 累加统计量
        pixel_num += pixels.shape[0]
        channel_sum += np.sum(pixels, axis=0)
        channel_sq_sum += np.sum(pixels ** 2, axis=0)

    # 计算均值和标准差
    # E[X]
    mean = channel_sum / pixel_num
    # std = sqrt(E[X^2] - (E[X])^2)
    std = np.sqrt(channel_sq_sum / pixel_num - mean ** 2)
    
    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    # 定义数据集图片目录 (只使用训练集计算)
    dataset_root = "/home/fs-ai/YOLO-crack-detection/dataset/crack500/CRACK500/traincrop/traincrop"
    # "/home/fs-ai/YOLO-crack-detection/dataset/youluoyuan-crack/youluoyuan"
    try:
        mean, std = calculate_mean_std([dataset_root])
        print("\n=== 计算完成 ===")
        print(f"Mean: {mean}")
        print(f"Std:  {std}")
        
        # 提取数据集名称（取倒数第三级目录，即 dataset/ultralytics 中的 'ultralytics'）
        dataset_name = Path(dataset_root).parent.parent.name
        
        # 保存到 json 文件
        save_path = "/home/fs-ai/YOLO-crack-detection/scripts/unet_seg/dataset_stats.json"
        stats = {
            "dataset_name": dataset_name,
            "dataset_path": dataset_root,
            "mean": mean,
            "std": std
        }
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=4)
            
        print(f"\n统计数据已保存至: {save_path}")
        
    except Exception as e:
        print(f"发生错误: {e}")
