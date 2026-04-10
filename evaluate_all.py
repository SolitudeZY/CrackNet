import sys
import os
import time
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO

# 添加根目录以便于模块导入，移除多余的子目录路径以避免 model.py 冲突
sys.path.insert(0, "/home/fs-ai/CrackNet")

from DeepLab_ResNet.script.model import build_model as build_deeplab
from DeepLab_ResNet.script.sliding_window import sliding_window_predict as sliding_window_predict_deeplab

# U2Net/UNet_MobileNetV2 兼容性
from U2Net.unet_seg_v3.U_Net_MobileNetV2_model import U_Net_MobileNetV2
from U2Net.unet_seg_v1.U_Net_MobileNetV2_model import U_Net_MobileNetV2 as UNet_v1_MobileNetV2

# 动态导入避免 sys.path 污染
import importlib.util
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

u2net_model_mod = load_module_from_path("u2net_full_model", "/home/fs-ai/CrackNet/U2Net/U2Net_full/model.py")
build_u2net = u2net_model_mod.build_model

def sliding_window_inference_u2net(model, image_rgb, device, img_size=512, stride=None, tta=False):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    h, w = image_rgb.shape[:2]
    window = img_size
    stride = stride or max(1, img_size * 2 // 3)

    def infer_patch(patch):
        resized = cv2.resize(patch, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        resized = resized.astype(np.float32) / 255.0
        for c in range(3):
            resized[:, :, c] = (resized[:, :, c] - mean[c]) / std[c]

        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(tensor)[0])
        pred = torch.nn.functional.interpolate(pred, size=(patch.shape[0], patch.shape[1]), mode="bilinear", align_corners=False)
        return pred.squeeze().cpu().numpy()

    if h <= window and w <= window:
        return infer_patch(image_rgb)

    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(1, h - window + 1), stride))
    xs = list(range(0, max(1, w - window + 1), stride))
    if ys[-1] != h - window:
        ys.append(max(0, h - window))
    if xs[-1] != w - window:
        xs.append(max(0, w - window))

    for y in ys:
        for x in xs:
            patch = image_rgb[y:y + window, x:x + window]
            pred = infer_patch(patch)
            ph, pw = pred.shape
            prob_map[y:y + ph, x:x + pw] += pred
            count_map[y:y + ph, x:x + pw] += 1.0

    return prob_map / np.maximum(count_map, 1.0)

DATASET_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/crack500/CRACK500")

def load_deeplab(weights_path, device):
    model = build_deeplab(pretrained=False)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if "ema_state_dict" in ckpt:
        state_dict = ckpt["ema_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device).eval()
    threshold = ckpt.get("val_metrics", {}).get("threshold", 0.5)
    return model, threshold

def load_u2net_full(weights_path, device):
    model = build_u2net(name="u2net")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, 0.5

def load_unet_v3(weights_path, device):
    model = U_Net_MobileNetV2(pretrained=False, attention_type="CBAM", deep_supervision=True)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, 0.5

def load_unet_v1(weights_path, device):
    model = UNet_v1_MobileNetV2(pretrained=False)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, 0.5

def calculate_metrics(pred_mask, gt_mask):
    """
    计算一张图片的指标
    """
    pred_flat = pred_mask.flatten().astype(bool)
    gt_flat = gt_mask.flatten().astype(bool)

    tp = np.sum(pred_flat & gt_flat)
    fp = np.sum(pred_flat & ~gt_flat)
    fn = np.sum(~pred_flat & gt_flat)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    
    return precision, recall, iou, dice

def get_test_samples():
    split_file = DATASET_ROOT / "test.txt"
    samples = []
    if not split_file.exists():
        print(f"找不到 {split_file}")
        return []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            img_path = DATASET_ROOT / parts[0]
            mask_path = DATASET_ROOT / parts[1]
            if not img_path.exists():
                subdir = parts[0].split("/")[0]
                fname = "/".join(parts[0].split("/")[1:])
                img_path = DATASET_ROOT / subdir / subdir / fname
                mask_path = DATASET_ROOT / subdir / subdir / "/".join(parts[1].split("/")[1:])
            if img_path.exists() and mask_path.exists():
                samples.append((img_path, mask_path))
    return samples

def evaluate_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    models_info = [
        {
            "name": "DeepLab_ResNet101 (全量高精度)",
            "type": "deeplab",
            "path": "/home/fs-ai/CrackNet/DeepLab_ResNet/script/runs/20260402_110122/weights/best_iou.pt"
        },
        {
            "name": "U2Net_full_CBAM (全量增强版)",
            "type": "u2net_full",
            "path": "/home/fs-ai/CrackNet/U2Net/U2Net_full/runs/train/exp_20260401_UNet_CBAM_ALL/weights/best_iou.pt"
        },
        {
            "name": "U-Net v3 (MobileNetV2 轻量版)",
            "type": "unet_v3",
            "path": "/home/fs-ai/CrackNet/U2Net/unet_seg_v3/runs/train/exp_20260325_150905/weights/best_iou.pt"
        },
        {
            "name": "U-Net v1 (EfficientNet-B0 基线版)",
            "type": "unet_v1",
            "path": "/home/fs-ai/CrackNet/U2Net/unet_seg_v1/checkpoints/best.pt"
        },
        {
            "name": "YOLO11n-seg (极轻量端侧版)",
            "type": "yolo",
            "path": "/home/fs-ai/YOLO-crack-detection/scripts/runs/train/crack_yolo11n-seg_final_opt/weights/best.pt"
        }
    ]

    samples = get_test_samples()
    print(f"测试集样本数量: {len(samples)}")
    
    # 因为跑全量测试会很慢，我们在 100 张图片上做随机采样测试以快速验证
    if len(samples) > 100:
        np.random.seed(42)
        idx = np.random.choice(len(samples), 100, replace=False)
        samples = [samples[i] for i in idx]
        print(f"抽样测试样本数量: 100")

    results = {}

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for m_info in models_info:
        print(f"\n{'='*50}\n正在评估模型: {m_info['name']}")
        
        if not os.path.exists(m_info["path"]):
            print(f"权重文件不存在: {m_info['path']}")
            continue
            
        if m_info["type"] == "deeplab":
            model, thr = load_deeplab(m_info["path"], device)
        elif m_info["type"] == "u2net_full":
            model, thr = load_u2net_full(m_info["path"], device)
        elif m_info["type"] == "unet_v3":
            model, thr = load_unet_v3(m_info["path"], device)
        elif m_info["type"] == "unet_v1":
            model, thr = load_unet_v1(m_info["path"], device)
        elif m_info["type"] == "yolo":
            model = YOLO(m_info["path"])
            thr = 0.5
        
        p_list, r_list, iou_list, dice_list = [], [], [], []
        
        for img_path, mask_path in tqdm(samples, desc=m_info["name"]):
            # Load GT mask
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                gt_mask = np.array(Image.open(mask_path).convert("L"))
            gt_mask = (gt_mask > 0).astype(np.uint8)
            h, w = gt_mask.shape

            # Inference
            if m_info["type"] == "yolo":
                res = model(str(img_path), verbose=False, imgsz=640)
                pred_mask = np.zeros((h, w), dtype=np.uint8)
                if res[0].masks is not None:
                    # Resize mask to original shape
                    m = res[0].masks.data.cpu().numpy()
                    if m.shape[0] > 0:
                        m_combined = np.max(m, axis=0)
                        pred_mask = cv2.resize(m_combined, (w, h), interpolation=cv2.INTER_LINEAR)
                pred_mask = (pred_mask > thr).astype(np.uint8)
                
            else:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if m_info["type"] == "deeplab":
                    # Use sliding window for deeplab
                    img_norm = (img.astype(np.float32) / 255.0 - mean) / std
                    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    with torch.no_grad():
                        prob = sliding_window_predict_deeplab(model, img_tensor, 512, 256, device, True, False)
                    pred_mask = (prob.cpu().numpy() > thr).astype(np.uint8)
                elif m_info["type"] == "u2net_full":
                    # U2Net full uses sliding window via its own function
                    prob = sliding_window_inference_u2net(model, img, device, img_size=512, tta=False)
                    pred_mask = (prob > thr).astype(np.uint8)
                else:
                    # Simple resize for UNet
                    img_resized = cv2.resize(img, (640, 640))
                    img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
                    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    with torch.no_grad():
                        out = model(img_tensor)
                        if isinstance(out, tuple):
                            out = out[0]
                        prob = torch.sigmoid(out).squeeze().cpu().numpy()
                    prob_resized = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
                    pred_mask = (prob_resized > thr).astype(np.uint8)
                    
            p, r, iou, dice = calculate_metrics(pred_mask, gt_mask)
            p_list.append(p)
            r_list.append(r)
            iou_list.append(iou)
            dice_list.append(dice)
            
        results[m_info["name"]] = {
            "Precision": np.mean(p_list),
            "Recall": np.mean(r_list),
            "IoU": np.mean(iou_list),
            "Dice": np.mean(dice_list)
        }
        
    print("\n" + "="*80)
    print(f"{'模型名称':<40} | {'Precision':<9} | {'Recall':<9} | {'IoU':<9} | {'Dice':<9}")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"{name:<40} | {metrics['Precision']:.4f}    | {metrics['Recall']:.4f}    | {metrics['IoU']:.4f}    | {metrics['Dice']:.4f}")
    print("="*80)

if __name__ == "__main__":
    evaluate_models()
