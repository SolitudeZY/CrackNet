import torch
import argparse
from pathlib import Path

from U_Net_MobileNetV2_model import U_Net_MobileNetV2

def auto_detect_model_config(state_dict):
    attention_type = "None"
    deep_supervision = False
    for key in state_dict.keys():
        if "ds_head" in key:
            deep_supervision = True
        if "attention.W_g" in key or "attention.psi" in key or "ag.W_g" in key or "ag.psi" in key:
            attention_type = "AG"
        elif "attention.ca.fc1" in key or "attention.sa.conv1" in key:
            attention_type = "CBAM"
        elif "attention.conv.weight" in key:
            attention_type = "ECA"
    return attention_type, deep_supervision

def export_to_onnx(checkpoint_path, output_path, img_size=640, batch_size=1):
    """
    将 PyTorch 模型导出为 ONNX 格式。
    
    参数:
        checkpoint_path: PyTorch 权重文件路径 (.pt)
        output_path: 输出的 ONNX 文件路径 (.onnx)
        img_size: 模型输入图片的尺寸 (默认 640)
        batch_size: 批处理大小 (默认 1)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 1. 加载模型
    device = torch.device("cpu") # 导出时通常在 CPU 上进行即可
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # 移除可能存在的 'module.' 前缀 (DataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # 兼容旧版本键名
        normalized_state_dict = {}
        for k, v in state_dict.items():
            if ".ag." in k:
                normalized_state_dict[k.replace(".ag.", ".attention.")] = v
            else:
                normalized_state_dict[k] = v

        attention_type, deep_supervision = auto_detect_model_config(normalized_state_dict)
        print(f"Auto-detected model config: Attention='{attention_type}', DeepSupervision={deep_supervision}")

        model = U_Net_MobileNetV2(
            pretrained=False,
            attention_type=attention_type,
            deep_supervision=deep_supervision
        )
        model.load_state_dict(normalized_state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # 设置为评估模式 (非常重要，影响 BatchNorm 和 Dropout 的行为)
    model.eval()

    # 2. 创建 Dummy Input (模拟输入)
    # 输入形状: (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # 3. 导出 ONNX
    print(f"Exporting to ONNX format (Input shape: {dummy_input.shape})...")
    
    # 我们支持动态的 batch size 和 图像尺寸
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }

    try:
        torch.onnx.export(
            model,                         # 要导出的模型
            dummy_input,                   # 模拟输入
            output_path,                   # 输出文件路径
            export_params=True,            # 是否导出参数权重
            opset_version=18,              # 使用较新的 opset 版本以避免降级转换错误
            do_constant_folding=True,      # 是否执行常量折叠优化
            dynamo=False,                  # 使用经典导出器，兼容 adaptive_max_pool2d
            input_names=['input'],         # 输入节点的名称
            output_names=['output'],       # 输出节点的名称
            dynamic_axes=dynamic_axes      # 设置动态维度
        )
        print(f"✅ ONNX export successful! Saved to: {output_path}")
        print("Note: The exported model supports dynamic batch sizes and image resolutions.")
    except Exception as e:
        print(f"❌ Error during ONNX export: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export U-Net model to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="Path to the PyTorch checkpoint (.pt file)")
    parser.add_argument("--output", type=str, default="U_Net.onnx", help="Output ONNX file path")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size (default: 640)")
    args = parser.parse_args()

    export_to_onnx(args.weights, args.output, args.img_size)
