"""
通用 ONNX 导出脚本: 支持 U_Net_MobileNetV2 和 DeepCrack_MobileNetV2。
自动从 checkpoint 的 state_dict key 中推断模型类型和配置。

用法:
  python export_onnx.py --weights best_iou.pt --output model.onnx
  python export_onnx.py --weights best_iou.pt --output model.onnx --img-size 512
"""
import argparse
import warnings

import torch
from pathlib import Path


def detect_model_config(state_dict: dict) -> dict:
    """从 state_dict 的 key 名自动推断模型类型和配置。"""
    keys = list(state_dict.keys())

    # 检测模型类型
    has_side = any("side" in k for k in keys)
    has_final_fuse = any("final_fuse" in k for k in keys)
    is_deepcrack = has_side and has_final_fuse

    # 检测注意力类型
    if any("attention.ca.fc1" in k for k in keys):
        attention = "CBAM"
    elif any("attention.W_g" in k for k in keys):
        attention = "AG"
    elif any("attention.conv" in k and "attention.ca" not in k for k in keys):
        attention = "ECA"
    else:
        attention = "None"

    # 检测深度监督 (仅 U_Net_MobileNetV2)
    has_ds = any("ds_head" in k for k in keys)

    config = {
        "model_class": "DeepCrack_MobileNetV2" if is_deepcrack else "U_Net_MobileNetV2",
        "attention_type": attention,
        "deep_supervision": has_ds and not is_deepcrack,
    }
    return config


def load_model(checkpoint_path: str, device: torch.device):
    """加载 checkpoint 并返回模型实例。"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # 移除 DataParallel 前缀
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    config = detect_model_config(state_dict)
    print(f"  检测到模型: {config['model_class']}")
    print(f"  注意力类型: {config['attention_type']}")
    if config["model_class"] == "U_Net_MobileNetV2":
        print(f"  深度监督: {config['deep_supervision']}")

    if config["model_class"] == "DeepCrack_MobileNetV2":
        from DeepCrack_MobileNetV2 import DeepCrack_MobileNetV2
        model = DeepCrack_MobileNetV2(
            pretrained=False,
            attention_type=config["attention_type"],
        )
    else:
        from U_Net_MobileNetV2_model import U_Net_MobileNetV2
        model = U_Net_MobileNetV2(
            pretrained=False,
            attention_type=config["attention_type"],
            deep_supervision=config["deep_supervision"],
        )

    # 加载权重 (strict=False 容忍 ds_head 等推理时不需要的层)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # ds_head 在推理模式下不会被用到，缺失是正常的
        real_missing = [k for k in missing if "ds_head" not in k]
        if real_missing:
            print(f"  警告: 缺失的 key: {real_missing}")
    if unexpected:
        print(f"  警告: 多余的 key: {unexpected}")

    model.eval()

    # 打印 epoch 和指标
    if "epoch" in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")
    if "val_metrics" in ckpt:
        m = ckpt["val_metrics"]
        print(f"  Val metrics: IoU={m.get('iou',0):.4f} Dice={m.get('dice',0):.4f} "
              f"P={m.get('precision',0):.4f} R={m.get('recall',0):.4f}")

    return model, config


def export_to_onnx(checkpoint_path: str, output_path: str, img_size: int = 640):
    print(f"加载 checkpoint: {checkpoint_path}")
    device = torch.device("cpu")

    model, config = load_model(checkpoint_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"},
    }

    print(f"\n导出 ONNX (input: [1, 3, {img_size}, {img_size}])...")
    warnings.filterwarnings(
        "ignore",
        message="You are using the legacy TorchScript-based ONNX export.*",
        category=DeprecationWarning,
    )
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,  # 使用 legacy TorchScript 导出器 (兼容 AdaptiveMaxPool2d 等 op)
    )

    # 验证
    import onnxruntime as ort
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    out = session.run(None, {"input": dummy_input.numpy()})
    print(f"验证通过: 输出 shape = {out[0].shape}")
    print(f"ONNX 已保存: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="PyTorch checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    args = parser.parse_args()

    export_to_onnx(args.weights, args.output, args.img_size)
