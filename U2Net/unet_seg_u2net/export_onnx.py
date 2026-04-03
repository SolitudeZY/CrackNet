"""
U2NETP ONNX 导出脚本。

导出 U2NETP 模型为 ONNX 格式，仅输出 d0 (fused output) + sigmoid。
适用于 GPU 受限环境的推理部署。

用法:
  python export_onnx.py --weights runs/.../weights/best_iou.pt
  python export_onnx.py --weights best_iou.pt --output u2netp.onnx --img-size 320
"""
import argparse
import os
import warnings

import torch
from pathlib import Path

from model import U2NETP, U2NETP_ForExport


def export_to_onnx(checkpoint_path: str, output_path: str, img_size: int = 320):
    print(f"加载 checkpoint: {checkpoint_path}")
    device = torch.device("cpu")

    # 加载权重
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = U2NETP(3, 1)
    model.load_state_dict(state_dict)
    model.eval()

    if "epoch" in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")
    if "val_metrics" in ckpt:
        m = ckpt["val_metrics"]
        print(f"  Val metrics: IoU={m.get('iou', 0):.4f} Dice={m.get('dice', 0):.4f} "
              f"P={m.get('precision', 0):.4f} R={m.get('recall', 0):.4f}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")
    print(f"  模型大小: {n_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    # 包装为导出模型 (仅输出 d0 + sigmoid)
    export_model = U2NETP_ForExport(model)
    export_model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    print(f"\n导出 ONNX (input: [1, 3, {img_size}, {img_size}])...")
    warnings.filterwarnings("ignore", message=".*ONNX export.*", category=DeprecationWarning)
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # 验证
    import onnxruntime as ort
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"input": dummy_input.numpy()})

    # 一致性检查
    with torch.no_grad():
        pt_out = export_model(dummy_input).numpy()
    max_diff = abs(pt_out - ort_out[0]).max()

    onnx_size = os.path.getsize(output_path) / 1024 / 1024

    print(f"\n验证通过:")
    print(f"  输出 shape: {ort_out[0].shape}")
    print(f"  PyTorch vs ONNX 最大差异: {max_diff:.6f}")
    print(f"  ONNX 文件大小: {onnx_size:.1f} MB")
    print(f"  ONNX 已保存: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U2NETP ONNX 导出")
    parser.add_argument("--weights", type=str, required=True, help="PyTorch checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="u2netp.onnx", help="输出 ONNX 路径")
    parser.add_argument("--img-size", type=int, default=320, help="输入图像尺寸")
    args = parser.parse_args()

    export_to_onnx(args.weights, args.output, args.img_size)
