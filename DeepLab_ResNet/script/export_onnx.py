"""
将 DeepLabV3+ ResNet101 的 .pt checkpoint 导出为 .onnx。

用法:
  python export_onnx.py --weights runs/20260402_110122/weights/best.pt
  python export_onnx.py --weights runs/20260402_110122/weights/last.pt --use-ema
  python export_onnx.py --weights best.pt --output cracknet.onnx --img-size 512
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch

from model import build_model


def load_checkpoint(checkpoint_path: str, use_ema: bool) -> tuple[torch.nn.Module, dict]:
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        if use_ema and "ema_state_dict" in ckpt:
            state_dict = ckpt["ema_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f"不支持的 checkpoint 格式: {type(ckpt)}")

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = build_model(pretrained=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"权重加载失败，missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}


def infer_default_output_path(weights_path: str) -> str:
    weights = Path(weights_path)
    return str(weights.with_suffix(".onnx"))


def verify_onnx_export(output_path: str, dummy_input: torch.Tensor) -> None:
    try:
        import onnx
        onnx.load(output_path)
        onnx.checker.check_model(output_path)
        print("ONNX 结构检查通过")
    except Exception as exc:
        print(f"ONNX 结构检查跳过或失败: {exc}")

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        outputs = session.run(None, {"input": dummy_input.numpy()})
        print(f"ONNXRuntime 验证通过: 输出 shape = {outputs[0].shape}")
    except Exception as exc:
        print(f"ONNXRuntime 验证跳过或失败: {exc}")


def export_to_onnx(
    weights_path: str,
    output_path: str,
    img_size: int,
    opset: int,
    dynamic: bool,
    use_ema: bool,
) -> None:
    print(f"加载 checkpoint: {weights_path}")
    model, ckpt = load_checkpoint(weights_path, use_ema=use_ema)

    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {params:,}")
    if ckpt:
        if "epoch" in ckpt:
            print(f"epoch: {ckpt['epoch']}")
        if "val_metrics" in ckpt and isinstance(ckpt["val_metrics"], dict):
            metrics = ckpt["val_metrics"]
            print(
                "val_metrics: "
                f"IoU={metrics.get('iou', 0):.4f} "
                f"Dice={metrics.get('dice', 0):.4f} "
                f"Precision={metrics.get('precision', 0):.4f} "
                f"Recall={metrics.get('recall', 0):.4f} "
                f"Thr={metrics.get('threshold', 0):.2f}"
            )

    dummy_input = torch.randn(1, 3, img_size, img_size, device="cpu")
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    output_dir = Path(output_path).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings(
        "ignore",
        message="You are using the legacy TorchScript-based ONNX export.*",
        category=DeprecationWarning,
    )

    print(f"开始导出 ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    print("导出完成")
    verify_onnx_export(output_path, dummy_input)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CrackNet DeepLab checkpoint to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="PyTorch checkpoint path (.pt)")
    parser.add_argument("--output", type=str, default="", help="Output ONNX path")
    parser.add_argument("--img-size", type=int, default=512, help="Dummy input size for export")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=True, help="Export dynamic batch/height/width axes")
    parser.add_argument("--use-ema", action="store_true", help="When exporting last.pt, prefer ema_state_dict if it exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or infer_default_output_path(args.weights)
    export_to_onnx(
        weights_path=args.weights,
        output_path=output_path,
        img_size=args.img_size,
        opset=args.opset,
        dynamic=args.dynamic,
        use_ema=args.use_ema,
    )


if __name__ == "__main__":
    main()
