"""
Export U2Net/U2NetP checkpoints (.pt/.pth) to ONNX.

Features:
  - supports train.py checkpoints and plain state_dict files
  - auto-detects model name and input size from checkpoint when possible
  - exports only the main output d0 after sigmoid
  - optional ONNX Runtime verification
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import torch

from model import U2NetForExport, build_model


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[dict, str, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_name = checkpoint.get("model_name") or checkpoint.get("config", {}).get("model", "u2net")
        img_size = int(checkpoint.get("img_size", checkpoint.get("config", {}).get("img_size", 512)))
    else:
        state_dict = checkpoint
        model_name = "u2net"
        img_size = 512

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return state_dict, model_name, img_size


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    model_name_override: str,
    img_size_override: int,
    opset: int,
    verify: bool,
) -> None:
    device = torch.device("cpu")
    state_dict, ckpt_model_name, ckpt_img_size = load_checkpoint(checkpoint_path, device)

    model_name = model_name_override or ckpt_model_name
    img_size = img_size_override or ckpt_img_size

    model = build_model(model_name, in_ch=3, out_ch=1)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    export_model = U2NetForExport(model).eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"},
    }

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {model_name}")
    print(f"Input size: {img_size}")
    print(f"Output: {output_path}")

    warnings.filterwarnings("ignore", message=".*ONNX export.*", category=DeprecationWarning)
    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        dynamo=False,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    print(f"ONNX saved: {output_path}")
    print(f"ONNX size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    if verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("Skip verification: onnxruntime is not installed.")
            return

        with torch.no_grad():
            torch_out = export_model(dummy_input).numpy()

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        ort_out = session.run(None, {"input": dummy_input.numpy()})[0]
        max_diff = abs(torch_out - ort_out).max()
        print(f"Verification passed. max_diff={max_diff:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export U2Net checkpoints to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt/.pth checkpoint")
    parser.add_argument("--output", type=str, default="", help="Output ONNX path")
    parser.add_argument("--model", type=str, default="", choices=["", "u2net", "u2netp"], help="Override model name")
    parser.add_argument("--img-size", type=int, default=0, help="Override input image size")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True, help="Verify exported ONNX with onnxruntime")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.weights).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = checkpoint_path.with_suffix(".onnx")

    export_to_onnx(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        model_name_override=args.model,
        img_size_override=args.img_size,
        opset=args.opset,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
