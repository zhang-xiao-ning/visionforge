"""Export PyTorch checkpoints to ONNX."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from main import EXPERIMENTS
from utils.path import CHECKPOINTS_PATH


def find_latest_checkpoint(experiment: str) -> Path:
    """找到某个 experiment 下最新的 checkpoint。"""
    candidates = sorted(
        CHECKPOINTS_PATH.glob(f"{experiment}_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for experiment '{experiment}'")
    return candidates[-1]


def load_model(checkpoint_path: Path, model_cls: type[nn.Module]) -> nn.Module:
    model = model_cls()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def export_to_onnx(
    model: nn.Module,
    onnx_path: Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    opset_version: int = 17,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset_version,
    )
    print(f"Exported ONNX to {onnx_path}")


def verify_onnx(
    onnx_path: Path,
    pytorch_model: nn.Module,
    input_shape: tuple[int, int, int, int] = (4, 3, 32, 32),
    atol: float = 1e-4,
) -> None:
    """对比 PyTorch 和 ONNX 的输出。"""
    dummy = torch.randn(*input_shape)

    with torch.no_grad():
        torch_out = pytorch_model(dummy).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    if max_diff > atol:
        raise AssertionError(f"ONNX output differs: max_diff={max_diff:.3e} > atol={atol}")
    print(f"Verification passed. max_diff = {max_diff:.2e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX.")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list(EXPERIMENTS.keys()),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="path to checkpoint; if not set, use latest for the experiment",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output ONNX path; default exports/<experiment>.onnx",
    )
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_latest_checkpoint(args.experiment)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_path = (
        Path(args.output)
        if args.output is not None
        else Path("exports") / f"{args.experiment}.onnx"
    )

    model_cls, _ = EXPERIMENTS[args.experiment]

    print(f"Experiment: {args.experiment}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output: {output_path}")

    model = load_model(ckpt_path, model_cls)
    export_to_onnx(model, output_path, opset_version=args.opset)
    verify_onnx(output_path, model)


if __name__ == "__main__":
    main()
