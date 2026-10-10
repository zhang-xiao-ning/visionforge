"""Export PyTorch checkpoints to ONNX."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from application.registry import EXPERIMENTS
from framework.utils import CHECKPOINTS_PATH


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
    """Rebuild the model from the checkpoint's stored `model_init`.

    `model_init` was saved by `ExperimentRunner._save_checkpoint` and
    contains exactly the kwargs needed to reconstruct the model.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = model_cls(**ckpt["model_init"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _export_spec(category: str) -> tuple[tuple[torch.Tensor, ...], dict]:
    """Return (dummy_args, onnx_io) for a given experiment category.

    onnx_io = {"input_names": [...], "output_names": [...], "dynamic_axes": {...}}
    """
    if category == "classification":
        dummy = torch.randn(1, 3, 32, 32)
        return (dummy,), {
            "input_names": ["input"],
            "output_names": ["output"],
            "dynamic_axes": {"input": {0: "batch"}, "output": {0: "batch"}},
        }
    if category == "captioning":
        dummy_images = torch.randn(1, 3, 224, 224)
        dummy_ids = torch.ones(1, 8, dtype=torch.long)
        return (dummy_images, dummy_ids), {
            "input_names": ["images", "input_ids"],
            "output_names": ["logits"],
            "dynamic_axes": {
                "images": {0: "batch"},
                "input_ids": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
        }
    raise ValueError(f"Unsupported category for ONNX export: {category}")


def export_to_onnx(
    model: nn.Module,
    onnx_path: Path,
    category: str,
    opset_version: int = 17,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_args, io = _export_spec(category)
    torch.onnx.export(
        model,
        dummy_args,
        str(onnx_path),
        input_names=io["input_names"],
        output_names=io["output_names"],
        dynamic_axes=io["dynamic_axes"],
        opset_version=opset_version,
    )
    print(f"Exported ONNX to {onnx_path}")


def verify_onnx(
    onnx_path: Path,
    pytorch_model: nn.Module,
    category: str,
    atol: float = 1e-4,
) -> None:
    """Compare PyTorch and ONNX outputs on a batch-4 dummy input."""
    if category == "classification":
        dummy = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            torch_out = pytorch_model(dummy).numpy()
        feed = {"input": dummy.numpy()}
    elif category == "captioning":
        dummy_images = torch.randn(4, 3, 224, 224)
        dummy_ids = torch.ones(4, 8, dtype=torch.long)
        with torch.no_grad():
            torch_out = pytorch_model(dummy_images, dummy_ids).numpy()
        feed = {"images": dummy_images.numpy(), "input_ids": dummy_ids.numpy()}
    else:
        raise ValueError(f"Unsupported category for ONNX verify: {category}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, feed)[0]

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

    experiment = EXPERIMENTS[args.experiment]
    model_cls = experiment.model
    category = experiment.category

    print(f"Experiment: {args.experiment}")
    print(f"Category:   {category}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output:     {output_path}")

    model = load_model(ckpt_path, model_cls)
    export_to_onnx(model, output_path, category=category, opset_version=args.opset)
    verify_onnx(output_path, model, category=category)


if __name__ == "__main__":
    main()
