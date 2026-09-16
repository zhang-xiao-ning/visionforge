"""ONNX-based inference."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from data.transforms import cifar10_test_transform

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class OnnxClassifier:
    def __init__(self, onnx_path: Path) -> None:
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.transform = cifar10_test_transform()
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: Image.Image, topk: int = 3) -> list[dict]:
        """Returns list of {class_id, class_name, confidence}, sorted desc."""
        tensor = self.transform(image)  # (3, 32, 32)
        batch = tensor.unsqueeze(0).numpy()  # (1, 3, 32, 32)

        logits = self.session.run(None, {self.input_name: batch})[0]  # (1, 10)
        probs = _softmax(logits[0])

        top_idx = np.argsort(probs)[::-1][:topk]
        return [
            {
                "class_id": int(i),
                "class_name": CIFAR10_CLASSES[int(i)],
                "confidence": float(probs[int(i)]),
            }
            for i in top_idx
        ]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
