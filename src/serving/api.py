"""FastAPI app."""

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from serving.inference import OnnxClassifier
from serving.schema import HealthResponse, Prediction, PredictionResponse

DEFAULT_ONNX_PATH = Path("exports/mlp.onnx")


def create_app(onnx_path: Path | None = None) -> FastAPI:
    if onnx_path is None:
        onnx_path = Path(os.environ.get("ONNX_PATH", str(DEFAULT_ONNX_PATH)))

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}. Run `make export EXP=mlp` first."
        )

    app = FastAPI(title="visionforge inference", version="0.1.0")
    classifier = OnnxClassifier(onnx_path)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model=onnx_path.name)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        topk: int = 3,
    ) -> PredictionResponse:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError as e:
            raise HTTPException(status_code=400, detail="Invalid image file") from e

        results = classifier.predict(image, topk=topk)
        return PredictionResponse(predictions=[Prediction(**r) for r in results])

    return app


app = create_app()
