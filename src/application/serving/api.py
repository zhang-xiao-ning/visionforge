"""FastAPI app."""

import io
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from application.serving.inference import OnnxClassifier
from application.serving.schema import HealthResponse, Prediction, PredictionResponse
from framework.runtime import DEFAULT_EXPERIMENT

DEFAULT_ONNX_PATH = Path(f"exports/{DEFAULT_EXPERIMENT}.onnx")


def create_app(onnx_path: Path | None = None) -> FastAPI:
    if onnx_path is None:
        onnx_path = Path(os.environ.get("ONNX_PATH", str(DEFAULT_ONNX_PATH)))

    classifier: OnnxClassifier | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal classifier
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                f"Run `make export EXP={DEFAULT_EXPERIMENT}` first, or set ONNX_PATH."
            )
        classifier = OnnxClassifier(onnx_path)
        yield

    app = FastAPI(title="visionforge inference", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model=onnx_path.name)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        topk: int = 3,
    ) -> PredictionResponse:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError as e:
            raise HTTPException(status_code=400, detail="Invalid image file") from e

        results = classifier.predict(image, topk=topk)
        return PredictionResponse(predictions=[Prediction(**r) for r in results])

    return app


app = create_app()
