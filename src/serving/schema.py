"""Pydantic schemas for the inference API."""

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    class_id: int = Field(..., description="0-9 CIFAR-10 class index")
    class_name: str = Field(..., description="human-readable class name")
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model: str
