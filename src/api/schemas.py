from typing import List

from pydantic import BaseModel, Field


class SensorSample(BaseModel):
    timestamp: float = Field(..., description="Unix timestamp in seconds.")
    x: float
    y: float
    z: float


class PredictionRequest(BaseModel):
    device_id: str = Field(..., min_length=1)
    samples: List[SensorSample] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    activity: str
    confidence: float


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    status: str
