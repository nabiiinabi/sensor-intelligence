from fastapi import FastAPI

from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.models.predict import predict_activity
from src.monitoring.metrics import get_metrics, increment_prediction_requests

app = FastAPI(title="Mobile Sensor Intelligence API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name="placeholder-activity-classifier",
        model_version="0.0.1",
        status="not-trained",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    increment_prediction_requests()
    return predict_activity(payload)


@app.get("/metrics")
def metrics() -> dict:
    return get_metrics()
