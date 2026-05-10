from src.api.schemas import PredictionRequest, PredictionResponse


def predict_activity(_: PredictionRequest) -> PredictionResponse:
    """
    Placeholder prediction function for milestone 1.
    A real model inference pipeline will replace this later.
    """
    return PredictionResponse(activity="walking", confidence=0.85)
