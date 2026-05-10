from typing import Dict


_METRICS: Dict[str, int] = {
    "prediction_requests_total": 0,
}


def increment_prediction_requests() -> None:
    _METRICS["prediction_requests_total"] += 1


def get_metrics() -> Dict[str, int]:
    return dict(_METRICS)
