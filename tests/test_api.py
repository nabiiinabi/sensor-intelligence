from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_endpoint() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == {
        "model_name": "placeholder-activity-classifier",
        "model_version": "0.0.1",
        "status": "not-trained",
    }


def test_predict_endpoint_returns_placeholder_prediction() -> None:
    payload = {
        "device_id": "phone-123",
        "samples": [
            {"timestamp": 1710000000.0, "x": 0.1, "y": 0.2, "z": 9.8},
            {"timestamp": 1710000000.1, "x": 0.2, "y": 0.1, "z": 9.7},
        ],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {"activity": "walking", "confidence": 0.85}


def test_metrics_endpoint() -> None:
    client.post(
        "/predict",
        json={
            "device_id": "phone-456",
            "samples": [{"timestamp": 1710000000.2, "x": 0.0, "y": 0.0, "z": 9.8}],
        },
    )
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prediction_requests_total" in response.json()
    assert response.json()["prediction_requests_total"] >= 1
