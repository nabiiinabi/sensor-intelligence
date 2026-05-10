# Mobile Sensor Intelligence Platform PRD

## Goal
Build a production-style ML service that classifies human activity from smartphone/smartwatch sensor data.

## MVP
- Use WISDM sensor data
- Train a baseline scikit-learn activity classifier
- Serve predictions through FastAPI
- Add request validation with Pydantic
- Add Docker
- Add pytest tests
- Add basic monitoring: /health, /model-info, /metrics, logs

## Stretch
- PyTorch model
- AWS deployment
- MLflow/Databricks
- robustness testing on noisy/missing sensor data

## Core API endpoints
- GET /health
- GET /model-info
- POST /predict
- GET /metrics

## Learning goal
I know Python and ML, but I am learning backend/SWE concepts. Code should be simple, modular, and explainable.