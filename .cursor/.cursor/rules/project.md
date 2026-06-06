# Project Rules

This is a learning-focused production ML project.

Project goal:
Build a Dockerized FastAPI ML inference service for mobile sensor activity classification.

The user knows Python and ML but is learning backend/SWE concepts. Prioritize readable, explainable code over clever abstractions.

Tech stack:
- Python
- FastAPI
- Pydantic
- pandas
- NumPy
- scikit-learn
- pytest
- Docker
- AWS later
- PyTorch optional later

Code organization:
- src/api: FastAPI app, routes, schemas
- src/data: data loading and validation
- src/features: feature engineering/windowing
- src/models: training, evaluation, prediction
- src/monitoring: logging and metrics
- tests: pytest tests

Rules:
- Do not build the whole project at once.
- Keep changes small and scoped.
- Keep API logic separate from model logic.
- Keep training code separate from inference code.
- Add tests when adding new behavior.
- Explain important backend concepts in comments or summaries when helpful.
- Prefer simple functions with clear names.