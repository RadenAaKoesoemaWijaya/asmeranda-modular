# Asmeranda AI

Asmeranda AI is a modular machine learning platform for end-to-end data science workflows: upload dataset, explore data, preprocess, train models, interpret results, and run time series forecasting/anomaly detection.

## Stack
- Backend: FastAPI + Pydantic + Polars + scikit-learn
- Frontend: Next.js App Router
- Storage: Parquet datasets with metadata sidecars in the `data/` folder
- Deployment: Docker Compose, with Azure-ready configuration files available

## Active runtime
The project currently runs on the FastAPI backend and the Next.js frontend as the primary application path. The legacy Streamlit UI is retained only for compatibility and is not the main runtime.

## Features
- Dataset upload and metadata management
- EDA summary, correlation matrix, and paginated table preview
- Preprocessing pipeline with scaling, encoding, and train/test split
- Model training for classification/regression/forecasting workflows
- SHAP and LIME interpretation
- Time series forecasting and anomaly detection
- Docker-based local setup

## Quick start

### Docker
```bash
docker compose up --build
```

Access points:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API docs: http://localhost:8000/docs

### Local development
```bash
# Backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

## Main folders
- `backend/` — FastAPI app, schemas, services, routes
- `frontend/` — Next.js app and API client
- `core/` — shared state and utilities
- `ml_engine/` — ML helper logic
- `data/` — dataset files and generated artifacts
- `pages/` — legacy Streamlit pages
- `tests_*.py` — smoke, behavior, and end-to-end validation scripts

## Notes
- Use [docker-compose.yml](docker-compose.yml) as the canonical local runtime.
- Deployment-specific files such as [docker-compose.azure.yml](docker-compose.azure.yml) are for cloud override scenarios.
- Keep generated runtime artifacts in `data/` or temp folders; remove caches/logs before packaging or sharing the project.
