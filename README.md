# Body Sizer AI

Body Sizer AI is a Python service for estimating body measurements from person photos. It provides a FastAPI inference endpoint that extracts pose landmarks using MediaPipe, computes geometric features, and predicts measurements using a trained ML model.

## Key Features

- FastAPI-based prediction API
- Image upload endpoint for body measurement inference
- MediaPipe pose landmark extraction
- Feature engineering for pose-based dimension estimates
- ML training pipeline with scikit-learn and MLflow
- Docker and Docker Compose support
- Prometheus metrics endpoint
- OpenTelemetry tracing support

## Repository Structure

- `api/` - FastAPI application logic, inference pipeline, feature engineering, and model loading
- `model/` - Trained model artifacts and preprocessing objects (not committed by default)
- `observability/` - Metrics, tracing, and logging helpers
- `scripts/` - Model training and logging script
- `docker/` - Dockerfile for containerized app
- `infra/` - Docker Compose setup for app + MLflow
- `test_images/` - Sample image assets for local testing
- `.github/workflows/ci.yml` - GitHub Actions CI pipeline
- `requirements.txt` - Python dependencies
- `.env.sample` - Example environment variables

## Getting Started

### Requirements

- Python 3.10
- `pip`
- `docker` and `docker-compose` (optional)

### Local Setup

1. Create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API locally:

```bash
python api/main.py
```

4. Open the API docs:

- `http://localhost:8001/docs`

### Environment Variables

Use `.env.sample` as a template. Example:

```text
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Docker Compose

Run the application and MLflow together:

```bash
docker compose -f infra/docker-compose.yml up --build
```

This starts:

- API service on `http://localhost:8001`
- MLflow tracking server on `http://localhost:5001`

## Training the Model

The repository includes a training script at `scripts/train_and_log.py`.

```bash
python scripts/train_and_log.py <path-to-training-csv>
```

The script saves model artifacts into `model/`:

- `best_models.pkl`
- `imputer.pkl`
- `scaler.pkl`
- `feature_cols.pkl`
- `target_cols.pkl`

> Note: `model/` artifacts are excluded from Git by `.gitignore`. Add your trained files locally before running the API.

## API Endpoints

### Health check

- `GET /health`
- Returns JSON `{ "status": "ok" }`

### Image prediction

- `POST /predict-image`
- Accepts a multipart file field named `image`
- Returns predicted measurements

Example using `curl`:

```bash
curl -X POST "http://localhost:8001/predict-image" \
  -F "image=@/path/to/photo.jpg"
```

## Observability

- Metrics endpoint: `GET /metrics`
- Prometheus-compatible metrics
- OpenTelemetry tracing configured via `OTEL_EXPORTER_OTLP_ENDPOINT`

## Continuous Integration

GitHub Actions CI is configured in `.github/workflows/ci.yml` and runs:

- dependency installation
- linting with `flake8`
- simple Pytest execution

## Notes

- The inference pipeline expects valid person pose landmarks from MediaPipe.
- The app loads model artifacts from the `model/` directory at runtime.
- If no model artifacts exist, the API will raise a `FileNotFoundError`.

## Contributing

See `CONTRIBUTING.md` for contribution guidelines.

## License

This project does not include a license file by default. Add a `LICENSE` file to define the terms for reuse.