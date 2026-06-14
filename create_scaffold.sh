#!/usr/bin/env bash
set -e

BASE="$(pwd)"

# api/main.py
cat > api/main.py <<'PY'
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import PredictionResponse, HealthResponse
from api.inference import predict_from_image_bytes
import uvicorn, os

app = FastAPI(title="Body Measurement Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}

@app.post("/predict-image", response_model=PredictionResponse)
async def predict_image(image: UploadFile = File(...)):
    contents = await image.read()
    try:
        preds = predict_from_image_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predictions": preds}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
PY

# api/schemas.py
cat > api/schemas.py <<'PY'
from pydantic import BaseModel
from typing import Dict

class HealthResponse(BaseModel):
    status: str

class PredictionResponse(BaseModel):
    predictions: Dict[str, float]
PY

# api/model_loader.py
cat > api/model_loader.py <<'PY'
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "model"

_model = None
_imputer = None
_scaler = None
_feature_cols = None
_target_cols = None

def load_artifacts():
    global _model, _imputer, _scaler, _feature_cols, _target_cols
    if _model is None:
        _model = joblib.load(MODELS_DIR / "best_models.pkl")
    if _imputer is None:
        _imputer = joblib.load(MODELS_DIR / "imputer.pkl")
    if _scaler is None:
        _scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    if _feature_cols is None:
        _feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    if _target_cols is None:
        try:
            _target_cols = joblib.load(MODELS_DIR / "target_cols.pkl")
        except Exception:
            _target_cols = ['BUST','WAIST','HIPS','HALF_LENGTH','FULL_LENGTH','SLEEVE_LENGTH']
    return _model, _imputer, _scaler, _feature_cols, _target_cols
PY

# api/feature_engineering.py
cat > api/feature_engineering.py <<'PY'
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
POSE = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

LM = {
    'left_shoulder':11,'right_shoulder':12,'left_hip':23,'right_hip':24,
    'left_wrist':15,'right_wrist':16,'left_ankle':27,'right_ankle':28
}

def extract_landmarks_from_bgr_image(image_bgr):
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = POSE.process(image_rgb)
    if not res.pose_landmarks:
        return None
    kps = []
    for lm in res.pose_landmarks.landmark:
        kps.append([lm.x * w, lm.y * h, lm.z * w, lm.visibility])
    return kps

def compute_features_from_landmarks(kps):
    if not kps:
        return None
    def dist(a,b): return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))
    try:
        ls = kps[LM['left_shoulder']]; rs = kps[LM['right_shoulder']]
        lh = kps[LM['left_hip']]; rh = kps[LM['right_hip']]
        lw = kps[LM['left_wrist']]; rw = kps[LM['right_wrist']]
        la = kps[LM['left_ankle']]; ra = kps[LM['right_ankle']]
    except Exception:
        return None

    shoulder_px = dist(ls, rs)
    hip_px = dist(lh, rh)
    sleeve_px = (dist(ls,lw) + dist(rs,rw))/2.0
    mid_shoulder = ((ls[0]+rs[0])/2.0, (ls[1]+rs[1])/2.0)
    lowest_ank_y = max(la[1], ra[1])
    height_px = abs(lowest_ank_y - mid_shoulder[1]) + 1e-6

    feats = {
        'shoulder_px': float(shoulder_px),
        'hip_px': float(hip_px),
        'sleeve_px': float(sleeve_px),
        'height_px': float(height_px),
        'shoulder_to_hip_ratio': float(shoulder_px / (hip_px + 1e-6)),
        'shoulder_to_height_ratio': float(shoulder_px / (height_px + 1e-6)),
        'sleeve_to_height_ratio': float(sleeve_px / (height_px + 1e-6)),
        'sleeve_to_shoulder_ratio': float(sleeve_px / (shoulder_px + 1e-6)),
        'shoulder_z': float((ls[2] + rs[2])/2.0),
        'hip_z': float((lh[2] + rh[2])/2.0)
    }
    return feats
PY

# api/inference.py
cat > api/inference.py <<'PY'
import cv2
import numpy as np
from api.feature_engineering import extract_landmarks_from_bgr_image, compute_features_from_landmarks
from api.model_loader import load_artifacts

def predict_from_image_bytes(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image")
    kps = extract_landmarks_from_bgr_image(image)
    if kps is None:
        raise ValueError("No pose detected")
    feats = compute_features_from_landmarks(kps)
    if feats is None:
        raise ValueError("Could not compute features from pose")

    model, imputer, scaler, feature_cols, target_cols = load_artifacts()
    X = np.array([feats.get(c, np.nan) for c in feature_cols], dtype=float).reshape(1, -1)
    X = imputer.transform(X)
    X = scaler.transform(X)
    y_pred = model.predict(X)
    arr = np.array(y_pred)
    preds = {t: float(arr[0, i]) for i, t in enumerate(target_cols)}
    return preds
PY

# scripts/train_and_log.py
cat > scripts/train_and_log.py <<'PY'
import argparse, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import mlflow
from sklearn.metrics import mean_absolute_error

MODEL_DIR = Path(__file__).resolve().parents[1] / 'model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main(input_csv):
    data = pd.read_csv(input_csv)
    feature_cols = [c for c in data.columns if c not in ['USERID','IMAGES','BUST','WAIST','HIPS','HALF_LENGTH','FULL_LENGTH','SLEEVE_LENGTH']]
    X = data[feature_cols].values
    Y = data[['BUST','WAIST','HIPS','HALF_LENGTH','FULL_LENGTH','SLEEVE_LENGTH']].values

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
    mlflow.set_experiment('body_measurement_experiments')
    with mlflow.start_run(run_name='rf_baseline'):
        rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
        rf.fit(X_train, Y_train)
        preds = rf.predict(X_test)
        mae = np.mean([mean_absolute_error(Y_test[:,i], preds[:,i]) for i in range(Y_test.shape[1])])
        mlflow.log_metric('mean_MAE', float(mae))
        mlflow.sklearn.log_model(rf, 'rf_baseline')
        joblib.dump(rf, MODEL_DIR / 'best_models.pkl')
        joblib.dump(imputer, MODEL_DIR / 'imputer.pkl')
        joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
        joblib.dump(feature_cols, MODEL_DIR / 'feature_cols.pkl')
        joblib.dump(['BUST','WAIST','HIPS','HALF_LENGTH','FULL_LENGTH','SLEEVE_LENGTH'], MODEL_DIR / 'target_cols.pkl')
        print('Saved artifacts to model/')
if __name__ == '__main__':
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else 'augmented_dataset/modeling_ready_augmented.csv'
    main(inp)
PY

# docker/Dockerfile
cat > docker/Dockerfile <<'PY'
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
PY

# infra/docker-compose.yml
cat > infra/docker-compose.yml <<'PY'
version: '3.8'
services:
  app:
    build: ..
    container_name: body_api
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ../model:/app/model
    depends_on:
      - mlflow

  mlflow:
    image: python:3.10-slim
    container_name: mlflow
    command: bash -lc "pip install mlflow && mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlruns --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    volumes:
      - ../mlruns:/mlruns
      - ../mlflow.db:/mlflow.db
PY

# requirements.txt
cat > requirements.txt <<'PY'
fastapi
uvicorn[standard]
numpy
pandas
opencv-python
mediapipe
scikit-learn
xgboost
joblib
pydantic
albumentations
mlflow
python-multipart
optuna
pytest
PY

# .env.sample
cat > .env.sample <<'PY'
MLFLOW_TRACKING_URI=http://localhost:5000
PY

# .gitignore
cat > .gitignore <<'PY'
__pycache__/
*.pyc
.env
model/*.pkl
mlruns/
mlflow.db
PY

echo "Scaffold created in $(pwd). Run 'code .' to open VS Code here."
