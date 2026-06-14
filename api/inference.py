import cv2
import numpy as np
from pathlib import Path

from api.feature_engineering import (
    compute_features_from_landmarks,
    extract_landmarks_from_bgr_image,
)
from api.model_loader import load_artifacts

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"


def predict_from_image_bytes(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image")

    kps = extract_landmarks_from_bgr_image(image)
    if kps is None:
        raise ValueError("No pose detected")

    feats = compute_features_from_landmarks(kps, image)
    if feats is None:
        raise ValueError("Could not compute features from pose")

    model, imputer, scaler, feature_cols, target_cols = load_artifacts()
    X = np.array([feats.get(c, np.nan) for c in feature_cols], dtype=float).reshape(1, -1)
    X = imputer.transform(X)
    X = scaler.transform(X)

    if isinstance(model, dict):
        import xgboost as xgb

        dm = xgb.DMatrix(X)
        preds = {}
        for t in target_cols:
            booster = model[t]
            preds[t] = float(booster.predict(dm)[0])
        return preds

    y_pred = model.predict(X)
    arr = np.array(y_pred)
    preds = {t: float(arr[0, i]) for i, t in enumerate(target_cols)}
    return preds
