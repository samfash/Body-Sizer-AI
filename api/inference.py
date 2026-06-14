import cv2
import numpy as np
import json
import os
from api.feature_engineering import extract_landmarks_from_bgr_image, compute_features_from_landmarks
from api.model_loader import load_artifacts
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"

# LOG_BUCKET = os.getenv("INFERENCE_LOG_S3_BUCKET")
# s3_client = boto3.client("s3")
# def capture_inference_log(features, preds, model_version, meta):
#     payload = {"features": features, "preds": preds, "model_version": model_version, "meta": meta}
    
#     # If no bucket is configured, skip logging
#     if not LOG_BUCKET:
#         return

#     try:
#         import time
#         # create a deterministic-ish key for the log object
#         key = f"inference_logs/{model_version or 'unknown'}_{int(time.time())}.json"
#         s3_client.put_object(Bucket=LOG_BUCKET, Key=key, Body=json.dumps(payload))
#     except Exception:
#         # Swallow any logging errors — inference should not fail due to logging
#         pass


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

    # Model predict: handle xgboost dict or sklearn multioutput
    if isinstance(model, dict):
        # per-target xgboost boosters
        preds = {}
        import xgboost as xgb
        dm = xgb.DMatrix(X)  # X already shaped (1, n_features)

        # # Load target cols from disk
        # try:
        #     import joblib
        #     target_cols = joblib.load(MODEL_DIR / "target_cols.pkl")
        # except:
        #     target_cols = ['BUST','WAIST','HIPS','HALF_LENGTH','FULL_LENGTH','SLEEVE_LENGTH']
    
        for t in target_cols:
            booster = model[t]   # xgb Booster
            pred_val = float(booster.predict(dm)[0])
            preds[t] = pred_val
    
        return preds
    else:
        y_pred = model.predict(X)  # shape (1, n_targets)
        arr = np.array(y_pred)
        # need to map index -> target names: try to load TARGET_COLS if exists
        try:
            import joblib, pathlib
            target_cols = joblib.load(Path(model.__module__).resolve().parents[1] / "model" / "target_cols.pkl")
        except Exception:
            # fallback - use default order
            target_cols = ['BUST','WAIST','HIPS','HALF_LENGTH','FULL_LENGTH','SLEEVE_LENGTH']
        preds = {t: float(arr[0, i]) for i, t in enumerate(target_cols)}
    
    return preds
