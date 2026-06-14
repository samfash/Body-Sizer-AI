import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"

_model = None
_imputer = None
_scaler = None
_feature_cols = None
_target_cols = None


def load_artifacts():
    """
    Returns: model, imputer, scaler, feature_cols (list), target_cols (list)
    model can be:
      - sklearn-like estimator with .predict(X) -> (n_samples, n_targets)
      - dict of { target_name: xgboost.Booster or xgb.XGBRegressor }
    """
    global _model, _imputer, _scaler, _feature_cols, _target_cols

    if _model is None:
        model_path = MODEL_DIR / "best_models.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        _model = joblib.load(model_path)

    if _imputer is None:
        imp_path = MODEL_DIR / "imputer.pkl"
        _imputer = joblib.load(imp_path) if imp_path.exists() else None

    if _scaler is None:
        scl_path = MODEL_DIR / "scaler.pkl"
        _scaler = joblib.load(scl_path) if scl_path.exists() else None

    if _feature_cols is None:
        fc_path = MODEL_DIR / "feature_cols.pkl"
        if fc_path.exists():
            _feature_cols = joblib.load(fc_path)
        else:
            raise FileNotFoundError("feature_cols.pkl not found in model directory")

    if _target_cols is None:
        tc_path = MODEL_DIR / "target_cols.pkl"
        if tc_path.exists():
            _target_cols = joblib.load(tc_path)
        else:
            _target_cols = [
                "SHOULDER_x",
                "BUST_x",
                "WAIST_x",
                "HIPS_x",
                "HALF_LENGTH_x",
                "FULL_LENGTH_x",
                "SLEEVE_LENGTH_x",
            ]

    return _model, _imputer, _scaler, _feature_cols, _target_cols
