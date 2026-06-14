import joblib
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main(input_csv):
    data = pd.read_csv(input_csv)
    feature_cols = [
        c
        for c in data.columns
        if c not in [
            "USERID",
            "IMAGES",
            "BUST",
            "WAIST",
            "HIPS",
            "HALF_LENGTH",
            "FULL_LENGTH",
            "SLEEVE_LENGTH",
        ]
    ]
    X = data[feature_cols].values
    Y = data[
        [
            "BUST",
            "WAIST",
            "HIPS",
            "HALF_LENGTH",
            "FULL_LENGTH",
            "SLEEVE_LENGTH",
        ]
    ].values

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.15, random_state=42
    )

    mlflow.set_experiment("body_measurement_experiments")
    with mlflow.start_run(run_name="rf_baseline"):
        rf = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        )
        rf.fit(X_train, Y_train)
        preds = rf.predict(X_test)
        mae = np.mean(
            [
                mean_absolute_error(Y_test[:, i], preds[:, i])
                for i in range(Y_test.shape[1])
            ]
        )
        mlflow.log_metric("mean_MAE", float(mae))
        mlflow.sklearn.log_model(rf, "rf_baseline")
        joblib.dump(rf, MODEL_DIR / "best_models.pkl")
        joblib.dump(imputer, MODEL_DIR / "imputer.pkl")
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
        joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")
        joblib.dump(
            [
                "BUST",
                "WAIST",
                "HIPS",
                "HALF_LENGTH",
                "FULL_LENGTH",
                "SLEEVE_LENGTH",
            ],
            MODEL_DIR / "target_cols.pkl",
        )
        print("Saved artifacts to model/")


if __name__ == "__main__":
    import sys

    inp = sys.argv[1] if len(sys.argv) > 1 else "augmented_dataset/modeling_ready_augmented.csv"
    main(inp)
