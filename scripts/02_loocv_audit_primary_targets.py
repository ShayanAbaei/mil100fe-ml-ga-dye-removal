"""
LOOCV audit for Reviewer 2 response.

Inputs:
    data/mil100fe_ml_dataset_primary_targets.csv
    models/GPR_target1_BET_surface_area_best_params.pkl
    models/GPR_target2_MB_removal_best_params.pkl

Outputs:
    results/loocv_metrics_primary_targets.csv
    results/loocv_predictions_primary_targets.csv
    results/key_gpr_dummy_linear_comparison.csv

Target mapping:
    Target 1 = BET_surface_area_m2_g
    Target 2 = MB_removal_percent
"""

from pathlib import Path
import pickle
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mil100fe_ml_dataset_primary_targets.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FEATURES = [
    "time_h",
    "temperature_C",
    "metal_to_ligand_molar_ratio_Fe_to_ligand",
    "metal_concentration_M",
]

TARGETS = {
    "Target 1 - BET surface area": {
        "column": "BET_surface_area_m2_g",
        "unit": "m2/g",
        "gpr_params": ROOT / "models" / "GPR_target1_BET_surface_area_best_params.pkl",
    },
    "Target 2 - MB removal": {
        "column": "MB_removal_percent",
        "unit": "%",
        "gpr_params": ROOT / "models" / "GPR_target2_MB_removal_best_params.pkl",
    },
}


def load_params(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_models(gpr_params_path: Path) -> dict:
    params = load_params(gpr_params_path)
    return {
        "Dummy Regressor": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(fit_intercept=False),
        "Gaussian Process Regressor": GaussianProcessRegressor(**params),
    }


def loocv_evaluate(X: pd.DataFrame, y: pd.Series, sample_ids, models: dict, target_name: str, unit: str):
    loo = LeaveOneOut()
    metric_rows = []
    prediction_rows = []

    for model_name, model in models.items():
        y_true, y_pred, heldout = [], [], []

        for train_idx, test_idx in loo.split(X):
            model_i = clone(model)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model_i.fit(X_train, y_train)
            pred = model_i.predict(X_test)[0]

            y_true.append(float(y_test.iloc[0]))
            y_pred.append(float(pred))
            heldout.append(sample_ids[test_idx[0]])

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metric_rows.append({
            "Target": target_name,
            "Unit": unit,
            "Model": model_name,
            "LOOCV_RMSE": rmse,
            "LOOCV_MSE": mse,
            "LOOCV_MAE": mae,
            "LOOCV_R2": r2,
        })

        for sid, obs, pred in zip(heldout, y_true, y_pred):
            prediction_rows.append({
                "Target": target_name,
                "Unit": unit,
                "Model": model_name,
                "Sample": sid,
                "Measured": obs,
                "LOOCV_Predicted": pred,
                "Residual_Measured_minus_Predicted": obs - pred,
                "Absolute_Error": abs(obs - pred),
                "Squared_Error": (obs - pred) ** 2,
            })

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES]
    sample_ids = df["sample_id"].tolist()

    all_metrics, all_predictions = [], []
    for target_name, cfg in TARGETS.items():
        y = df[cfg["column"]]
        models = make_models(cfg["gpr_params"])
        metrics, preds = loocv_evaluate(X, y, sample_ids, models, target_name, cfg["unit"])
        all_metrics.append(metrics)
        all_predictions.append(preds)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    preds_df = pd.concat(all_predictions, ignore_index=True)

    metrics_df.to_csv(RESULTS / "loocv_metrics_primary_targets.csv", index=False)
    preds_df.to_csv(RESULTS / "loocv_predictions_primary_targets.csv", index=False)

    key = metrics_df[metrics_df["Model"].isin([
        "Dummy Regressor", "Linear Regression", "Gaussian Process Regressor"
    ])].copy()
    key.to_csv(RESULTS / "key_gpr_dummy_linear_comparison.csv", index=False)

    print("LOOCV metrics")
    print(metrics_df.sort_values(["Target", "LOOCV_RMSE"]))
    print("\nSaved outputs in:", RESULTS)


if __name__ == "__main__":
    main()
