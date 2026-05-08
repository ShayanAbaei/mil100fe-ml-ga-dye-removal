"""
Created for Reviewer 2 response
Original project: MIL-100(Fe) dye adsorption ML-GA revision

Author: Shayan Abaei
Purpose:
    Leave-One-Out Cross-Validation (LOOCV) evaluation for:
        Target 1 = BET surface area
        Target 2 = MB removal

    This script keeps the original data paths and best-parameter paths,
    but replaces the 10-fold CV evaluation with LOOCV.
"""

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0. Original paths from your source code
# ============================================================

save_path = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11'

file_path = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/Data_Set_1.11.xlsx'

best_params_gpr_target1_path = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Gaussian Process Regressor_target1_best_params.pkl'
best_params_gpr_target2_path = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Gaussian Process Regressor_target2_best_params.pkl'
best_params_lsvr_target1_path = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Linear SVR_target1_best_params.pkl'
best_params_lsvr_target2_path = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11/Linear SVR_target2_best_params.pkl'


# ============================================================
# 1. Imports
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import sqrt

from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

import xgboost as xgb


# ============================================================
# 2. Output folder
# ============================================================

loocv_path = os.path.join(save_path, "LOOCV_review_response")
os.makedirs(loocv_path, exist_ok=True)

print("\nLOOCV output folder:")
print(loocv_path)


# ============================================================
# 3. Load data exactly like your source code
# ============================================================

train_data = pd.read_excel(file_path, sheet_name="Train Set", skiprows=2)
test_data = pd.read_excel(file_path, sheet_name="Test Set", skiprows=2)

# Drop extra empty column if it exists
for df in [train_data, test_data]:
    if "Unnamed: 5" in df.columns:
        df.drop(columns=["Unnamed: 5"], inplace=True)

# Combine original train and test sheets into the full 11-point dataset
data = pd.concat([train_data, test_data], ignore_index=True)
data = data.reset_index(drop=True)

# Save machine-readable dataset for reviewer / GitHub / SI
data.to_csv(os.path.join(loocv_path, "mil100fe_full_ml_dataset_from_excel.csv"), index=False)

# Define X and targets
X = data.drop(columns=["Unnamed: 0", "target 1", "target 2"])

# IMPORTANT:
# target 1 = BET surface area
# target 2 = MB removal
y_target1 = data["target 1"]
y_target2 = data["target 2"]

# Sample labels
if "Unnamed: 0" in data.columns:
    sample_names = []
    for value in data["Unnamed: 0"].tolist():
        value_str = str(value)
        if value_str.startswith("S"):
            sample_names.append(value_str)
        else:
            sample_names.append("S" + value_str)
else:
    sample_names = [f"S{i+1}" for i in range(len(data))]

print("\nDetected input features:")
print(list(X.columns))

print("\nNumber of samples:", len(data))
print("Sample names:", sample_names)

print("\nTarget 1 = BET surface area")
print("Target 2 = MB removal")


# ============================================================
# 4. Load original best parameters
# ============================================================

def load_best_params(path):
    with open(path, 'rb') as file:
        best_params = pickle.load(file)
    return best_params

best_params_gpr_target1 = load_best_params(best_params_gpr_target1_path)
best_params_gpr_target2 = load_best_params(best_params_gpr_target2_path)
best_params_lsvr_target1 = load_best_params(best_params_lsvr_target1_path)
best_params_lsvr_target2 = load_best_params(best_params_lsvr_target2_path)

print("\nLoaded best parameters successfully.")


# ============================================================
# 5. Helper for LinearSVR parameters
# Avoid duplicate keyword errors if random_state/max_iter already exist
# ============================================================

def make_lsvr(params):
    params_safe = params.copy()
    params_safe.setdefault("random_state", 42)
    params_safe.setdefault("max_iter", 100000)
    return LinearSVR(**params_safe)


# ============================================================
# 6. Define models separately for each target
# This fixes the danger in the old code where Target 1 could be
# evaluated using Target 2 model settings.
# ============================================================

# ---------------------------
# Target 1: BET surface area
# ---------------------------

models_target1 = {
    "Linear Regression": LinearRegression(fit_intercept=False, n_jobs=None),

    "Ridge Regression": Ridge(alpha=100, solver='saga', random_state=42),

    "Lasso Regression": Lasso(alpha=100, selection='random', random_state=42),

    "Decision Trees": DecisionTreeRegressor(
        criterion='friedman_mse',
        max_depth=6,
        min_samples_split=4,
        splitter='random',
        max_features='sqrt',
        min_samples_leaf=2,
        random_state=1671
    ),

    "Random Forest": RandomForestRegressor(
        bootstrap=False,
        n_estimators=10,
        max_depth=5,
        max_features='sqrt',
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=1810
    ),

    "Support Vector Machines": SVR(
        C=0.1,
        epsilon=0.01,
        gamma='scale',
        kernel='linear',
        degree=2
    ),

    "K-Nearest Neighbors": KNeighborsRegressor(
        algorithm='auto',
        n_neighbors=7,
        weights='uniform',
        leaf_size=20,
        p=2
    ),

    # Reviewer 2's requested mean-property baseline
    "Dummy Regressor": DummyRegressor(),

    "XGBoost": xgb.XGBRegressor(
        colsample_bytree=0.3,
        gamma=0,
        learning_rate=0.04,
        max_depth=2,
        n_estimators=100,
        subsample=0.5,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42
    ),

    "Gaussian Process Regressor": GaussianProcessRegressor(**best_params_gpr_target1),

    "Linear SVR": make_lsvr(best_params_lsvr_target1)
}


# ---------------------------
# Target 2: MB removal
# ---------------------------

models_target2 = {
    "Linear Regression": LinearRegression(fit_intercept=False, n_jobs=None),

    "Ridge Regression": Ridge(alpha=100, solver='saga', random_state=42),

    "Lasso Regression": Lasso(alpha=10, selection='cyclic', random_state=42),

    "Decision Trees": DecisionTreeRegressor(
        criterion='friedman_mse',
        max_depth=11,
        min_samples_split=5,
        splitter='best',
        max_features='sqrt',
        min_samples_leaf=1,
        random_state=512
    ),

    "Random Forest": RandomForestRegressor(
        bootstrap=True,
        n_estimators=10,
        max_depth=9,
        max_features='log2',
        min_samples_leaf=1,
        min_samples_split=4,
        random_state=481
    ),

    "Support Vector Machines": SVR(
        C=464.15888336127773,
        epsilon=0.01,
        gamma='scale',
        kernel='rbf',
        degree=2
    ),

    "K-Nearest Neighbors": KNeighborsRegressor(
        algorithm='auto',
        n_neighbors=3,
        weights='uniform',
        leaf_size=20,
        p=1
    ),

    # Reviewer 2's requested mean-property baseline
    "Dummy Regressor": DummyRegressor(),

    "XGBoost": xgb.XGBRegressor(
        colsample_bytree=1,
        gamma=0,
        learning_rate=0.04,
        max_depth=3,
        n_estimators=100,
        subsample=0.5,
        reg_alpha=0,
        reg_lambda=0.5,
        random_state=42
    ),

    "Gaussian Process Regressor": GaussianProcessRegressor(**best_params_gpr_target2),

    "Linear SVR": make_lsvr(best_params_lsvr_target2)
}


# ============================================================
# 7. LOOCV evaluation function
# ============================================================

def loocv_evaluate_models(X, y, models, target_name, target_unit, sample_names):
    """
    Leave-One-Out Cross-Validation.
    Each sample is held out once as the TEST point.
    The model is fitted only on the remaining 10 points.
    Metrics are calculated from all 11 held-out predictions together.
    """

    loo = LeaveOneOut()

    all_metrics = []
    all_predictions = []

    for model_name, model in models.items():

        y_true_all = []
        y_pred_all = []
        heldout_sample_all = []
        train_rmse_all = []
        train_mae_all = []

        print(f"\nRunning LOOCV: {target_name} | {model_name}")

        for train_idx, test_idx in loo.split(X):

            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()

            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()

            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            y_pred_test = model_clone.predict(X_test)
            y_pred_train = model_clone.predict(X_train)

            true_value = float(y_test.iloc[0])
            pred_value = float(y_pred_test[0])

            y_true_all.append(true_value)
            y_pred_all.append(pred_value)
            heldout_sample_all.append(sample_names[test_idx[0]])

            train_rmse_all.append(sqrt(mean_squared_error(y_train, y_pred_train)))
            train_mae_all.append(mean_absolute_error(y_train, y_pred_train))

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        mse = mean_squared_error(y_true_all, y_pred_all)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_true_all, y_pred_all)

        # Correct way for LOOCV:
        # calculate R2 over the pooled held-out predictions,
        # not fold-by-fold, because each fold has only one test sample.
        r2 = r2_score(y_true_all, y_pred_all)

        metrics_row = {
            "Target": target_name,
            "Unit": target_unit,
            "Model": model_name,
            "LOOCV_MSE": mse,
            "LOOCV_RMSE": rmse,
            "LOOCV_MAE": mae,
            "LOOCV_R2": r2,
            "Average_Training_RMSE": np.mean(train_rmse_all),
            "Average_Training_MAE": np.mean(train_mae_all)
        }

        all_metrics.append(metrics_row)

        pred_df_model = pd.DataFrame({
            "Target": target_name,
            "Unit": target_unit,
            "Model": model_name,
            "Sample": heldout_sample_all,
            "Measured": y_true_all,
            "LOOCV_Predicted": y_pred_all,
            "Residual_Measured_minus_Predicted": y_true_all - y_pred_all,
            "Absolute_Error": np.abs(y_true_all - y_pred_all),
            "Squared_Error": (y_true_all - y_pred_all) ** 2
        })

        all_predictions.append(pred_df_model)

    metrics_df = pd.DataFrame(all_metrics).sort_values(by="LOOCV_RMSE")
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    return metrics_df, predictions_df


# ============================================================
# 8. Run LOOCV for both targets
# ============================================================

metrics_target1, predictions_target1 = loocv_evaluate_models(
    X=X,
    y=y_target1,
    models=models_target1,
    target_name="Target 1 - BET surface area",
    target_unit="m2/g",
    sample_names=sample_names
)

metrics_target2, predictions_target2 = loocv_evaluate_models(
    X=X,
    y=y_target2,
    models=models_target2,
    target_name="Target 2 - MB removal",
    target_unit="%",
    sample_names=sample_names
)

all_metrics = pd.concat([metrics_target1, metrics_target2], ignore_index=True)
all_predictions = pd.concat([predictions_target1, predictions_target2], ignore_index=True)


# ============================================================
# 9. Save LOOCV metrics and predictions
# ============================================================

metrics_target1.to_csv(
    os.path.join(loocv_path, "loocv_metrics_target1_BET_surface_area.csv"),
    index=False
)

metrics_target2.to_csv(
    os.path.join(loocv_path, "loocv_metrics_target2_MB_removal.csv"),
    index=False
)

all_metrics.to_csv(
    os.path.join(loocv_path, "loocv_metrics_all_models_both_targets.csv"),
    index=False
)

predictions_target1.to_csv(
    os.path.join(loocv_path, "loocv_predictions_target1_BET_surface_area.csv"),
    index=False
)

predictions_target2.to_csv(
    os.path.join(loocv_path, "loocv_predictions_target2_MB_removal.csv"),
    index=False
)

all_predictions.to_csv(
    os.path.join(loocv_path, "loocv_predictions_all_models_both_targets.csv"),
    index=False
)


# ============================================================
# 10. Extract key reviewer comparison:
# GPR vs Dummy Regressor vs Linear Regression
# ============================================================

key_models = ["Dummy Regressor", "Linear Regression", "Gaussian Process Regressor"]

key_target1 = metrics_target1[metrics_target1["Model"].isin(key_models)].copy()
key_target2 = metrics_target2[metrics_target2["Model"].isin(key_models)].copy()

key_comparison = pd.concat([key_target1, key_target2], ignore_index=True)

key_comparison.to_csv(
    os.path.join(loocv_path, "KEY_REVIEWER2_GPR_vs_Dummy_vs_Linear.csv"),
    index=False
)

print("\n\n============================================================")
print("KEY REVIEWER 2 COMPARISON")
print("Target 1 = BET surface area")
print("Target 2 = MB removal")
print("============================================================")

print("\nTarget 1 - BET surface area:")
print(key_target1[["Model", "LOOCV_RMSE", "LOOCV_MSE", "LOOCV_MAE", "LOOCV_R2"]])

print("\nTarget 2 - MB removal:")
print(key_target2[["Model", "LOOCV_RMSE", "LOOCV_MSE", "LOOCV_MAE", "LOOCV_R2"]])


# ============================================================
# 11. Plot functions
# ============================================================

def make_parity_plot(predictions_df, target_filename, target_title, y_label, model_name):
    """
    Makes a measured-vs-predicted parity plot for one model.
    """

    df = predictions_df[predictions_df["Model"] == model_name].copy()

    measured = df["Measured"].values
    predicted = df["LOOCV_Predicted"].values

    min_val = min(measured.min(), predicted.min())
    max_val = max(measured.max(), predicted.max())

    if max_val == min_val:
        pad = 1.0
    else:
        pad = 0.07 * (max_val - min_val)

    lims = [min_val - pad, max_val + pad]

    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=600)

    ax.scatter(
        measured,
        predicted,
        s=75,
        edgecolor="black",
        linewidth=0.8
    )

    for _, row in df.iterrows():
        ax.text(
            row["Measured"],
            row["LOOCV_Predicted"],
            str(row["Sample"]),
            fontsize=8,
            ha="left",
            va="bottom"
        )

    ax.plot(lims, lims, linestyle="--", linewidth=1.2)

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(f"Measured {y_label}", fontsize=11)
    ax.set_ylabel(f"LOOCV-predicted {y_label}", fontsize=11)
    ax.set_title(f"{target_title}\n{model_name}", fontsize=11)

    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()

    png_path = os.path.join(loocv_path, f"parity_{target_filename}_{model_name.replace(' ', '_')}.png")
    svg_path = os.path.join(loocv_path, f"parity_{target_filename}_{model_name.replace(' ', '_')}.svg")

    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()

    return png_path, svg_path


def make_baseline_bar_plot(metrics_df, target_filename, target_title, y_unit):
    """
    Makes a reviewer-facing RMSE comparison:
    Dummy Regressor vs Linear Regression vs GPR.
    """

    df = metrics_df[metrics_df["Model"].isin(key_models)].copy()

    order = ["Dummy Regressor", "Linear Regression", "Gaussian Process Regressor"]
    df["Model"] = pd.Categorical(df["Model"], categories=order, ordered=True)
    df = df.sort_values("Model")

    fig, ax = plt.subplots(figsize=(6.0, 4.3), dpi=600)

    bars = ax.bar(
        df["Model"].astype(str),
        df["LOOCV_RMSE"],
        edgecolor="black",
        linewidth=0.8
    )

    for bar, value in zip(bars, df["LOOCV_RMSE"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    ax.set_ylabel(f"LOOCV RMSE {y_unit}", fontsize=11)
    ax.set_title(target_title, fontsize=11)

    ax.tick_params(axis="x", labelsize=9, rotation=18)
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()

    png_path = os.path.join(loocv_path, f"baseline_RMSE_{target_filename}.png")
    svg_path = os.path.join(loocv_path, f"baseline_RMSE_{target_filename}.svg")

    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()

    return png_path, svg_path


# ============================================================
# 12. Generate reviewer-ready figures
# ============================================================

# Target 1: BET surface area
make_parity_plot(
    predictions_df=predictions_target1,
    target_filename="target1_BET_surface_area",
    target_title="Target 1: BET surface area",
    y_label="BET surface area (m² g⁻¹)",
    model_name="Gaussian Process Regressor"
)

make_baseline_bar_plot(
    metrics_df=metrics_target1,
    target_filename="target1_BET_surface_area",
    target_title="Target 1: BET surface area baseline comparison",
    y_unit="(m² g⁻¹)"
)

# Target 2: MB removal
make_parity_plot(
    predictions_df=predictions_target2,
    target_filename="target2_MB_removal",
    target_title="Target 2: MB removal",
    y_label="MB removal (%)",
    model_name="Gaussian Process Regressor"
)

make_baseline_bar_plot(
    metrics_df=metrics_target2,
    target_filename="target2_MB_removal",
    target_title="Target 2: MB removal baseline comparison",
    y_unit="(%)"
)


# ============================================================
# 13. Save a short text summary for manuscript/response letter
# ============================================================

def get_metric(metrics_df, model_name, metric_name):
    return float(metrics_df.loc[metrics_df["Model"] == model_name, metric_name].iloc[0])

summary_lines = []

summary_lines.append("Reviewer 2 LOOCV summary")
summary_lines.append("Target 1 = BET surface area")
summary_lines.append("Target 2 = MB removal")
summary_lines.append("")

for target_label, metrics_df in [
    ("Target 1 - BET surface area", metrics_target1),
    ("Target 2 - MB removal", metrics_target2)
]:
    summary_lines.append(target_label)
    summary_lines.append("-" * len(target_label))

    for model_name in key_models:
        rmse = get_metric(metrics_df, model_name, "LOOCV_RMSE")
        mse = get_metric(metrics_df, model_name, "LOOCV_MSE")
        mae = get_metric(metrics_df, model_name, "LOOCV_MAE")
        r2 = get_metric(metrics_df, model_name, "LOOCV_R2")

        summary_lines.append(
            f"{model_name}: RMSE = {rmse:.4f}, MSE = {mse:.4f}, "
            f"MAE = {mae:.4f}, R2 = {r2:.4f}"
        )

    summary_lines.append("")

summary_text = "\n".join(summary_lines)

summary_txt_path = os.path.join(loocv_path, "COPY_THIS_LOOCV_summary_for_response_letter.txt")

with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print("\n\n============================================================")
print("LOOCV SUMMARY FOR RESPONSE LETTER")
print("============================================================")
print(summary_text)

print("\nAll files saved in:")
print(loocv_path)

print("\nImportant files to send/use:")
print("1. KEY_REVIEWER2_GPR_vs_Dummy_vs_Linear.csv")
print("2. loocv_metrics_all_models_both_targets.csv")
print("3. loocv_predictions_all_models_both_targets.csv")
print("4. parity_target1_BET_surface_area_Gaussian_Process_Regressor.png")
print("5. parity_target2_MB_removal_Gaussian_Process_Regressor.png")
print("6. baseline_RMSE_target1_BET_surface_area.png")
print("7. baseline_RMSE_target2_MB_removal.png")
print("8. COPY_THIS_LOOCV_summary_for_response_letter.txt")



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
save_path = 'C:/Users/abaei/Desktop/report/mehrdad-ML-dye-introduction/version 1/python code/11'
loocv_path = os.path.join(save_path, "LOOCV_review_response")

metrics_t1_path = os.path.join(loocv_path, "loocv_metrics_target1_BET_surface_area.csv")
metrics_t2_path = os.path.join(loocv_path, "loocv_metrics_target2_MB_removal.csv")
preds_t1_path = os.path.join(loocv_path, "loocv_predictions_target1_BET_surface_area.csv")
preds_t2_path = os.path.join(loocv_path, "loocv_predictions_target2_MB_removal.csv")

metrics_t1 = pd.read_csv(metrics_t1_path)
metrics_t2 = pd.read_csv(metrics_t2_path)
preds_t1 = pd.read_csv(preds_t1_path)
preds_t2 = pd.read_csv(preds_t2_path)

# =========================
# Global style
# =========================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# =========================
# Select data
# =========================
gpr_t1 = preds_t1[preds_t1["Model"] == "Gaussian Process Regressor"].copy()
gpr_t2 = preds_t2[preds_t2["Model"] == "Gaussian Process Regressor"].copy()

key_models = ["Dummy Regressor", "Linear Regression", "Gaussian Process Regressor"]

bar_t1 = metrics_t1[metrics_t1["Model"].isin(key_models)].copy()
bar_t2 = metrics_t2[metrics_t2["Model"].isin(key_models)].copy()

order = ["Dummy Regressor", "Linear Regression", "Gaussian Process Regressor"]
bar_t1["Model"] = pd.Categorical(bar_t1["Model"], categories=order, ordered=True)
bar_t2["Model"] = pd.Categorical(bar_t2["Model"], categories=order, ordered=True)

bar_t1 = bar_t1.sort_values("Model")
bar_t2 = bar_t2.sort_values("Model")

# =========================
# Create 2x2 subplot
# =========================
fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=600)

# ------------------------------------------------
# (A) Parity plot for Target 1: BET surface area
# ------------------------------------------------
ax = axes[0, 0]

x = gpr_t1["Measured"].values
y = gpr_t1["LOOCV_Predicted"].values

min_val = min(x.min(), y.min())
max_val = max(x.max(), y.max())
pad = 0.07 * (max_val - min_val)
lims = [min_val - pad, max_val + pad]

ax.scatter(x, y, s=120, edgecolor='black', linewidth=0.8)
ax.plot(lims, lims, '--', linewidth=1.5)

for _, row in gpr_t1.iterrows():
    ax.text(row["Measured"], row["LOOCV_Predicted"], str(row["Sample"]),
            fontsize=10, ha='left', va='bottom')

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Measured BET surface area (m² g$^{-1}$)")
ax.set_ylabel("LOOCV-predicted BET surface area (m² g$^{-1}$)")
ax.set_title("(A) Target 1: BET surface area parity plot")

# ------------------------------------------------
# (B) Parity plot for Target 2: MB removal
# ------------------------------------------------
ax = axes[0, 1]

x = gpr_t2["Measured"].values
y = gpr_t2["LOOCV_Predicted"].values

min_val = min(x.min(), y.min())
max_val = max(x.max(), y.max())
pad = 0.07 * (max_val - min_val)
lims = [min_val - pad, max_val + pad]

ax.scatter(x, y, s=120, edgecolor='black', linewidth=0.8)
ax.plot(lims, lims, '--', linewidth=1.5)

for _, row in gpr_t2.iterrows():
    ax.text(row["Measured"], row["LOOCV_Predicted"], str(row["Sample"]),
            fontsize=10, ha='left', va='bottom')

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Measured MB removal (%)")
ax.set_ylabel("LOOCV-predicted MB removal (%)")
ax.set_title("(B) Target 2: MB removal parity plot")

# ------------------------------------------------
# (C) Baseline comparison for Target 1
# ------------------------------------------------
ax = axes[1, 0]

bars = ax.bar(bar_t1["Model"].astype(str), bar_t1["LOOCV_RMSE"],
              edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, bar_t1["LOOCV_RMSE"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{val:.1f}", ha='center', va='bottom', fontsize=11)

ax.set_ylabel("LOOCV RMSE (m² g$^{-1}$)")
ax.set_title("(C) Target 1: baseline comparison")
ax.tick_params(axis='x', rotation=18)

# ------------------------------------------------
# (D) Baseline comparison for Target 2
# ------------------------------------------------
ax = axes[1, 1]

bars = ax.bar(bar_t2["Model"].astype(str), bar_t2["LOOCV_RMSE"],
              edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, bar_t2["LOOCV_RMSE"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{val:.2f}", ha='center', va='bottom', fontsize=11)

ax.set_ylabel("LOOCV RMSE (%)")
ax.set_title("(D) Target 2: baseline comparison")
ax.tick_params(axis='x', rotation=18)

plt.tight_layout()

fig_png = os.path.join(loocv_path, "Figure_R2_LOOCV_2x2.png")
fig_svg = os.path.join(loocv_path, "Figure_R2_LOOCV_2x2.svg")

plt.savefig(fig_png, dpi=600, bbox_inches="tight")
plt.savefig(fig_svg, bbox_inches="tight")
plt.show()

print("Saved:")
print(fig_png)
print(fig_svg)

