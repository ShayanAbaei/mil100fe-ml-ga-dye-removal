"""Create Figure S1b: LOOCV parity plots and baseline RMSE comparison."""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

metrics = pd.read_csv(RESULTS / "loocv_metrics_primary_targets.csv")
preds = pd.read_csv(RESULTS / "loocv_predictions_primary_targets.csv")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 13
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=300)

# Parity data
for ax, target, xlabel, ylabel, title in [
    (axes[0, 0], "Target 1 - BET surface area", "Measured BET surface area (m² g$^{-1}$)", "LOOCV-predicted BET surface area (m² g$^{-1}$)", "(A) Target 1: BET surface area parity plot"),
    (axes[0, 1], "Target 2 - MB removal", "Measured MB removal (%)", "LOOCV-predicted MB removal (%)", "(B) Target 2: MB removal parity plot"),
]:
    d = preds[(preds["Target"] == target) & (preds["Model"] == "Gaussian Process Regressor")]
    x, y = d["Measured"], d["LOOCV_Predicted"]
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.07 * (hi - lo) if hi > lo else 1
    lims = [lo - pad, hi + pad]
    ax.scatter(x, y, s=110, edgecolor="black", linewidth=0.8)
    ax.plot(lims, lims, "--", linewidth=1.4)
    for _, row in d.iterrows():
        ax.text(row["Measured"], row["LOOCV_Predicted"], row["Sample"], fontsize=10, ha="left", va="bottom")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)

# Bar plots
for ax, target, ylabel, title in [
    (axes[1, 0], "Target 1 - BET surface area", "LOOCV RMSE (m² g$^{-1}$)", "(C) Target 1: baseline comparison"),
    (axes[1, 1], "Target 2 - MB removal", "LOOCV RMSE (%)", "(D) Target 2: baseline comparison"),
]:
    order = ["Dummy Regressor", "Linear Regression", "Gaussian Process Regressor"]
    d = metrics[(metrics["Target"] == target) & (metrics["Model"].isin(order))].copy()
    d["Model"] = pd.Categorical(d["Model"], categories=order, ordered=True)
    d = d.sort_values("Model")
    bars = ax.bar(d["Model"].astype(str), d["LOOCV_RMSE"], edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, d["LOOCV_RMSE"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.tick_params(axis="x", rotation=18)

plt.tight_layout()
fig.savefig(FIGURES / "Figure_S1b_LOOCV_audit.png", dpi=600, bbox_inches="tight")
fig.savefig(FIGURES / "Figure_S1b_LOOCV_audit.svg", bbox_inches="tight")
print("Saved Figure S1b in", FIGURES)
