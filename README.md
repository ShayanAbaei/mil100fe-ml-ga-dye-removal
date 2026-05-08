# MIL-100(Fe) ML-GA Dye Removal Reproducibility Kit

This repository contains the machine-readable dataset, GPR model-parameter files, cleaned Python scripts, and archived development scripts used for the MIL-100(Fe) low-data machine-learning analysis and the Reviewer 2 leave-one-out cross-validation (LOOCV) audit.

The repository is organized to make the modeling workflow transparent and easy to inspect. The clean scripts in `scripts/` are the recommended files for reproducing the reviewer-response LOOCV audit. The original exploratory scripts are retained separately in `main_archive_scripts/` for transparency.

## Repository structure

```text
data/
  data_dictionary.csv
  mil100fe_ml_dataset_full_six_targets.csv
  mil100fe_ml_dataset_primary_targets.csv

models/
  GPR_target1_BET_surface_area_best_params.pkl
  GPR_target2_MB_removal_best_params.pkl
  GPR_target3_total_pore_volume_best_params.pkl
  GPR_target4_average_crystallite_size_best_params.pkl
  GPR_target5_crystallinity_best_params.pkl
  GPR_target6_yield_best_params.pkl
  gpr_best_params_summary_all_targets.csv

scripts/
  01_make_dataset_from_tables.py
  02_loocv_audit_primary_targets.py
  03_plot_loocv_figure.py

main_archive_scripts/
  legacy_exploratory_full_workflow_original.py
  LOOCV_reviewer2_response_original_absolute_paths.py
```

## Dataset

The dataset contains 11 MIL-100(Fe) synthesis trials.

The four synthesis input variables are:

1. synthesis time (h),
2. temperature (°C),
3. metal-to-ligand molar ratio,
4. metal concentration (M).

The measured output variables are:

1. BET surface area,
2. crystallinity,
3. total pore volume,
4. average crystallite size,
5. yield,
6. methylene blue (MB) removal.

The file `data/mil100fe_ml_dataset_full_six_targets.csv` contains all six measured targets. The file `data/mil100fe_ml_dataset_primary_targets.csv` contains the two primary targets used in the Reviewer 2 LOOCV audit: BET surface area and MB removal.

## Model parameter files

The `models/` folder contains saved Gaussian Process Regressor (GPR) parameter files for all six targets. The file `models/gpr_best_params_summary_all_targets.csv` provides a human-readable summary of the GPR settings, including kernel, alpha, normalization setting, optimizer, optimizer restarts, and random state.

## Reproducing the LOOCV audit

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the LOOCV audit for the two primary targets:

```bash
python scripts/02_loocv_audit_primary_targets.py
```

This script treats each sample once as the held-out test point and compares the GPR model with a training-mean Dummy Regressor and a Linear Regression baseline. Running the script will create a local `results/` folder containing the LOOCV metric and prediction tables.

Optional: recreate the LOOCV parity/baseline-comparison figure:

```bash
python scripts/03_plot_loocv_figure.py
```

Running this script will create a local `figures/` folder containing the generated figure files.

## Archived scripts

The `main_archive_scripts/` folder contains the original exploratory workflow and the original reviewer-response LOOCV script with absolute local paths. These files are preserved for transparency, but the cleaned scripts in `scripts/` should be used for reproduction.

## Citation

If you use this dataset or code, please cite the associated manuscript and this repository.
