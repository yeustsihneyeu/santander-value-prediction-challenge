# Santander Value Prediction

Notebook-first ML project for the Santander Value Prediction regression task. The goal is to predict the target transaction value from a high-dimensional, sparse, mostly numerical feature set.

The current best local experiment is an Optuna-tuned `LightGBM` model trained on `log1p(target)` with sparse feature filtering and row-wise aggregate features.

## Current Best Result

From `09_final_optuna_lgbm.ipynb`:

| metric | value |
| --- | ---: |
| CV RMSLE mean | 1.3576 |
| Train RMSLE | 1.1052 |
| Test RMSLE | 1.3767 |
| Test RMSE | 7,072,245.79 |
| Test MAE | 3,963,023.64 |
| Test R2 | 0.2163 |

Model setup:

- Target transform: `log1p(target)`
- Model: `LGBMRegressor`
- Optimizer: `Optuna`
- CV: 5-fold `KFold`
- Test size: `0.33`
- Sparse threshold: `0.9875`
- Base features kept: `2,482`
- Row-wise features added: `8`
- Final feature count: `2,490`
- Optuna trials: `50`
- Final `n_estimators`: `499`

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv
│   └── processed_data.csv        # local processed dataset
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 01_eda_report.md
│   ├── 02_base_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_remove_near_constnat.ipynb
│   ├── 05_sparsity_thresholds_lgbm.ipynb
│   ├── 06_add_features_lgbm.ipynb
│   ├── 07_feature_importance_top_lgbm.ipynb
│   ├── 08_lgbm_tuning_param_per_time.ipynb
│   ├── 09_final_optuna_lgbm.ipynb
│   ├── 10_pca_lgbm.ipynb
│   └── 11_truncated_svd_lgbm.ipynb
├── src/
│   ├── features.py
│   ├── loader.py
│   ├── metrics.py
│   ├── modeling.py
│   └── preprocessing.py
├── reports/
│   ├── experiments_comparison_report.md
│   └── final_experiment_report.md
```

## Notebook Flow

Run notebooks in this order when rebuilding the experiment from scratch:

1. `01_eda.ipynb` - exploratory analysis and dataset diagnostics.
2. `02_base_preprocessing.ipynb` - remove `ID` and duplicate feature columns.
3. `03_baseline.ipynb` - baseline model comparison on `log1p(target)`.
4. `04_remove_near_constnat.ipynb` - test near-constant sparse feature removal.
5. `05_sparsity_thresholds_lgbm.ipynb` - compare sparse feature thresholds.
6. `06_add_features_lgbm.ipynb` - add row-wise aggregate features.
7. `07_feature_importance_top_lgbm.ipynb` - compare top-k feature sets selected by LightGBM importance.
8. `08_lgbm_tuning_param_per_time.ipynb` - controlled LightGBM parameter sweeps.
9. `09_final_optuna_lgbm.ipynb` - final Optuna run using sparse filtering plus row-wise features.
10. `10_pca_lgbm.ipynb` - test whether PCA components can replace the original processed feature set.
11. `11_truncated_svd_lgbm.ipynb` - test whether sparse-friendly TruncatedSVD components can replace the filtered feature set.

Most notebook reports are embedded as final markdown cells. The EDA report is also kept as `notebooks/01_eda_report.md`.

Final reports:

- `reports/final_experiment_report.md` - short narrative summary of every notebook and final conclusion.
- `reports/experiments_comparison_report.md` - detailed comparison table with features, model setup, hyperparameters, and RMSLE metrics.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name santander
```

## Data

`data/train.csv` is the raw training dataset. `data/processed_data.csv` is the processed local dataset used by the modeling notebooks.

Notebook results are displayed in memory as tables, metrics, and final `summary` dictionaries.

## Current Findings

- The dataset is high-dimensional: the number of features exceeds the number of samples.
- The target is strongly right-skewed, and `log1p(target)` improves modeling stability.
- Many features are sparse and dominated by zeros.
- Removing duplicate features and filtering highly sparse features improves the LightGBM setup.
- Row-wise aggregate features add useful signal.
- Feature-importance top-k selection improved CV RMSLE but worsened held-out test RMSLE.
- PCA-only and TruncatedSVD-only compression lose too much sparse target-relevant signal.
- Final model selection should rely primarily on CV RMSLE; held-out test RMSLE is a sanity check.

