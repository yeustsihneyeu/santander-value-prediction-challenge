# 07 Optuna LGBM Tuning Report

## Goal

The goal of this notebook was to tune LightGBM with Optuna.

## What Was Done

- Loaded the processed data.
- Added row-wise features.
- Used `log1p(target)`.
- Ran Optuna for `60` trials.
- Optimized CV RMSLE.
- Trained the best model on the training split.
- Checked the model on the holdout split.
- Saved results to `artifacts/optuna_lgbm/`.

## Main Results

Saved artifact result:

| metric | value |
| --- | ---: |
| Best CV RMSLE | 1.4347 |
| Holdout RMSLE | 1.4644 |
| Holdout RMSE | 7,357,560.56 |
| Holdout MAE | 4,128,461.53 |
| Holdout R2 | 0.1518 |

Best Optuna parameters included:

- `n_estimators`: `439`
- `learning_rate`: `0.0193`
- `num_leaves`: `32`
- `max_depth`: `11`
- `colsample_bytree`: `0.7408`

## Conclusion

Optuna found a tuned LightGBM setup. The saved artifact result was better than the first baseline, but weaker than the later row-wise and sparsity experiments.
