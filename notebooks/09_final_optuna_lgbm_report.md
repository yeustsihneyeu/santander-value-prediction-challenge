# 09 Final Optuna LGBM Report

## Goal

The goal of this notebook was to run the final LightGBM tuning.

## What Was Done

- Loaded the processed data.
- Used the best sparse threshold from notebook `08`.
- Used row-wise features.
- Used `log1p(target)`.
- Ran Optuna for `20` trials.
- Selected the best parameters by CV RMSLE.
- Trained the final model.
- Checked holdout metrics.
- Saved final artifacts to `artifacts/final_optuna_lgbm/`.

## Main Results

| metric | value |
| --- | ---: |
| Sparse threshold | 0.9875 |
| Base features kept | 2,482 |
| Final features | 2,490 |
| Best CV RMSLE | 1.3569 |
| Holdout RMSLE | 1.3786 |
| Holdout RMSE | 6,948,547.59 |
| Holdout MAE | 3,904,504.43 |
| Holdout R2 | 0.2435 |

Best final parameters included:

- `n_estimators`: `497`
- `learning_rate`: `0.0082`
- `num_leaves`: `81`
- `max_depth`: `11`
- `colsample_bytree`: `0.7662`
- `reg_alpha`: `1.5556`

## Conclusion

This is the best current local result. It has the best CV RMSLE and the best holdout RMSLE in the project so far. It should be the main model for the next step.
