# 05 LGBM Tuning One Parameter At A Time Report

## Goal

The goal of this notebook was to tune LightGBM in a controlled way.

## What Was Done

- Loaded the processed data.
- Used `log1p(target)`.
- Tested one LightGBM parameter at a time.
- Compared candidates by CV RMSLE.
- Built a model with the best selected parameters.
- Saved tuning results to `artifacts/baseline/`.

## Main Results

Best selected parameters:

- `learning_rate`: `0.03`
- `n_estimators`: `100`
- `num_leaves`: `31`
- `min_child_samples`: `20`
- `subsample`: `1.0`
- `colsample_bytree`: `0.8`
- `reg_alpha`: `0.05`
- `reg_lambda`: `0.05`

Final holdout result:

| metric | value |
| --- | ---: |
| Holdout RMSLE | 1.4643 |
| Holdout RMSE | 7,513,960.89 |
| Holdout MAE | 4,226,236.71 |
| Holdout R2 | 0.1154 |

## Conclusion

Manual tuning improved RMSLE compared with the simple baseline. But the raw-space metrics and R2 became weaker, so the result should be checked against later experiments.
