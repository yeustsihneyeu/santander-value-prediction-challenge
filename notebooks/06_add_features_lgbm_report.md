# 06 Add Row-Wise Features Report

## Goal

The goal of this notebook was to test simple row-wise features.

## What Was Done

- Loaded the processed data.
- Added row-wise features:
  - `non_zero_count`
  - `non_zero_ratio`
  - `row_sum`
  - `row_mean`
  - `row_std`
  - `row_max`
  - `nz_mean`
  - `nz_std`
- Used the tuned LightGBM parameters from the Optuna artifact.
- Trained the model with `log1p(target)`.
- Saved results to `artifacts/rowwise_features_lgbm/`.

## Main Results

- Final feature count: `4,739`.

| metric | value |
| --- | ---: |
| CV RMSLE mean | 1.3819 |
| CV RMSLE std | 0.0405 |
| Holdout RMSLE | 1.3931 |
| Holdout RMSE | 6,977,905.17 |
| Holdout MAE | 3,919,146.04 |
| Holdout R2 | 0.2371 |

## Conclusion

The row-wise features helped a lot. The model became much better than the earlier baseline on RMSLE and also improved raw-space metrics.
