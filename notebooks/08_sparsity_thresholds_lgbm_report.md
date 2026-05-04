# 08 Sparsity Thresholds LGBM Report

## Goal

The goal of this notebook was to find a good sparse feature threshold.

## What Was Done

- Loaded the processed data.
- Used row-wise features.
- Tested several zero-share thresholds:
  - `none`
  - `0.975`
  - `0.98`
  - `0.9825`
  - `0.985`
  - `0.9875`
  - `0.99`
- For each threshold, selected columns inside the CV fold.
- Trained LightGBM and compared CV RMSLE.
- Saved results to `artifacts/sparsity_thresholds_rowwise_lgbm/`.

## Main Results

Best result by CV RMSLE:

| metric | value |
| --- | ---: |
| Best threshold | 0.9875 |
| Base features kept | 2,482 |
| Final features | 2,490 |
| CV RMSLE mean | 1.3644 |
| CV RMSLE std | 0.0302 |
| Holdout RMSLE | 1.3908 |
| Holdout RMSE | 6,942,104.74 |
| Holdout MAE | 3,895,660.22 |
| Holdout R2 | 0.2449 |

## Conclusion

Sparse feature filtering improved the model. The best CV threshold was `0.9875`. This threshold was used in the final Optuna notebook.
