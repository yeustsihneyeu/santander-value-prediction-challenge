# 04 Remove Near-Constant Features Report

## Goal

The goal of this notebook was to test if removing very sparse features improves the model.

## What Was Done

- Loaded the processed data.
- Split the data into train and holdout parts.
- Removed features with more than `99%` zero values.
- Fitted this filter only on the training split.
- Trained a LightGBM model with `log1p(target)`.
- Checked cross-validation and holdout metrics.

## Main Results

- Removed near-constant sparse features: `2,104`.
- Feature count changed from `4,731` to `2,627`.

| metric | value |
| --- | ---: |
| CV RMSLE mean | 1.465 |
| CV RMSLE std | 0.030 |
| Holdout RMSLE | 1.483 |
| Holdout RMSE | 7,295,799.33 |
| Holdout MAE | 4,185,374.14 |
| Holdout R2 | 0.166 |

## Conclusion

Removing very sparse features gave a small CV improvement, but the holdout RMSLE was slightly worse than the simple baseline. This experiment was useful, but it was not clearly better than notebook `03`.
