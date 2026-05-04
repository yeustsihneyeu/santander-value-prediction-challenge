# 03 Baseline Report

## Goal

The goal of this notebook was to build a first modeling baseline.

## What Was Done

- Loaded `data/processed_data.csv`.
- Split the data into train and holdout parts.
- Tested baseline models with cross-validation.
- Used `log1p(target)` for the main baseline.
- Trained the best baseline model.
- Checked the result on the holdout split.
- Saved baseline artifacts to `artifacts/baseline/`.

## Main Results

The best baseline model was `LightGBM` with `log1p(target)`.

| metric | value |
| --- | ---: |
| CV RMSLE mean | 1.4719 |
| CV RMSLE std | 0.0351 |
| Holdout RMSLE | 1.4821 |
| Holdout RMSE | 7,326,665.62 |
| Holdout MAE | 4,216,347.88 |
| Holdout R2 | 0.1589 |

## Conclusion

The best simple baseline is LightGBM trained on `log1p(target)`. This result is the first reference for later experiments.
