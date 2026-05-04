# 01 EDA Report

## Goal

The goal of this notebook was to understand the training data before modeling.

## What Was Done

- Loaded `data/train.csv`.
- Checked the dataset size and column types.
- Studied the target variable.
- Checked missing values.
- Checked duplicate rows and duplicate columns.
- Checked constant features.
- Studied sparsity: many values are zero.
- Checked simple feature-target relations with correlation and mutual information.
- Checked feature-feature correlation.

## Main Results

- The dataset has `4,459` rows and `4,993` columns before preprocessing.
- The target is numeric and positive.
- The target is very right-skewed.
- `log1p(target)` makes the target distribution more stable.
- There are no missing values.
- There are no duplicate rows.
- There are duplicate feature columns.
- At least one feature is constant.
- After removing constant features in the EDA check, the feature matrix had `4,730` columns.
- Many features are sparse and contain many zeros.

## Conclusion

The data needs preprocessing before modeling. The target should be modeled with `log1p(target)`. Duplicate features, constant features, and sparse features should be checked in later experiments.
