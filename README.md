# Santander Value Prediction

This repository contains exploratory data analysis for the Santander value prediction task. The current project state is focused on understanding the training data, identifying data quality issues, and preparing hypotheses for the next modeling stage.

## Project Goal

Build a regression pipeline that predicts the target value from a high-dimensional, sparse, mostly numerical feature set.

At the moment, the repository includes:

- the training dataset
- an EDA notebook
- a written EDA report
- an empty `src/` directory reserved for reusable project code

## Repository Structure

```text
.
├── README.md
├── data/
│   └── train.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 01_eda_report.md
└── src/
```

## What Is Already Done

- Loaded and inspected the training dataset
- Analyzed target distribution and tested `log1p(target)`
- Checked missing values, duplicate rows, duplicate columns, and constant features
- Investigated feature sparsity
- Explored feature-target relationships with Pearson, Spearman, and mutual information
- Reviewed feature-feature correlation patterns

## Current Key Findings

- The dataset is high-dimensional: the number of features exceeds the number of samples
- The target is strongly right-skewed
- Many features are sparse and dominated by zeros
- Some duplicate and constant features can be removed during preprocessing
- Mutual information suggests that at least part of the signal may be non-linear
# santander-value-prediction-challenge
