# 02 Base Preprocessing Report

## Goal

The goal of this notebook was to create a clean base dataset for modeling.

## What Was Done

- Loaded `data/train.csv`.
- Removed the `ID` column.
- Checked duplicate rows.
- Removed duplicate feature columns.
- Tested the same logic with the `Processor` class from `src/preprocessing.py`.
- Saved the result to `data/processed_data.csv`.

## Main Results

- Original data shape: `4,459` rows and `4,993` columns.
- Duplicate rows found: `0`.
- After removing `ID` and duplicate feature columns, the data shape was `4,459` rows and `4,732` columns.
- The processed dataset was saved as `data/processed_data.csv`.

## Conclusion

The notebook created the base processed dataset. This file is used by the next modeling notebooks.
