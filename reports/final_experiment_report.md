# Final Experiment Report

## Project Goal

The goal of this project was to build a regression model for the Santander value prediction task.

The data is difficult because:

- there are only `4,459` rows;
- there are many features;
- many features are sparse and contain many zeros;
- the target is very right-skewed;
- feature names are anonymous.

The main metric was `RMSLE`. Because of this, most good experiments used `log1p(target)`.

## What Was Done

The work was done step by step:

1. Exploratory data analysis.
2. Base preprocessing.
3. First baseline models.
4. Near-constant sparse feature filtering.
5. Manual LightGBM tuning.
6. Row-wise feature engineering.
7. Optuna tuning.
8. Sparse threshold search.
9. Final Optuna LightGBM model.

## Experiment Comparison

| notebook | experiment | features | target | CV RMSLE | holdout RMSLE | holdout RMSE | holdout MAE | holdout R2 | result |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `03` | Raw-target LGBM baseline | 4,731 | raw | 3.4122 | 3.2909 | 6,988,302 | 4,798,946 | 0.2348 | weak RMSLE |
| `03` | Log-target LGBM baseline | 4,731 | `log1p` | 1.4719 | 1.4821 | 7,326,666 | 4,216,348 | 0.1589 | good baseline |
| `04` | Log-target + near-constant removal | 2,627 | `log1p` | 1.4655 | 1.4833 | 7,295,799 | 4,185,374 | 0.1660 | small CV gain, no holdout gain |
| `05` | Manual one-parameter tuning | 4,731 | `log1p` | 1.4493 | 1.4643 | 7,513,961 | 4,226,237 | 0.1154 | better RMSLE, worse raw metrics |
| `06` | Row-wise features + tuned LGBM | 4,739 | `log1p` | 1.3819 | 1.3931 | 6,977,905 | 3,919,146 | 0.2371 | strong improvement |
| `07` | Optuna LGBM tuning | 4,739 | `log1p` | 1.4347 | 1.4644 | 7,357,561 | 4,128,462 | 0.1518 | not better than row-wise setup |
| `08` | Best sparse threshold search | 2,490 | `log1p` | 1.3644 | 1.3908 | 6,942,105 | 3,895,660 | 0.2449 | improved CV and raw metrics |
| `09` | Final sparse + row-wise Optuna LGBM | 2,490 | `log1p` | 1.3569 | 1.3786 | 6,948,548 | 3,904,504 | 0.2435 | best current RMSLE |

## What Helped

### `log1p(target)`

This was the most important early decision.

The raw target model had very bad RMSLE. After using `log1p(target)`, RMSLE became much better and the optimization target matched the competition metric better.

### Row-Wise Features

Row-wise features helped a lot.

Useful added features were:

- number of non-zero values in a row;
- share of non-zero values;
- row sum;
- row mean;
- row standard deviation;
- row max;
- mean of non-zero values;
- standard deviation of non-zero values.

These features worked well because the dataset is very sparse. They describe the row structure, not only single columns.

### Sparse Feature Filtering

Sparse feature filtering also helped.

The best threshold was `0.9875`. It kept `2,482` base features and gave `2,490` final features after adding row-wise features.

This reduced noise and improved CV RMSLE.

### Final Optuna Tuning

Final Optuna tuning on the filtered feature space gave the best result:

- CV RMSLE: `1.3569`
- holdout RMSLE: `1.3786`

This is the best local model in the project so far.

## What Did Not Help Much

### Raw Target Modeling

Raw target modeling did not work well for RMSLE.

The target is too skewed, so the model had problems with the competition metric.

### Near-Constant Filtering Alone

Removing features with more than `99%` zeros gave a small CV improvement, but it did not improve holdout RMSLE.

It was useful as an idea, but it was not enough alone.

### Manual One-Parameter Tuning

Manual tuning improved RMSLE, but raw-space metrics became weaker.

It was useful for understanding parameter effects, but it was not the final best setup.

### First Optuna Run

The first Optuna run did not beat the row-wise feature experiment.

It was still useful because its parameters were reused in later experiments.

## Best Current Model

The best current model is from `09_final_optuna_lgbm.ipynb`.

Setup:

- model: `LGBMRegressor`;
- target: `log1p(target)`;
- sparse threshold: `0.9875`;
- base features kept: `2,482`;
- final features: `2,490`;
- row-wise features: yes;
- Optuna trials: `20`.

Best parameters included:

- `n_estimators`: `497`;
- `learning_rate`: `0.0082`;
- `num_leaves`: `81`;
- `max_depth`: `11`;
- `colsample_bytree`: `0.7662`;
- `reg_alpha`: `1.5556`.

## Final Conclusion

The project improved from a simple log-target baseline with holdout RMSLE `1.4821` to a final model with holdout RMSLE `1.3786`.

The biggest improvements came from:

1. using `log1p(target)`;
2. adding row-wise sparse features;
3. filtering sparse columns;
4. tuning LightGBM on the final feature space.

The next best step is to make the final training pipeline reproducible outside notebooks and then build a submission or inference script.
