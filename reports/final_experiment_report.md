# Final Experiment Report

## Project Goal

The goal was to build a regression model for the Santander value prediction task and optimize the competition-aligned metric `RMSLE`.

The dataset is challenging because it has only `4,459` rows, thousands of anonymous numeric features, many sparse columns, and a strongly right-skewed target. Because `RMSLE` penalizes relative error, the modeling notebooks mainly use `log1p(target)` and evaluate RMSE in log space.

## Notebook Results

| notebook | what was done | why it was done | result |
| --- | --- | --- | --- |
| `01_eda.ipynb` | Explored target distribution, feature sparsity, duplicate/constant columns, feature-target relations, and high-correlation features. | To understand the data before modeling and decide which preprocessing/modeling choices are reasonable. | Found a very skewed target, many sparse anonymous features, duplicate columns, at least one constant feature, and weak simple feature-target correlations. This justified `log1p(target)`, duplicate removal, sparse filtering, and tree models. |
| `02_base_preprocessing.ipynb` | Removed `ID` and duplicate feature columns, then saved the processed modeling table. | `ID` is not predictive, and duplicate columns add noise and unnecessary dimensionality. | Produced `data/processed_data.csv` with `4,459` rows and `4,732` columns, including `target`; this means `4,731` model features. |
| `03_baseline.ipynb` | Compared first baseline regressors using `log1p(target)` and selected LightGBM for held-out test evaluation. | To establish a practical baseline and confirm that log-target modeling matches the RMSLE objective. | Selected `LGBMRegressor` baseline: CV RMSLE `1.4719`, train RMSLE `0.7724`, test RMSLE `1.4821`. |
| `04_remove_near_constnat.ipynb` | Removed features with zero share greater than `0.99` and trained default LightGBM. | To test whether very sparse near-constant features hurt generalization. | Reduced features from `4,731` to `2,669`. CV RMSLE improved slightly to `1.4693`, but test RMSLE was `1.4826`, so filtering alone did not beat the baseline on test. |
| `05_sparsity_thresholds_lgbm.ipynb` | Compared sparse zero-share thresholds and selected the best threshold by CV. | To choose a less arbitrary sparse-column cutoff before feature engineering. | Best CV threshold was `0.9875`, keeping `2,482` features. CV RMSLE improved to `1.4500`; test RMSLE was `1.4635`. |
| `06_add_features_lgbm.ipynb` | Added row-wise aggregate features after sparse filtering: non-zero counts/ratios, row sums, means, stds, maxima, and non-zero summaries. | The dataset is sparse, so row-level structure can carry signal that individual anonymous columns do not expose directly. | Final feature count became `2,490`. CV RMSLE improved strongly to `1.3757`, train RMSLE `0.7593`, test RMSLE `1.3851`. This was the largest practical improvement. |
| `07_feature_importance_top_lgbm.ipynb` | Ranked the `2,490` sparse-filtered row-wise features by LightGBM gain importance and tested top `100`, `250`, and `500` subsets. | To check whether a smaller high-importance subset can improve generalization and reduce feature noise. | Best CV subset was `top_k=250`: CV RMSLE `1.3313`, train RMSLE `0.7396`, test RMSLE `1.4079`. CV improved, but test worsened versus notebook `06`, suggesting over-selection to the validation folds. |
| `08_lgbm_tuning_param_per_time.ipynb` | Tuned LightGBM one parameter at a time on the sparse-filtered row-wise feature setup. | To improve the model in a controlled and interpretable way before running Optuna. | Final manual parameters reached CV RMSLE `1.3571`, train RMSLE `1.0155`, test RMSLE `1.3776`. This improved test RMSLE versus notebook `06`. |
| `09_final_optuna_lgbm.ipynb` | Ran a 50-trial Optuna search on the selected sparse-filtered row-wise setup with early stopping. | To search parameter interactions more broadly than manual one-parameter tuning. | Best current held-out RMSLE: CV RMSLE `1.3576`, train RMSLE `1.1052`, test RMSLE `1.3767`. The gain over notebook `08` is small but it is the best test RMSLE in the current run. |
| `10_pca_lgbm.ipynb` | Replaced the original processed feature set with PCA components only and trained LightGBM. | To test whether dense linear compression can reduce noise and dimensionality. | Best PCA size was `50` components. CV RMSLE `1.5471`, train RMSLE `1.0739`, test RMSLE `1.5283`; PCA-only is worse than the baseline and should not be used. |
| `11_truncated_svd_lgbm.ipynb` | Applied sparse filtering, then replaced the filtered features with TruncatedSVD components only and trained LightGBM. | TruncatedSVD works without centering and is better suited than PCA for sparse matrices, so it was tested as a sparse-friendly compression method. | Best SVD size was `50` components. CV RMSLE `1.5258`, train RMSLE `1.0628`, test RMSLE `1.5066`; SVD-only beats PCA-only but is still worse than original sparse-filtered row-wise features. |

## Experiment Comparison

| notebook | feature setup | features before -> after | model | CV RMSLE | train RMSLE | test RMSLE | result |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `03` | processed base features | 4,731 -> 4,731 | `LGBMRegressor` | 1.4719 | 0.7724 | 1.4821 | baseline |
| `04` | zero share `>0.99` removed | 4,731 -> 2,669 | `LGBMRegressor` | 1.4693 | 0.7717 | 1.4826 | tiny CV gain, no test gain |
| `05` | threshold search, best `0.9875` | 4,731 -> 2,482 | `LGBMRegressor` | 1.4500 | 0.8764 | 1.4635 | useful sparse filtering |
| `06` | sparse filtering plus row-wise features | 4,731 -> 2,490 | `LGBMRegressor` | 1.3757 | 0.7593 | 1.3851 | biggest improvement |
| `07` | top gain-importance subset | 2,490 -> 250 | `LGBMRegressor` | 1.3313 | 0.7396 | 1.4079 | CV improved, test worsened |
| `08` | manual tuned sparse row-wise setup | 4,731 -> 2,490 | `LGBMRegressor` | 1.3571 | 1.0155 | 1.3776 | strong tuned model |
| `09` | Optuna tuned sparse row-wise setup | 4,731 -> 2,490 | `LGBMRegressor` | 1.3576 | 1.1052 | 1.3767 | best test RMSLE |
| `10` | PCA-only components | 4,731 -> 50 | `PCA` + `LGBMRegressor` | 1.5471 | 1.0739 | 1.5283 | worse than baseline |
| `11` | sparse filter plus SVD-only components | 4,731 -> 50 | `TruncatedSVD` + `LGBMRegressor` | 1.5258 | 1.0628 | 1.5066 | better than PCA-only, still weak |

## What Helped

The most useful data-side change was adding row-wise sparse aggregate features after sparse filtering. The test RMSLE improved from `1.4635` in notebook `05` to `1.3851` in notebook `06`.

Sparse filtering also helped. The best threshold was `0.9875`, which kept `2,482` base features and improved CV RMSLE compared with using all processed base features.

LightGBM tuning helped after the feature setup was fixed. Manual tuning and Optuna both improved held-out RMSLE, with notebook `09` giving the best current test RMSLE of `1.3767`.

## What Did Not Help

Near-constant filtering with a fixed `0.99` threshold gave only a tiny CV improvement and did not improve test RMSLE.

Top-k feature-importance selection looked strong in CV but hurt held-out test RMSLE. It likely removed useful lower-ranked features or overfit to the validation folds.

PCA-only and SVD-only dimensionality reduction lost too much target-relevant sparse signal. SVD was better than PCA, but both were worse than the sparse-filtered row-wise LightGBM setup.

## Best Current Model

The best current model is from `09_final_optuna_lgbm.ipynb`.

Setup:

- target: `log1p(target)`;
- model: `LGBMRegressor`;
- sparse threshold: `0.9875`;
- base features kept: `2,482`;
- row-wise features added: `8`;
- final features: `2,490`;
- Optuna trials: `50`;
- final `n_estimators`: `499` from early-stopping fold best iterations.

Main metrics:

- CV RMSLE: `1.3576`;
- train RMSLE: `1.1052`;
- test RMSLE: `1.3767`;
- test RMSE: `7,072,245.79`;
- test MAE: `3,963,023.64`;
- test R2: `0.2163`.

## Final Conclusion

The project improved from the LightGBM log-target baseline test RMSLE `1.4821` to the best current test RMSLE `1.3767`.

The strongest recipe is:

1. remove duplicate columns and `ID`;
2. model `log1p(target)`;
3. remove very sparse columns with threshold `0.9875`;
4. add row-wise sparse aggregate features;
5. tune LightGBM on that fixed feature space.

The next step is to turn the best notebook workflow into a reproducible training/inference pipeline and use it for final submission generation.
