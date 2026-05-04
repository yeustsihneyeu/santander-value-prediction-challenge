from __future__ import annotations

import numpy as np
import pandas as pd


ROWWISE_FEATURES = (
    "non_zero_count",
    "non_zero_ratio",
    "row_sum",
    "row_mean",
    "row_std",
    "row_max",
    "nz_mean",
    "nz_std",
)


def get_sparse_columns_to_keep(
    X_fit: pd.DataFrame,
    zero_share_threshold: float | None,
) -> list[str]:
    if zero_share_threshold is None:
        return X_fit.columns.tolist()

    zero_share = X_fit.eq(0).mean(axis=0)
    return zero_share[zero_share <= zero_share_threshold].index.tolist()


def add_rowwise_features(X: pd.DataFrame) -> pd.DataFrame:
    X_aug = X.copy()
    values = X.to_numpy(dtype=np.float32, copy=False)
    non_zero_mask = values != 0
    non_zero_count = non_zero_mask.sum(axis=1)
    total_features = values.shape[1]

    row_sum = values.sum(axis=1)
    row_mean = values.mean(axis=1)
    row_std = values.std(axis=1)
    row_max = values.max(axis=1)

    nz_sum = np.where(non_zero_mask, values, 0).sum(axis=1)
    nz_mean = np.divide(
        nz_sum,
        non_zero_count,
        out=np.zeros_like(nz_sum, dtype=np.float32),
        where=non_zero_count > 0,
    )

    centered_non_zero = np.where(non_zero_mask, values - nz_mean[:, None], 0)
    nz_var = np.divide(
        (centered_non_zero**2).sum(axis=1),
        non_zero_count,
        out=np.zeros_like(nz_sum, dtype=np.float32),
        where=non_zero_count > 0,
    )

    X_aug["non_zero_count"] = non_zero_count
    X_aug["non_zero_ratio"] = non_zero_count / total_features
    X_aug["row_sum"] = row_sum
    X_aug["row_mean"] = row_mean
    X_aug["row_std"] = row_std
    X_aug["row_max"] = row_max
    X_aug["nz_mean"] = nz_mean
    X_aug["nz_std"] = np.sqrt(nz_var)

    return X_aug
