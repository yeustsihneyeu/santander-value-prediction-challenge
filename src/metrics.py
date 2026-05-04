from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmsle": float(root_mean_squared_log_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def metric_table(metrics: Mapping[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": list(metrics.keys()),
            "value": list(metrics.values()),
        }
    )


def format_metric_value(value) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.4f}"
