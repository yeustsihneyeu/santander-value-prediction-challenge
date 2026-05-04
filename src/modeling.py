from __future__ import annotations

from lightgbm import LGBMRegressor


def build_lgbm_regressor(
    params: dict,
    seed: int = 42,
    n_jobs: int = -1,
    verbosity: int = -1,
) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=verbosity,
        **params,
    )
