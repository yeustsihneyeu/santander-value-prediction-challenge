from dataclasses import dataclass

import pandas as pd

from src.features import add_rowwise_features


@dataclass
class Processor:
    df: pd.DataFrame

    def _remove_id(self, df):
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
        return df

    def _remove_duplicate_features(self, df):
        duplicated = df.columns[df.T.duplicated()]
        return df.drop(columns=duplicated)

    def base_preprocess(self) -> pd.DataFrame:
        df = self.df.copy()
        df = self._remove_id(df)
        df = self._remove_duplicate_features(df)
        return df


@dataclass
class FeaturePreprocessor:
    remove_id: bool = False
    remove_duplicates: bool = False
    remove_constant: bool = False
    zero_share_threshold: float | None = None
    add_rowwise: bool = False

    def __post_init__(self) -> None:
        self.columns_to_keep_: list[str] | None = None
        self.removed_duplicate_columns_: list[str] = []
        self.removed_constant_columns_: list[str] = []
        self.removed_sparse_columns_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        X_fit = X.copy()

        if self.remove_id and "ID" in X_fit.columns:
            X_fit = X_fit.drop(columns=["ID"])

        if self.remove_duplicates:
            duplicated = X_fit.columns[X_fit.T.duplicated()]
            self.removed_duplicate_columns_ = duplicated.tolist()
            X_fit = X_fit.drop(columns=self.removed_duplicate_columns_)

        if self.remove_constant:
            constant = X_fit.columns[X_fit.nunique(dropna=False) <= 1]
            self.removed_constant_columns_ = constant.tolist()
            X_fit = X_fit.drop(columns=self.removed_constant_columns_)

        if self.zero_share_threshold is not None:
            zero_share = X_fit.eq(0).mean(axis=0)
            sparse = zero_share[zero_share > self.zero_share_threshold].index
            self.removed_sparse_columns_ = sparse.tolist()
            X_fit = X_fit.drop(columns=self.removed_sparse_columns_)

        self.columns_to_keep_ = X_fit.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_to_keep_ is None:
            raise ValueError("FeaturePreprocessor must be fitted before transform.")

        missing_columns = [col for col in self.columns_to_keep_ if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Missing columns during transform: {missing_columns[:5]}")

        X_transformed = X.loc[:, self.columns_to_keep_].copy()

        if self.add_rowwise:
            X_transformed = add_rowwise_features(X_transformed)

        return X_transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
