from dataclasses import dataclass

import pandas as pd


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
