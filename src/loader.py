from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Loader:


    def load(self, path: Path | str) -> pd.DataFrame:
        return pd.read_csv(path)


    def to_csv(self, df: pd.DataFrame, path: Path | str) -> None:
        df.to_csv(path, index=False)
