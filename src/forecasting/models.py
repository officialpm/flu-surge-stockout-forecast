import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


def build_baseline_features(lag_weeks: tuple[int, ...]) -> list[str]:
    return [f"sales_lag_{n}" for n in lag_weeks] + ["week_sin", "week_cos"]


def build_augmented_features(lag_weeks: tuple[int, ...]) -> list[str]:
    return build_baseline_features(lag_weeks) + ["ili_rate"] + [
        f"ili_rate_lag_{n}" for n in lag_weeks
    ]


BASELINE_FEATURES = build_baseline_features((1, 2, 3, 4))
AUGMENTED_FEATURES = build_augmented_features((1, 2, 3, 4))


class DemandForecaster:
    """LightGBM-based weekly demand forecaster.

    min_child_samples=1 is set because weekly-per-region panels are
    small (tens to low hundreds of rows), unlike LightGBM's typical
    large-dataset defaults.
    """

    def __init__(self, feature_cols: list[str], random_state: int = 42):
        self.feature_cols = feature_cols
        self.model = LGBMRegressor(
            min_child_samples=1, verbosity=-1, random_state=random_state
        )

    def fit(self, train_df: pd.DataFrame, target_col: str = "sales") -> None:
        X = train_df[self.feature_cols]
        y = train_df[target_col]
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols]
        return self.model.predict(X)
