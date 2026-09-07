import math

import pandas as pd


def add_lag_features(df: pd.DataFrame, lag_weeks: tuple[int, ...]) -> pd.DataFrame:
    """Add sales_lag_N and ili_rate_lag_N columns, shifted within each region.

    Assumes df is one row per (region, year, week); sorts by (region, year,
    week) before shifting so lags never leak across regions.
    """
    out = df.sort_values(["region", "year", "week"]).reset_index(drop=True)
    for n in lag_weeks:
        out[f"sales_lag_{n}"] = out.groupby("region")["sales"].shift(n)
        out[f"ili_rate_lag_{n}"] = out.groupby("region")["ili_rate"].shift(n)
    return out


def add_cyclical_week_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add week_sin/week_cos cyclical encodings of ISO week number."""
    out = df.copy()
    out["week_sin"] = out["week"].apply(lambda w: math.sin(2 * math.pi * w / 52))
    out["week_cos"] = out["week"].apply(lambda w: math.cos(2 * math.pi * w / 52))
    return out
