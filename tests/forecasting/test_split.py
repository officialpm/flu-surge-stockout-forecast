import pandas as pd

from src.forecasting.split import time_series_split


def test_time_series_split_holds_out_most_recent_weeks():
    df = pd.DataFrame(
        {
            "region": [9, 9, 9, 6, 6, 6],
            "year": [2015] * 6,
            "week": [1, 2, 3, 1, 2, 3],
            "sales": [10, 20, 30, 5, 6, 7],
        }
    )

    train, test = time_series_split(df, test_weeks=1)

    assert set(test["week"].unique()) == {3}
    assert set(train["week"].unique()) == {1, 2}
    assert len(test) == 2  # one row per region for week 3
    assert len(train) == 4


def test_time_series_split_multiple_test_weeks():
    df = pd.DataFrame(
        {
            "region": [9, 9, 9, 9],
            "year": [2015] * 4,
            "week": [1, 2, 3, 4],
            "sales": [10, 20, 30, 40],
        }
    )

    train, test = time_series_split(df, test_weeks=2)

    assert set(test["week"].unique()) == {3, 4}
    assert set(train["week"].unique()) == {1, 2}
