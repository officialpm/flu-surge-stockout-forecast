import math

import pandas as pd

from src.forecasting.features import add_cyclical_week_features, add_lag_features


def test_add_lag_features_shifts_within_region():
    df = pd.DataFrame(
        {
            "region": [9, 9, 6, 6],
            "year": [2015, 2015, 2015, 2015],
            "week": [2, 3, 2, 3],
            "sales": [70, 140, 35, 35],
            "ili_rate": [1.5, 2.0, 1.0, 1.0],
        }
    )

    out = add_lag_features(df, lag_weeks=(1,))

    r9_w2 = out[(out["region"] == 9) & (out["week"] == 2)].iloc[0]
    r9_w3 = out[(out["region"] == 9) & (out["week"] == 3)].iloc[0]

    assert pd.isna(r9_w2["sales_lag_1"])
    assert r9_w3["sales_lag_1"] == 70
    assert r9_w3["ili_rate_lag_1"] == 1.5


def test_add_lag_features_does_not_leak_across_regions():
    df = pd.DataFrame(
        {
            "region": [9, 6],
            "year": [2015, 2015],
            "week": [3, 3],
            "sales": [140, 35],
            "ili_rate": [2.0, 1.0],
        }
    )
    out = add_lag_features(df, lag_weeks=(1,))
    # both are the only row for their region -> lag must be NaN, not borrowed
    # from the other region's row
    assert out["sales_lag_1"].isna().all()


def test_add_cyclical_week_features():
    df = pd.DataFrame({"week": [13]})
    out = add_cyclical_week_features(df)

    expected_sin = math.sin(2 * math.pi * 13 / 52)
    expected_cos = math.cos(2 * math.pi * 13 / 52)

    assert math.isclose(out.iloc[0]["week_sin"], expected_sin, rel_tol=1e-9)
    assert math.isclose(out.iloc[0]["week_cos"], expected_cos, rel_tol=1e-9)
