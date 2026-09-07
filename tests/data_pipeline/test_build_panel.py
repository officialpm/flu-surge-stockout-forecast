import pandas as pd

from src.data_pipeline.build_panel import build_weekly_panel


def test_build_weekly_panel_joins_on_region_year_week():
    ili_df = pd.DataFrame(
        {
            "region": [9, 9, 6, 6, 5, 5],
            "year": [2015] * 6,
            "week": [2, 3, 2, 3, 2, 3],
            "ili_rate": [1.5, 2.0, 1.0, 1.0, 1.2, 1.3],
        }
    )
    m5_df = pd.DataFrame(
        {
            "state_id": ["CA", "CA", "TX", "TX", "WI", "WI"],
            "year": [2015] * 6,
            "week": [2, 3, 2, 3, 2, 3],
            "sales": [70, 140, 35, 35, 56, 56],
        }
    )

    panel = build_weekly_panel(ili_df, m5_df)

    assert list(panel.columns) == ["region", "year", "week", "sales", "ili_rate"]
    assert len(panel) == 6

    r9_w3 = panel[(panel["region"] == 9) & (panel["week"] == 3)].iloc[0]
    assert r9_w3["sales"] == 140
    assert r9_w3["ili_rate"] == 2.0

    # sorted by region, year, week
    assert panel["region"].tolist() == sorted(panel["region"].tolist())


def test_build_weekly_panel_sums_multiple_states_in_same_region():
    # NY and NJ are both HHS region 2
    ili_df = pd.DataFrame(
        {"region": [2], "year": [2015], "week": [2], "ili_rate": [1.0]}
    )
    m5_df = pd.DataFrame(
        {
            "state_id": ["NY", "NJ"],
            "year": [2015, 2015],
            "week": [2, 2],
            "sales": [10, 20],
        }
    )

    panel = build_weekly_panel(ili_df, m5_df)

    assert len(panel) == 1
    assert panel.iloc[0]["sales"] == 30
