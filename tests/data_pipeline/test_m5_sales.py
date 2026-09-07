from src.data_pipeline.m5_sales import load_m5_weekly_state_sales


def test_load_m5_weekly_state_sales_aggregates_by_week():
    df = load_m5_weekly_state_sales(
        "tests/fixtures/m5_sales_sample.csv",
        "tests/fixtures/m5_calendar_sample.csv",
        categories=("HOUSEHOLD",),
    )

    assert list(df.columns) == ["state_id", "year", "week", "sales"]
    assert len(df) == 6  # 3 states x 2 weeks

    ca_week2 = df[(df["state_id"] == "CA") & (df["week"] == 2)].iloc[0]
    ca_week3 = df[(df["state_id"] == "CA") & (df["week"] == 3)].iloc[0]
    assert ca_week2["year"] == 2015
    assert ca_week2["sales"] == 70
    assert ca_week3["sales"] == 140

    tx_week2 = df[(df["state_id"] == "TX") & (df["week"] == 2)].iloc[0]
    assert tx_week2["sales"] == 35

    wi_week3 = df[(df["state_id"] == "WI") & (df["week"] == 3)].iloc[0]
    assert wi_week3["sales"] == 56


def test_load_m5_weekly_state_sales_excludes_other_categories():
    df = load_m5_weekly_state_sales(
        "tests/fixtures/m5_sales_sample.csv",
        "tests/fixtures/m5_calendar_sample.csv",
        categories=("HOUSEHOLD",),
    )
    # FOODS row (sales=99/day) must not leak into CA's HOUSEHOLD total
    ca_week2 = df[(df["state_id"] == "CA") & (df["week"] == 2)].iloc[0]
    assert ca_week2["sales"] == 70
