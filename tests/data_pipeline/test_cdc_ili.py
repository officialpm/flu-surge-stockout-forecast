from src.data_pipeline.cdc_ili import load_ili_data


def test_load_ili_data_parses_region_and_columns():
    df = load_ili_data("tests/fixtures/ili_sample.csv")

    assert list(df.columns) == ["region", "year", "week", "ili_rate"]
    assert len(df) == 6

    row = df[(df["region"] == 9) & (df["week"] == 3)].iloc[0]
    assert row["year"] == 2015
    assert row["ili_rate"] == 2.0


def test_load_ili_data_all_regions_present():
    df = load_ili_data("tests/fixtures/ili_sample.csv")
    assert set(df["region"].unique()) == {9, 6, 5}


def test_load_ili_data_ignores_non_hhs_region_rows():
    df = load_ili_data("tests/fixtures/ili_sample.csv")
    # Fixture includes a "National" REGION TYPE row with a non-numeric
    # REGION value. Loading must not crash, and the row must be excluded.
    assert len(df) == 6
    assert set(df["region"].unique()) == {9, 6, 5}
