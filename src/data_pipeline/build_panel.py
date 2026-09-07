import pandas as pd

from src.data_pipeline.region_mapping import get_hhs_region


def build_weekly_panel(ili_df: pd.DataFrame, m5_df: pd.DataFrame) -> pd.DataFrame:
    """Join CDC ILI data with M5 sales, aggregated to (region, year, week).

    Inner join: both signals must be present, so baseline and
    health-augmented models are compared on the exact same rows.
    """
    m5_with_region = m5_df.copy()
    m5_with_region["region"] = m5_with_region["state_id"].apply(get_hhs_region)

    region_sales = (
        m5_with_region.groupby(["region", "year", "week"])["sales"]
        .sum()
        .reset_index()
    )

    panel = region_sales.merge(ili_df, on=["region", "year", "week"], how="inner")
    panel = panel.sort_values(["region", "year", "week"]).reset_index(drop=True)
    return panel[["region", "year", "week", "sales", "ili_rate"]]
