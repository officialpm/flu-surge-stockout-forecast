import pandas as pd


def time_series_split(
    df: pd.DataFrame, test_weeks: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on the most recent `test_weeks` distinct (year, week) combos.

    The cutoff is global (same period held out across all regions) so
    accuracy comparisons across regions are on the same time window.
    """
    period_keys = (
        df[["year", "week"]]
        .drop_duplicates()
        .sort_values(["year", "week"])
        .reset_index(drop=True)
    )
    test_periods = period_keys.tail(test_weeks)

    is_test = df.set_index(["year", "week"]).index.isin(
        test_periods.set_index(["year", "week"]).index
    )

    test_df = df[is_test].reset_index(drop=True)
    train_df = df[~is_test].reset_index(drop=True)
    return train_df, test_df
