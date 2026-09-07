import pandas as pd


def load_m5_weekly_state_sales(
    sales_path: str,
    calendar_path: str,
    categories: tuple[str, ...] = ("HOUSEHOLD",),
) -> pd.DataFrame:
    """Load M5 sales, filter to categories, aggregate to weekly per-state totals.

    M5's sales_train_validation.csv is wide (one column per day, d_1..d_n).
    We melt it to long form, attach real dates via calendar.csv, derive
    ISO year/week, and sum sales per (state_id, year, week).
    """
    sales = pd.read_csv(sales_path)
    sales = sales[sales["cat_id"].isin(categories)]

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long_df = sales.melt(
        id_vars=["state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    calendar = pd.read_csv(calendar_path, usecols=["date", "d"])
    long_df = long_df.merge(calendar, on="d", how="left")
    long_df["date"] = pd.to_datetime(long_df["date"])

    iso = long_df["date"].dt.isocalendar()
    long_df["year"] = iso["year"].astype(int)
    long_df["week"] = iso["week"].astype(int)

    weekly = (
        long_df.groupby(["state_id", "year", "week"])["sales"]
        .sum()
        .reset_index()
    )
    return weekly
