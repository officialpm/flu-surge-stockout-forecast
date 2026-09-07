import pandas as pd


def load_ili_data(csv_path: str) -> pd.DataFrame:
    """Load a CDC FluView ILINet HHS-region export.

    Expects the standard ILINet CSV columns REGION TYPE, REGION, YEAR,
    WEEK, and %UNWEIGHTED ILI. Real exports include non-HHS rows (e.g.
    a "National" row) that we must drop before parsing, since REGION
    is non-numeric for those rows. REGION values for HHS rows look
    like "Region 9"; we extract the integer HHS region number.
    """
    raw = pd.read_csv(csv_path)
    raw = raw[raw["REGION TYPE"] == "HHS Regions"].reset_index(drop=True)
    region = raw["REGION"].str.extract(r"(\d+)").astype(int)[0]

    df = pd.DataFrame(
        {
            "region": region,
            "year": raw["YEAR"].astype(int),
            "week": raw["WEEK"].astype(int),
            "ili_rate": pd.to_numeric(raw["%UNWEIGHTED ILI"], errors="coerce"),
        }
    )
    return df.reset_index(drop=True)
