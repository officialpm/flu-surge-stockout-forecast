"""Download CDC ILINet HHS-region flu surveillance data.

Uses the public Delphi Epidata API (https://api.delphi.cmu.edu/epidata/fluview/),
a free, no-login mirror of CDC FluView ILINet, and reshapes its response into
the same columns `src.data_pipeline.cdc_ili.load_ili_data` expects from a
manually exported FluView CSV: REGION TYPE, REGION, YEAR, WEEK, %UNWEIGHTED ILI.

Only HHS regions 5, 6, and 9 are fetched by default -- the M5 sales dataset
covers Walmart stores in WI, TX, and CA respectively (see
src/data_pipeline/region_mapping.py), so surveillance data outside those
regions is never joined against sales and would be dead weight.
"""

import argparse
import sys

import requests

DELPHI_FLUVIEW_URL = "https://api.delphi.cmu.edu/epidata/fluview/"
DEFAULT_HHS_REGIONS = (5, 6, 9)
DEFAULT_EPIWEEK_RANGE = "201001-202352"


def fetch_ili_rows(hhs_regions: tuple[int, ...], epiweek_range: str) -> list[dict]:
    region_params = ",".join(f"hhs{r}" for r in hhs_regions)
    response = requests.get(
        DELPHI_FLUVIEW_URL,
        params={"regions": region_params, "epiweeks": epiweek_range},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("result") != 1:
        raise RuntimeError(f"Delphi Epidata API returned no data: {payload.get('message')}")

    return payload["epidata"]


def rows_to_fluview_csv_rows(rows: list[dict]) -> list[dict]:
    """Reshape Delphi epidata rows into ILINet CSV export column names."""
    out = []
    for row in rows:
        epiweek = row["epiweek"]
        year = epiweek // 100
        week = epiweek % 100
        region_num = row["region"].replace("hhs", "")
        out.append(
            {
                "REGION TYPE": "HHS Regions",
                "REGION": f"Region {region_num}",
                "YEAR": year,
                "WEEK": week,
                "% WEIGHTED ILI": row["wili"],
                "%UNWEIGHTED ILI": row["ili"],
                "ILITOTAL": row["num_ili"],
                "NUM. OF PROVIDERS": row["num_providers"],
                "TOTAL PATIENTS": row["num_patients"],
            }
        )
    return out


def write_csv(rows: list[dict], output_path: str) -> None:
    import csv

    if not rows:
        raise RuntimeError("No rows to write -- refusing to create an empty CSV")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Download CDC ILINet HHS-region data via the public Delphi Epidata API"
    )
    parser.add_argument("--output", default="data/raw/ili_ilinet.csv")
    parser.add_argument(
        "--hhs-regions",
        default=",".join(str(r) for r in DEFAULT_HHS_REGIONS),
        help="Comma-separated HHS region numbers (default: 5,6,9 -- the regions "
        "covered by the M5 sales dataset's WI/TX/CA stores)",
    )
    parser.add_argument("--epiweek-range", default=DEFAULT_EPIWEEK_RANGE)
    args = parser.parse_args()

    hhs_regions = tuple(int(r) for r in args.hhs_regions.split(","))

    print(f"Fetching ILINet data for HHS regions {hhs_regions}, weeks {args.epiweek_range}...")
    raw_rows = fetch_ili_rows(hhs_regions, args.epiweek_range)
    csv_rows = rows_to_fluview_csv_rows(raw_rows)
    write_csv(csv_rows, args.output)
    print(f"Wrote {len(csv_rows)} rows to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as e:
        print(f"Network error contacting Delphi Epidata API: {e}", file=sys.stderr)
        sys.exit(1)
