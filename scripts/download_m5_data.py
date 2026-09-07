"""Download M5 Forecasting Accuracy competition data from Kaggle.

Requires a Kaggle API token (https://www.kaggle.com/settings/api) exported
as KAGGLE_API_TOKEN, or KAGGLE_USERNAME + KAGGLE_KEY for the legacy
kaggle.json format. See https://www.kaggle.com/docs/api for details.

Only sales_train_validation.csv and calendar.csv are needed by
src.data_pipeline.m5_sales.load_m5_weekly_state_sales -- the competition
also ships sales_train_evaluation.csv, sell_prices.csv, and a submission
template, which this project does not use.
"""

import argparse
import os
import zipfile

COMPETITION = "m5-forecasting-accuracy"
NEEDED_FILES = ("sales_train_validation.csv", "calendar.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Download M5 sales_train_validation.csv and calendar.csv from Kaggle"
    )
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    if not (os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")):
        raise SystemExit(
            "No Kaggle credentials found. Export KAGGLE_API_TOKEN (new-format "
            "token from https://www.kaggle.com/settings/api), or "
            "KAGGLE_USERNAME + KAGGLE_KEY for the legacy kaggle.json format."
        )

    # Imported after the credential check: the kaggle package authenticates
    # at import time and raises immediately if no credentials are configured.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in NEEDED_FILES:
        print(f"Downloading {filename} from {COMPETITION}...")
        api.competition_download_file(COMPETITION, filename, path=args.output_dir)

        zip_path = os.path.join(args.output_dir, filename + ".zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(args.output_dir)
            os.remove(zip_path)

        final_path = os.path.join(args.output_dir, filename)
        print(f"  -> {final_path}")


if __name__ == "__main__":
    main()
