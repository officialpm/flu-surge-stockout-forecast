import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

from src.config import load_config
from src.data_pipeline.build_panel import build_weekly_panel
from src.data_pipeline.cdc_ili import load_ili_data
from src.data_pipeline.m5_sales import load_m5_weekly_state_sales
from src.eval.faithfulness import faithfulness_precision_recall
from src.eval.metrics import compute_mape, paired_significance_test
from src.forecasting.features import add_cyclical_week_features, add_lag_features
from src.forecasting.models import DemandForecaster, build_augmented_features, build_baseline_features
from src.forecasting.shap_utils import compute_shap_mean_abs_importance, compute_shap_top_features
from src.forecasting.split import time_series_split
from src.genai_advisory.advisory_generator import extract_cited_drivers, generate_advisory
from src.genai_advisory.llm_client import AnthropicLLMClient
from src.risk_scoring.risk_model import compute_stockout_risk, estimate_current_stock_proxy


def run_pipeline(config: dict, llm_client) -> dict:
    ili_df = load_ili_data(config["data"]["ili_csv"])
    m5_df = load_m5_weekly_state_sales(
        config["data"]["m5_sales_csv"],
        config["data"]["m5_calendar_csv"],
        categories=tuple(config["data"]["categories"]),
    )
    panel = build_weekly_panel(ili_df, m5_df)

    lag_weeks = tuple(config["forecasting"]["lag_weeks"])
    panel = add_lag_features(panel, lag_weeks=lag_weeks)
    panel = add_cyclical_week_features(panel)
    # Note: rows with NaN lag features (the earliest period(s) per region,
    # before enough history exists) are intentionally kept rather than
    # dropped. LightGBM natively handles missing values, and dropping them
    # can starve the training split entirely when only a few periods of
    # history are available (as in small fixtures/early-stage datasets).

    train_df, test_df = time_series_split(panel, test_weeks=config["forecasting"]["test_weeks"])

    baseline_features = build_baseline_features(lag_weeks)
    augmented_features = build_augmented_features(lag_weeks)
    random_state = config["forecasting"].get("random_state", 42)

    baseline = DemandForecaster(feature_cols=baseline_features, random_state=random_state)
    baseline.fit(train_df)
    baseline_preds = baseline.predict(test_df)

    augmented = DemandForecaster(feature_cols=augmented_features, random_state=random_state)
    augmented.fit(train_df)
    augmented_preds = augmented.predict(test_df)

    baseline_mape = compute_mape(test_df["sales"].values, baseline_preds)
    augmented_mape = compute_mape(test_df["sales"].values, augmented_preds)

    baseline_errors = abs(test_df["sales"].values - baseline_preds)
    augmented_errors = abs(test_df["sales"].values - augmented_preds)
    significance = paired_significance_test(baseline_errors, augmented_errors)

    top_k = config["forecasting"]["top_k_shap"]
    shap_top = compute_shap_top_features(augmented, test_df[augmented_features], top_k=top_k)
    shap_importance = compute_shap_mean_abs_importance(augmented, test_df[augmented_features])

    predictions = []
    for i, (_, row) in enumerate(test_df.reset_index(drop=True).iterrows()):
        predictions.append(
            {
                "region": int(row["region"]),
                "year": int(row["year"]),
                "week": int(row["week"]),
                "actual_sales": float(row["sales"]),
                "baseline_pred": float(baseline_preds[i]),
                "augmented_pred": float(augmented_preds[i]),
            }
        )

    faithfulness_results = []
    for i, (_, row) in enumerate(test_df.reset_index(drop=True).iterrows()):
        shap_features_for_row = shap_top.iloc[i].tolist()
        stock_proxy = estimate_current_stock_proxy(
            train_df[train_df["region"] == row["region"]]["sales"],
            weeks_of_cover=config["risk"]["weeks_of_cover"],
        )
        risk_tier = compute_stockout_risk(
            forecast_demand=augmented_preds[i],
            current_stock_proxy=stock_proxy,
            reorder_lead_time_weeks=config["risk"]["reorder_lead_time_weeks"],
        )
        advisory_text = generate_advisory(
            llm_client,
            region=f"Region {row['region']}",
            risk_tier=risk_tier,
            forecast_demand=augmented_preds[i],
            top_shap_features=shap_features_for_row,
        )
        cited = extract_cited_drivers(advisory_text, augmented_features)
        scores = faithfulness_precision_recall(cited, shap_features_for_row)
        faithfulness_results.append(
            {
                "region": int(row["region"]),
                "forecast_demand": float(augmented_preds[i]),
                "risk_tier": risk_tier,
                "shap_top_features": shap_features_for_row,
                "cited_drivers": cited,
                "advisory_text": advisory_text,
                **scores,
            }
        )

    return {
        "baseline_mape": baseline_mape,
        "augmented_mape": augmented_mape,
        "significance": significance,
        "faithfulness": faithfulness_results,
        "predictions": predictions,
        "shap_importance": shap_importance,
    }


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_run_artifact(config: dict, result: dict, output_path: str) -> None:
    """Write a structured, reproducible run record: config + provenance + results.

    A paper's headline numbers must be traceable to the exact config and code
    version that produced them, not scraped from stdout.
    """
    artifact = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "config": config,
        "result": result,
    }
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run the flu-surge stockout pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write a structured JSON run artifact (config, git commit, "
        "and full results) for reproducible reporting. Defaults to "
        "results/run_<UTC-timestamp>.json.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    llm_client = AnthropicLLMClient(model=config["llm"]["model"])
    result = run_pipeline(config, llm_client)

    print(f"Baseline MAPE: {result['baseline_mape']:.2f}%")
    print(f"Health-augmented MAPE: {result['augmented_mape']:.2f}%")
    print(f"Significance: {result['significance']}")
    print(f"Faithfulness (per test row): {result['faithfulness']}")

    output_path = args.output
    if output_path is None:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = f"results/run_{timestamp}.json"
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    write_run_artifact(config, result, output_path)
    print(f"Run artifact written to {output_path}")


if __name__ == "__main__":
    main()
