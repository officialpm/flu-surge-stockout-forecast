import json

from scripts.run_pipeline import run_pipeline, write_run_artifact
from src.forecasting.models import build_augmented_features
from src.genai_advisory.llm_client import FakeLLMClient


def _test_config():
    return {
        "data": {
            "ili_csv": "tests/fixtures/ili_sample.csv",
            "m5_sales_csv": "tests/fixtures/m5_sales_sample.csv",
            "m5_calendar_csv": "tests/fixtures/m5_calendar_sample.csv",
            "categories": ["HOUSEHOLD"],
        },
        "forecasting": {
            "lag_weeks": [1],
            "test_weeks": 1,
            "top_k_shap": 2,
            "random_state": 42,
        },
        "risk": {
            "weeks_of_cover": 1,
            "reorder_lead_time_weeks": 1,
        },
    }


def test_run_pipeline_end_to_end_with_fixtures():
    config = _test_config()
    fake_client = FakeLLMClient(
        canned_response="Flu (ILI) rate is rising sharply, restock now."
    )

    result = run_pipeline(config, fake_client)

    assert "baseline_mape" in result
    assert "augmented_mape" in result
    assert "significance" in result
    assert "faithfulness" in result
    assert isinstance(result["faithfulness"], list)
    assert len(result["faithfulness"]) == 3  # one row per region in test set
    for entry in result["faithfulness"]:
        assert "precision" in entry
        assert "recall" in entry
        assert "region" in entry
        assert "forecast_demand" in entry
        assert "risk_tier" in entry
        assert "shap_top_features" in entry
        assert "cited_drivers" in entry
        assert "advisory_text" in entry

    assert "predictions" in result
    assert len(result["predictions"]) == 3
    for entry in result["predictions"]:
        assert "region" in entry
        assert "year" in entry
        assert "week" in entry
        assert "actual_sales" in entry
        assert "baseline_pred" in entry
        assert "augmented_pred" in entry

    assert "shap_importance" in result
    expected_features = build_augmented_features(tuple(config["forecasting"]["lag_weeks"]))
    assert set(result["shap_importance"].keys()) == set(expected_features)
    assert all(isinstance(v, float) for v in result["shap_importance"].values())


def test_run_pipeline_same_config_and_seed_is_reproducible():
    # A paper's headline MAPE numbers must be exactly reproducible from the
    # same config, not merely close, across repeated runs.
    config = _test_config()
    fake_client = FakeLLMClient(canned_response="Restock now.")

    result_a = run_pipeline(config, fake_client)
    result_b = run_pipeline(config, fake_client)

    assert result_a["baseline_mape"] == result_b["baseline_mape"]
    assert result_a["augmented_mape"] == result_b["augmented_mape"]


def test_write_run_artifact_includes_config_provenance_and_result(tmp_path):
    config = _test_config()
    fake_client = FakeLLMClient(canned_response="Restock now.")
    result = run_pipeline(config, fake_client)

    output_path = tmp_path / "run.json"
    write_run_artifact(config, result, str(output_path))

    with open(output_path) as f:
        artifact = json.load(f)

    assert artifact["config"] == config
    assert artifact["result"]["baseline_mape"] == result["baseline_mape"]
    assert "git_commit" in artifact
    assert "generated_at_utc" in artifact
