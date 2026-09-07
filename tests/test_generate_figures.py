import os

from scripts.generate_figures import generate_all_figures


def _fake_artifact():
    return {
        "config": {},
        "git_commit": "abc123",
        "result": {
            "baseline_mape": 6.5,
            "augmented_mape": 5.9,
            "significance": {"statistic": 12.0, "p_value": 0.03},
            "predictions": [
                {
                    "region": 5,
                    "year": 2016,
                    "week": w,
                    "actual_sales": 100.0 + w,
                    "baseline_pred": 98.0 + w,
                    "augmented_pred": 99.0 + w,
                }
                for w in range(1, 4)
            ]
            + [
                {
                    "region": 6,
                    "year": 2016,
                    "week": w,
                    "actual_sales": 200.0 + w,
                    "baseline_pred": 195.0 + w,
                    "augmented_pred": 198.0 + w,
                }
                for w in range(1, 4)
            ],
            "shap_importance": {
                "sales_lag_1": 12.5,
                "sales_lag_2": 8.0,
                "week_sin": 2.0,
                "ili_rate": 5.5,
                "ili_rate_lag_1": 3.0,
            },
            "faithfulness": [
                {
                    "region": 5,
                    "risk_tier": "low",
                    "precision": 1.0,
                    "recall": 1.0,
                },
                {
                    "region": 5,
                    "risk_tier": "medium",
                    "precision": 0.5,
                    "recall": 0.5,
                },
                {
                    "region": 6,
                    "risk_tier": "high",
                    "precision": 0.0,
                    "recall": 0.0,
                },
            ],
        },
    }


def test_generate_all_figures_writes_png_and_pdf_for_every_figure(tmp_path):
    artifact = _fake_artifact()
    output_dir = str(tmp_path / "figures")

    written = generate_all_figures(artifact, output_dir)

    expected_names = {
        "mape_comparison",
        "actual_vs_predicted",
        "shap_importance",
        "faithfulness_by_region",
        "risk_tier_distribution",
    }
    written_names = {os.path.splitext(os.path.basename(p))[0] for p in written}
    assert written_names == expected_names

    for path in written:
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        assert path.endswith(".png") or path.endswith(".pdf")
