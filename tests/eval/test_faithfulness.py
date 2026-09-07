from src.eval.faithfulness import faithfulness_precision_recall


def test_faithfulness_perfect_match():
    result = faithfulness_precision_recall(
        cited_drivers=["ili_rate", "sales_lag_1"],
        shap_top_features=["ili_rate", "sales_lag_1"],
    )
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_faithfulness_partial_match():
    # cited 2, 1 correct (in shap top) -> precision 0.5
    # shap top has 3, 1 covered -> recall 1/3
    result = faithfulness_precision_recall(
        cited_drivers=["ili_rate", "week_sin"],
        shap_top_features=["ili_rate", "sales_lag_1", "sales_lag_2"],
    )
    assert result["precision"] == 0.5
    assert abs(result["recall"] - (1 / 3)) < 1e-9


def test_faithfulness_no_cited_drivers_returns_zero_precision():
    result = faithfulness_precision_recall(
        cited_drivers=[],
        shap_top_features=["ili_rate"],
    )
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_faithfulness_week_sin_cos_not_double_counted():
    result = faithfulness_precision_recall(
        cited_drivers=["week_sin", "week_cos", "ili_rate"],
        shap_top_features=["ili_rate", "week_sin"],
    )
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
