import numpy as np

from src.eval.metrics import compute_mape, compute_rmse, paired_significance_test


def test_compute_mape():
    y_true = np.array([100.0, 200.0, 50.0])
    y_pred = np.array([110.0, 180.0, 55.0])
    mape = compute_mape(y_true, y_pred)
    # errors: 10/100=0.10, 20/200=0.10, 5/50=0.10 -> mean 0.10 -> 10.0%
    assert abs(mape - 10.0) < 1e-6


def test_compute_rmse():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    rmse = compute_rmse(y_true, y_pred)
    # sqrt((9+16)/2) = sqrt(12.5)
    assert abs(rmse - (12.5**0.5)) < 1e-6


def test_paired_significance_test_returns_statistic_and_pvalue():
    baseline_errors = np.array([10, 12, 11, 13, 9, 14, 15, 10])
    augmented_errors = np.array([5, 6, 4, 7, 3, 8, 6, 5])

    result = paired_significance_test(baseline_errors, augmented_errors)

    assert "statistic" in result
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0
