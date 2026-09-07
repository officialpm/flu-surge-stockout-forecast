import numpy as np
from scipy.stats import wilcoxon


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def paired_significance_test(
    baseline_errors: np.ndarray, augmented_errors: np.ndarray
) -> dict:
    """Wilcoxon signed-rank test comparing per-sample absolute errors.

    Non-parametric, appropriate since per-region/per-week error
    distributions are not assumed normal and sample sizes are small.
    """
    statistic, p_value = wilcoxon(baseline_errors, augmented_errors)
    return {"statistic": float(statistic), "p_value": float(p_value)}
