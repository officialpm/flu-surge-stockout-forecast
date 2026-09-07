import numpy as np
import pandas as pd
import shap

from src.forecasting.models import DemandForecaster


def _as_2d_shap_matrix(shap_values, n_rows: int, n_features: int) -> np.ndarray:
    """Normalize shap.TreeExplainer output to a (n_rows, n_features) array.

    Depending on the shap/LightGBM version, `explainer.shap_values(X)` for a
    single-output regressor may return:
      - a 2D array of shape (n_rows, n_features) (typical case), or
      - a list containing one such 2D array (multi-output-style API), or
      - a 3D array of shape (n_rows, n_features, n_outputs) with n_outputs == 1.
    This helper collapses any of those into the plain 2D case.
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    arr = np.asarray(shap_values)

    if arr.ndim == 3:
        # Shape is either (n_rows, n_features, n_outputs) or
        # (n_outputs, n_rows, n_features); pick the axis that isn't
        # n_rows/n_features and squeeze it out.
        if arr.shape[0] == n_rows and arr.shape[1] == n_features:
            arr = arr[:, :, 0]
        elif arr.shape[1] == n_rows and arr.shape[2] == n_features:
            arr = arr[0, :, :]
        else:
            arr = arr.reshape(n_rows, n_features)

    return arr


def _compute_shap_matrix(forecaster: DemandForecaster, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(forecaster.model)
    raw_shap_values = explainer.shap_values(X)
    return _as_2d_shap_matrix(raw_shap_values, n_rows=len(X), n_features=len(X.columns))


def compute_shap_top_features(
    forecaster: DemandForecaster, X: pd.DataFrame, top_k: int = 3
) -> pd.DataFrame:
    """Rank features by |SHAP value| per row, return the top_k names per row."""
    shap_matrix = _compute_shap_matrix(forecaster, X)
    feature_names = X.columns.tolist()

    records = []
    for row in shap_matrix:
        order = np.argsort(-np.abs(row))[:top_k]
        records.append([feature_names[i] for i in order])

    cols = [f"top_feature_{i + 1}" for i in range(top_k)]
    return pd.DataFrame(records, columns=cols)


def compute_shap_mean_abs_importance(forecaster: DemandForecaster, X: pd.DataFrame) -> dict[str, float]:
    """Mean |SHAP value| per feature across all rows in X.

    Unlike compute_shap_top_features (which keeps only per-row rankings and
    discards magnitude), this preserves the magnitude needed for a global
    feature-importance figure.
    """
    shap_matrix = _compute_shap_matrix(forecaster, X)
    feature_names = X.columns.tolist()
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    return {name: float(value) for name, value in zip(feature_names, mean_abs)}
