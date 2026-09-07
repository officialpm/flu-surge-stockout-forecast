import numpy as np
import pandas as pd

from src.forecasting.models import DemandForecaster
from src.forecasting.shap_utils import compute_shap_mean_abs_importance, compute_shap_top_features


def test_compute_shap_top_features_identifies_dominant_feature():
    rng = np.random.default_rng(7)
    n = 60
    feature_a = rng.uniform(0, 10, n)
    # constant features contribute ~0 SHAP value regardless of tree quirks
    feature_b = np.full(n, 5.0)
    feature_c = np.full(n, 1.0)
    target = feature_a * 100  # feature_a dominates the target completely

    df = pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_c": feature_c,
            "sales": target,
        }
    )

    forecaster = DemandForecaster(feature_cols=["feature_a", "feature_b", "feature_c"])
    forecaster.fit(df)

    top_features = compute_shap_top_features(forecaster, df[forecaster.feature_cols], top_k=1)

    assert list(top_features.columns) == ["top_feature_1"]
    assert (top_features["top_feature_1"] == "feature_a").all()


from src.forecasting.shap_utils import _as_2d_shap_matrix


def test_compute_shap_mean_abs_importance_ranks_dominant_feature_highest():
    rng = np.random.default_rng(7)
    n = 60
    feature_a = rng.uniform(0, 10, n)
    feature_b = np.full(n, 5.0)
    target = feature_a * 100

    df = pd.DataFrame({"feature_a": feature_a, "feature_b": feature_b, "sales": target})

    forecaster = DemandForecaster(feature_cols=["feature_a", "feature_b"])
    forecaster.fit(df)

    importance = compute_shap_mean_abs_importance(forecaster, df[forecaster.feature_cols])

    assert set(importance.keys()) == {"feature_a", "feature_b"}
    assert importance["feature_a"] > importance["feature_b"]


def test_as_2d_shap_matrix_passthrough_for_plain_2d_array():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = _as_2d_shap_matrix(arr, n_rows=2, n_features=2)
    np.testing.assert_array_equal(result, arr)


def test_as_2d_shap_matrix_unwraps_list_wrapped_array():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = _as_2d_shap_matrix([arr], n_rows=2, n_features=2)
    np.testing.assert_array_equal(result, arr)


def test_as_2d_shap_matrix_squeezes_trailing_output_dim():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    arr_3d = arr[:, :, None]  # shape (2, 2, 1)
    result = _as_2d_shap_matrix(arr_3d, n_rows=2, n_features=2)
    np.testing.assert_array_equal(result, arr)
