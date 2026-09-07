import numpy as np
import pandas as pd

from src.forecasting.models import (
    AUGMENTED_FEATURES,
    BASELINE_FEATURES,
    DemandForecaster,
)


def _synthetic_training_frame(n=40):
    rng = np.random.default_rng(42)
    sales_lag_1 = rng.uniform(10, 100, n)
    ili_rate = rng.uniform(0.5, 3.0, n)
    week_sin = rng.uniform(-1, 1, n)
    week_cos = rng.uniform(-1, 1, n)
    target = sales_lag_1 * 1.2 + rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "sales_lag_1": sales_lag_1,
            "sales_lag_2": sales_lag_1 * 0.9,
            "sales_lag_3": sales_lag_1 * 0.8,
            "sales_lag_4": sales_lag_1 * 0.7,
            "ili_rate": ili_rate,
            "ili_rate_lag_1": ili_rate * 0.9,
            "ili_rate_lag_2": ili_rate * 0.8,
            "ili_rate_lag_3": ili_rate * 0.7,
            "ili_rate_lag_4": ili_rate * 0.6,
            "week_sin": week_sin,
            "week_cos": week_cos,
            "sales": target,
        }
    )


def test_baseline_features_exclude_ili():
    assert all("ili" not in f for f in BASELINE_FEATURES)


def test_augmented_features_include_ili_and_baseline():
    assert set(BASELINE_FEATURES).issubset(set(AUGMENTED_FEATURES))
    assert "ili_rate" in AUGMENTED_FEATURES


def test_demand_forecaster_fit_predict_returns_correct_length():
    train_df = _synthetic_training_frame()
    forecaster = DemandForecaster(feature_cols=AUGMENTED_FEATURES)
    forecaster.fit(train_df)

    preds = forecaster.predict(train_df)

    assert len(preds) == len(train_df)
    assert forecaster.model is not None


def test_demand_forecaster_same_seed_is_reproducible():
    # A paper's reported numbers must be exactly reproducible from the same
    # config on repeated runs, not merely close.
    train_df = _synthetic_training_frame()

    forecaster_a = DemandForecaster(feature_cols=AUGMENTED_FEATURES, random_state=42)
    forecaster_a.fit(train_df)
    preds_a = forecaster_a.predict(train_df)

    forecaster_b = DemandForecaster(feature_cols=AUGMENTED_FEATURES, random_state=42)
    forecaster_b.fit(train_df)
    preds_b = forecaster_b.predict(train_df)

    np.testing.assert_array_equal(preds_a, preds_b)
