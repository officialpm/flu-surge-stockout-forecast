import pandas as pd

from src.risk_scoring.risk_model import compute_stockout_risk, estimate_current_stock_proxy


def test_estimate_current_stock_proxy_uses_recent_average():
    historical_sales = pd.Series([10, 20, 30, 40, 100])  # last 4: 20,30,40,100 -> mean 47.5
    proxy = estimate_current_stock_proxy(historical_sales, weeks_of_cover=4)
    assert proxy == 47.5 * 4


def test_compute_stockout_risk_low():
    # projected_need = 10 * 2 = 20; ratio = 20/100 = 0.2 -> low
    assert compute_stockout_risk(forecast_demand=10, current_stock_proxy=100, reorder_lead_time_weeks=2) == "low"


def test_compute_stockout_risk_medium():
    # projected_need = 40 * 2 = 80; ratio = 80/100 = 0.8 -> medium
    assert compute_stockout_risk(forecast_demand=40, current_stock_proxy=100, reorder_lead_time_weeks=2) == "medium"


def test_compute_stockout_risk_high():
    # projected_need = 60 * 2 = 120; ratio = 120/100 = 1.2 -> high
    assert compute_stockout_risk(forecast_demand=60, current_stock_proxy=100, reorder_lead_time_weeks=2) == "high"


def test_compute_stockout_risk_zero_stock_proxy_is_high():
    assert (
        compute_stockout_risk(forecast_demand=10, current_stock_proxy=0, reorder_lead_time_weeks=1)
        == "high"
    )


def test_compute_stockout_risk_negative_stock_proxy_is_high():
    assert (
        compute_stockout_risk(forecast_demand=10, current_stock_proxy=-5, reorder_lead_time_weeks=1)
        == "high"
    )
