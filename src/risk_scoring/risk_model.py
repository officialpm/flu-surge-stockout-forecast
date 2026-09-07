import pandas as pd


def estimate_current_stock_proxy(historical_sales: pd.Series, weeks_of_cover: int = 4) -> float:
    """Simulated current-stock estimate: recent avg weekly sales x target weeks of cover.

    This is a documented simulation assumption (spec section 7) since no
    real pharmacy inventory data is used — it approximates "how much
    stock a retailer nominally holds to cover N weeks of typical demand."
    """
    recent_avg = historical_sales.tail(weeks_of_cover).mean()
    return float(recent_avg * weeks_of_cover)


def compute_stockout_risk(
    forecast_demand: float, current_stock_proxy: float, reorder_lead_time_weeks: int
) -> str:
    """Tiered stockout risk based on projected need over the reorder lead time."""
    if current_stock_proxy <= 0:
        # No stock buffer at all -- treat as the worst case rather than
        # dividing by zero.
        return "high"

    projected_need = forecast_demand * reorder_lead_time_weeks
    ratio = projected_need / current_stock_proxy

    if ratio < 0.7:
        return "low"
    if ratio < 1.0:
        return "medium"
    return "high"
