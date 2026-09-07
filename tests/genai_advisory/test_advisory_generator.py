from src.genai_advisory.advisory_generator import (
    build_advisory_prompt,
    extract_cited_drivers,
    generate_advisory,
)
from src.genai_advisory.llm_client import FakeLLMClient


def test_build_advisory_prompt_includes_key_facts():
    prompt = build_advisory_prompt(
        region="Region 9",
        risk_tier="high",
        forecast_demand=150.0,
        top_shap_features=["ili_rate", "sales_lag_1"],
    )
    assert "Region 9" in prompt
    assert "high" in prompt
    assert "150" in prompt


def test_generate_advisory_calls_llm_and_returns_text():
    fake_client = FakeLLMClient(canned_response="Restock now.")
    result = generate_advisory(
        fake_client,
        region="Region 9",
        risk_tier="high",
        forecast_demand=150.0,
        top_shap_features=["ili_rate"],
    )
    assert result == "Restock now."
    assert "Region 9" in fake_client.last_prompt


def test_extract_cited_drivers_parses_structured_line():
    advisory_text = (
        "Region 9 shows high stockout risk driven by a sharp rise in the "
        "regional flu rate over recent weeks.\n"
        "CITED_DRIVERS: ili_rate"
    )
    candidates = ["ili_rate", "week_sin", "sales_lag_1"]

    cited = extract_cited_drivers(advisory_text, candidates)

    assert cited == ["ili_rate"]


def test_extract_cited_drivers_returns_empty_when_declared_none():
    advisory_text = "Stock levels look fine, no action needed.\nCITED_DRIVERS: none"
    candidates = ["ili_rate", "sales_lag_1"]

    cited = extract_cited_drivers(advisory_text, candidates)

    assert cited == []


def test_extract_cited_drivers_returns_empty_when_no_structured_line():
    advisory_text = "Stock levels look fine, no action needed."
    candidates = ["ili_rate", "sales_lag_1"]

    cited = extract_cited_drivers(advisory_text, candidates)

    assert cited == []


def test_extract_cited_drivers_ignores_hallucinated_tokens():
    advisory_text = "Demand is stable.\nCITED_DRIVERS: ili_rate, made_up_feature"
    candidates = ["ili_rate", "sales_lag_1"]

    cited = extract_cited_drivers(advisory_text, candidates)

    assert cited == ["ili_rate"]
