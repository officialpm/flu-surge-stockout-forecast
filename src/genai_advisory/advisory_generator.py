from src.genai_advisory.llm_client import LLMClient

CITED_DRIVERS_PREFIX = "CITED_DRIVERS:"


def build_advisory_prompt(
    region: str, risk_tier: str, forecast_demand: float, top_shap_features: list[str]
) -> str:
    drivers = ", ".join(top_shap_features)
    return (
        f"You are a supply-chain advisor for a pharmacy chain.\n"
        f"Region: {region}\n"
        f"Stockout risk tier: {risk_tier}\n"
        f"Forecasted demand (next period): {forecast_demand:.0f} units\n"
        f"Top model-identified drivers of this forecast: {drivers}\n\n"
        "Write a short, plain-language advisory (2-3 sentences) for the "
        "pharmacy operations team: what's happening, why (referencing the "
        "drivers above in plain language), and a concrete restock action.\n\n"
        "After the advisory, on its own final line, list exactly which of "
        f"the drivers above you actually referenced, using their literal "
        f"names from this set: {drivers}. Format that line as:\n"
        f"{CITED_DRIVERS_PREFIX} <comma-separated literal driver names, or "
        "none if you referenced none of them>"
    )


def generate_advisory(
    llm_client: LLMClient,
    region: str,
    risk_tier: str,
    forecast_demand: float,
    top_shap_features: list[str],
) -> str:
    prompt = build_advisory_prompt(region, risk_tier, forecast_demand, top_shap_features)
    return llm_client.complete(prompt)


def extract_cited_drivers(advisory_text: str, candidate_features: list[str]) -> list[str]:
    """Return which candidate features the LLM declared it cited.

    Parses the structured `CITED_DRIVERS: ...` line the prompt requires,
    rather than fuzzy-matching prose -- free-form phrasing ("driven by the
    past 1, 3, and 4 periods") varies too much across LLM calls to match
    reliably, and the paper's faithfulness metric depends on precise
    citation extraction. Declared names are filtered against
    candidate_features so a hallucinated or malformed token can't be
    scored as a citation.
    """
    for line in advisory_text.splitlines():
        line = line.strip()
        if not line.startswith(CITED_DRIVERS_PREFIX):
            continue
        declared = line[len(CITED_DRIVERS_PREFIX) :].strip()
        if declared.lower() == "none":
            return []
        named = {token.strip() for token in declared.split(",") if token.strip()}
        return [feature for feature in candidate_features if feature in named]
    return []
