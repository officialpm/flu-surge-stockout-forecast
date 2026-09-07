# Feature groups that represent one underlying concept split across
# multiple model features (e.g. a cyclical encoding's sin/cos halves).
# Faithfulness scoring must treat citing either member as citing the same
# single concept -- otherwise a driver split into N correlated features
# is worth up to N independent "credits" and distorts precision/recall.
_EQUIVALENT_FEATURE_GROUPS = [frozenset({"week_sin", "week_cos"})]


def _canonicalize(features: list[str]) -> set[str]:
    canonical = set()
    for feature in features:
        for group in _EQUIVALENT_FEATURE_GROUPS:
            if feature in group:
                canonical.add(min(group))
                break
        else:
            canonical.add(feature)
    return canonical


def faithfulness_precision_recall(
    cited_drivers: list[str], shap_top_features: list[str]
) -> dict:
    """Overlap between what the LLM cited and the model's actual SHAP drivers.

    precision: fraction of cited drivers that are real top drivers.
    recall: fraction of real top drivers that were cited.

    Cited/shap feature names are canonicalized first via
    _EQUIVALENT_FEATURE_GROUPS so that a multi-feature concept (e.g.
    week_sin/week_cos) is not double-counted.
    """
    if not cited_drivers:
        return {"precision": 0.0, "recall": 0.0}

    cited_set = _canonicalize(cited_drivers)
    shap_set = _canonicalize(shap_top_features)
    overlap = cited_set & shap_set

    precision = len(overlap) / len(cited_set)
    recall = len(overlap) / len(shap_set) if shap_set else 0.0
    return {"precision": precision, "recall": recall}
