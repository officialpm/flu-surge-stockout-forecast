# Flu-Surge Stockout Forecast

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22632723.svg)](https://doi.org/10.5281/zenodo.22632723)
[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fofficialpm%2Fflu-surge-stockout-forecast%2F&countColor=%23007ec6&style=flat)](https://github.com/officialpm/flu-surge-stockout-forecast)

An end-to-end, fully reproducible pipeline that fuses CDC influenza
surveillance with retail sales to forecast demand, scores the forecast into
a stockout-risk tier, and generates a plain-language LLM advisory whose
stated reasoning is audited against the forecasting model's real SHAP
feature attributions instead of being trusted at face value.

This is a research proof-of-concept, not a production forecasting system.
It uses `HOUSEHOLD`-category retail sales from the M5 competition as a
public proxy for OTC-relevant demand (no public dataset pairs real OTC
pharmacy sales with regional epidemiological surveillance), and a simulated
inventory policy in place of real pharmacy stock data. See
[Scope and limitations](#scope-and-limitations) below and the paper's
Limitations section for the full account.

## Why

Reorder lead times for OTC cold-and-flu inventory run one to three weeks.
A forecast built only on sales history notices a flu-driven demand surge
only after it has already begun eroding stock. CDC FluView publishes
weekly, region-level influenza-like-illness (ILI) data that leads exactly
this kind of demand shock, but it has no connection to any retail inventory
pipeline. This project builds that connection and, separately, tests
whether an LLM's natural-language explanation of a forecast can be held to
the same standard of evidence as the forecast itself.

## What it does

1. **Data layer** — joins CDC FluView ILINet (HHS-region, weekly ILI rate)
   with M5 retail sales (Walmart, daily → weekly, state → HHS region) into
   a single weekly panel.
2. **Forecasting layer** — trains a sales-only baseline and a
   health-augmented LightGBM forecaster on identical rows, compares
   accuracy with a paired Wilcoxon signed-rank test, and computes SHAP
   feature attributions for the augmented model.
3. **Risk-scoring layer** — converts each forecast into a `low` / `medium`
   / `high` stockout-risk tier under an explicit, documented (simulated)
   inventory policy.
4. **GenAI advisory layer** — prompts an LLM for a short, plain-language
   advisory per forecast, then runs a **faithfulness audit**: the LLM must
   declare its cited drivers in a machine-parseable trailer line, which is
   scored against the model's real top-SHAP drivers using exact
   precision/recall, not fuzzy text matching.

Every stage reads only from the previous stage's structured output. A
single pipeline run produces one JSON artifact (config + git commit +
full results) that every number and figure in the paper is generated from
— nothing is hand-computed or transcribed.

```
CDC ILI ──┐
          ├─→ weekly panel ─→ baseline / health-augmented forecaster ─→ SHAP top-k
M5 sales ─┘                                                              │
                                                                          ▼
                                              stockout risk tier ←── forecast
                                                      │
                                                      ▼
                                     LLM advisory + CITED_DRIVERS
                                                      │
                                                      ▼
                                     faithfulness audit (precision/recall
                                     vs. SHAP top-k)
```

## Repository layout

```
config/                 Pipeline configuration (config.yaml)
scripts/
  download_ili_data.py    Fetch CDC ILINet data via the public Delphi Epidata API
  download_m5_data.py     Fetch M5 sales_train_validation.csv + calendar.csv from Kaggle
  run_pipeline.py         Run the full pipeline end-to-end, write a JSON run artifact
  generate_figures.py     Regenerate all paper figures from a run artifact
src/
  config.py                Config loading
  data_pipeline/            ILI loader, M5 loader, HHS-region mapping, panel builder
  forecasting/               Feature engineering, LightGBM forecaster, SHAP, time split
  risk_scoring/              Stockout risk tier logic
  eval/                        MAPE/RMSE, paired significance test, faithfulness scoring
  genai_advisory/               LLM client abstraction, advisory generation + citation parsing
tests/                  Mirrors src/ + scripts/; fixtures under tests/fixtures/
paper/                  LaTeX paper source (not tracked in this repo; see below)
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Get the data

Both source datasets are public. You do not need Kaggle credentials if you
download the M5 files by hand from the website instead of using the
provided script.

**CDC FluView ILINet** (HHS regions, all seasons) — either:
- Run `python -m scripts.download_ili_data` to pull it automatically from
  the public [Delphi Epidata API](https://api.delphi.cmu.edu/epidata/fluview/)
  (no account needed), which writes `data/raw/ili_ilinet.csv`; or
- Export it manually from the
  [CDC FluView Interactive dashboard](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html)
  and save it as `data/raw/ili_ilinet.csv`.

**M5 Forecasting Accuracy dataset** (Kaggle) — either:
- Export a Kaggle API token from https://www.kaggle.com/settings/api,
  set `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`), and run
  `python -m scripts.download_m5_data`; or
- Download `sales_train_validation.csv` and `calendar.csv` manually from
  https://www.kaggle.com/c/m5-forecasting-accuracy/data and place them
  under `data/raw/`.

### Set the LLM API key

The GenAI advisory stage calls the Anthropic Messages API.

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

`ANTHROPIC_BASE_URL` is optional and only needed if you route requests
through a self-hosted gateway instead of `api.anthropic.com`.

## Run

```bash
python -m scripts.run_pipeline --config config/config.yaml
```

This prints the headline numbers (baseline vs. health-augmented MAPE,
significance test, faithfulness scores) and writes a timestamped JSON
artifact to `results/run_<UTC-timestamp>.json`.

Then regenerate the figures from that artifact:

```bash
python -m scripts.generate_figures --run results/run_<timestamp>.json
```

Figures are written to `results/figures/` and are not committed to
git — they are always derived from a run artifact, never hand-edited.

## Test

```bash
pytest
```

The full suite (46 tests) runs entirely offline against fixtures in
`tests/fixtures/` and a `FakeLLMClient` test double — no network access or
API key is required to verify correctness.

## Configuration

All pipeline parameters live in `config/config.yaml`:

| Key | Meaning |
|---|---|
| `data.categories` | M5 product category used as the retail-demand proxy (default: `HOUSEHOLD`) |
| `forecasting.lag_weeks` | Which weekly sales lags to use as features |
| `forecasting.test_weeks` | Held-out weeks per region for evaluation |
| `forecasting.top_k_shap` | Number of top SHAP drivers surfaced to the LLM per forecast |
| `risk.weeks_of_cover` | Simulated stock-proxy window (trailing average sales) |
| `risk.reorder_lead_time_weeks` | Lead time used to compute the demand-to-stock ratio |
| `llm.model` | Anthropic model used for advisory generation |

## Scope and limitations

Stated plainly, not buried in a footnote:

- **Retail category is a proxy, not real OTC sales.** The M5 taxonomy has
  no cold-and-flu or pharmacy category. `HOUSEHOLD` is used because prior
  work ([Miliou et al., 2021](https://doi.org/10.1371/journal.pcbi.1009087))
  shows this category's illness-driven stocking behavior (tissues,
  cleaning and sanitizing products, paper goods) responds to influenza
  activity. Forecast-accuracy results should be read as evidence about
  health-augmented retail forecasting on this proxy category, not as a
  claim validated on actual pharmacy SKUs.
- **The stockout-risk tier uses a simulated inventory policy.** No real
  pharmacy inventory data was available; current stock is approximated as
  a trailing average of weekly sales. This bounds the risk-scoring layer's
  claims to the assumed policy, not a validated one.
- **CDC ILI is syndromic surveillance**, a broader proxy than confirmed
  influenza incidence — a well-documented property of this data stream
  generally, not specific to how it's used here.
- **Sample size is small** (3 HHS regions, 8 held-out weeks each, n=24
  overall); the paper does not claim the accuracy improvement is
  universally significant, only that it is directionally consistent with
  each region's ILI signal variance in the test window.

## Citation

If this pipeline or its faithfulness-audit methodology is useful in your
own work, please cite the accompanying paper — see [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
