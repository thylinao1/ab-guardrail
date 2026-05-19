# Automated A/B Testing & Causal Inference Agent

A command-line guardrail agent for online experiments. Point it at an A/B-test
CSV and it will:

1. Use **Anthropic Claude with tool-use** to orchestrate the analysis — Claude
   inspects the dataset, decides which column is the variant, which are
   outcomes, and which are covariates, then calls statistical tools to run
   the appropriate checks.
2. Run a **Chi-square Sample Ratio Mismatch (SRM)** test on the allocation —
   the single most under-checked failure mode in industry A/B testing.
3. Run **Welch's t-test + Mann-Whitney U** on continuous metrics, and a
   **2-proportion chi-square** + **Newcombe hybrid-score 95% CI** (Newcombe,
   1998) on binary metrics, with standardised effect sizes.
4. Apply **Benjamini-Hochberg FDR** correction (or Bonferroni) across
   metric tests to avoid false discoveries when checking many metrics.
5. Estimate the **Average Treatment effect on the Treated (ATT)** via
   **Propensity Score Matching** (1-NN, with replacement, caliper =
   0.2 × SD of logit propensity, Austin 2011), with **common-support
   trimming**, **bootstrap standard errors** (because the naive paired-t
   SE is biased under matching with replacement; Abadie & Imbens 2006),
   and **Rosenbaum sensitivity bounds** to quantify how strong an
   unmeasured confounder would have to be to overturn the finding
   (Rosenbaum 2002).
6. Optionally apply **CUPED** variance reduction (Deng et al. WSDM 2013)
   using a pre-experiment covariate, to tighten confidence intervals.
7. Emit a structured Markdown report with a launch verdict —
   `SAFE TO ROLL OUT`, `EXPERIMENT COMPROMISED`, or `NO SIGNIFICANT EFFECT` —
   a love-plot PNG of pre/post covariate balance, and an LLM-written
   executive summary that is constrained to the numerical results.

The LLM is intentionally kept *outside* the statistical loop. It decides
*what* to compute; every number reported comes from `scipy.stats` /
`scikit-learn`.

---

## Why this design

A surprisingly large fraction of A/B tests in production are interpretable
junk: feature flags drop traffic, randomisation seeds get re-used, network
effects bleed between arms, and someone reads off a p-value anyway. The
Microsoft Bing experimentation paper (Fabijan et al., KDD 2019) reports
that ~6% of their A/B tests exhibit SRM, and SRM-affected tests have
effect estimates that are systematically biased.

This tool does the four things every serious experimentation review needs:

| Check | What it catches | Why naive analysis misses it |
|---|---|---|
| Chi-square SRM | broken randomiser, leaky feature flag | a t-test on the metric runs fine even when the split is wrong |
| Welch + Mann-Whitney + Newcombe CI | heavy-tailed metric, near-boundary proportion | a Wald CI undercovers; vanilla t-test understates variance |
| BH-FDR correction | metric-shopping across many secondary metrics | each individual test is fine; collectively the false-discovery rate explodes |
| PSM ATT + bootstrap + Rosenbaum | confounding from leaky randomisation, unmeasured bias | the naive Δ-mean estimate is biased; matched-pair t-SE is biased under replacement |

---

## Three agent modes

| Mode | What it does | When to use |
|---|---|---|
| `--agent tools` (**default**) | Claude is given three tools (`run_srm_check`, `run_metric_test`, `run_propensity_score_match`) and orchestrates the analysis over multiple turns. | Production use. Demonstrates real agentic tool-use. |
| `--agent simple` | Two LLM calls: one to infer schema, one to write the narrative. | API-cheaper for routine batch use. |
| `--agent none` | Fully deterministic — column-name heuristic for schema, template-based narrative. | CI, testing, offline use, when no API key is available. |

In every mode the statistical numbers are computed deterministically by
`scipy.stats` and `scikit-learn`. The LLM never computes a p-value.

---

## Architecture

```
                CSV
                 │
                 ▼
       ┌──────────────────┐
       │  data_loader.py  │  schema validation + dataset profile
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────────────────┐
       │   agent.py  (Claude)         │
       │  ── infer schema OR          │
       │  ── orchestrate tool calls   │
       └────────┬─────────────────────┘
                │
   ┌────────────┼────────────────────────┐
   ▼            ▼                        ▼
 srm.py    metric_tests.py           causal.py
  χ²       Welch + MW-U +            PSM with:
            Newcombe CI +             ├─ common-support trimming
            BH-FDR + CUPED            ├─ bootstrap SE
                                      └─ Rosenbaum bounds
   └────────────┼────────────────────────┘
                ▼
       ┌──────────────────┐
       │     report.py    │  verdict logic + Markdown render
       │                  │  love-plot of pre/post SMD
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │   agent.py       │  LLM: executive summary
       │   (or template)  │
       └────────┬─────────┘
                │
                ▼
            report.md + love_plot.png + terminal verdict
```

---

## Project layout

```
ab_testing_agent/
├── README.md                 ← you are here
├── LICENSE                   MIT
├── pyproject.toml
├── .env.example
├── .github/workflows/ci.yml  pytest + ruff on push
├── data/
│   ├── clean_experiment.csv          generated; healthy A/B test
│   └── compromised_experiment.csv    generated; SRM + confounding
├── reports/                  ← generated Markdown reports + love-plots
├── scripts/
│   └── generate_data.py      synthetic data generator
├── src/
│   ├── cli.py                argparse entry-point
│   ├── agent.py              tool-use orchestration, schema inference, narrative
│   ├── data_loader.py        CSV ingestion + profile
│   ├── exceptions.py         typed errors
│   ├── report.py             verdict logic + Markdown + love-plot
│   └── guardrails/
│       ├── srm.py            chi-square SRM
│       ├── metric_tests.py   Welch / MW / Newcombe / BH-FDR / CUPED
│       └── causal.py         PSM with trimming, bootstrap SE, Rosenbaum
└── tests/                    pytest suite (21 tests)
    ├── conftest.py
    ├── test_srm.py
    ├── test_metric_tests.py
    ├── test_psm.py
    └── test_verdict.py
```

---

## Quick start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Set your API key (or skip and use --agent none)
cp .env.example .env
# edit .env to add your ANTHROPIC_API_KEY

# 3. Generate the demo datasets
python scripts/generate_data.py

# 4. Run the agent
ab-guardrail data/clean_experiment.csv
ab-guardrail data/compromised_experiment.csv

# 5. Run the tests
pytest
```

Reports land in `reports/<csv_stem>_report.md` alongside a
`<csv_stem>_love_plot.png` of pre/post covariate balance.

### Without an API key

```bash
ab-guardrail data/compromised_experiment.csv --agent none
```

`--agent none` uses a deterministic column-name heuristic and a
template-based narrative. Verdicts and statistical numbers are identical
with or without the LLM — the LLM never sees raw data, only column names
and a few sample rows.

### Variance reduction with CUPED

```bash
ab-guardrail data/clean_experiment.csv --cuped pre_signup_value
```

CUPED uses a pre-experiment covariate to reduce variance of the primary
metric, producing tighter confidence intervals on the same effect. When
CUPED is applied, PSM is skipped for the adjusted metric (CUPED has
already done the covariate adjustment parametrically).

### Bring your own CSV

```bash
ab-guardrail my_data.csv \
    --variant-column arm \
    --control-label A \
    --treatment-label B \
    --primary-metric purchased \
    --covariates "signup_age_days,tier,country" \
    --expected-ratio '{"A": 0.5, "B": 0.5}' \
    --correction bh
```

---

## What the two demo datasets look like

Both datasets share the schema `(user_id, variant, pre_signup_value,
device, country, converted, revenue)` with 12,000 users.

**`clean_experiment.csv`** — properly randomised. The agent should output
**SAFE TO ROLL OUT**: SRM not detected, conversion lift ≈ +3.4pp
(p ≈ 7 × 10⁻⁸), PSM-adjusted ATT in the same direction, Rosenbaum
critical Γ around 1.25 (mild robustness to hidden bias).

**`compromised_experiment.csv`** — two simultaneous problems:
- **SRM**: 61/39 split instead of 50/50, because variant assignment
  is correlated with `pre_signup_value` and `device`.
- **Confounding**: the true treatment effect is zero, but treatment
  systematically gets higher-value users.

The agent should output **EXPERIMENT COMPROMISED**, primarily citing the
SRM (p ≈ 2 × 10⁻¹³²), and the PSM step shows the naive revenue lift of
+$0.57 shrinks to +$0.42 (bootstrap CI [+0.13, +0.73]) after adjustment.

---

## CLI reference

```
ab-guardrail <csv>
             [--agent {tools,simple,none}]    LLM mode (default: tools)
             [--correction {bh,bonferroni,none}]
             [--cuped COL]                    variance-reduction covariate
             [--expected-ratio JSON]          {"A": 0.5, "B": 0.5}
             [--variant-column NAME]
             [--control-label LABEL] [--treatment-label LABEL]
             [--primary-metric NAME] [--secondary-metrics A,B]
             [--covariates A,B,C]
             [--out PATH] [--json]
```

---

## Verdict logic

The verdict is decided deterministically — the LLM does not vote.

1. SRM detected (χ² p < 0.001) → **COMPROMISED**. Nothing else matters.
2. PSM produces a *sign flip* on the primary metric while the naive
   estimate was significant → **COMPROMISED** (confounding).
3. Naive estimate is significant AND PSM ATT collapsed to less than 50%
   of naive AND PSM bootstrap 95% CI covers zero → **COMPROMISED** (effect
   not robust).
4. Primary metric is significant at 5% (after multiple-testing
   correction) and no guardrail fired → **SAFE**.
5. Otherwise → **NO SIGNIFICANT EFFECT**.

---

## Caveats

- PSM with replacement, 1-NN, caliper 0.2 × SD logit is the Austin (2011)
  default. It is *not* a substitute for randomisation. The agent reports
  PSM as a robustness check, not as a get-out-of-jail card.
- The bootstrap SE resamples *treated* units; this matches the most
  common matched-with-replacement bootstrap recipe but does not equal
  the analytical SE of Abadie & Imbens (2006). For published causal
  estimates, consider an analytical implementation.
- Rosenbaum bounds assume one hidden binary confounder with odds-ratio
  influence Γ; they are an *upper bound* on the worst-case p-value at
  that level of hidden bias. The bound is conservative.
- The agent does not detect: novelty effects, primacy effects, peeking,
  network/SUTVA violations between arms, or post-treatment selection.
  These require experiment design, not analysis.

---

## References

- Fabijan, A., Gupchup, J., Gupta, S., Omhover, J., Qin, W., Vermeer,
  L., & Dmitriev, P. (2019). *Diagnosing Sample Ratio Mismatch in
  Online Controlled Experiments*. KDD 2019.
  [https://doi.org/10.1145/3292500.3330722](https://doi.org/10.1145/3292500.3330722)
- Rosenbaum, P. R., & Rubin, D. B. (1983). *The central role of the
  propensity score in observational studies for causal effects*.
  Biometrika, 70(1), 41-55.
  [https://doi.org/10.1093/biomet/70.1.41](https://doi.org/10.1093/biomet/70.1.41)
- Austin, P. C. (2011). *An introduction to propensity score methods
  for reducing the effects of confounding in observational studies*.
  Multivariate Behavioral Research, 46(3), 399-424.
  [https://doi.org/10.1080/00273171.2011.568786](https://doi.org/10.1080/00273171.2011.568786)
- Abadie, A., & Imbens, G. W. (2006). *Large sample properties of
  matching estimators for average treatment effects*. Econometrica,
  74(1), 235-267.
  [https://doi.org/10.1111/j.1468-0262.2006.00655.x](https://doi.org/10.1111/j.1468-0262.2006.00655.x)
- Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer.
- Newcombe, R. G. (1998). *Interval estimation for the difference
  between independent proportions: comparison of eleven methods*.
  Statistics in Medicine, 17(8), 873-890.
- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). *Improving the
  Sensitivity of Online Controlled Experiments by Utilizing
  Pre-Experiment Data*. WSDM 2013.
  [https://doi.org/10.1145/2433396.2433413](https://doi.org/10.1145/2433396.2433413)
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled
  Experiments: A Practical Guide to A/B Testing*. Cambridge University
  Press.

---

## License

MIT — see [LICENSE](LICENSE).
