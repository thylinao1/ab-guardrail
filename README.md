# Automated A/B Testing & Causal Inference Guardrail

A command-line guardrail for online experiments. Point it at an A/B-test
CSV and it will:

1. **Load and clean the file defensively** - skip malformed rows, drop
   exact-duplicate (double-logged) rows and all-null columns, coerce dirty
   numeric columns, and report every repair in a data-quality summary.
2. **Route deterministically** - a hardcoded heuristic identifies the
   variant column, metrics, and covariates. No LLM in this hot path.
3. Run a **Chi-square Sample Ratio Mismatch (SRM)** test on the allocation -
   the single most under-checked failure mode in industry A/B testing.
4. Run **Welch's t-test + Mann-Whitney U** on continuous metrics, and a
   **2-proportion chi-square** + **Newcombe hybrid-score 95% CI** (Newcombe,
   1998) on binary metrics, with standardised effect sizes.
5. Apply **Benjamini-Hochberg FDR** correction (or Bonferroni) across
   metric tests to avoid false discoveries when checking many metrics.
6. Estimate the **Average Treatment effect on the Treated (ATT)** via
   **Propensity Score Matching** (1-NN, with replacement, caliper =
   0.2 × SD of logit propensity, Austin 2011), with **common-support
   trimming**, **bootstrap standard errors** (because the naive paired-t
   SE is biased under matching with replacement; Abadie & Imbens 2006),
   and **Rosenbaum sensitivity bounds** for hidden bias (Rosenbaum 2002).
7. Optionally apply **CUPED** variance reduction (Deng et al. WSDM 2013).
8. Emit a structured Markdown report with a launch verdict -
   `SAFE TO ROLL OUT`, `EXPERIMENT COMPROMISED`, or `NO SIGNIFICANT EFFECT` -
   a love-plot PNG of pre/post covariate balance, and a plain-English
   executive summary.

**Where the LLM fits.** Routing and statistics are a plain, deterministic
Python pipeline - paying LLM latency and cost to dispatch data to three
deterministic functions is poor production engineering. The LLM is invoked
**exactly once, at the very end**, to turn the finished results JSON into
the executive summary. A tool-use *agent* mode is available opt-in
(`--mode agent`) for datasets whose column roles cannot be inferred
deterministically - useful, but deliberately not the default. The LLM
never computes a number; every statistic comes from `scipy.stats` /
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

## Three execution modes

| Mode | Routing | Final summary | When to use |
|---|---|---|---|
| `--mode pipeline` (**default**) | deterministic heuristic - no LLM | one LLM call on the finished JSON | Production. No LLM in the hot path. |
| `--mode agent` (opt-in) | Claude tool-use loop | agent's closing turn | Unfamiliar schemas where deterministic routing fails. Slower, costlier. |
| `--mode offline` | deterministic heuristic - no LLM | template (no API) | CI, offline runs. Byte-stable. |

In every mode the statistical numbers are computed deterministically by
`scipy.stats` and `scikit-learn`. The LLM never computes a p-value, and on
the default path it is not on the routing loop at all.

---

## Architecture

```
                CSV
                 │
                 ▼
       ┌──────────────────┐
       │  data_loader.py  │  defensive load: skip malformed rows, drop
       │                  │  duplicates + all-null cols, coerce dirty
       │                  │  numerics → DataQualityReport
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────────────────┐
       │  routing (deterministic)     │  hardcoded heuristic picks the
       │  infer_schema_heuristic()    │  variant / metrics / covariates.
       │  [--mode agent: tool-use]    │  No LLM on the default path.
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
       │     report.py    │  deterministic verdict logic + Markdown
       │                  │  render + love-plot of pre/post SMD
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │  narrate()       │  ◀── the ONLY LLM call on the default path:
       │  (1 LLM call)    │      summarises the finished results JSON
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
│   ├── compromised_experiment.csv    generated; SRM + confounding
│   └── messy_experiment.csv          generated; clean test, dirty file
├── reports/                  ← generated Markdown reports + love-plots
├── scripts/
│   ├── generate_data.py      synthetic data generator (3 datasets)
│   └── criteo_adapter.py     maps the real Criteo Uplift dataset → schema
├── src/
│   ├── cli.py                argparse entry-point, --mode {pipeline,agent,offline}
│   ├── agent.py              deterministic routing heuristic, narrate(), tool-use agent
│   ├── data_loader.py        defensive CSV load + DataQualityReport
│   ├── exceptions.py         typed errors
│   ├── report.py             verdict logic + Markdown + love-plot
│   └── guardrails/
│       ├── srm.py            chi-square SRM
│       ├── metric_tests.py   Welch / MW / Newcombe / BH-FDR / CUPED
│       └── causal.py         PSM with trimming, bootstrap SE, Rosenbaum
└── tests/                    pytest suite (36 tests)
    ├── conftest.py
    ├── test_srm.py
    ├── test_metric_tests.py
    ├── test_psm.py
    ├── test_data_loader.py   defensive-loading / messy-data tests
    └── test_verdict.py
```

---

## Quick start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. (Optional) set an API key for the closing summary
cp .env.example .env       # add ANTHROPIC_API_KEY - or skip and use --mode offline

# 3. Generate the three demo datasets
python scripts/generate_data.py

# 4. Run it
ab-guardrail data/clean_experiment.csv
ab-guardrail data/compromised_experiment.csv
ab-guardrail data/messy_experiment.csv

# 5. Run the tests
pytest
```

Reports land in `reports/<csv_stem>_report.md` alongside a
`<csv_stem>_love_plot.png` of pre/post covariate balance.

Default `--mode pipeline` routes deterministically and makes a single LLM
call for the closing summary. If `ANTHROPIC_API_KEY` is unset it silently
falls back to a template summary - the verdict and every statistic are
identical either way.

### Fully offline

```bash
ab-guardrail data/compromised_experiment.csv --mode offline
```

No API calls at all. Byte-stable; used in CI.

### Variance reduction with CUPED

```bash
ab-guardrail data/clean_experiment.csv --cuped pre_signup_value
```

CUPED uses a pre-experiment covariate to reduce variance of the primary
metric - tighter confidence intervals on the same effect. When CUPED is
applied, PSM is skipped for the adjusted metric (CUPED has already done the
covariate adjustment parametrically).

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

## The three demo datasets

All three share the schema `(user_id, variant, pre_signup_value, device,
country, converted, revenue)` with 12,000 users.

**`clean_experiment.csv`** - properly randomised. Verdict **SAFE TO ROLL
OUT**: SRM not detected, conversion lift ≈ +3.4pp (p ≈ 7 × 10⁻⁸),
PSM-adjusted ATT in the same direction.

**`compromised_experiment.csv`** - two simultaneous problems: a 61/39 split
(SRM) and variant assignment correlated with `pre_signup_value` and
`device` (confounding). Verdict **EXPERIMENT COMPROMISED**, mainly
citing the SRM (p ≈ 2 × 10⁻¹³²); PSM shows the naive revenue lift shrink
under adjustment.

**`messy_experiment.csv`** - the experiment is *clean* (50/50, a real
lift); the **file** is broken, the way a real export is: ~8% missing
`pre_signup_value`, ~5% missing `device`, dirty `ERROR`/`NULL` tokens in a
numeric column, an all-null `experiment_notes` column, 25 duplicate rows,
and 3 malformed rows. The loader repairs all of it, reports each repair,
and the pipeline still returns **SAFE TO ROLL OUT** - demonstrating that
data-quality defects do not silently corrupt the verdict.

---

## Messy real-world data

Production e-commerce A/B logs are not tidy. `data_loader.py` ingests CSVs
defensively and surfaces a `DataQualityReport` so nothing is repaired
silently:

| Pathology | Handling |
|---|---|
| malformed rows (wrong field count) | skipped at parse time, counted |
| double-logged rows | dropped *only* when the file has a per-row identifier and rows repeat on it - without an ID, identical rows (low-cardinality covariates) are kept |
| all-null columns (instrumented, never populated) | dropped, named |
| dirty numeric tokens (`ERROR`, `NULL`, `""`) | column coerced to numeric; bad tokens → missing |
| missing covariate values | reported per column; PSM / metric tests drop incomplete rows per analysis |

Every repair is printed to the terminal and written into a **Data quality**
section of the Markdown report.

### Validating against the Criteo Uplift dataset

The synthetic demos have a known ground truth; real data does not.
`scripts/criteo_adapter.py` maps the public **Criteo Uplift Modeling
dataset** (~13.98M-row randomised advertising experiment, 12 anonymised
features) onto this tool's schema:

```bash
# download criteo-uplift-v2.1.csv.gz from
# https://ailab.criteo.com/criteo-uplift-prediction-dataset/  then:
python scripts/criteo_adapter.py /path/to/criteo-uplift-v2.1.csv \
    --out data/criteo_ready.csv --sample 300000

ab-guardrail data/criteo_ready.csv --mode pipeline \
    --primary-metric conversion --secondary-metrics visit \
    --covariates f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11 \
    --expected-ratio '{"control": 0.15, "treatment": 0.85}'
```

Two real-data lessons are baked into the adapter. First, Criteo encodes
treatment as a 0/1 column - the loader deliberately excludes numeric 0/1
columns from variant candidates, so the adapter remaps it to string
labels. Second, **Criteo is designed with an ~85/15 split, not 50/50** -
the SRM check compares observed against *planned*, so the planned ratio
must be passed via `--expected-ratio` or the check false-positives. The
adapter prints the exact command, including the correct ratio, after it
runs.

#### Result on a 300k-row Criteo sample

Run against a stratified 300,204-row sample of the 13.98M-row experiment,
the guardrail returns **SAFE TO ROLL OUT**, which is the correct call for
a genuine RCT:

- **SRM passes** (χ² p ≈ 0.98) once the 85/15 design ratio is supplied.
- **Both metrics show a significant lift:** conversion 0.18% → 0.31%
  (+0.13pp, p ≈ 2 × 10⁻⁶); visit 3.69% → 4.86% (+1.17pp, p ≈ 4 × 10⁻²⁷).
- **PSM barely moves the estimate** - ATT 0.0013 → 0.0010 (conversion) and
  0.0117 → 0.0101 (visit) across 255,123 matched pairs. That small
  adjustment is the expected signature of a *truly randomised* experiment
  (little confounding to remove), and contrasts sharply with the synthetic
  `compromised_experiment.csv`, where PSM collapses the naive estimate.
  Rosenbaum Γ ≈ 1.5 indicates moderate robustness to hidden bias.

Validating on Criteo also surfaced - and the test suite now guards
against - three real-data bugs the synthetic demos could not: identical
rows wrongly dropped when covariates are low-cardinality, variant labels
misread from the first few rows under a skewed split, and an
index-alignment fault in the adapter's sampling path.

---

## CLI reference

```
ab-guardrail <csv>
             [--mode {pipeline,agent,offline}]   default: pipeline
             [--correction {bh,bonferroni,none}]
             [--cuped COL]                       variance-reduction covariate
             [--expected-ratio JSON]             {"A": 0.5, "B": 0.5}
             [--variant-column NAME]
             [--control-label LABEL] [--treatment-label LABEL]
             [--primary-metric NAME] [--secondary-metrics A,B]
             [--covariates A,B,C]
             [--out PATH] [--json]
```

---

## Verdict logic

The verdict is decided deterministically - the LLM does not vote.

1. SRM detected (χ² p < 0.001) → **COMPROMISED**. Nothing else matters.
2. PSM produces a *sign flip* on the primary metric while the naive
   estimate was significant → **COMPROMISED** (confounding).
3. Naive estimate is significant AND PSM ATT collapsed to less than 50%
   of naive AND PSM bootstrap 95% CI covers zero → **COMPROMISED** (effect
   does not hold up).
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
- The tool does not detect: novelty effects, primacy effects, peeking,
  network/SUTVA violations between arms, or post-treatment selection.
  These require experiment design, not analysis.
- The Criteo adapter is shipped and unit-tested against a synthetic
  Criteo-shaped CSV, but a full run needs the ~297 MB dataset downloaded
  locally - it is not exercised in CI. The three bundled demo datasets are
  synthetic, with a known ground truth, by design.

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

MIT - see [LICENSE](LICENSE).
