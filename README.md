# ab-guardrail

A command-line guardrail for online experiments. Point it at an A/B-test CSV and it
checks the traffic allocation for Sample Ratio Mismatch, runs the metric test that
suits each metric's shape, corrects for multiple testing, re-estimates the effect
under propensity score matching, and writes a Markdown report ending in one of three
verdicts: `SAFE TO ROLL OUT`, `EXPERIMENT COMPROMISED`, or `NO SIGNIFICANT EFFECT`.

It has been run against the Criteo Uplift dataset, a 13.98M-row randomised advertising
experiment. On a stratified 300,204-row sample it returns `SAFE TO ROLL OUT`, with the
SRM check passing at chi-square p ~ 0.98 once the planned 85/15 allocation is supplied.
Numbers for that run are in [Results](#results).

## Install

Python 3.10 or newer.

```bash
pip install -e ".[dev]"
```

The `anthropic` and `python-dotenv` packages are runtime dependencies, but no API key
is needed to run the tool. `--mode offline` skips the network entirely.

## Run

```bash
# generate the three demo datasets
python scripts/generate_data.py

# analyse them
ab-guardrail data/clean_experiment.csv
ab-guardrail data/compromised_experiment.csv
ab-guardrail data/messy_experiment.csv

# tests
pytest
```

Each run writes `reports/<csv_stem>_report.md` and a `<csv_stem>_love_plot.png` showing
pre/post covariate balance.

Without an `ANTHROPIC_API_KEY`, the closing summary falls back to a template. The
verdict and every statistic are identical either way. To skip the API call explicitly:

```bash
ab-guardrail data/compromised_experiment.csv --mode offline
```

CUPED variance reduction on a pre-experiment covariate:

```bash
ab-guardrail data/clean_experiment.csv --cuped pre_signup_value
```

Your own CSV, with the column roles pinned by hand:

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

Full flag list:

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

## Method

### Loading

`data_loader.py` parses the CSV defensively and returns a `DataQualityReport` alongside
the frame, so no repair happens silently. Every repair is printed to the terminal and
written into a "Data quality" section of the report.

| Pathology | Handling |
|---|---|
| malformed rows (wrong field count) | skipped at parse time, counted |
| double-logged rows | dropped only when the file has a per-row identifier and rows repeat on it; without an ID, identical rows are kept, since low-cardinality covariates make genuine collisions common |
| all-null columns (instrumented, never populated) | dropped and named |
| dirty numeric tokens (`ERROR`, `NULL`, `""`) | column coerced to numeric, bad tokens become missing |
| missing covariate values | reported per column; metric tests and PSM drop incomplete rows per analysis |

### Routing

A hardcoded heuristic in `agent.py` picks the variant column, the metrics, and the
pre-treatment covariates from the profiled frame. Column names carrying a covariate
prefix (`pre_`, `prior_`, `baseline_`) win over outcome tokens, so `pre_signup_value`
is treated as a covariate despite containing "value". Numeric 0/1 columns are excluded
from variant candidates, because a binary outcome would otherwise be mistaken for the
assignment column. Any of these choices can be overridden from the CLI.

### Sample Ratio Mismatch

A chi-square goodness-of-fit test compares the observed allocation against the planned
one. It declares an SRM at p < 0.001 rather than 0.05, because an SRM investigation is
expensive and a 5% false-positive rate would trigger far too many of them. Cells with an
expected count below 5 raise instead of returning a number.

An SRM survives an otherwise correct analysis: the t-test on the metric runs fine on a
broken split and prints a p-value. Fabijan et al. (KDD 2019) report that around 6% of
the online tests they studied carried one, with systematically distorted effect
estimates.

### Metric tests

Binary metrics get a two-proportion chi-square on the 2x2 table plus a Newcombe (1998)
hybrid-score 95% CI on the difference of proportions, which covers considerably better
than a Wald interval when a proportion sits near 0 or 1. Continuous metrics get Welch's
t-test plus a Mann-Whitney U that does not assume normality. Both p-values are reported.
Effect sizes are standardised: Cohen's h for proportions, Cohen's d for means.

Across a family of metrics, `apply_multiple_testing_correction()` applies
Benjamini-Hochberg FDR by default, or Bonferroni, or nothing. Large experimentation
platforms usually correct across guardrail or secondary metrics and leave the primary
uncorrected; here the choice is a CLI flag.

CUPED (Deng et al., WSDM 2013) is available through `--cuped COL`. It subtracts the part
of the outcome that a pre-experiment covariate already explained, leaving the point
estimate unchanged in expectation and cutting variance by a factor of (1 - rho^2). When
CUPED is applied, PSM is skipped for the adjusted metric, since the covariate adjustment
has already been done parametrically.

### Propensity score matching

`causal.py` estimates the Average Treatment effect on the Treated. A logistic regression
predicts P(treated | covariates); treated units whose propensity falls outside the range
of the control propensities are trimmed away, because no causal claim is defensible
outside the overlap region; each surviving treated unit is matched to its nearest
control on the propensity score (1-NN, with replacement) within a caliper of 0.2 times
the SD of the logit propensity, the Austin (2011) default.

Two things are reported that a plain matched-pair analysis leaves out. The standard
error comes from a bootstrap over treated units with re-matching at each of 500
resamples, because matching with replacement re-uses controls and breaks the
independence the paired-t SE assumes (Abadie & Imbens, 2006). The report prints the
paired-t SE next to it so the size of the bias is visible. Rosenbaum (2002) bounds then
sweep a hidden-bias grid and report the smallest Gamma at which the worst-case
Wilcoxon p-value crosses 0.05.

Covariate balance is reported as standardised mean differences before and after
matching, in a table and as a love plot.

### Verdict

`report.py` decides the verdict deterministically. The LLM does not vote.

1. SRM detected (chi-square p < 0.001), so `EXPERIMENT COMPROMISED`. Nothing below is
   evaluated.
2. PSM produces a sign flip on the primary metric while the naive estimate was
   significant, so `EXPERIMENT COMPROMISED` (confounding).
3. The naive estimate is significant, the PSM ATT collapsed below 50% of it, and the
   bootstrap 95% CI covers zero, so `EXPERIMENT COMPROMISED`.
4. The primary metric is significant at 5% after correction and no guardrail fired, so
   `SAFE TO ROLL OUT`.
5. Otherwise `NO SIGNIFICANT EFFECT`.

### Where the LLM fits

Routing and statistics are a deterministic Python pipeline. The LLM is called once, at
the very end, to turn the finished results JSON into a plain-English executive summary,
and it is instructed not to introduce a number that is absent from the payload or to
contradict the verdict. Every statistic comes from `scipy.stats` or `scikit-learn`.

`--mode agent` puts Claude in charge of routing through a tool-use loop over the same
three deterministic functions. That mode exists for schemas the heuristic cannot read,
and it is slower and more expensive, so it is opt-in.

| Mode | Routing | Final summary |
|---|---|---|
| `pipeline` (default) | deterministic heuristic | one LLM call on the finished JSON |
| `agent` | Claude tool-use loop | the agent's closing turn |
| `offline` | deterministic heuristic | template, no API call, byte-stable |

## Repository layout

| Path | Contents |
|---|---|
| `src/cli.py` | argparse entry point, `--mode {pipeline,agent,offline}` |
| `src/agent.py` | routing heuristic, `narrate()`, tool-use agent |
| `src/data_loader.py` | defensive CSV load and `DataQualityReport` |
| `src/report.py` | verdict logic, Markdown render, love plot |
| `src/guardrails/srm.py` | chi-square SRM |
| `src/guardrails/metric_tests.py` | Welch, Mann-Whitney, Newcombe, BH-FDR, CUPED |
| `src/guardrails/causal.py` | PSM with trimming, bootstrap SE, Rosenbaum bounds |
| `scripts/generate_data.py` | synthetic data generator for the three demo files |
| `scripts/criteo_adapter.py` | maps the Criteo Uplift dataset onto this tool's schema |
| `data/` | the three generated demo CSVs |
| `reports/` | generated reports and love plots |
| `tests/` | pytest suite, 36 tests |

## Results

### Demo datasets

All three carry the schema `(user_id, variant, pre_signup_value, device, country,
converted, revenue)` with 12,000 users.

| Dataset | Injected defect | Verdict |
|---|---|---|
| `clean_experiment.csv` | none | `SAFE TO ROLL OUT` |
| `compromised_experiment.csv` | 61/39 split, and assignment correlated with `pre_signup_value` and `device` | `EXPERIMENT COMPROMISED` |
| `messy_experiment.csv` | none in the experiment, several in the file | `SAFE TO ROLL OUT` |

On the clean file, SRM is not detected and conversion rises about 3.4pp
(p ~ 7e-8), with the PSM-adjusted ATT pointing the same way. On the compromised file the
SRM check fires at p ~ 2e-132 and ends the analysis; PSM also shows the naive revenue
lift shrinking under adjustment.

The messy file is the interesting one. The experiment underneath it is sound (50/50, a
real lift), and what is broken is the export: about 8% missing `pre_signup_value`, about
5% missing `device`, `ERROR` and `NULL` tokens inside a numeric column, an all-null
`experiment_notes` column, 25 duplicate rows, and 3 malformed rows. The loader repairs
all of it, reports each repair, and the verdict is unchanged.

### Criteo Uplift dataset

The synthetic demos have a known ground truth; real data does not.
`scripts/criteo_adapter.py` maps the public Criteo Uplift Modeling dataset, a 13.98M-row
randomised advertising experiment with 12 anonymised features, onto this tool's schema.

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

On a stratified 300,204-row sample:

| Check | Result |
|---|---|
| SRM, planned 85/15 | chi-square p ~ 0.98, passes |
| conversion | 0.18% to 0.31%, +0.13pp, p ~ 2e-6 |
| visit | 3.69% to 4.86%, +1.17pp, p ~ 4e-27 |
| PSM ATT, conversion | naive 0.0013, adjusted 0.0010 |
| PSM ATT, visit | naive 0.0117, adjusted 0.0101 |
| matched pairs | 255,123 |
| Rosenbaum critical Gamma | ~1.5 |
| Verdict | `SAFE TO ROLL OUT` |

That is the correct call for a genuine RCT. The small PSM adjustment is what a truly
randomised experiment should look like: there is little confounding left to remove, in
contrast to `compromised_experiment.csv`, where matching collapses the naive estimate.
A critical Gamma of about 1.5 means a hidden binary confounder would have to shift the
odds of assignment by roughly half again before the ATT stopped being significant.

Two properties of the real file are handled in the adapter rather than the tool. Criteo
encodes treatment as a 0/1 column, and the loader deliberately excludes numeric 0/1
columns from variant candidates, so the adapter remaps it to string labels. And Criteo
is designed with an 85/15 split, so the planned ratio has to be passed through
`--expected-ratio` or the SRM check false-positives. The adapter prints the exact
command, with the right ratio, after it runs.

The Criteo run also exposed three bugs the synthetic demos could not surface, all of
which now have tests: identical rows wrongly dropped when covariates are low-cardinality,
variant labels misread from the first few rows under a skewed split, and an
index-alignment fault in the adapter's sampling path.

## Limitations

PSM with replacement, 1-NN, caliper 0.2 SD on the logit is the Austin (2011) default.
It adjusts for observed covariates only, which makes it a sensitivity check on top of a
randomised experiment rather than a replacement for randomisation.

The bootstrap SE resamples treated units. That is the common matched-with-replacement
recipe; it differs from the analytical estimator of Abadie & Imbens (2006). For a
published causal estimate, the analytical form would be the next step.

Rosenbaum bounds assume one hidden binary confounder with odds-ratio influence Gamma,
and they give an upper bound on the worst-case p-value at that level of bias. The bound
is conservative by construction.

The tool does not detect novelty effects, primacy effects, peeking, network or SUTVA
violations between arms, or post-treatment selection. Those are properties of the
experiment design and cannot be recovered from the logs afterwards.

The Criteo adapter is unit-tested against a synthetic Criteo-shaped CSV, but a full run
needs the ~297 MB dataset downloaded locally, so it is not exercised in CI. The three
bundled demo datasets are synthetic on purpose, so their ground truth is known.

## References

- Fabijan, A., Gupchup, J., Gupta, S., Omhover, J., Qin, W., Vermeer, L., & Dmitriev, P.
  (2019). *Diagnosing Sample Ratio Mismatch in Online Controlled Experiments*. KDD 2019.
  https://doi.org/10.1145/3292500.3330722
- Rosenbaum, P. R., & Rubin, D. B. (1983). *The central role of the propensity score in
  observational studies for causal effects*. Biometrika, 70(1), 41-55.
  https://doi.org/10.1093/biomet/70.1.41
- Austin, P. C. (2011). *An introduction to propensity score methods for reducing the
  effects of confounding in observational studies*. Multivariate Behavioral Research,
  46(3), 399-424. https://doi.org/10.1080/00273171.2011.568786
- Abadie, A., & Imbens, G. W. (2006). *Large sample properties of matching estimators for
  average treatment effects*. Econometrica, 74(1), 235-267.
  https://doi.org/10.1111/j.1468-0262.2006.00655.x
- Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer.
- Newcombe, R. G. (1998). *Interval estimation for the difference between independent
  proportions: comparison of eleven methods*. Statistics in Medicine, 17(8), 873-890.
- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). *Improving the Sensitivity of Online
  Controlled Experiments by Utilizing Pre-Experiment Data*. WSDM 2013.
  https://doi.org/10.1145/2433396.2433413
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A
  Practical Guide to A/B Testing*. Cambridge University Press.

## License

MIT. See [LICENSE](LICENSE).
