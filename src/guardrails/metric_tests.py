"""Metric-shift tests between control and treatment.

We pick the test based on the metric's shape:

* Binary (0/1) metric  -> two-proportion test via chi-square on a 2x2
  contingency table, with a **Newcombe (1998) hybrid-score 95% CI** on the
  difference of proportions. The Newcombe interval has materially better
  coverage than a Wald interval, especially when proportions are near 0
  or 1 — see Newcombe, *Statistics in Medicine* 17:873 (1998).
* Continuous metric    -> Welch's t-test for the mean shift, plus a
  Mann-Whitney U as a non-parametric robustness check that does NOT
  assume normality. Both p-values are reported.

We report a 95% confidence interval on the mean (or proportion) difference,
plus a standardised effect size (Cohen's h for proportions, Cohen's d for
continuous), because a p-value alone tells the reader whether the effect
is non-zero but not whether it is meaningful.

`apply_multiple_testing_correction()` adjusts a batch of p-values across
metrics using Benjamini-Hochberg FDR (default) or Bonferroni. Industry
practice at large experimentation platforms (Microsoft, Booking) is to
correct only across *guardrail* metrics or *secondary* metrics, not the
primary; here we expose it as opt-in via the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import StatisticalCheckError


MetricKind = Literal["binary", "continuous"]


@dataclass
class MetricTestResult:
    metric: str
    kind: MetricKind
    n_control: int
    n_treatment: int
    control_mean: float
    treatment_mean: float
    absolute_diff: float
    relative_diff: float
    ci_low: float
    ci_high: float
    ci_method: str
    effect_size: float       # Cohen's h or d
    effect_size_kind: str
    primary_test: str
    primary_p_value: float
    secondary_test: str | None = None
    secondary_p_value: float | None = None
    # Populated by `apply_multiple_testing_correction`.
    adjusted_p_value: float | None = None
    adjustment_method: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def significant_at_5pct(self) -> bool:
        p = self.adjusted_p_value if self.adjusted_p_value is not None else self.primary_p_value
        return bool(p < 0.05)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _detect_kind(series: pd.Series) -> MetricKind:
    unique = pd.unique(series.dropna())
    if len(unique) == 2 and set(np.asarray(unique).astype(float).tolist()) <= {0.0, 1.0}:
        return "binary"
    return "continuous"


def metric_test(
    df: pd.DataFrame,
    metric: str,
    variant_column: str,
    control_label: str,
    treatment_label: str,
) -> MetricTestResult:
    """Run the appropriate metric-shift test for one metric column."""
    if metric not in df.columns:
        raise StatisticalCheckError(f"metric column '{metric}' not in DataFrame.")
    if variant_column not in df.columns:
        raise StatisticalCheckError(
            f"variant column '{variant_column}' not in DataFrame."
        )

    control = df.loc[df[variant_column] == control_label, metric].dropna()
    treatment = df.loc[df[variant_column] == treatment_label, metric].dropna()

    if len(control) < 30 or len(treatment) < 30:
        raise StatisticalCheckError(
            f"Sample too small for metric '{metric}' "
            f"(control={len(control)}, treatment={len(treatment)})."
        )

    if not np.issubdtype(control.dtype, np.number):
        raise StatisticalCheckError(f"metric '{metric}' is not numeric.")

    kind: MetricKind = _detect_kind(df[metric])

    if kind == "binary":
        return _binary_test(metric, control, treatment)
    return _continuous_test(metric, control, treatment)


# ---------- binary ---------------------------------------------------------

def _wilson_score_interval(s: float, n: float, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a single proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(1 - alpha / 2))
    p_hat = s / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return float(centre - half), float(centre + half)


def _newcombe_diff_ci(
    s1: float, n1: float, s2: float, n2: float, alpha: float = 0.05
) -> tuple[float, float]:
    """Newcombe (1998) hybrid-score CI for (p2 - p1).

    Method 10 in the original paper — converts Wilson intervals on the
    two individual proportions into an interval on the difference.
    Reference: Newcombe RG, Statistics in Medicine 17:873-890 (1998).
    """
    l1, u1 = _wilson_score_interval(s1, n1, alpha)
    l2, u2 = _wilson_score_interval(s2, n2, alpha)
    p1, p2 = s1 / n1, s2 / n2
    diff = p2 - p1
    # Newcombe's adjusted half-widths.
    lo = diff - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    return float(lo), float(hi)


def _binary_test(
    metric: str, control: pd.Series, treatment: pd.Series
) -> MetricTestResult:
    n_c, n_t = len(control), len(treatment)
    s_c, s_t = float(control.sum()), float(treatment.sum())
    p_c, p_t = s_c / n_c, s_t / n_t

    # Chi-square on the 2x2 contingency table = two-proportion z-test (squared).
    contingency = np.array(
        [
            [s_c, n_c - s_c],
            [s_t, n_t - s_t],
        ],
        dtype=float,
    )
    if (contingency < 5).any():
        raise StatisticalCheckError(
            f"Cells below 5 in 2x2 table for metric '{metric}'; chi-square is unreliable."
        )
    chi2, p_chi, _, _ = stats.chi2_contingency(contingency, correction=False)

    # Newcombe hybrid-score 95% CI on (p_t - p_c) — much better coverage
    # than Wald, especially near 0/1 (Newcombe 1998).
    diff = p_t - p_c
    ci_low, ci_high = _newcombe_diff_ci(s_c, n_c, s_t, n_t, alpha=0.05)

    # Cohen's h
    h = 2 * (np.arcsin(np.sqrt(p_t)) - np.arcsin(np.sqrt(p_c)))

    rel = diff / p_c if p_c > 0 else float("nan")

    return MetricTestResult(
        metric=metric,
        kind="binary",
        n_control=n_c,
        n_treatment=n_t,
        control_mean=p_c,
        treatment_mean=p_t,
        absolute_diff=diff,
        relative_diff=rel,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method="Newcombe hybrid-score (1998)",
        effect_size=float(h),
        effect_size_kind="Cohen's h",
        primary_test="chi-square (2x2)",
        primary_p_value=float(p_chi),
        notes=[f"chi2 = {chi2:.3f}"],
    )


# ---------- continuous -----------------------------------------------------

def _continuous_test(
    metric: str, control: pd.Series, treatment: pd.Series
) -> MetricTestResult:
    n_c, n_t = len(control), len(treatment)
    mean_c, mean_t = float(control.mean()), float(treatment.mean())
    var_c, var_t = float(control.var(ddof=1)), float(treatment.var(ddof=1))

    if var_c == 0.0 and var_t == 0.0:
        raise StatisticalCheckError(
            f"metric '{metric}' has zero variance in both groups."
        )

    # Welch's t-test (does NOT assume equal variances)
    t_stat, p_t = stats.ttest_ind(treatment, control, equal_var=False)

    # Mann-Whitney U as non-parametric robustness check
    try:
        u_stat, p_u = stats.mannwhitneyu(treatment, control, alternative="two-sided")
        u_note = f"U = {u_stat:.0f}"
    except ValueError as exc:
        u_stat, p_u = float("nan"), float("nan")
        u_note = f"Mann-Whitney failed: {exc}"

    # Welch-Satterthwaite degrees of freedom for the CI
    se = float(np.sqrt(var_c / n_c + var_t / n_t))
    df_w = (var_c / n_c + var_t / n_t) ** 2 / (
        (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
    )
    crit = float(stats.t.ppf(0.975, df_w))
    diff = mean_t - mean_c
    ci_low, ci_high = diff - crit * se, diff + crit * se

    # Pooled SD for Cohen's d
    pooled_sd = float(np.sqrt(((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2)))
    d = diff / pooled_sd if pooled_sd > 0 else float("nan")

    rel = diff / mean_c if mean_c != 0 else float("nan")

    return MetricTestResult(
        metric=metric,
        kind="continuous",
        n_control=n_c,
        n_treatment=n_t,
        control_mean=mean_c,
        treatment_mean=mean_t,
        absolute_diff=diff,
        relative_diff=rel,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method="Welch (1.96 * SE under Welch-Satterthwaite df)",
        effect_size=float(d),
        effect_size_kind="Cohen's d",
        primary_test="Welch's t-test",
        primary_p_value=float(p_t),
        secondary_test="Mann-Whitney U",
        secondary_p_value=float(p_u),
        notes=[f"t = {float(t_stat):.3f}", u_note],
    )


# ---------- multiple testing -----------------------------------------------

CorrectionMethod = Literal["bh", "bonferroni", "none"]


def apply_multiple_testing_correction(
    results: list[MetricTestResult],
    method: CorrectionMethod = "bh",
) -> list[MetricTestResult]:
    """Adjust the `primary_p_value` of each result for multiple testing,
    storing the adjusted value in `adjusted_p_value`.

    Methods
    -------
    "bh" : Benjamini-Hochberg FDR control at 5% (default).
    "bonferroni" : family-wise error control.
    "none" : no adjustment.
    """
    if not results:
        return results
    if method == "none":
        return results

    pvals = np.array([r.primary_p_value for r in results], dtype=float)
    m = len(pvals)

    if method == "bonferroni":
        adjusted = np.minimum(pvals * m, 1.0)
        for r, p in zip(results, adjusted, strict=True):
            r.adjusted_p_value = float(p)
            r.adjustment_method = "Bonferroni"
        return results

    if method == "bh":
        order = np.argsort(pvals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, m + 1)
        # BH adjusted p = min over k>=i of (p_(k) * m / k)
        sorted_p = pvals[order]
        scaled = sorted_p * m / np.arange(1, m + 1)
        # Enforce monotonicity from the largest p downward.
        adj_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
        adjusted = np.clip(adj_sorted[ranks - 1], 0.0, 1.0)
        for r, p in zip(results, adjusted, strict=True):
            r.adjusted_p_value = float(p)
            r.adjustment_method = "Benjamini-Hochberg FDR"
        return results

    raise ValueError(f"Unknown correction method: {method!r}")


# ---------- variance reduction (CUPED) -------------------------------------

@dataclass
class CUPEDResult:
    metric: str
    covariate: str
    theta: float
    variance_reduction: float    # 1 - Var(Y - theta*X) / Var(Y)
    adjusted_metric: str         # name of the new column added to the DataFrame


def apply_cuped(
    df: pd.DataFrame,
    metric: str,
    covariate: str,
    in_place: bool = False,
) -> tuple[pd.DataFrame, CUPEDResult]:
    """CUPED variance reduction (Deng et al., WSDM 2013).

    Computes the adjusted metric Y' = Y - theta * (X - mean(X)), where
    theta = Cov(Y, X) / Var(X), and X is a pre-experiment covariate.
    The mean of Y' equals the mean of Y in expectation, but the variance
    is reduced by a factor of (1 - rho^2) where rho is the correlation
    between Y and X.

    The adjusted column is added to the DataFrame so downstream
    `metric_test()` calls can use it directly.

    Reference: Deng, Xu, Kohavi, Walker, *Improving the Sensitivity of
    Online Controlled Experiments by Utilizing Pre-Experiment Data*,
    WSDM 2013.
    """
    if metric not in df.columns or covariate not in df.columns:
        raise StatisticalCheckError("metric or covariate missing for CUPED.")
    y = df[metric].astype(float).to_numpy()
    x = df[covariate].astype(float).to_numpy()
    if np.var(x, ddof=1) == 0.0:
        raise StatisticalCheckError(
            f"CUPED covariate '{covariate}' has zero variance."
        )
    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    y_adj = y - theta * (x - x.mean())
    var_red = 1.0 - float(np.var(y_adj, ddof=1) / np.var(y, ddof=1))

    out = df if in_place else df.copy()
    new_col = f"{metric}__cuped"
    out[new_col] = y_adj

    return out, CUPEDResult(
        metric=metric,
        covariate=covariate,
        theta=theta,
        variance_reduction=var_red,
        adjusted_metric=new_col,
    )
