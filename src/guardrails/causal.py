"""Propensity Score Matching (PSM) with sensitivity analysis.

The naive treatment effect is just the difference in outcome means between
the variant groups. When variant assignment is correlated with covariates
(broken randomisation, leaky feature flag, etc.) that estimate is biased.

PSM corrects for this by:

1. Fitting a logistic regression that predicts P(treated | covariates).
2. **Common-support trimming**: discarding treated units whose propensity
   lies outside the [min, max] of the control propensities (and vice
   versa). Outside that overlap region, no causal claim is defensible.
3. For each surviving treated unit, finding the control unit with the
   nearest propensity score (1-NN, **with replacement**) within a
   caliper of `caliper_sd` * SD(logit propensity).
4. Computing the average outcome difference across matched pairs —
   the Average Treatment effect on the Treated (ATT).
5. **Bootstrap SE on the ATT**: the matched-with-replacement design
   re-uses controls and so violates the i.i.d. assumption of the paired-t
   SE. We resample the *treated* units with replacement, re-match, and
   take the empirical SD of the ATT distribution. Abadie & Imbens (2006,
   *Econometrica*) is the canonical analytical alternative.
6. **Rosenbaum bounds** for hidden-bias sensitivity: how strongly would
   an unmeasured confounder have to bias treatment assignment (odds ratio
   Gamma) before the ATT's significance disappears? See Rosenbaum, 2002,
   *Observational Studies*.

We deliberately keep PSM scope-limited: continuous & one-hot-encoded
categorical covariates only, with caliper matching. That's the standard
recipe and is enough to demonstrate the agent corrects for confounding
when randomisation breaks.

References
----------
* Rosenbaum & Rubin, *Biometrika* 70:41 (1983).
* Austin, *Statistics in Medicine* 30:171 (2011).
* Abadie & Imbens, *Econometrica* 74:235 (2006).
* Rosenbaum, *Observational Studies*, Springer (2002).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..exceptions import StatisticalCheckError


@dataclass
class RosenbaumBounds:
    """Sensitivity of the matched ATT to a hidden binary confounder with
    odds ratio Gamma on treatment assignment. We report the smallest
    Gamma at which the upper-bound p-value (Wilcoxon signed-rank) exceeds
    0.05 — i.e. the strength of unmeasured bias required to overturn the
    finding.
    """
    gamma_grid: list[float]
    upper_p_values: list[float]
    gamma_critical: float | None  # smallest Gamma where p > 0.05


@dataclass
class PSMResult:
    metric: str
    covariates: list[str]
    n_treated: int
    n_control_pool: int
    n_matched_pairs: int
    naive_effect: float
    psm_att: float                          # average treatment effect on the treated
    psm_se_paired_t: float                  # naive paired-t SE (biased; for comparison)
    psm_se_bootstrap: float                 # block-bootstrap SE — recommended
    psm_ci_low: float                       # CI from bootstrap
    psm_ci_high: float
    psm_p_value: float                      # bootstrap two-sided p
    caliper_sd: float
    discarded_treated: int                  # outside caliper
    trimmed_treated: int                    # outside common support
    trimmed_control: int
    balance_before: dict[str, float]        # standardised mean diff per covariate
    balance_after: dict[str, float]
    rosenbaum: RosenbaumBounds | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _design_matrix(
    df: pd.DataFrame, covariates: list[str]
) -> tuple[np.ndarray, list[str]]:
    """One-hot-encode categorical covariates and return a numeric matrix."""
    parts: list[pd.DataFrame] = []
    names: list[str] = []
    for col in covariates:
        if col not in df.columns:
            raise StatisticalCheckError(f"covariate '{col}' not in DataFrame.")
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            parts.append(s.to_frame())
            names.append(col)
        else:
            dummies = pd.get_dummies(s, prefix=col, drop_first=True, dtype=float)
            parts.append(dummies)
            names.extend(dummies.columns.tolist())
    if not parts:
        raise StatisticalCheckError("No covariates provided for PSM.")
    X = pd.concat(parts, axis=1).to_numpy(dtype=float)
    return X, names


def _smd(x: np.ndarray, t: np.ndarray, names: list[str]) -> dict[str, float]:
    """Standardised mean difference per column, treated vs. control."""
    treated, control = x[t == 1], x[t == 0]
    out: dict[str, float] = {}
    for j in range(x.shape[1]):
        m_t, m_c = treated[:, j].mean(), control[:, j].mean()
        s_t, s_c = treated[:, j].std(ddof=1), control[:, j].std(ddof=1)
        pooled = np.sqrt((s_t**2 + s_c**2) / 2.0)
        key = names[j] if j < len(names) else str(j)
        out[key] = float((m_t - m_c) / pooled) if pooled > 0 else 0.0
    return out


def _match_1nn_with_caliper(
    logit_t: np.ndarray, logit_c: np.ndarray, caliper: float
) -> tuple[list[int], list[int], int]:
    """Match each treated index to its nearest control by logit propensity,
    within `caliper`. Indices are *positional* within `logit_t`/`logit_c`.

    Returns (treated_pos, matched_control_pos, discarded_count).
    """
    order = np.argsort(logit_c)
    sorted_c = logit_c[order]
    treated_pos: list[int] = []
    control_pos: list[int] = []
    discarded = 0
    for i, logit_val in enumerate(logit_t):
        pos = int(np.searchsorted(sorted_c, logit_val))
        # Candidates are the two neighbours around pos.
        candidates = []
        if pos < len(sorted_c):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        best: tuple[float, int] | None = None
        for c in candidates:
            dist = abs(sorted_c[c] - logit_val)
            if dist <= caliper and (best is None or dist < best[0]):
                best = (dist, int(order[c]))
        if best is None:
            discarded += 1
            continue
        treated_pos.append(i)
        control_pos.append(best[1])
    return treated_pos, control_pos, discarded


def _rosenbaum_bounds(
    pair_diffs: np.ndarray,
    gamma_grid: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0),
) -> RosenbaumBounds:
    """Wilcoxon signed-rank Rosenbaum bounds.

    For each Γ, computes the upper-bound TWO-SIDED p-value under the
    worst-case pattern of hidden bias. The bound is the maximum of the
    two one-sided p-values evaluated under the two worst-case hidden-bias
    directions:

      p_+ = Γ / (1 + Γ)   (treated more likely to be the "winner")
      p_+ = 1 / (1 + Γ)   (treated less likely to be the "winner")

    Doubling the larger one-sided p gives the two-sided upper bound,
    which is monotonically non-decreasing in Γ. When that bound first
    exceeds 0.05, the corresponding Γ is the critical sensitivity
    threshold.
    """
    d = pair_diffs[pair_diffs != 0]
    n = len(d)
    if n < 5:
        return RosenbaumBounds([], [], None)
    ranks = stats.rankdata(np.abs(d))
    positive = (d > 0).astype(int)
    T = float((positive * ranks).sum())
    total = float(ranks.sum())
    sumsq = float((ranks**2).sum())

    def _one_sided_p(p_plus: float, observed: float) -> float:
        mean = p_plus * total
        var = p_plus * (1 - p_plus) * sumsq
        if var <= 0:
            return 1.0
        z = (observed - mean) / np.sqrt(var)
        # One-sided p in the direction of the observed effect.
        # For an observed positive Wilcoxon T (treated > control), we
        # care about Pr(T >= observed) — i.e. survival.
        return float(stats.norm.sf(z))

    upper_ps: list[float] = []
    critical: float | None = None
    for gamma in gamma_grid:
        p_plus_high = gamma / (1 + gamma)
        p_plus_low = 1.0 / (1 + gamma)
        p_high = _one_sided_p(p_plus_high, T)
        p_low = _one_sided_p(p_plus_low, T)
        # Worst-case two-sided p-value is 2 * max one-sided p, capped at 1.
        two_sided = float(min(1.0, 2.0 * max(p_high, p_low)))
        upper_ps.append(two_sided)
        if critical is None and two_sided > 0.05:
            critical = float(gamma)
    return RosenbaumBounds(list(gamma_grid), upper_ps, critical)


def propensity_score_match(
    df: pd.DataFrame,
    metric: str,
    variant_column: str,
    control_label: str,
    treatment_label: str,
    covariates: list[str],
    caliper_sd: float = 0.2,
    bootstrap_samples: int = 500,
    rosenbaum_grid: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
    random_state: int = 0,
) -> PSMResult:
    """Estimate the ATT via 1-NN PSM with caliper, common-support trimming,
    bootstrap SE, and Rosenbaum sensitivity analysis.

    Raises :class:`StatisticalCheckError` when the data is unfit for PSM
    (e.g. perfect separation, no matchable treated units).
    """
    if not covariates:
        raise StatisticalCheckError("PSM requires at least one covariate.")
    if metric not in df.columns:
        raise StatisticalCheckError(f"metric '{metric}' not in DataFrame.")

    work = df.dropna(subset=[metric, variant_column, *covariates]).copy()
    mask = work[variant_column].isin([control_label, treatment_label])
    work = work.loc[mask].reset_index(drop=True)

    t = (work[variant_column] == treatment_label).astype(int).to_numpy()
    y = work[metric].to_numpy(dtype=float)
    X_raw, dummy_names = _design_matrix(work, covariates)

    if t.sum() == 0 or (1 - t).sum() == 0:
        raise StatisticalCheckError("Need both treated and control rows for PSM.")
    if len(np.unique(t)) < 2:
        raise StatisticalCheckError("variant column collapses to one level after filtering.")

    # --- propensity model ---------------------------------------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    try:
        model = LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=random_state
        )
        model.fit(X, t)
    except Exception as exc:
        raise StatisticalCheckError(f"Propensity model failed to fit: {exc}") from exc

    p_hat = model.predict_proba(X)[:, 1]
    p_clip = np.clip(p_hat, 1e-6, 1 - 1e-6)
    logit = np.log(p_clip / (1 - p_clip))

    # --- common support trimming -------------------------------------------
    # Drop treated units with p above max(p_control), and control units with
    # p below min(p_treated). This is the standard Crump/Imbens-style
    # support restriction, applied symmetrically.
    p_treated = p_hat[t == 1]
    p_control = p_hat[t == 0]
    p_low, p_high = float(p_treated.min()), float(p_control.max())
    in_support = (p_hat >= p_low) & (p_hat <= p_high)
    trimmed_treated_count = int(((t == 1) & ~in_support).sum())
    trimmed_control_count = int(((t == 0) & ~in_support).sum())

    if not in_support.any():
        raise StatisticalCheckError(
            "No units inside the common-support region — covariate "
            "distributions do not overlap."
        )

    # `work_s` is intentionally elided — once we have aligned numpy views,
    # we don't need the dropped-DataFrame any further in this function.
    t_s = t[in_support]
    y_s = y[in_support]
    X_raw_s = X_raw[in_support]
    logit_s = logit[in_support]

    if t_s.sum() == 0 or (1 - t_s).sum() == 0:
        raise StatisticalCheckError(
            "Common-support trimming removed all of one arm — refusing to match."
        )

    caliper = caliper_sd * float(np.std(logit_s, ddof=1))

    # --- 1-NN matching (with replacement) ----------------------------------
    treated_idx = np.where(t_s == 1)[0]
    control_idx = np.where(t_s == 0)[0]
    logit_t = logit_s[treated_idx]
    logit_c = logit_s[control_idx]
    t_pos, c_pos, discarded = _match_1nn_with_caliper(logit_t, logit_c, caliper)
    if not t_pos:
        raise StatisticalCheckError(
            f"No treated unit found a control match within caliper "
            f"({caliper:.4f}) after common-support trimming."
        )

    matched_treated_idx = treated_idx[t_pos]
    matched_control_idx = control_idx[c_pos]
    pair_diffs = y_s[matched_treated_idx] - y_s[matched_control_idx]
    att = float(pair_diffs.mean())

    # --- SE: naive paired-t (biased) AND bootstrap (recommended) ----------
    se_paired = float(pair_diffs.std(ddof=1) / np.sqrt(len(pair_diffs)))

    rng = np.random.default_rng(random_state)
    n_treated_s = len(logit_t)
    # Adaptive resample count: matching is O(n_treated) per resample, so on a
    # 250k-treated dataset 500 resamples is needlessly slow. Cap the total
    # matching work at a fixed budget; never go below 100 resamples, which is
    # still enough for a stable bootstrap SE / percentile CI.
    _WORK_BUDGET = 20_000_000
    effective_bootstrap = bootstrap_samples
    if n_treated_s * bootstrap_samples > _WORK_BUDGET:
        effective_bootstrap = max(100, _WORK_BUDGET // max(n_treated_s, 1))

    boot_atts = np.empty(effective_bootstrap)
    for b in range(effective_bootstrap):
        idx_b = rng.integers(0, n_treated_s, size=n_treated_s)
        tb, cb, _ = _match_1nn_with_caliper(logit_t[idx_b], logit_c, caliper)
        if not tb:
            boot_atts[b] = np.nan
            continue
        # Recover original positions of resampled treated.
        boot_pair_diffs = y_s[treated_idx[idx_b[tb]]] - y_s[control_idx[cb]]
        boot_atts[b] = float(boot_pair_diffs.mean())
    boot_atts = boot_atts[~np.isnan(boot_atts)]
    if len(boot_atts) < 2:
        raise StatisticalCheckError("Bootstrap produced too few valid resamples.")
    se_boot = float(boot_atts.std(ddof=1))
    ci_low = float(np.quantile(boot_atts, 0.025))
    ci_high = float(np.quantile(boot_atts, 0.975))
    # Two-sided bootstrap p-value: fraction crossing zero, doubled.
    p_one = float((boot_atts <= 0).mean() if att > 0 else (boot_atts >= 0).mean())
    p_value = float(min(1.0, 2 * p_one))

    # --- Naive contrast ----------------------------------------------------
    naive_effect = float(y[t == 1].mean() - y[t == 0].mean())

    # --- Balance diagnostics ------------------------------------------------
    smd_before = _smd(X_raw, t, dummy_names)
    stacked_x = np.vstack([X_raw_s[matched_treated_idx], X_raw_s[matched_control_idx]])
    stacked_t = np.array([1] * len(t_pos) + [0] * len(t_pos))
    smd_after = _smd(stacked_x, stacked_t, dummy_names)

    # --- Sensitivity analysis ----------------------------------------------
    rosenbaum = _rosenbaum_bounds(pair_diffs, rosenbaum_grid)

    return PSMResult(
        metric=metric,
        covariates=covariates,
        n_treated=int(t.sum()),
        n_control_pool=int((1 - t).sum()),
        n_matched_pairs=len(t_pos),
        naive_effect=naive_effect,
        psm_att=att,
        psm_se_paired_t=se_paired,
        psm_se_bootstrap=se_boot,
        psm_ci_low=ci_low,
        psm_ci_high=ci_high,
        psm_p_value=p_value,
        caliper_sd=float(caliper_sd),
        discarded_treated=int(discarded),
        trimmed_treated=trimmed_treated_count,
        trimmed_control=trimmed_control_count,
        balance_before=smd_before,
        balance_after=smd_after,
        rosenbaum=rosenbaum,
        notes=[
            f"caliper (logit units) = {caliper:.4f}",
            f"bootstrap resamples = {effective_bootstrap}",
        ],
    )
