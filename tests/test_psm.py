"""Tests for propensity score matching, bootstrap inference, and Rosenbaum bounds."""

from __future__ import annotations

import pandas as pd
import pytest

from src.exceptions import StatisticalCheckError
from src.guardrails import propensity_score_match


def test_psm_recovers_zero_effect_under_confounding(confounded_experiment: pd.DataFrame):
    """In confounded_experiment, the TRUE effect on revenue is zero but the
    naive estimate is non-zero. PSM should shrink the estimate toward zero."""
    result = propensity_score_match(
        confounded_experiment,
        metric="revenue",
        variant_column="variant",
        control_label="control",
        treatment_label="treatment",
        covariates=["pre_signup_value"],
        bootstrap_samples=200,
    )
    # Naive estimate should be substantially positive due to confounding.
    assert result.naive_effect > 0
    # PSM ATT should be much smaller in magnitude than naive.
    assert abs(result.psm_att) < abs(result.naive_effect) * 0.6
    # Bootstrap CI should exist and be finite.
    assert result.psm_ci_low < result.psm_ci_high
    # Common-support trimming may or may not fire on this data; field should populate.
    assert result.trimmed_treated >= 0
    assert result.trimmed_control >= 0
    # Bootstrap SE should be reported alongside the (biased) paired-t SE.
    assert result.psm_se_bootstrap > 0
    assert result.psm_se_paired_t > 0


def test_psm_balance_improves_after_matching(confounded_experiment: pd.DataFrame):
    result = propensity_score_match(
        confounded_experiment,
        metric="revenue",
        variant_column="variant",
        control_label="control",
        treatment_label="treatment",
        covariates=["pre_signup_value"],
        bootstrap_samples=100,
    )
    # The dominant covariate's |SMD| should drop after matching.
    smd_before = abs(result.balance_before["pre_signup_value"])
    smd_after = abs(result.balance_after["pre_signup_value"])
    assert smd_after < smd_before


def test_psm_rosenbaum_bounds_populated(balanced_experiment: pd.DataFrame):
    result = propensity_score_match(
        balanced_experiment,
        metric="revenue",
        variant_column="variant",
        control_label="control",
        treatment_label="treatment",
        covariates=["pre_signup_value", "device"],
        bootstrap_samples=100,
    )
    assert result.rosenbaum is not None
    assert len(result.rosenbaum.gamma_grid) >= 3
    # Upper-bound p-values are monotonically non-decreasing in gamma.
    ps = result.rosenbaum.upper_p_values
    for a, b in zip(ps, ps[1:], strict=False):
        assert a <= b + 1e-9


def test_psm_rejects_no_covariates(balanced_experiment: pd.DataFrame):
    with pytest.raises(StatisticalCheckError, match="at least one covariate"):
        propensity_score_match(
            balanced_experiment,
            metric="revenue",
            variant_column="variant",
            control_label="control",
            treatment_label="treatment",
            covariates=[],
        )


def test_psm_handles_unbalanced_arms(srm_experiment: pd.DataFrame):
    """The SRM dataset has no covariates, so we synthesize one and confirm
    PSM still runs without crashing."""
    df = srm_experiment.copy()
    # Add a constant-ish covariate (varies enough not to be degenerate).
    rng = pd.Series(range(len(df))).mod(3).rename("device")
    df["device"] = rng.values
    result = propensity_score_match(
        df,
        metric="converted",
        variant_column="variant",
        control_label="control",
        treatment_label="treatment",
        covariates=["device"],
        bootstrap_samples=50,
    )
    assert result.n_matched_pairs > 0
