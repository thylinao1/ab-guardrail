"""Tests for the metric-shift tests and multiple-testing correction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.guardrails import metric_test
from src.guardrails.metric_tests import (
    apply_cuped,
    apply_multiple_testing_correction,
)


def test_continuous_metric_detects_real_shift(balanced_experiment: pd.DataFrame):
    r = metric_test(balanced_experiment, "revenue", "variant", "control", "treatment")
    assert r.kind == "continuous"
    assert r.primary_test == "Welch's t-test"
    assert r.absolute_diff > 0
    assert r.primary_p_value < 0.05
    # 95% CI should not contain zero when there's a real lift.
    assert not (r.ci_low <= 0 <= r.ci_high)
    # Mann-Whitney also returned.
    assert r.secondary_p_value is not None


def test_binary_metric_uses_newcombe_ci(balanced_experiment: pd.DataFrame):
    r = metric_test(balanced_experiment, "converted", "variant", "control", "treatment")
    assert r.kind == "binary"
    assert "Newcombe" in r.ci_method
    assert r.primary_p_value < 0.05
    # CI should be tighter and not zero-containing for a real lift this size.
    assert r.ci_low > 0


def test_bh_correction_increases_pvalues_monotonically():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "variant": rng.choice(["a", "b"], size=400, p=[0.5, 0.5]),
            "metric": rng.normal(size=400),
        }
    )
    # Make several copies with random shifts so we have a spread of p-values.
    df["m1"] = df["metric"]
    df["m2"] = df["metric"] + 0.3 * (df["variant"] == "b")
    df["m3"] = df["metric"] + 0.1 * (df["variant"] == "b")
    results = [
        metric_test(df, m, "variant", "a", "b") for m in ["m1", "m2", "m3"]
    ]
    apply_multiple_testing_correction(results, method="bh")
    for r in results:
        assert r.adjusted_p_value is not None
        assert r.adjusted_p_value >= r.primary_p_value - 1e-12
        assert r.adjustment_method == "Benjamini-Hochberg FDR"


def test_bonferroni_correction():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "variant": rng.choice(["a", "b"], size=400, p=[0.5, 0.5]),
            "m1": rng.normal(size=400),
            "m2": rng.normal(size=400),
        }
    )
    results = [metric_test(df, m, "variant", "a", "b") for m in ["m1", "m2"]]
    apply_multiple_testing_correction(results, method="bonferroni")
    for r in results:
        assert r.adjusted_p_value is not None
        # Bonferroni multiplies by m; for m=2 the adjusted should be 2× raw, capped at 1.
        assert r.adjusted_p_value <= min(1.0, 2 * r.primary_p_value) + 1e-12
        assert r.adjustment_method == "Bonferroni"


def test_cuped_reduces_variance(balanced_experiment: pd.DataFrame):
    out, cuped = apply_cuped(balanced_experiment, "revenue", "pre_signup_value")
    assert cuped.variance_reduction > 0.0
    # The adjusted column should exist and have lower variance than the original.
    assert cuped.adjusted_metric in out.columns
    assert out[cuped.adjusted_metric].var() < out["revenue"].var()
