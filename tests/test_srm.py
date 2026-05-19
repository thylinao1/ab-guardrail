"""Tests for the chi-square SRM check."""

from __future__ import annotations

import pandas as pd
import pytest

from src.exceptions import StatisticalCheckError
from src.guardrails import srm_check


def test_balanced_50_50_passes(balanced_experiment: pd.DataFrame):
    result = srm_check(balanced_experiment, "variant")
    assert not result.srm_detected
    assert result.p_value > 0.01


def test_unbalanced_40_60_fires(srm_experiment: pd.DataFrame):
    result = srm_check(srm_experiment, "variant")
    assert result.srm_detected
    assert result.p_value < 1e-6
    # observed proportions should reflect the true 40/60 generation
    obs = result.observed_ratio
    assert abs(obs["control"] - 0.40) < 0.03
    assert abs(obs["treatment"] - 0.60) < 0.03


def test_expected_ratio_validation():
    df = pd.DataFrame({"variant": ["a"] * 100 + ["b"] * 100})
    # Sums to != 1.0
    with pytest.raises(StatisticalCheckError, match="sum to 1"):
        srm_check(df, "variant", expected_ratio={"a": 0.3, "b": 0.3})


def test_missing_variant_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(StatisticalCheckError, match="not in DataFrame"):
        srm_check(df, "variant")


def test_single_level_variant_rejected():
    df = pd.DataFrame({"variant": ["a"] * 100})
    with pytest.raises(StatisticalCheckError, match="fewer than 2 levels"):
        srm_check(df, "variant")


def test_small_cells_rejected():
    df = pd.DataFrame({"variant": ["a", "a", "b", "b", "b"]})
    with pytest.raises(StatisticalCheckError, match="Expected counts below 5"):
        srm_check(df, "variant")
