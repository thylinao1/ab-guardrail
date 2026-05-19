"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Make `src` importable for tests run from the project root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def balanced_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """5,000-user clean A/B test with real 2pp lift and no confounding."""
    n = 5_000
    variant = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])
    treated = (variant == "treatment").astype(int)
    pre = np.clip(rng.gamma(2, 15, size=n), 0, 500).round(2)
    device = rng.choice(["mobile", "desktop"], size=n, p=[0.65, 0.35])
    p_base = 0.08 + 0.001 * pre + 0.02 * (device == "desktop")
    p = np.clip(p_base + treated * 0.04, 0.01, 0.99)
    converted = rng.binomial(1, p)
    revenue = np.where(
        converted == 1,
        np.clip(rng.gamma(2, 12, size=n) + treated * 3.0, 0, 500),
        0.0,
    ).round(2)
    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "variant": variant,
            "pre_signup_value": pre,
            "device": device,
            "converted": converted.astype(int),
            "revenue": revenue,
        }
    )


@pytest.fixture
def srm_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """Experiment with a deliberate 40/60 split — should fire SRM."""
    n = 5_000
    variant = rng.choice(["control", "treatment"], size=n, p=[0.4, 0.6])
    converted = rng.binomial(1, 0.10, size=n)
    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "variant": variant,
            "converted": converted.astype(int),
        }
    )


@pytest.fixture
def confounded_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """Experiment where variant assignment leaks from pre_signup_value.

    The true treatment effect on `revenue` is zero, but treatment users
    systematically have higher `pre_signup_value`, which itself drives
    revenue — so the naive comparison shows a large positive lift.
    PSM matching on pre_signup_value should shrink the estimate toward 0.
    """
    n = 5_000
    pre = np.clip(rng.gamma(2, 25, size=n), 0, 500).round(2)
    # Strong confounding: propensity ranges roughly 0.30 to 0.90 in pre.
    propensity = np.clip(0.30 + 0.010 * pre, 0.05, 0.95)
    treated = rng.binomial(1, propensity)
    variant = np.where(treated == 1, "treatment", "control")
    # Revenue strongly driven by pre_signup_value; no treatment effect.
    revenue = np.clip(
        rng.gamma(2, 10, size=n) + 0.15 * pre, 0, 500
    ).round(2)
    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "variant": variant,
            "pre_signup_value": pre,
            "revenue": revenue,
        }
    )
