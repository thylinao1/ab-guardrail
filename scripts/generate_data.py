"""
Generate two synthetic A/B-test datasets for the experimentation guardrail agent.

1. clean_experiment.csv
   - 50/50 allocation between control and treatment
   - No confounding between covariates and variant
   - Small but real treatment effect on conversion + revenue

2. compromised_experiment.csv
   - 45/55 allocation between control and treatment  (Sample Ratio Mismatch)
   - Treatment users skew toward higher pre_signup_value
     (the variant assignment is correlated with a covariate, breaking
      the random-assignment assumption)
   - The *naive* lift looks huge; once we propensity-match on the covariates,
     most or all of the lift evaporates.

Both files share the same schema, so the agent's schema-inference logic
sees a consistent shape regardless of which one is fed in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA_COLUMNS = [
    "user_id",
    "variant",
    "pre_signup_value",   # covariate (USD spent in prior 30 days)
    "device",              # covariate (mobile / desktop)
    "country",             # covariate
    "converted",           # primary metric (binary)
    "revenue",             # secondary metric (continuous, $)
]


def _make_users(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate baseline user covariates that are independent of variant."""
    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "pre_signup_value": np.clip(
                rng.gamma(shape=2.0, scale=15.0, size=n), 0, 500
            ).round(2),
            "device": rng.choice(
                ["mobile", "desktop"], size=n, p=[0.65, 0.35]
            ),
            "country": rng.choice(
                ["SG", "MY", "ID", "PH", "VN"],
                size=n,
                p=[0.30, 0.20, 0.25, 0.15, 0.10],
            ),
        }
    )


def _outcome(
    p_base: np.ndarray,
    treated: np.ndarray,
    lift: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (converted, revenue) for a user given baseline propensity + treatment lift."""
    p = np.clip(p_base + treated * lift, 0.01, 0.99)
    converted = rng.binomial(1, p)
    # Revenue is zero if not converted, otherwise a noisy positive draw
    # with a small treatment uplift.
    revenue = np.where(
        converted == 1,
        np.clip(
            rng.gamma(shape=2.0, scale=12.0, size=len(p))
            + treated * 3.0,
            0,
            500,
        ),
        0.0,
    ).round(2)
    return converted, revenue


def generate_clean(n: int = 12_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = _make_users(n, rng)

    # Random 50/50 assignment, independent of everything else
    variant = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])
    treated = (variant == "treatment").astype(int)

    # Baseline conversion probability driven by covariates, no leakage
    p_base = (
        0.08
        + 0.0010 * users["pre_signup_value"].values   # higher spenders convert more
        + 0.02 * (users["device"].values == "desktop")
    )
    converted, revenue = _outcome(p_base, treated, lift=0.04, rng=rng)

    out = users.copy()
    out["variant"] = variant
    out["converted"] = converted
    out["revenue"] = revenue
    return out[SCHEMA_COLUMNS]


def generate_compromised(n: int = 12_000, seed: int = 7) -> pd.DataFrame:
    """
    Two things are wrong with this experiment:

    1. SRM — the assignment rate is 45/55 instead of 50/50 (e.g. a broken
       feature flag dropped some control users).
    2. Confounding — the assignment probability depends on
       pre_signup_value AND device, so treatment systematically gets
       higher-value users. Naive comparisons will overstate the lift.
    """
    rng = np.random.default_rng(seed)
    users = _make_users(n, rng)

    # Variant assignment is NOT random — it leaks from covariates.
    propensity = (
        0.45
        + 0.0020 * users["pre_signup_value"].values
        + 0.10 * (users["device"].values == "desktop")
    )
    propensity = np.clip(propensity, 0.05, 0.95)
    # Bias the global rate so we also get an SRM
    propensity = propensity * 1.12
    propensity = np.clip(propensity, 0.05, 0.95)

    treated = rng.binomial(1, propensity)
    variant = np.where(treated == 1, "treatment", "control")

    # The TRUE treatment effect here is essentially zero — any apparent lift
    # comes from the confounding above.
    p_base = (
        0.08
        + 0.0010 * users["pre_signup_value"].values
        + 0.02 * (users["device"].values == "desktop")
    )
    converted, revenue = _outcome(p_base, treated, lift=0.0, rng=rng)

    out = users.copy()
    out["variant"] = variant
    out["converted"] = converted
    out["revenue"] = revenue
    return out[SCHEMA_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Where to write the CSV files.",
    )
    parser.add_argument("--n", type=int, default=12_000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    clean = generate_clean(n=args.n)
    compromised = generate_compromised(n=args.n)

    clean_path = args.out_dir / "clean_experiment.csv"
    comp_path = args.out_dir / "compromised_experiment.csv"
    clean.to_csv(clean_path, index=False)
    compromised.to_csv(comp_path, index=False)

    print(f"Wrote {clean_path}  ({len(clean):,} rows)")
    print(f"Wrote {comp_path}  ({len(compromised):,} rows)")
    print("\nClean split:")
    print(clean["variant"].value_counts(normalize=True).round(4))
    print("\nCompromised split:")
    print(compromised["variant"].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()
