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


def write_messy_experiment(path: Path, n: int = 12_000, seed: int = 19) -> dict:
    """Write messy_experiment.csv: a properly randomised test wrapped in the
    data-quality pathologies a real e-commerce export carries.

    The underlying experiment is clean (50/50, a real conversion lift), so a
    pipeline that correctly *cleans* the file should still return SAFE TO
    ROLL OUT. What is deliberately broken is the file, not the experiment:

    * ~8% missing values in the `pre_signup_value` covariate (consent gates,
      late joiners)
    * ~5% missing values in the `device` covariate
    * a handful of dirty numeric tokens ("ERROR", "NULL") in
      `pre_signup_value`, forcing the column to text dtype on read
    * an all-null `experiment_notes` column (instrumented, never populated)
    * ~25 exact-duplicate rows (an at-least-once logging pipeline)
    * 3 malformed rows with the wrong field count (a broken log writer)

    Returns a dict of the injected-defect counts for the caller to print.
    """
    rng = np.random.default_rng(seed)
    users = _make_users(n, rng)
    variant = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])
    treated = (variant == "treatment").astype(int)
    p_base = (
        0.08
        + 0.0010 * users["pre_signup_value"].values
        + 0.02 * (users["device"].values == "desktop")
    )
    converted, revenue = _outcome(p_base, treated, lift=0.04, rng=rng)

    df = users.copy()
    df["variant"] = variant
    df["converted"] = converted
    df["revenue"] = revenue
    df = df[SCHEMA_COLUMNS].copy()

    # --- inject data-quality defects ---------------------------------------
    # pre_signup_value: ~8% missing
    miss_pre = rng.choice(n, size=int(0.08 * n), replace=False)
    df.loc[miss_pre, "pre_signup_value"] = np.nan
    # pre_signup_value: a few dirty tokens (forces object dtype on read)
    dirty_idx = rng.choice(
        [i for i in range(n) if i not in set(miss_pre)], size=6, replace=False
    )
    df["pre_signup_value"] = df["pre_signup_value"].astype(object)
    for j, i in enumerate(dirty_idx):
        df.loc[i, "pre_signup_value"] = "ERROR" if j % 2 == 0 else "NULL"
    # device: ~5% missing
    miss_dev = rng.choice(n, size=int(0.05 * n), replace=False)
    df.loc[miss_dev, "device"] = np.nan
    # an all-null instrumented column
    df["experiment_notes"] = np.nan

    # ~25 exact-duplicate rows (double-logged events)
    n_dupes = 25
    dupe_rows = df.iloc[rng.choice(n, size=n_dupes, replace=False)]
    df = pd.concat([df, dupe_rows], ignore_index=True)

    # --- serialise, then splice in malformed lines -------------------------
    # All malformed lines carry MORE fields than the header so pandas'
    # on_bad_lines="skip" cleanly drops them. (Lines with *fewer* fields are
    # NaN-padded rather than skipped, which would leak junk into the data.)
    csv_text = df.to_csv(index=False)
    lines = csv_text.splitlines()
    header = lines[0]
    n_fields = header.count(",") + 1
    malformed = [
        ",".join(["999999", "treatment"] + ["junk"] * (n_fields + 1)),
        ",".join(["999998", "control"] + ["x"] * (n_fields + 4)),
        ",".join(["not", "a", "valid", "row"] + ["extra"] * n_fields),
    ]
    # insert the malformed lines at spread-out positions in the body
    body = lines[1:]
    for offset, bad in zip((1000, 5000, 9000), malformed, strict=True):
        body.insert(min(offset, len(body)), bad)
    path.write_text("\n".join([header, *body]) + "\n", encoding="utf-8")

    return {
        "rows_written": len(body) + 1,            # incl. malformed, excl. header
        "duplicates": n_dupes,
        "malformed": len(malformed),
        "missing_pre_signup_value": len(miss_pre) + len(dirty_idx),
        "missing_device": len(miss_dev),
    }


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
    messy_path = args.out_dir / "messy_experiment.csv"
    clean.to_csv(clean_path, index=False)
    compromised.to_csv(comp_path, index=False)
    messy_stats = write_messy_experiment(messy_path, n=args.n)

    print(f"Wrote {clean_path}  ({len(clean):,} rows)")
    print(f"Wrote {comp_path}  ({len(compromised):,} rows)")
    print(f"Wrote {messy_path}  ({messy_stats['rows_written']:,} rows incl. defects)")
    print("\nClean split:")
    print(clean["variant"].value_counts(normalize=True).round(4))
    print("\nCompromised split:")
    print(compromised["variant"].value_counts(normalize=True).round(4))
    print("\nMessy dataset injected defects:")
    for k, v in messy_stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
