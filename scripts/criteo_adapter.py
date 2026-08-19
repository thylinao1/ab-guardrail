"""Adapter: Criteo Uplift Modeling dataset -> ab-guardrail schema.

The Criteo Uplift dataset is a real, large-scale (~13.98M-row) randomised
advertising experiment released by Criteo. It is the closest public stand-in
for the kind of e-commerce A/B log this tool is built for, and it is far
messier than the synthetic demos: 12 anonymised numeric features, heavy
class imbalance, and a treatment column encoded as 0/1.

This script converts the raw Criteo CSV into a CSV that ab-guardrail can
consume directly.

Raw Criteo schema
-----------------
    f0 ... f11    12 anonymised numeric features (pre-treatment covariates)
    treatment     0 / 1   - randomised ad-targeting assignment
    conversion    0 / 1   - did the user convert (primary outcome)
    visit         0 / 1   - did the user visit       (secondary outcome)
    exposure      0 / 1   - did the user actually see an ad (dropped:
                            it is post-randomisation and not an outcome
                            we want the agent to test)

Mapping applied
---------------
    treatment 0/1     ->  variant column with string labels
                          0 -> "control", 1 -> "treatment"
                          (string labels matter: the loader deliberately
                           excludes numeric 0/1 columns from variant
                           candidates so an outcome can't be mistaken for
                           the assignment.)
    conversion        ->  primary metric   (kept as 0/1)
    visit             ->  secondary metric (kept as 0/1)
    f0 ... f11        ->  covariates for propensity score matching
    exposure          ->  dropped

Why a separate adapter
----------------------
ab-guardrail's loader cleans malformed rows, duplicates and dirty numerics,
but it will not *reinterpret semantics*. It cannot know that Criteo's
`treatment` is the variant, or that `exposure` must be excluded. Semantic
mapping is a per-dataset job and belongs in an adapter, not in the tool.

Get the data
------------
The dataset is distributed by Criteo at
https://ailab.criteo.com/criteo-uplift-prediction-dataset/  (~297 MB gzipped,
free, registration-free). Download `criteo-uplift-v2.1.csv.gz`, gunzip it,
then:

    python scripts/criteo_adapter.py /path/to/criteo-uplift-v2.1.csv \
        --out data/criteo_ready.csv --sample 300000

    ab-guardrail data/criteo_ready.csv --mode pipeline

Sampling is recommended: PSM with a bootstrap on 13.98M rows is slow. A
stratified 300k-row sample preserves the treatment split and the (low)
conversion rate while keeping a run to a few seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_FEATURES = [f"f{i}" for i in range(12)]
RAW_REQUIRED = {*RAW_FEATURES, "treatment", "conversion", "visit"}
OUTPUT_COLUMNS = ["variant", "conversion", "visit", *RAW_FEATURES]


def adapt_criteo(
    raw_path: str | Path,
    out_path: str | Path,
    sample: int | None = 300_000,
    chunksize: int = 1_000_000,
    seed: int = 0,
) -> dict:
    """Convert the raw Criteo Uplift CSV into an ab-guardrail-ready CSV.

    Parameters
    ----------
    raw_path : path to the un-gzipped Criteo CSV.
    out_path : where to write the adapted CSV.
    sample   : if set, take a stratified random sample of this many rows
               (stratified on the treatment column so the split is held).
               Pass None to keep every row.
    chunksize: rows per read chunk; keeps memory flat on the 13.98M-row file.
    seed     : RNG seed for reproducible sampling.

    Returns a dict of summary stats.
    """
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Criteo CSV not found: {raw_path}\n"
            "Download criteo-uplift-v2.1.csv.gz from "
            "https://ailab.criteo.com/criteo-uplift-prediction-dataset/, "
            "gunzip it, and point this script at the result."
        )

    rng = np.random.default_rng(seed)
    kept_chunks: list[pd.DataFrame] = []
    total_rows = 0
    malformed_skipped = 0

    # Sampling probability per row, so the final size is ~`sample` without
    # holding the whole file in memory.
    # First, do a fast pass to count rows (cheap: just len of each chunk).
    reader = pd.read_csv(raw_path, chunksize=chunksize, on_bad_lines="skip")
    file_rows = 0
    for chunk in reader:
        file_rows += len(chunk)
    keep_frac = 1.0 if (sample is None or sample >= file_rows) else sample / file_rows

    reader = pd.read_csv(raw_path, chunksize=chunksize, on_bad_lines="skip")
    first = True
    for chunk in reader:
        if first:
            missing = RAW_REQUIRED - set(chunk.columns)
            if missing:
                raise ValueError(
                    f"Criteo CSV is missing expected columns: {sorted(missing)}. "
                    f"Got columns: {list(chunk.columns)}"
                )
            first = False
        total_rows += len(chunk)
        if keep_frac < 1.0:
            mask = rng.random(len(chunk)) < keep_frac
            chunk = chunk.loc[mask]
        kept_chunks.append(_map_chunk(chunk))

    out = pd.concat(kept_chunks, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    split = out["variant"].value_counts(normalize=True).round(4).to_dict()
    return {
        "raw_rows": total_rows,
        "rows_written": len(out),
        "malformed_skipped": malformed_skipped,
        "treatment_split": split,
        "conversion_rate": round(float(out["conversion"].mean()), 5),
        "visit_rate": round(float(out["visit"].mean()), 5),
        "out_path": str(out_path),
    }


def _map_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Apply the Criteo -> ab-guardrail column mapping to one chunk.

    Every value is extracted with ``.to_numpy()`` so the new frame is built
    positionally. The incoming chunk has a non-contiguous index after the
    sampling mask; assigning index-bearing Series would re-align against a
    fresh RangeIndex and silently NaN out almost every row.
    """
    t = pd.to_numeric(chunk["treatment"], errors="coerce").to_numpy()
    data: dict[str, np.ndarray] = {
        "variant": np.where(t == 1, "treatment", "control"),
        "conversion": pd.to_numeric(chunk["conversion"], errors="coerce").to_numpy(),
        "visit": pd.to_numeric(chunk["visit"], errors="coerce").to_numpy(),
    }
    for f in RAW_FEATURES:
        data[f] = pd.to_numeric(chunk[f], errors="coerce").to_numpy()
    return pd.DataFrame(data)[OUTPUT_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapt the Criteo Uplift dataset for ab-guardrail.",
    )
    parser.add_argument("raw_csv", type=Path, help="Path to the un-gzipped Criteo CSV.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/criteo_ready.csv"),
        help="Output CSV path (default: data/criteo_ready.csv).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=300_000,
        help="Stratified sample size (default 300k). Pass 0 to keep all rows.",
    )
    args = parser.parse_args()

    try:
        stats = adapt_criteo(
            args.raw_csv,
            args.out,
            sample=None if args.sample == 0 else args.sample,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    split = stats["treatment_split"]
    print(f"Wrote {stats['out_path']}")
    print(f"  raw rows       : {stats['raw_rows']:,}")
    print(f"  rows written   : {stats['rows_written']:,}")
    print(f"  treatment split: {split}")
    print(f"  conversion rate: {stats['conversion_rate']}")
    print(f"  visit rate     : {stats['visit_rate']}")
    print()
    # Criteo is DESIGNED with an unequal (~85/15) split. The SRM check
    # compares observed vs *planned*, so the planned ratio must be passed
    # explicitly, or the default 50/50 assumption false-positives.
    expected = json.dumps(
        {"control": split.get("control", 0.0), "treatment": split.get("treatment", 0.0)}
    )
    print("Next (Criteo's planned split is NOT 50/50, so pass it):")
    print(f"  ab-guardrail {stats['out_path']} --mode pipeline \\")
    print("      --primary-metric conversion --secondary-metrics visit \\")
    print("      --covariates " + ",".join(RAW_FEATURES) + " \\")
    print(f"      --expected-ratio '{expected}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
