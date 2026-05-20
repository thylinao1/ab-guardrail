"""Tests for the Criteo Uplift dataset adapter.

The adapter has no real Criteo file in CI, so these build a tiny
Criteo-shaped CSV and exercise the mapping. The key regression guarded
here: when sampling leaves a non-contiguous index, the column mapping
must stay positional - index-aligned assignment silently NaNs out almost
every row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.criteo_adapter import OUTPUT_COLUMNS, adapt_criteo


def _criteo_shaped_csv(path: Path, n: int = 6000, seed: int = 0) -> None:
    """Write a CSV with the real Criteo Uplift schema (f0..f11, treatment,
    conversion, visit, exposure)."""
    rng = np.random.default_rng(seed)
    data = {f"f{i}": rng.integers(0, 4, size=n).astype(float) for i in range(12)}
    data["treatment"] = rng.choice([0, 1], size=n, p=[0.15, 0.85])
    data["conversion"] = rng.binomial(1, 0.03, size=n)
    data["visit"] = rng.binomial(1, 0.06, size=n)
    data["exposure"] = rng.binomial(1, 0.5 * data["treatment"])
    pd.DataFrame(data).to_csv(path, index=False)


def test_adapter_maps_schema_without_sampling(tmp_path: Path):
    raw = tmp_path / "criteo.csv"
    _criteo_shaped_csv(raw, n=4000)
    out = tmp_path / "ready.csv"
    stats = adapt_criteo(raw, out, sample=None)

    df = pd.read_csv(out)
    assert list(df.columns) == OUTPUT_COLUMNS
    assert len(df) == 4000
    # treatment 0/1 was remapped to string labels
    assert set(df["variant"].unique()) == {"control", "treatment"}
    # exposure column dropped
    assert "exposure" not in df.columns
    assert stats["raw_rows"] == 4000


def test_adapter_no_nan_after_sampling(tmp_path: Path):
    """Regression: with a sampling mask the index is non-contiguous; the
    mapping must stay positional so no column NaNs out."""
    raw = tmp_path / "criteo.csv"
    _criteo_shaped_csv(raw, n=6000)
    out = tmp_path / "ready.csv"
    adapt_criteo(raw, out, sample=2000)

    df = pd.read_csv(out)
    # the bug produced ~99.8% NaN across every mapped column
    assert df.isna().sum().sum() == 0
    assert 1500 < len(df) < 2500          # ~2000, stratified-random
    assert set(df["variant"].unique()) == {"control", "treatment"}


def test_adapter_rejects_missing_columns(tmp_path: Path):
    raw = tmp_path / "bad.csv"
    pd.DataFrame({"treatment": [0, 1] * 50, "conversion": [0, 1] * 50}).to_csv(
        raw, index=False
    )
    out = tmp_path / "ready.csv"
    with pytest.raises(ValueError, match="missing expected columns"):
        adapt_criteo(raw, out, sample=None)


def test_adapter_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not found"):
        adapt_criteo(tmp_path / "nope.csv", tmp_path / "out.csv", sample=None)
