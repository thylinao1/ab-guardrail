"""Tests for defensive CSV loading and the data-quality report.

These exercise the messy-real-world-data path: malformed rows, duplicates,
all-null columns, dirty numeric tokens, and missing covariates, the
pathologies a production e-commerce export carries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agent import infer_schema_heuristic
from src.data_loader import load_experiment_csv
from src.exceptions import DataValidationError


def _base_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "variant": rng.choice(["control", "treatment"], size=n),
            "pre_value": rng.gamma(2, 15, size=n).round(2),
            "device": rng.choice(["mobile", "desktop"], size=n),
            "converted": rng.binomial(1, 0.12, size=n),
        }
    )


def test_clean_csv_loads_without_quality_issues(tmp_path: Path):
    p = tmp_path / "clean.csv"
    _base_frame().to_csv(p, index=False)
    df, profile = load_experiment_csv(p)
    assert len(df) == 400
    assert not profile.quality.has_issues
    assert "variant" in profile.binary_candidates


def test_duplicate_rows_dropped_when_keyed_on_identifier(tmp_path: Path):
    """With a user_id column, repeated rows are double-logged events and
    are dropped on the identifier."""
    base = _base_frame(n=300)
    withdupes = pd.concat([base, base.iloc[:20]], ignore_index=True)
    p = tmp_path / "dupes.csv"
    withdupes.to_csv(p, index=False)
    df, profile = load_experiment_csv(p)
    assert profile.quality.duplicate_rows_dropped == 20
    assert len(df) == 300


def test_identical_rows_kept_when_no_identifier_column(tmp_path: Path):
    """Without a per-row identifier, two distinct entities can legitimately
    share an identical row (low-cardinality covariates, e.g. Criteo). Such
    rows must NOT be dropped. This is the bug real Criteo data exposed."""
    rng = np.random.default_rng(1)
    n = 600
    # low-cardinality covariates -> many genuinely identical rows
    frame = pd.DataFrame(
        {
            "variant": rng.choice(["control", "treatment"], size=n),
            "f0": rng.integers(0, 3, size=n),
            "f1": rng.integers(0, 3, size=n),
            "converted": rng.binomial(1, 0.1, size=n),
        }
    )
    p = tmp_path / "no_id.csv"
    frame.to_csv(p, index=False)
    df, profile = load_experiment_csv(p)
    # no identifier column -> nothing dropped, even though rows repeat
    assert profile.quality.duplicate_rows_dropped == 0
    assert len(df) == n


def test_all_null_column_is_dropped(tmp_path: Path):
    base = _base_frame()
    base["notes"] = np.nan
    p = tmp_path / "nullcol.csv"
    base.to_csv(p, index=False)
    df, profile = load_experiment_csv(p)
    assert "notes" not in df.columns
    assert "notes" in profile.quality.dropped_all_null_columns


def test_malformed_rows_are_skipped(tmp_path: Path):
    base = _base_frame(n=200)
    p = tmp_path / "malformed.csv"
    text = base.to_csv(index=False)
    lines = text.splitlines()
    n_fields = lines[0].count(",") + 1
    # two rows with too many fields -> skipped by the parser
    bad = ",".join(["x"] * (n_fields + 3))
    lines.insert(50, bad)
    lines.insert(120, bad)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    df, profile = load_experiment_csv(p)
    assert profile.quality.malformed_rows_skipped == 2
    assert len(df) == 200


def test_dirty_numeric_column_is_coerced(tmp_path: Path):
    base = _base_frame(n=300)
    base["pre_value"] = base["pre_value"].astype(object)
    base.loc[5, "pre_value"] = "ERROR"
    base.loc[9, "pre_value"] = "NULL"
    p = tmp_path / "dirty.csv"
    base.to_csv(p, index=False)
    df, profile = load_experiment_csv(p)
    # column is numeric again; dirty tokens became NaN (missingness)
    assert pd.api.types.is_numeric_dtype(df["pre_value"])
    assert "pre_value" in profile.quality.coerced_to_numeric
    assert df["pre_value"].isna().sum() == 2
    assert profile.quality.missing_by_column.get("pre_value", 0) > 0


def test_missing_covariate_values_are_reported(tmp_path: Path):
    base = _base_frame(n=500)
    base.loc[base.index[:40], "pre_value"] = np.nan
    p = tmp_path / "missing.csv"
    base.to_csv(p, index=False)
    df, profile = load_experiment_csv(p)
    frac = profile.quality.missing_by_column["pre_value"]
    assert abs(frac - 40 / 500) < 1e-9


def test_skewed_split_labels_inferred_from_full_column(tmp_path: Path):
    """Regression: a heavily skewed split (e.g. Criteo's 85/15) can leave
    the first rows all carrying one label. Variant labels must be read from
    the full column (profile.binary_levels), not from a head() sample, or
    deterministic routing fails to find two distinct variants."""
    n = 500
    frame = pd.DataFrame(
        {
            "user_id": np.arange(n),
            # sorted so the first rows are entirely 'treatment'
            "variant": ["treatment"] * 425 + ["control"] * 75,
            "pre_value": np.random.default_rng(0).normal(size=n),
            "converted": np.random.default_rng(1).binomial(1, 0.1, size=n),
        }
    )
    p = tmp_path / "skewed.csv"
    frame.to_csv(p, index=False)
    df, profile = load_experiment_csv(p)
    assert set(profile.binary_levels["variant"]) == {"control", "treatment"}
    plan = infer_schema_heuristic(profile)
    assert {plan.control_label, plan.treatment_label} == {"control", "treatment"}


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(DataValidationError, match="not found"):
        load_experiment_csv(tmp_path / "does_not_exist.csv")


def test_no_numeric_column_raises(tmp_path: Path):
    p = tmp_path / "no_numeric.csv"
    pd.DataFrame({"variant": ["a", "b"] * 50, "label": ["x", "y"] * 50}).to_csv(
        p, index=False
    )
    with pytest.raises(DataValidationError, match="numeric"):
        load_experiment_csv(p)


def test_messy_dataset_survives_full_pipeline(tmp_path: Path):
    """End-to-end: a messy CSV (malformed + dupes + null col + dirty
    numerics + missing covariates) loads and profiles without raising."""
    base = _base_frame(n=600)
    base["pre_value"] = base["pre_value"].astype(object)
    base.loc[3, "pre_value"] = "ERROR"
    base.loc[base.index[:50], "device"] = np.nan
    base["notes"] = np.nan
    withdupes = pd.concat([base, base.iloc[:15]], ignore_index=True)
    text = withdupes.to_csv(index=False)
    lines = text.splitlines()
    lines.insert(100, ",".join(["x"] * (lines[0].count(",") + 5)))
    p = tmp_path / "messy.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    df, profile = load_experiment_csv(p)
    q = profile.quality
    assert q.has_issues
    assert q.malformed_rows_skipped == 1
    assert q.duplicate_rows_dropped == 15
    assert "notes" in q.dropped_all_null_columns
    assert "pre_value" in q.coerced_to_numeric
    assert "device" in q.missing_by_column
    # still usable downstream
    assert profile.binary_candidates
    assert profile.numeric_columns
