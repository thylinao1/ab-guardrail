"""CSV loading, cleaning, and schema profiling.

Real experiment exports are not tidy. A production e-commerce A/B log
(Shopee, TikTok Shop, Criteo) routinely carries:

* malformed rows  - a logging job wrote a line with the wrong field count
* duplicate rows  - an at-least-once delivery pipeline double-logged events
* all-null columns - a feature was instrumented but never populated
* missing covariates - consent gates, late joins, schema drift
* dirty numerics  - a covariate column read as text because of a stray
                    "ERROR" / "NULL" / "" token

This module ingests the CSV defensively, repairs what is safely
repairable, and returns a :class:`DataQualityReport` so nothing is fixed
silently. Downstream guardrails treat the remaining missingness honestly:
metric tests and PSM drop incomplete rows per analysis and report the
counts.

Baseline invariants still enforced (raise DataValidationError):
* the file exists and parses
* after cleaning, at least one usable variant column and one numeric
  metric column remain
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import DataValidationError

# An object column is coerced to numeric when at least this fraction of
# its non-null values parse as numbers. Stray non-numeric tokens become
# NaN and are counted as missingness, not silently dropped.
_NUMERIC_COERCE_THRESHOLD = 0.80


@dataclass
class DataQualityReport:
    """What the loader had to repair. Surfaced in the CLI and report so
    no cleaning step is invisible."""
    rows_in_file: int
    rows_loaded: int
    malformed_rows_skipped: int
    duplicate_rows_dropped: int
    dropped_all_null_columns: list[str] = field(default_factory=list)
    coerced_to_numeric: list[str] = field(default_factory=list)
    missing_by_column: dict[str, float] = field(default_factory=dict)

    @property
    def has_issues(self) -> bool:
        return bool(
            self.malformed_rows_skipped
            or self.duplicate_rows_dropped
            or self.dropped_all_null_columns
            or self.coerced_to_numeric
            or self.missing_by_column
        )

    def summary_lines(self) -> list[str]:
        """Human-readable one-liners for the CLI / report."""
        lines: list[str] = []
        if self.malformed_rows_skipped:
            lines.append(
                f"{self.malformed_rows_skipped} malformed row(s) skipped at parse time"
            )
        if self.duplicate_rows_dropped:
            lines.append(
                f"{self.duplicate_rows_dropped} row(s) dropped as double-logged "
                "(repeated on the identifier column)"
            )
        if self.dropped_all_null_columns:
            lines.append(
                f"dropped {len(self.dropped_all_null_columns)} all-null column(s): "
                f"{', '.join(self.dropped_all_null_columns)}"
            )
        if self.coerced_to_numeric:
            lines.append(
                f"coerced {len(self.coerced_to_numeric)} text column(s) to numeric "
                f"(dirty tokens -> missing): {', '.join(self.coerced_to_numeric)}"
            )
        for col, frac in sorted(
            self.missing_by_column.items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"column '{col}' is {frac:.1%} missing")
        return lines


@dataclass
class DatasetProfile:
    """Structural summary of the cleaned CSV, used for schema inference."""
    n_rows: int
    columns: list[str]
    dtypes: dict[str, str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    binary_candidates: list[str]      # columns with exactly 2 distinct values
    binary_levels: dict[str, list[Any]]  # candidate column -> its 2 values
    sample_rows: list[dict[str, Any]]
    quality: DataQualityReport


# Column names that identify a per-row entity. A genuine double-logged
# event is the SAME entity appearing twice, keyed on one of these.
_IDENTIFIER_TOKENS = {
    "user_id", "id", "uid", "userid", "session_id", "event_id", "customer_id",
}


def _find_identifier_column(df: pd.DataFrame) -> str | None:
    """Return a per-row identifier column if one plausibly exists.

    A column qualifies when its name matches a known identifier token AND
    it is mostly unique (so a categorical column happening to be named
    'id' is not mistaken for one)."""
    n = max(len(df), 1)
    for col in df.columns:
        if (
            col.lower() in _IDENTIFIER_TOKENS
            and df[col].notna().any()
            and df[col].nunique(dropna=True) / n > 0.5
        ):
            return col
    return None


def load_experiment_csv(
    path: str | Path,
    *,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, DatasetProfile]:
    """Load, defensively clean, and profile an experiment CSV.

    Parameters
    ----------
    path : CSV path.
    drop_duplicates : drop double-logged rows (default True). Duplicates are
        dropped ONLY when the file has a per-row identifier column
        (user_id, id, ...) and rows repeat on it. Without an identifier,
        two distinct entities can legitimately share an identical row -
        common when covariates are low-cardinality (e.g. the Criteo
        features) - so full-row duplicates are NOT dropped.

    Raises
    ------
    DataValidationError
        If the file cannot be read or, after cleaning, has no usable
        variant column or no numeric metric column.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise DataValidationError(f"CSV not found: {csv_path}")

    # --- parse, tolerating malformed lines ---------------------------------
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
    except Exception as exc:  # pandas raises many different things
        raise DataValidationError(f"Failed to parse CSV {csv_path}: {exc}") from exc

    if df.empty:
        raise DataValidationError(f"CSV {csv_path} is empty.")

    rows_parsed = len(df)
    # Best-effort malformed-row count: raw non-empty data lines minus parsed
    # rows. Assumes no embedded newlines in quoted fields (true for the
    # tool's CSV inputs); reported as an estimate either way.
    malformed = _estimate_malformed_rows(csv_path, rows_parsed)

    # --- drop double-logged rows (only when keyed on an identifier) --------
    dupes = 0
    id_col = _find_identifier_column(df)
    if drop_duplicates and id_col is not None:
        before = len(df)
        df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
        dupes = before - len(df)

    # --- drop all-null columns ---------------------------------------------
    all_null = [c for c in df.columns if df[c].isna().all()]
    if all_null:
        df = df.drop(columns=all_null)
    if df.columns.empty:
        raise DataValidationError(f"CSV {csv_path} has no non-empty columns.")

    # --- coerce dirty numeric columns --------------------------------------
    coerced = _coerce_numeric_columns(df)

    # --- missingness -------------------------------------------------------
    missing = {
        c: float(df[c].isna().mean())
        for c in df.columns
        if df[c].isna().any()
    }

    quality = DataQualityReport(
        rows_in_file=rows_parsed + malformed,
        rows_loaded=len(df),
        malformed_rows_skipped=malformed,
        duplicate_rows_dropped=dupes,
        dropped_all_null_columns=all_null,
        coerced_to_numeric=coerced,
        missing_by_column=missing,
    )

    profile = _profile(df, quality)

    if not profile.binary_candidates:
        raise DataValidationError(
            "No column with exactly two distinct values was found; "
            "cannot identify a variant/treatment column."
        )
    if not profile.numeric_columns:
        raise DataValidationError(
            "No numeric columns found; the agent has nothing to measure."
        )

    return df, profile


def _estimate_malformed_rows(csv_path: Path, rows_parsed: int) -> int:
    """Estimate rows the parser skipped, by counting raw data lines."""
    try:
        with open(csv_path, encoding="utf-8", errors="replace") as fh:
            raw_lines = sum(1 for line in fh if line.strip())
    except OSError:
        return 0
    raw_data_lines = max(0, raw_lines - 1)  # minus header
    return max(0, raw_data_lines - rows_parsed)


def _coerce_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Coerce object columns that are mostly numeric in place.

    A covariate column carrying a stray 'ERROR' or 'NULL' token reads as
    object dtype. If >= _NUMERIC_COERCE_THRESHOLD of its non-null values
    parse as numbers, coerce it - the dirty tokens become NaN (counted as
    missingness downstream) rather than poisoning the whole column.
    """
    coerced: list[str] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue
        frac_numeric = pd.to_numeric(non_null, errors="coerce").notna().mean()
        if frac_numeric >= _NUMERIC_COERCE_THRESHOLD:
            # Mostly numeric -> coerce; dirty tokens become NaN.
            df[col] = pd.to_numeric(s, errors="coerce")
            coerced.append(col)
    return coerced


def _profile(
    df: pd.DataFrame, quality: DataQualityReport | None = None
) -> DatasetProfile:
    if quality is None:
        # Allows building a profile straight from a DataFrame (tests,
        # programmatic use) without going through the CSV loader.
        quality = DataQualityReport(
            rows_in_file=len(df),
            rows_loaded=len(df),
            malformed_rows_skipped=0,
            duplicate_rows_dropped=0,
        )
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    binary_candidates: list[str] = []
    binary_levels: dict[str, list[Any]] = {}
    for col in df.columns:
        # nunique excludes NaN by default
        if df[col].nunique(dropna=True) != 2:
            continue
        # A 0/1 numeric column is almost always an outcome (converted,
        # purchased, etc.), not a variant assignment. Excluding these
        # prevents an outcome being mistaken for the variant column.
        if col in numeric_cols:
            uniq = set(
                np.asarray(df[col].dropna().unique()).astype(float).tolist()
            )
            if uniq <= {0.0, 1.0}:
                continue
        binary_candidates.append(col)
        # The two distinct values, computed from the FULL column - not from
        # head(5), which on a skewed split (e.g. Criteo's 85/15) can show
        # only one of the two labels.
        levels = sorted(df[col].dropna().unique().tolist(), key=str)
        binary_levels[col] = levels

    return DatasetProfile(
        n_rows=int(len(df)),
        columns=list(df.columns),
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        binary_candidates=binary_candidates,
        binary_levels=binary_levels,
        sample_rows=df.head(5).where(pd.notna(df.head(5)), None).to_dict(orient="records"),
        quality=quality,
    )
