"""CSV loading and schema validation.

The agent is schema-flexible — the LLM picks which columns play which
role — but we still enforce some baseline invariants here:

* the file exists and parses as CSV
* at least one column has exactly two distinct non-null values (so we can
  pick a variant column)
* at least one numeric column (so we can pick a metric)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import DataValidationError


@dataclass
class DatasetProfile:
    """Lightweight summary of the loaded CSV, used by the agent for schema inference."""
    n_rows: int
    columns: list[str]
    dtypes: dict[str, str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    binary_candidates: list[str]    # columns with exactly 2 distinct values
    sample_rows: list[dict[str, Any]]


def load_experiment_csv(path: str | Path) -> tuple[pd.DataFrame, DatasetProfile]:
    """Load a CSV and return the DataFrame plus a structural profile.

    Raises
    ------
    DataValidationError
        If the file cannot be read or violates baseline shape requirements.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise DataValidationError(f"CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pandas raises many different things
        raise DataValidationError(f"Failed to parse CSV {csv_path}: {exc}") from exc

    if df.empty:
        raise DataValidationError(f"CSV {csv_path} is empty.")

    profile = _profile(df)

    if not profile.binary_candidates:
        raise DataValidationError(
            "No column with exactly two distinct values was found; "
            "the agent cannot identify a variant/treatment column."
        )
    if not profile.numeric_columns:
        raise DataValidationError(
            "No numeric columns found; the agent has nothing to measure."
        )

    return df, profile


def _profile(df: pd.DataFrame) -> DatasetProfile:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    binary_candidates: list[str] = []
    for col in df.columns:
        # nunique excludes NaN by default
        if df[col].nunique(dropna=True) != 2:
            continue
        # A 0/1 numeric column is almost always an outcome (converted,
        # purchased, etc.), not a variant assignment. Excluding these
        # prevents the schema agent from picking an outcome as the
        # variant.
        if col in numeric_cols:
            uniq = set(np.asarray(df[col].dropna().unique()).astype(float).tolist())
            if uniq <= {0.0, 1.0}:
                continue
        binary_candidates.append(col)

    return DatasetProfile(
        n_rows=int(len(df)),
        columns=list(df.columns),
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        binary_candidates=binary_candidates,
        sample_rows=df.head(5).to_dict(orient="records"),
    )
