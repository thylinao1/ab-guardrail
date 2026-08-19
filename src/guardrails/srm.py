"""Sample Ratio Mismatch (SRM) check.

The randomizer in a properly functioning experiment should produce an
observed allocation indistinguishable from the planned allocation. We
test that with a one-degree-of-freedom chi-square goodness-of-fit test.

A p-value below 0.001 is the conventional "this is broken" threshold -
it is intentionally stricter than 0.05 because, under the null, even a
correctly-running test will occasionally hit p<0.05 by chance, and an SRM
investigation is expensive enough that we don't want to chase false alarms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import StatisticalCheckError

SRM_PVALUE_THRESHOLD = 0.001


@dataclass
class SRMResult:
    variant_column: str
    expected_ratio: dict[str, float]
    observed_counts: dict[str, int]
    observed_ratio: dict[str, float]
    chi2: float
    p_value: float
    srm_detected: bool
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def srm_check(
    df: pd.DataFrame,
    variant_column: str,
    expected_ratio: dict[str, float] | None = None,
    threshold: float = SRM_PVALUE_THRESHOLD,
) -> SRMResult:
    """Run a chi-square goodness-of-fit test on the variant allocation.

    Parameters
    ----------
    df : DataFrame containing the experiment.
    variant_column : column to test (typically "variant").
    expected_ratio : mapping {variant: expected_proportion}. Must sum to 1.
        Defaults to uniform across observed levels.
    threshold : SRM is declared when p < threshold (default 0.001).
    """
    if variant_column not in df.columns:
        raise StatisticalCheckError(
            f"variant column '{variant_column}' not in DataFrame."
        )

    counts = df[variant_column].value_counts().to_dict()
    levels = sorted(counts.keys(), key=str)
    if len(levels) < 2:
        raise StatisticalCheckError(
            f"variant column '{variant_column}' has fewer than 2 levels: {levels!r}"
        )

    if expected_ratio is None:
        expected_ratio = {lvl: 1.0 / len(levels) for lvl in levels}
    else:
        if set(expected_ratio) != set(levels):
            raise StatisticalCheckError(
                f"expected_ratio keys {sorted(expected_ratio)} don't match "
                f"observed variants {levels}."
            )
        total = sum(expected_ratio.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise StatisticalCheckError(
                f"expected_ratio must sum to 1.0 (got {total:.6f})."
            )

    n = int(sum(counts.values()))
    observed = np.array([counts[lvl] for lvl in levels], dtype=float)
    expected = np.array([expected_ratio[lvl] * n for lvl in levels], dtype=float)

    if (expected < 5).any():
        # Chi-square approximation degrades with tiny expected cells.
        raise StatisticalCheckError(
            "Expected counts below 5 in at least one cell; the sample is too "
            "small for a reliable chi-square SRM test."
        )

    chi2, p_value = stats.chisquare(observed, f_exp=expected)

    return SRMResult(
        variant_column=variant_column,
        expected_ratio={lvl: float(expected_ratio[lvl]) for lvl in levels},
        observed_counts={lvl: int(counts[lvl]) for lvl in levels},
        observed_ratio={lvl: float(counts[lvl] / n) for lvl in levels},
        chi2=float(chi2),
        p_value=float(p_value),
        srm_detected=bool(p_value < threshold),
        threshold=float(threshold),
    )
