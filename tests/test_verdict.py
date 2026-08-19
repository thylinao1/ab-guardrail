"""Tests for the end-to-end verdict pipeline."""

from __future__ import annotations

import pandas as pd

from src.agent import deterministic_narrative, infer_schema_heuristic
from src.data_loader import _profile  # noqa: PLC2701 -- intentionally testing helpers
from src.guardrails import metric_test, propensity_score_match, srm_check
from src.report import (
    VERDICT_COMPROMISED,
    VERDICT_SAFE,
    build_report,
    decide_verdict,
)


def _run_pipeline(df: pd.DataFrame, primary: str, covariates: list[str]):
    srm = srm_check(df, "variant")
    metric_results = [metric_test(df, primary, "variant", "control", "treatment")]
    psm_results = [
        propensity_score_match(
            df,
            metric=primary,
            variant_column="variant",
            control_label="control",
            treatment_label="treatment",
            covariates=covariates,
            bootstrap_samples=100,
        )
    ]
    return srm, metric_results, psm_results


def test_clean_dataset_is_safe(balanced_experiment: pd.DataFrame):
    srm, metric_results, psm_results = _run_pipeline(
        balanced_experiment, "revenue", ["pre_signup_value", "device"]
    )
    verdict, _ = decide_verdict(srm, metric_results, psm_results, "revenue")
    assert verdict == VERDICT_SAFE


def test_srm_dataset_is_compromised(srm_experiment: pd.DataFrame):
    srm = srm_check(srm_experiment, "variant")
    verdict, reasons = decide_verdict(srm, [], [], "converted")
    assert verdict == VERDICT_COMPROMISED
    assert any("SRM" in r for r in reasons)


def test_schema_heuristic_picks_pre_columns_as_covariates(
    balanced_experiment: pd.DataFrame,
):
    profile = _profile(balanced_experiment)
    plan = infer_schema_heuristic(profile)
    assert plan.variant_column == "variant"
    assert plan.control_label == "control"
    assert plan.treatment_label == "treatment"
    assert "pre_signup_value" in plan.covariates
    # `revenue` and `converted` are outcomes; they must not appear in covariates.
    assert "revenue" not in plan.covariates
    assert "converted" not in plan.covariates


def test_build_report_returns_narrative_and_markdown(balanced_experiment: pd.DataFrame):
    srm, metric_results, psm_results = _run_pipeline(
        balanced_experiment, "revenue", ["pre_signup_value", "device"]
    )
    report = build_report(
        csv_path="dummy.csv",
        schema_rationale="test",
        srm=srm,
        metric_results=metric_results,
        psm_results=psm_results,
        primary_metric="revenue",
        narrative_provider=deterministic_narrative,
    )
    assert report.verdict in {VERDICT_SAFE, VERDICT_COMPROMISED}
    assert "Verdict" in report.markdown
    assert len(report.narrative) > 0


def test_deterministic_narrative_mentions_srm():
    payload = {
        "verdict": "EXPERIMENT COMPROMISED",
        "srm": {
            "srm_detected": True,
            "p_value": 1e-50,
            "observed_ratio": {"control": 0.4, "treatment": 0.6},
            "expected_ratio": {"control": 0.5, "treatment": 0.5},
        },
        "metric_tests": [],
        "psm": [],
    }
    text = deterministic_narrative(payload)
    assert "Sample Ratio Mismatch" in text
    assert "EXPERIMENT COMPROMISED" in text
