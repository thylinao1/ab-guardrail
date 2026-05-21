"""Render the final Markdown statistical report.

The report is intentionally opinionated: it leads with the verdict,
then surfaces the SRM check (because if SRM fires, nothing else matters),
then the metric tests, then PSM diagnostics with bootstrap-based
inference and Rosenbaum sensitivity. The same structured payload that
drives the report is also what the LLM narrative sees, so the two are
guaranteed to be consistent.

A "love plot" of pre/post covariate balance is saved alongside the
markdown when matplotlib is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .guardrails import MetricTestResult, PSMResult, SRMResult

VERDICT_SAFE = "SAFE TO ROLL OUT"
VERDICT_COMPROMISED = "EXPERIMENT COMPROMISED"
VERDICT_NULL = "NO SIGNIFICANT EFFECT"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"

# Single source of truth for the PSM-shrinkage threshold used by the verdict
# logic. ATT below this fraction of the naive estimate is "severe" shrinkage.
SHRINKAGE_THRESHOLD = 0.5


@dataclass
class FinalReport:
    verdict: str
    reasons: list[str]
    payload: dict[str, Any]
    markdown: str
    narrative: str = ""
    love_plot_path: Path | None = None


def decide_verdict(
    srm: SRMResult,
    metric_results: list[MetricTestResult],
    psm_results: list[PSMResult],
    primary_metric: str,
) -> tuple[str, list[str]]:
    """Decide the launch verdict and explain why.

    Rules, applied in order:
    1. SRM detected -> COMPROMISED. Stop.
    2. PSM ATT and naive estimate have opposite signs, *and* the naive
       estimate was significant -> COMPROMISED (confounding).
    3. Naive estimate is significant AND PSM ATT is < SHRINKAGE_THRESHOLD
       of naive AND PSM 95% CI covers 0 -> COMPROMISED (effect not
       robust to covariate adjustment).
    4. Primary metric's adjusted p-value is significant at 5% AND PSM
       ATT is in the same direction -> SAFE.
    5. Otherwise -> NO SIGNIFICANT EFFECT.
    """
    reasons: list[str] = []

    if srm.srm_detected:
        reasons.append(
            f"SRM detected: chi-square p={srm.p_value:.2g} < {srm.threshold:g}; "
            f"observed split {srm.observed_ratio} vs expected {srm.expected_ratio}."
        )
        return VERDICT_COMPROMISED, reasons

    primary_psm = next((p for p in psm_results if p.metric == primary_metric), None)
    primary_metric_test = next(
        (m for m in metric_results if m.metric == primary_metric), None
    )

    if primary_psm is not None:
        naive, att = primary_psm.naive_effect, primary_psm.psm_att
        sign_flip = (naive * att) < 0 and abs(naive) > 1e-9
        severe_shrink = (
            abs(naive) > 1e-9 and abs(att) < abs(naive) * SHRINKAGE_THRESHOLD
        )
        ci_contains_zero = (primary_psm.psm_ci_low <= 0 <= primary_psm.psm_ci_high)
        naive_appears_significant = (
            primary_metric_test is not None and primary_metric_test.significant_at_5pct
        )

        if sign_flip and naive_appears_significant:
            reasons.append(
                f"PSM sign flip for {primary_metric}: naive={naive:+.4f}, ATT={att:+.4f}. "
                "A statistically significant naive effect reverses after adjustment - "
                "confounding likely."
            )
            return VERDICT_COMPROMISED, reasons
        if naive_appears_significant and severe_shrink and ci_contains_zero:
            reasons.append(
                f"Naive estimate for {primary_metric} ({naive:+.4f}) collapsed to "
                f"{att:+.4f} after PSM (<{int(SHRINKAGE_THRESHOLD * 100)}% of naive) "
                f"and its bootstrap 95% CI "
                f"[{primary_psm.psm_ci_low:+.4f}, {primary_psm.psm_ci_high:+.4f}] "
                "covers zero - the effect is not robust to covariate adjustment."
            )
            return VERDICT_COMPROMISED, reasons

    if primary_metric_test is not None and primary_metric_test.significant_at_5pct:
        adj_method = primary_metric_test.adjustment_method or "uncorrected"
        p_used = (
            primary_metric_test.adjusted_p_value
            if primary_metric_test.adjusted_p_value is not None
            else primary_metric_test.primary_p_value
        )
        reasons.append(
            f"Primary metric ({primary_metric}) is significant at 5% "
            f"({primary_metric_test.primary_test}, {adj_method} p={p_used:.3g}); "
            "no guardrails fired."
        )
        # Also flag if PSM disagrees with sign even when no verdict-flipping rule triggered.
        if primary_psm is not None:
            naive, att = primary_psm.naive_effect, primary_psm.psm_att
            if (naive * att) < 0 and abs(naive) > 1e-9:
                reasons.append(
                    f"Caveat: PSM-adjusted estimate has opposite sign "
                    f"(ATT={att:+.4f} vs naive {naive:+.4f}); review covariates."
                )
        return VERDICT_SAFE, reasons

    reasons.append(
        f"Primary metric ({primary_metric}) is not significant at 5%; "
        "no guardrails fired but no detectable effect either."
    )
    return VERDICT_NULL, reasons


# ---------- formatting helpers ---------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_pvalue(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _srm_section(srm: SRMResult) -> str:
    flag = ":rotating_light:" if srm.srm_detected else ":white_check_mark:"
    lines = [
        "## 1. Sample Ratio Mismatch (SRM)",
        "",
        f"{flag} **{'SRM DETECTED' if srm.srm_detected else 'No SRM'}** "
        f"(chi-square p = {_fmt_pvalue(srm.p_value)}, threshold {srm.threshold}).",
        "",
        "| Variant | Expected | Observed | Count |",
        "| --- | ---: | ---: | ---: |",
    ]
    for v in sorted(srm.observed_counts):
        lines.append(
            f"| {v} | {_fmt_pct(srm.expected_ratio[v])} | "
            f"{_fmt_pct(srm.observed_ratio[v])} | {srm.observed_counts[v]:,} |"
        )
    lines.append("")
    return "\n".join(lines)


def _metric_section(metrics: list[MetricTestResult]) -> str:
    lines = ["## 2. Metric Shifts (control vs treatment)", ""]
    show_adjusted = any(m.adjusted_p_value is not None for m in metrics)
    headers = ["Metric", "Kind", "Control", "Treatment", "Δ", "95% CI", "Primary p"]
    if show_adjusted:
        headers.append("Adj. p")
    headers += ["Secondary p", "Effect size"]
    lines.append("| " + " | ".join(headers) + " |")
    align = ["---", "---", "---:", "---:", "---:", "---", "---:"]
    if show_adjusted:
        align.append("---:")
    align += ["---:", "---:"]
    lines.append("| " + " | ".join(align) + " |")
    for m in metrics:
        ci = f"[{m.ci_low:+.4f}, {m.ci_high:+.4f}]"
        sec_p = "-" if m.secondary_p_value is None else _fmt_pvalue(m.secondary_p_value)
        row = [
            f"`{m.metric}`",
            m.kind,
            f"{m.control_mean:.4f}",
            f"{m.treatment_mean:.4f}",
            f"{m.absolute_diff:+.4f}",
            ci,
            _fmt_pvalue(m.primary_p_value),
        ]
        if show_adjusted:
            row.append("-" if m.adjusted_p_value is None else _fmt_pvalue(m.adjusted_p_value))
        row += [sec_p, f"{m.effect_size:+.3f} ({m.effect_size_kind})"]
        lines.append("| " + " | ".join(row) + " |")
    if metrics and metrics[0].kind == "binary":
        lines.append("")
        lines.append(
            "_CI method:_ " + metrics[0].ci_method + " for binary metrics; "
            "Welch / Welch-Satterthwaite for continuous metrics."
        )
    if show_adjusted:
        method = next(
            (m.adjustment_method for m in metrics if m.adjustment_method),
            None,
        )
        if method:
            lines.append(f"_Multiple-testing correction:_ {method}.")
    lines.append("")
    return "\n".join(lines)


def _psm_section(psm_results: list[PSMResult]) -> str:
    lines = ["## 3. Propensity Score Matching (causal adjustment)", ""]
    for p in psm_results:
        shrink = (
            abs(p.psm_att) / abs(p.naive_effect)
            if abs(p.naive_effect) > 1e-9
            else float("nan")
        )
        lines.append(f"### `{p.metric}` (covariates: {', '.join(p.covariates)})")
        lines.append("")
        lines.append(f"- Naive effect (unadjusted): **{p.naive_effect:+.4f}**")
        lines.append(
            f"- PSM ATT: **{p.psm_att:+.4f}**  "
            f"(bootstrap 95% CI [{p.psm_ci_low:+.4f}, {p.psm_ci_high:+.4f}], "
            f"p = {_fmt_pvalue(p.psm_p_value)})"
        )
        lines.append(
            f"- Bootstrap SE: **{p.psm_se_bootstrap:.4f}**  "
            f"(naive paired-t SE: {p.psm_se_paired_t:.4f} - biased under "
            "matching-with-replacement; reported for comparison only)"
        )
        lines.append(
            f"- Ratio ATT / Naive: **{shrink:.0%}**  "
            f"(treated={p.n_treated:,}, matched pairs={p.n_matched_pairs:,}, "
            f"trimmed off support: {p.trimmed_treated} treated / "
            f"{p.trimmed_control} control, discarded by caliper: {p.discarded_treated})"
        )
        if p.balance_before and p.balance_after:
            lines.append("")
            lines.append(
                "Covariate balance (standardised mean difference, |SMD| < 0.1 is good):"
            )
            lines.append("")
            lines.append("| Covariate | Before | After |")
            lines.append("| --- | ---: | ---: |")
            for k in p.balance_before:
                lines.append(
                    f"| `{k}` | {p.balance_before[k]:+.3f} | "
                    f"{p.balance_after.get(k, float('nan')):+.3f} |"
                )
        if p.rosenbaum is not None and p.rosenbaum.gamma_grid:
            lines.append("")
            lines.append(
                "**Rosenbaum bounds (sensitivity to hidden bias):** at each Γ, "
                "the upper-bound p-value under the worst-case unmeasured "
                "confounder."
            )
            lines.append("")
            lines.append("| Γ | Upper-bound p |")
            lines.append("| ---: | ---: |")
            for g, pv in zip(p.rosenbaum.gamma_grid, p.rosenbaum.upper_p_values, strict=True):
                lines.append(f"| {g:.2f} | {_fmt_pvalue(pv)} |")
            if p.rosenbaum.gamma_critical is None:
                lines.append("")
                lines.append(
                    "_All Γ in grid keep p ≤ 0.05 - the finding is robust to the tested levels of hidden bias._"
                )
            else:
                lines.append("")
                lines.append(
                    f"_Critical Γ ≈ {p.rosenbaum.gamma_critical:.2f}: a hidden binary "
                    "confounder would need to bias treatment assignment by at least this "
                    "odds ratio to overturn the significance of the ATT._"
                )
        lines.append("")
    return "\n".join(lines)


def render_love_plot(
    psm_results: list[PSMResult], out_path: Path
) -> Path | None:
    """Render a love-plot (pre vs post SMD) of the first PSM result.

    Returns the file path on success, None if matplotlib isn't available
    or the PSM result has no balance data.
    """
    if not psm_results:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    p = psm_results[0]
    if not p.balance_before:
        return None
    keys = list(p.balance_before.keys())
    before = [p.balance_before[k] for k in keys]
    after = [p.balance_after.get(k, 0.0) for k in keys]

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(keys))))
    y = list(range(len(keys)))
    ax.scatter(before, y, label="Before matching", marker="o", color="#d62728")
    ax.scatter(after, y, label="After matching", marker="s", color="#2ca02c")
    for i, (b, a) in enumerate(zip(before, after, strict=True)):
        ax.plot([b, a], [i, i], color="gray", linewidth=0.6, alpha=0.6)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.1, color="black", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.axvline(-0.1, color="black", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_yticks(y, keys)
    ax.set_xlabel("Standardised mean difference")
    ax.set_title(f"Covariate balance - `{p.metric}`")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_report(
    *,
    csv_path: str | Path,
    schema_rationale: str,
    srm: SRMResult,
    metric_results: list[MetricTestResult],
    psm_results: list[PSMResult],
    primary_metric: str,
    narrative_provider,  # callable(payload: dict) -> str
    love_plot_path: Path | None = None,
    quality=None,  # DataQualityReport | None
) -> FinalReport:
    """Stitch the structured results together into a Markdown document.

    The narrative is computed *after* the verdict is known so the LLM
    sees the same payload that decided the verdict - we pass a callable
    rather than a string so callers don't have to do the dance themselves.
    """
    verdict, reasons = decide_verdict(srm, metric_results, psm_results, primary_metric)

    payload: dict[str, Any] = {
        "verdict": verdict,
        "reasons": reasons,
        "schema_rationale": schema_rationale,
        "srm": srm.to_dict(),
        "metric_tests": [m.to_dict() for m in metric_results],
        "psm": [p.to_dict() for p in psm_results],
    }
    if quality is not None and quality.has_issues:
        payload["data_quality"] = quality.summary_lines()
    narrative = narrative_provider(payload)

    verdict_emoji = {
        VERDICT_SAFE: ":white_check_mark:",
        VERDICT_COMPROMISED: ":rotating_light:",
        VERDICT_NULL: ":no_entry_sign:",
        VERDICT_INCONCLUSIVE: ":grey_question:",
    }.get(verdict, "")

    md_parts = [
        "# Experimentation Guardrail Report",
        "",
        f"_Source CSV:_ `{csv_path}`  ",
        f"_Generated:_ {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"## Verdict: {verdict_emoji} **{verdict}**",
        "",
    ]
    for r in reasons:
        md_parts.append(f"- {r}")
    md_parts.extend(["", "### Narrative", "", narrative, "", "---", ""])
    md_parts.append(f"_Schema rationale:_ {schema_rationale}")
    md_parts.append("")
    if quality is not None and quality.has_issues:
        md_parts.append("### Data quality")
        md_parts.append("")
        md_parts.append(
            f"Loaded {quality.rows_loaded:,} of {quality.rows_in_file:,} rows. "
            "The loader repaired the following before analysis:"
        )
        md_parts.append("")
        for line in quality.summary_lines():
            md_parts.append(f"- {line}")
        md_parts.append("")
    if love_plot_path is not None:
        md_parts.append(f"![Love plot]({love_plot_path.name})")
        md_parts.append("")
    md_parts.append(_srm_section(srm))
    md_parts.append(_metric_section(metric_results))
    md_parts.append(_psm_section(psm_results))

    return FinalReport(
        verdict=verdict,
        reasons=reasons,
        payload=payload,
        markdown="\n".join(md_parts),
        narrative=narrative,
        love_plot_path=love_plot_path,
    )
