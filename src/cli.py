"""Command-line entry point.

Examples:

    ab-guardrail data/clean_experiment.csv
    ab-guardrail data/compromised_experiment.csv --out reports/report.md
    ab-guardrail data/compromised_experiment.csv --agent none      # deterministic
    ab-guardrail data/clean_experiment.csv --agent tools           # LLM orchestrates
    ab-guardrail data/clean_experiment.csv --cuped pre_signup_value
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a, **_k):  # type: ignore[no-redef]
        return False

from .agent import (
    AgentRun,
    SchemaPlan,
    deterministic_narrative,
    infer_schema,
    infer_schema_heuristic,
    narrate,
    run_tool_agent,
)
from .data_loader import DatasetProfile, load_experiment_csv
from .exceptions import AgentError, GuardrailError
from .guardrails import (
    metric_test,
    propensity_score_match,
    srm_check,
)
from .guardrails.metric_tests import (
    apply_cuped,
    apply_multiple_testing_correction,
)
from .report import (
    VERDICT_COMPROMISED,
    VERDICT_NULL,
    VERDICT_SAFE,
    _maybe_render_love_plot,
    build_report,
)


def _c(code: str) -> str:
    return code if sys.stdout.isatty() else ""

_RESET = _c("\033[0m")
_BOLD = _c("\033[1m")
_RED = _c("\033[91m")
_GREEN = _c("\033[92m")
_YELLOW = _c("\033[93m")
_CYAN = _c("\033[96m")
_DIM = _c("\033[2m")


VERDICT_COLOUR = {
    VERDICT_SAFE: _GREEN,
    VERDICT_COMPROMISED: _RED,
    VERDICT_NULL: _YELLOW,
}


def _csv_list(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [t.strip() for t in s.split(",") if t.strip()]


def _apply_overrides(
    plan: SchemaPlan, args: argparse.Namespace, profile: DatasetProfile
) -> SchemaPlan:
    """Apply CLI flag overrides on top of the inferred schema."""
    variant_column = args.variant_column or plan.variant_column
    control_label = args.control_label or plan.control_label
    treatment_label = args.treatment_label or plan.treatment_label
    primary_metric = args.primary_metric or plan.primary_metric
    sec_override = _csv_list(args.secondary_metrics)
    cov_override = _csv_list(args.covariates)
    secondary = sec_override if sec_override is not None else plan.secondary_metrics
    covariates = cov_override if cov_override is not None else plan.covariates

    if variant_column not in profile.columns:
        raise GuardrailError(f"--variant-column {variant_column!r} not in CSV.")
    if primary_metric not in profile.numeric_columns:
        raise GuardrailError(
            f"--primary-metric {primary_metric!r} is not a numeric column."
        )
    for sm in secondary:
        if sm not in profile.numeric_columns:
            raise GuardrailError(f"secondary metric {sm!r} is not numeric.")
    for cv in covariates:
        if cv not in profile.columns:
            raise GuardrailError(f"covariate {cv!r} not in CSV.")
    if primary_metric in covariates or any(s in covariates for s in secondary):
        raise GuardrailError(
            "covariates must not overlap with metrics (would bias the PSM model)."
        )

    rationale = plan.rationale
    if any([args.variant_column, args.primary_metric, sec_override, cov_override]):
        rationale = f"{rationale} (overridden by CLI flags)"

    return SchemaPlan(
        variant_column=variant_column,
        control_label=control_label,
        treatment_label=treatment_label,
        primary_metric=primary_metric,
        secondary_metrics=secondary,
        covariates=covariates,
        rationale=rationale,
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(
        prog="ab-guardrail",
        description=(
            "Automated experimentation guardrail agent. Reads an A/B-test CSV, "
            "checks for SRM, runs metric-shift tests, applies propensity-score "
            "matching with bootstrap SE and Rosenbaum bounds, and writes a "
            "Markdown report."
        ),
    )
    p.add_argument("csv", type=Path, help="Path to the experiment CSV.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the Markdown report (default: reports/<csv_stem>_report.md).",
    )
    p.add_argument(
        "--agent",
        choices=["tools", "simple", "none"],
        default="tools",
        help=(
            "LLM agent mode. "
            "'tools' (default): Claude orchestrates the analysis via tool calls. "
            "'simple': two LLM calls (schema + narrative). "
            "'none': fully deterministic — no API calls."
        ),
    )
    p.add_argument(
        "--correction",
        choices=["bh", "bonferroni", "none"],
        default="bh",
        help="Multiple-testing correction across metric tests (default: Benjamini-Hochberg FDR).",
    )
    p.add_argument(
        "--cuped",
        type=str,
        default=None,
        help="Pre-experiment covariate name for CUPED variance reduction "
        "of the primary metric (Deng et al. WSDM 2013).",
    )
    p.add_argument(
        "--expected-ratio",
        type=str,
        default=None,
        help='Expected allocation, JSON e.g. \'{"control": 0.5, "treatment": 0.5}\'.',
    )
    p.add_argument("--variant-column", help="Override the variant column name.")
    p.add_argument("--control-label", help="Override the control label.")
    p.add_argument("--treatment-label", help="Override the treatment label.")
    p.add_argument("--primary-metric", help="Override the primary metric column.")
    p.add_argument(
        "--secondary-metrics", help="Comma-separated list of secondary metric columns."
    )
    p.add_argument(
        "--covariates",
        help="Comma-separated list of covariate columns (for PSM).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Also print the structured results payload as JSON on stdout.",
    )
    args = p.parse_args(argv)

    try:
        return _run(args)
    except GuardrailError as exc:
        print(f"{_RED}{_BOLD}Error:{_RESET} {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"\n{_YELLOW}Aborted.{_RESET}", file=sys.stderr)
        return 130


def _run(args: argparse.Namespace) -> int:
    print(f"{_BOLD}{_CYAN}>> Loading{_RESET} {args.csv}")
    df, profile = load_experiment_csv(args.csv)
    print(f"   {profile.n_rows:,} rows, {len(profile.columns)} columns.")

    expected_ratio = None
    if args.expected_ratio:
        try:
            expected_ratio = json.loads(args.expected_ratio)
        except json.JSONDecodeError as exc:
            raise GuardrailError(f"--expected-ratio is not valid JSON: {exc}") from exc

    agent_run: AgentRun | None = None
    plan: SchemaPlan

    if args.agent == "tools":
        try:
            print(f"{_BOLD}{_CYAN}>> Agent (tools mode){_RESET}  Claude orchestrates the analysis")
            agent_run = run_tool_agent(df, profile, progress=lambda s: print(f"   {_DIM}{s}{_RESET}"))
            plan = agent_run.plan
        except AgentError as exc:
            print(f"   {_YELLOW}tools mode unavailable ({exc}); falling back to simple{_RESET}",
                  file=sys.stderr)
            args.agent = "simple"

    if args.agent in {"simple", "none"} and agent_run is None:
        print(f"{_BOLD}{_CYAN}>> Inferring schema{_RESET} "
              f"({'heuristic' if args.agent == 'none' else 'Claude'})")
        if args.agent == "none":
            plan = infer_schema_heuristic(profile)
        else:
            try:
                plan = infer_schema(profile)
            except AgentError as exc:
                print(f"   {_YELLOW}LLM schema failed ({exc}); using heuristic{_RESET}",
                      file=sys.stderr)
                plan = infer_schema_heuristic(profile)
        plan = _apply_overrides(plan, args, profile)

    # If we ran tools, still let CLI overrides win.
    if agent_run is not None:
        plan = _apply_overrides(plan, args, profile)

    print(f"   variant={plan.variant_column} "
          f"({plan.control_label}/{plan.treatment_label}); "
          f"primary={plan.primary_metric}; "
          f"secondary={plan.secondary_metrics}; "
          f"covariates={plan.covariates}")

    # Optional CUPED variance reduction.
    cuped_note = ""
    if args.cuped:
        if args.cuped not in profile.columns:
            raise GuardrailError(f"--cuped covariate {args.cuped!r} not in CSV.")
        if args.cuped in plan.covariates:
            # Don't double-count: drop it from PSM covariates if we use it for CUPED.
            plan = SchemaPlan(
                **{**plan.to_dict(),
                   "covariates": [c for c in plan.covariates if c != args.cuped]}
            )
        df, cuped = apply_cuped(df, plan.primary_metric, args.cuped, in_place=True)
        cuped_note = (
            f" CUPED on {args.cuped}: θ={cuped.theta:.4f}, "
            f"variance reduction {cuped.variance_reduction:.1%}"
        )
        # Swap the primary metric to the adjusted column.
        plan = SchemaPlan(
            **{**plan.to_dict(), "primary_metric": cuped.adjusted_metric}
        )
        print(f"{_BOLD}{_CYAN}>> CUPED{_RESET}  {cuped_note.strip()}")

    # Statistical pipeline — re-run regardless of agent mode so the same
    # final numbers always make it into the report.
    if agent_run is None or args.cuped:
        print(f"{_BOLD}{_CYAN}>> SRM check{_RESET}")
        srm = srm_check(df, plan.variant_column, expected_ratio=expected_ratio)
        metric_cols = [plan.primary_metric, *plan.secondary_metrics]
        print(f"{_BOLD}{_CYAN}>> Metric tests{_RESET}  ({', '.join(metric_cols)})")
        metric_results = [
            metric_test(df, m, plan.variant_column, plan.control_label, plan.treatment_label)
            for m in metric_cols
        ]
        # CUPED already performs a covariate adjustment, so running PSM on
        # the CUPED-adjusted metric (with the remaining covariates) is an
        # over-correction. Skip PSM for the adjusted metric specifically.
        psm_metrics = [
            m for m in metric_cols
            if not (args.cuped and m.endswith("__cuped"))
        ]
        if psm_metrics and plan.covariates:
            print(f"{_BOLD}{_CYAN}>> Propensity score matching{_RESET}  ({', '.join(psm_metrics)})")
            psm_results = [
                propensity_score_match(
                    df,
                    metric=m,
                    variant_column=plan.variant_column,
                    control_label=plan.control_label,
                    treatment_label=plan.treatment_label,
                    covariates=plan.covariates,
                )
                for m in psm_metrics
            ]
        else:
            psm_results = []
            print(f"{_BOLD}{_CYAN}>> Propensity score matching{_RESET}  skipped "
                  f"(CUPED already adjusts for the covariate)")
    else:
        srm = agent_run.srm
        metric_results = agent_run.metric_results
        psm_results = agent_run.psm_results

    # Multiple-testing correction across metric tests.
    metric_results = apply_multiple_testing_correction(metric_results, method=args.correction)

    for m in metric_results:
        sig = (
            f"{_GREEN}sig{_RESET}"
            if m.significant_at_5pct
            else f"{_YELLOW}n.s.{_RESET}"
        )
        adj = f" adj.p={m.adjusted_p_value:.3g}" if m.adjusted_p_value is not None else ""
        print(f"   {m.metric}: Δ={m.absolute_diff:+.4f} p={m.primary_p_value:.3g}{adj} [{sig}]")
    for psm in psm_results:
        gamma = psm.rosenbaum.gamma_critical if psm.rosenbaum else None
        gstr = f" Γ_crit={gamma:.2f}" if gamma else " Γ_crit=robust"
        print(
            f"   PSM {psm.metric}: naive={psm.naive_effect:+.4f}  "
            f"ATT={psm.psm_att:+.4f} [{psm.psm_ci_low:+.4f},{psm.psm_ci_high:+.4f}]  "
            f"({psm.n_matched_pairs:,} pairs;{gstr})"
        )

    # Narrative provider — closes over agent mode.
    def _narrative_provider(payload):
        if args.agent == "tools" and agent_run is not None and not args.cuped:
            return agent_run.narrative or deterministic_narrative(payload)
        if args.agent == "none":
            return deterministic_narrative(payload)
        try:
            return narrate(payload)
        except AgentError as exc:
            print(f"   {_YELLOW}narration failed ({exc}); using deterministic{_RESET}",
                  file=sys.stderr)
            return deterministic_narrative(payload)

    out_path = args.out or Path("reports") / f"{args.csv.stem}_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    love_plot_path = _maybe_render_love_plot(
        psm_results, out_path.with_name(f"{out_path.stem}_love_plot.png")
    )

    print(f"{_BOLD}{_CYAN}>> Writing report{_RESET}")
    report = build_report(
        csv_path=args.csv,
        schema_rationale=plan.rationale + cuped_note,
        srm=srm,
        metric_results=metric_results,
        psm_results=psm_results,
        primary_metric=plan.primary_metric,
        narrative_provider=_narrative_provider,
        love_plot_path=love_plot_path,
    )
    out_path.write_text(report.markdown, encoding="utf-8")
    print(f"   wrote {out_path}")
    if love_plot_path:
        print(f"   wrote {love_plot_path}")

    colour = VERDICT_COLOUR.get(report.verdict, "")
    print()
    print(f"{_BOLD}Verdict:{_RESET} {colour}{_BOLD}{report.verdict}{_RESET}")
    for r in report.reasons:
        print(f"  - {r}")
    print()
    print(f"{_BOLD}Narrative:{_RESET}")
    print(report.narrative)
    print()

    if args.json:
        print(json.dumps(report.payload, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
