"""Command-line entry point.

Three execution modes, set with --mode:

    pipeline  (default)  Deterministic routing. Column roles are inferred
                         by a hardcoded heuristic; SRM / metric tests / PSM
                         run as a plain Python pipeline. The LLM is invoked
                         exactly once, at the very end, to turn the
                         deterministic JSON into a plain-English summary.
                         No LLM latency or cost sits in the routing loop.

    agent     (opt-in)   Claude drives routing through a tool-use loop.
                         Useful when column roles cannot be inferred
                         deterministically (unfamiliar schema). Slower and
                         more expensive, so it is not the default.

    offline              Fully deterministic. No API calls at all. The
                         summary is rendered from a template. Used in CI
                         and offline runs; byte-stable.

Examples:

    ab-guardrail data/clean_experiment.csv
    ab-guardrail data/compromised_experiment.csv --mode offline
    ab-guardrail data/messy_experiment.csv --mode pipeline
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
    infer_schema_heuristic,
    narrate,
    run_tool_agent,
)
from .data_loader import DatasetProfile, load_experiment_csv
from .exceptions import AgentError, GuardrailError
from .guardrails import metric_test, propensity_score_match, srm_check
from .guardrails.metric_tests import apply_cuped, apply_multiple_testing_correction
from .report import (
    VERDICT_COMPROMISED,
    VERDICT_NULL,
    VERDICT_SAFE,
    build_report,
    render_love_plot,
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
            "Automated experimentation guardrail. Reads an A/B-test CSV, "
            "checks for SRM, runs metric-shift tests, applies propensity-score "
            "matching with bootstrap SE and Rosenbaum bounds, and writes a "
            "Markdown report. Routing is deterministic by default; the LLM "
            "only writes the closing summary."
        ),
    )
    p.add_argument("csv", type=Path, help="Path to the experiment CSV.")
    p.add_argument(
        "--mode",
        choices=["pipeline", "agent", "offline"],
        default="pipeline",
        help=(
            "Execution mode. "
            "'pipeline' (default): deterministic routing, LLM writes only the "
            "final summary. "
            "'agent': Claude tool-use loop drives routing (opt-in; for "
            "unfamiliar schemas). "
            "'offline': no API calls at all."
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
        "--covariates", help="Comma-separated list of covariate columns (for PSM)."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the Markdown report (default: reports/<csv_stem>_report.md).",
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
    # --- load + clean ------------------------------------------------------
    print(f"{_BOLD}{_CYAN}>> Loading{_RESET} {args.csv}")
    df, profile = load_experiment_csv(args.csv)
    print(f"   {profile.n_rows:,} rows, {len(profile.columns)} columns.")
    if profile.quality.has_issues:
        print(f"   {_YELLOW}data-quality notes:{_RESET}")
        for line in profile.quality.summary_lines():
            print(f"     {_DIM}- {line}{_RESET}")

    expected_ratio = None
    if args.expected_ratio:
        try:
            expected_ratio = json.loads(args.expected_ratio)
        except json.JSONDecodeError as exc:
            raise GuardrailError(f"--expected-ratio is not valid JSON: {exc}") from exc

    # --- routing -----------------------------------------------------------
    agent_run: AgentRun | None = None
    plan: SchemaPlan

    if args.mode == "agent":
        try:
            print(f"{_BOLD}{_CYAN}>> Routing (agent){_RESET}  "
                  f"Claude tool-use loop")
            agent_run = run_tool_agent(
                df, profile, progress=lambda s: print(f"   {_DIM}{s}{_RESET}")
            )
            plan = agent_run.plan
        except AgentError as exc:
            print(f"   {_YELLOW}agent routing unavailable ({exc}); "
                  f"using deterministic routing{_RESET}", file=sys.stderr)
            plan = infer_schema_heuristic(profile)
    else:
        print(f"{_BOLD}{_CYAN}>> Routing (deterministic){_RESET}  "
              f"hardcoded heuristic, no LLM")
        plan = infer_schema_heuristic(profile)

    plan = _apply_overrides(plan, args, profile)
    print(f"   variant={plan.variant_column} "
          f"({plan.control_label}/{plan.treatment_label}); "
          f"primary={plan.primary_metric}; "
          f"secondary={plan.secondary_metrics}; "
          f"covariates={plan.covariates}")

    # --- optional CUPED variance reduction --------------------------------
    cuped_note = ""
    if args.cuped:
        if args.cuped not in profile.columns:
            raise GuardrailError(f"--cuped covariate {args.cuped!r} not in CSV.")
        if args.cuped in plan.covariates:
            plan = SchemaPlan(
                **{**plan.to_dict(),
                   "covariates": [c for c in plan.covariates if c != args.cuped]}
            )
        df, cuped = apply_cuped(df, plan.primary_metric, args.cuped, in_place=True)
        cuped_note = (
            f" CUPED on {args.cuped}: theta={cuped.theta:.4f}, "
            f"variance reduction {cuped.variance_reduction:.1%}"
        )
        plan = SchemaPlan(**{**plan.to_dict(), "primary_metric": cuped.adjusted_metric})
        print(f"{_BOLD}{_CYAN}>> CUPED{_RESET}  {cuped_note.strip()}")

    # --- statistical pipeline (always deterministic) ----------------------
    print(f"{_BOLD}{_CYAN}>> SRM check{_RESET}")
    srm = srm_check(df, plan.variant_column, expected_ratio=expected_ratio)

    metric_cols = [plan.primary_metric, *plan.secondary_metrics]
    print(f"{_BOLD}{_CYAN}>> Metric tests{_RESET}  ({', '.join(metric_cols)})")
    metric_results = [
        metric_test(df, m, plan.variant_column, plan.control_label, plan.treatment_label)
        for m in metric_cols
    ]
    metric_results = apply_multiple_testing_correction(
        metric_results, method=args.correction
    )

    psm_metrics = [
        m for m in metric_cols if not (args.cuped and m.endswith("__cuped"))
    ]
    if psm_metrics and plan.covariates:
        print(f"{_BOLD}{_CYAN}>> Propensity score matching{_RESET}  "
              f"({', '.join(psm_metrics)})")
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
              f"({'CUPED already adjusts for the covariate' if args.cuped else 'no covariates'})")

    for m in metric_results:
        sig = f"{_GREEN}sig{_RESET}" if m.significant_at_5pct else f"{_YELLOW}n.s.{_RESET}"
        adj = f" adj.p={m.adjusted_p_value:.3g}" if m.adjusted_p_value is not None else ""
        print(f"   {m.metric}: delta={m.absolute_diff:+.4f} "
              f"p={m.primary_p_value:.3g}{adj} [{sig}]")
    for psm in psm_results:
        gamma = psm.rosenbaum.gamma_critical if psm.rosenbaum else None
        gstr = f" gamma_crit={gamma:.2f}" if gamma else " gamma_crit=above grid"
        print(f"   PSM {psm.metric}: naive={psm.naive_effect:+.4f}  "
              f"ATT={psm.psm_att:+.4f} [{psm.psm_ci_low:+.4f},{psm.psm_ci_high:+.4f}]  "
              f"({psm.n_matched_pairs:,} pairs;{gstr})")

    # --- closing summary: the ONLY place the LLM is invoked ---------------
    def _narrative_provider(payload):
        if args.mode == "offline":
            return deterministic_narrative(payload)
        if args.mode == "agent" and agent_run is not None and not args.cuped:
            return agent_run.narrative or deterministic_narrative(payload)
        # pipeline mode: single LLM call on the finished, deterministic JSON.
        try:
            return narrate(payload)
        except AgentError as exc:
            print(f"   {_YELLOW}summary LLM call failed ({exc}); "
                  f"using template{_RESET}", file=sys.stderr)
            return deterministic_narrative(payload)

    out_path = args.out or Path("reports") / f"{args.csv.stem}_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    love_plot_path = render_love_plot(
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
        quality=profile.quality,
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
    print(f"{_BOLD}Summary:{_RESET}")
    print(report.narrative)
    print()

    if args.json:
        print(json.dumps(report.payload, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
