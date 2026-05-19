"""LLM agent layer (Anthropic Claude).

Three modes, increasing in agentic-ness:

* ``mode="none"``    — no LLM. Deterministic schema heuristic + deterministic
                       narrative. Used in CI, tests, and when no API key is set.
* ``mode="simple"``  — two LLM calls: one to infer the schema, one to write
                       the executive summary. Statistical computation stays
                       deterministic.
* ``mode="tools"``   — *agentic.* Claude is given three tools
                       (``run_srm_check``, ``run_metric_test``,
                       ``run_propensity_score_match``) and decides which to
                       call, on which columns, in which order. The LLM
                       drives the analysis; the host runs the math and
                       returns structured results. Final narrative is
                       Claude's last message. This is the closest to an
                       "AI Agent" in the modern (tool-using, multi-turn)
                       sense.

In every mode the *statistical numbers* in the report come from
``scipy.stats`` / ``scikit-learn``. The LLM never computes a p-value.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Callable

import pandas as pd

from .data_loader import DatasetProfile
from .exceptions import AgentError
from .guardrails import (
    MetricTestResult,
    PSMResult,
    SRMResult,
    metric_test,
    propensity_score_match,
    srm_check,
)


SCHEMA_MODEL = "claude-sonnet-4-5"
NARRATIVE_MODEL = "claude-sonnet-4-5"
AGENT_MODEL = "claude-sonnet-4-5"


@dataclass
class SchemaPlan:
    variant_column: str
    control_label: str
    treatment_label: str
    primary_metric: str
    secondary_metrics: list[str]
    covariates: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRun:
    """Result of a tool-use agent run."""
    plan: SchemaPlan
    srm: SRMResult
    metric_results: list[MetricTestResult]
    psm_results: list[PSMResult]
    narrative: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


# ===========================================================================
# Schema-inference heuristics (used as fallback when no LLM is available)
# ===========================================================================

_OUTCOME_TOKENS = {
    "converted", "conversion", "click", "clicked", "purchase", "purchased",
    "order", "ordered", "signup", "signed_up", "subscribe", "subscribed",
    "revenue", "spend", "spent", "value", "ltv", "gmv", "session", "retention",
}
_COVARIATE_PREFIXES = (
    "pre_", "prior_", "baseline_", "user_", "device", "country", "region",
)
_IDENTIFIER_TOKENS = {"user_id", "id", "uid", "userid", "session_id", "event_id"}


def _classify_column(name: str) -> str:
    """Return one of {'outcome', 'covariate', 'id', 'unknown'} based on name.

    Covariate prefixes (pre_, prior_, baseline_) win over outcome tokens in
    the name, since `pre_signup_value` is unambiguously a pre-treatment
    covariate despite containing "value".
    """
    lower = name.lower()
    if lower in _IDENTIFIER_TOKENS:
        return "id"
    if any(lower.startswith(p) for p in _COVARIATE_PREFIXES):
        return "covariate"
    if any(tok in lower for tok in _OUTCOME_TOKENS):
        return "outcome"
    return "unknown"


def infer_schema_heuristic(profile: DatasetProfile) -> SchemaPlan:
    """Deterministic fallback when no LLM is available."""
    preferred_names = {"variant", "group", "arm", "treatment", "bucket"}
    variant_col = next(
        (c for c in profile.binary_candidates if c.lower() in preferred_names),
        profile.binary_candidates[0] if profile.binary_candidates else None,
    )
    if variant_col is None:
        raise AgentError("No binary candidate columns to pick a variant from.")
    sample_vals = list(
        {row[variant_col] for row in profile.sample_rows if variant_col in row}
    )
    sample_vals_sorted = sorted(
        sample_vals, key=lambda v: (str(v).lower() != "control", str(v))
    )
    if len(sample_vals_sorted) < 2:
        raise AgentError(
            "Could not infer two distinct labels for the variant column "
            "from sample rows."
        )
    control_label, treatment_label = sample_vals_sorted[0], sample_vals_sorted[1]

    outcomes: list[str] = []
    covariates: list[str] = []
    for c in profile.columns:
        if c == variant_col:
            continue
        cls = _classify_column(c)
        if cls == "outcome" and c in profile.numeric_columns:
            outcomes.append(c)
        elif cls == "covariate":
            covariates.append(c)
        elif cls == "id":
            continue
        else:
            if c in profile.numeric_columns or c in profile.categorical_columns:
                covariates.append(c)

    if not outcomes:
        raise AgentError(
            "Heuristic schema inference could not identify an outcome column. "
            "Pass --primary-metric explicitly or set ANTHROPIC_API_KEY."
        )

    primary = next(
        (c for c in outcomes if c in profile.binary_candidates),
        outcomes[0],
    )
    secondary = [c for c in outcomes if c != primary]

    return SchemaPlan(
        variant_column=variant_col,
        control_label=str(control_label),
        treatment_label=str(treatment_label),
        primary_metric=primary,
        secondary_metrics=secondary,
        covariates=covariates,
        rationale="Heuristic fallback (no LLM call); classification by column name.",
    )


# ===========================================================================
# Simple mode: schema inference + narrative as two LLM calls
# ===========================================================================

_SCHEMA_SYSTEM = """You are an experimentation analyst inspecting a CSV from
an A/B test. Given the column names, dtypes, and sample rows, decide which
column is the variant/treatment assignment, which numeric columns are
outcome metrics, and which are pre-treatment covariates suitable for
propensity-score matching.

Return ONLY a JSON object with this exact shape — no prose, no markdown fences:

{
  "variant_column": "<column name>",
  "control_label": "<value in that column>",
  "treatment_label": "<value in that column>",
  "primary_metric": "<column name>",
  "secondary_metrics": ["<column name>", ...],
  "covariates": ["<column name>", ...],
  "rationale": "<one or two sentences>"
}

Rules:
- variant_column MUST be one of the binary_candidates.
- primary_metric and secondary_metrics MUST be numeric columns; they MUST
  NOT include the variant column or any covariate.
- covariates must be PRE-treatment (e.g. signup_value, device, country).
  Never use the outcome columns as covariates.
- A binary outcome (e.g. "converted") is preferred as primary_metric when
  present; otherwise prefer a continuous outcome with clear business
  meaning (e.g. revenue).
"""


def infer_schema(
    profile: DatasetProfile,
    *,
    allow_fallback: bool = True,
) -> SchemaPlan:
    """Use Claude to map the dataset's columns onto experiment roles.

    Falls back to :func:`infer_schema_heuristic` if ``ANTHROPIC_API_KEY``
    is not set (and ``allow_fallback=True``).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        if allow_fallback:
            return infer_schema_heuristic(profile)
        raise AgentError("ANTHROPIC_API_KEY not set and allow_fallback=False.")

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        if allow_fallback:
            return infer_schema_heuristic(profile)
        raise AgentError(f"anthropic SDK not installed: {exc}") from exc

    client = Anthropic(api_key=api_key)
    user_payload = {
        "n_rows": profile.n_rows,
        "columns": profile.columns,
        "dtypes": profile.dtypes,
        "numeric_columns": profile.numeric_columns,
        "binary_candidates": profile.binary_candidates,
        "sample_rows": profile.sample_rows,
    }

    try:
        msg = client.messages.create(
            model=SCHEMA_MODEL,
            max_tokens=600,
            system=_SCHEMA_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(user_payload, default=str)}],
        )
    except Exception as exc:
        raise AgentError(f"Schema inference call failed: {exc}") from exc

    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    plan_dict = _extract_json(text)
    return _validate_plan(plan_dict, profile)


_NARRATIVE_SYSTEM = """You are writing the executive summary for an A/B
test review. You will be given JSON containing structured statistical
results (SRM check, metric tests, propensity-matched ATT). Produce a
concise 4-6 sentence summary suitable for a launch-decision document.

Strict rules:
- Do NOT introduce numbers that are not in the JSON.
- Do NOT contradict the verdict field.
- If srm_detected is true, OPEN with the SRM finding — it dominates
  everything else.
- Mention the gap between naive_effect and psm_att when it is meaningful
  (>= 50% relative shrinkage or sign change).
- If a Rosenbaum gamma_critical is present, mention how robust the ATT
  is to hidden bias.
- Be plain English. No bullets, no markdown headings.
"""


def narrate(results_payload: dict[str, Any], *, allow_fallback: bool = True) -> str:
    """Render a plain-English narrative of the statistical results."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        if allow_fallback:
            return deterministic_narrative(results_payload)
        raise AgentError("ANTHROPIC_API_KEY not set and allow_fallback=False.")

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        if allow_fallback:
            return deterministic_narrative(results_payload)
        raise AgentError(f"anthropic SDK not installed: {exc}") from exc

    client = Anthropic(api_key=api_key)
    try:
        msg = client.messages.create(
            model=NARRATIVE_MODEL,
            max_tokens=600,
            system=_NARRATIVE_SYSTEM,
            messages=[
                {"role": "user", "content": json.dumps(results_payload, default=str)}
            ],
        )
    except Exception as exc:
        raise AgentError(f"Narrative generation failed: {exc}") from exc

    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


# ===========================================================================
# Tools mode: Claude orchestrates the analysis via tool calls
# ===========================================================================

_AGENT_SYSTEM = """You are an experimentation guardrail agent. You have
three tools that run statistical procedures on the user's A/B-test data:

1. `run_srm_check` — chi-square test for Sample Ratio Mismatch on the
   variant allocation.
2. `run_metric_test` — Welch's t-test + Mann-Whitney U (continuous) or
   chi-square + Newcombe CI (binary) for a single metric column.
3. `run_propensity_score_match` — 1-NN PSM with caliper, bootstrap SE,
   and Rosenbaum sensitivity bounds for a single outcome metric.

Strategy (follow this exactly):

Step 1. From the dataset profile in the user message, identify the
        variant column, control/treatment labels, primary metric, any
        secondary metrics, and pre-treatment covariates. Do not pick
        outcome columns as covariates or vice versa.
Step 2. Call `run_srm_check` first — if SRM is detected, the experiment
        is compromised regardless of metric results.
Step 3. Call `run_metric_test` for the primary metric, then each
        secondary metric.
Step 4. Call `run_propensity_score_match` for the primary metric (and
        secondary metrics if they're continuous).
Step 5. After all tools return, write a 4-6 sentence executive summary
        in plain English. Lead with SRM if detected; mention the
        naive-vs-ATT shrinkage when it is meaningful; mention Rosenbaum
        critical Γ when reported.

Do NOT compute numbers yourself; trust the tool outputs. Do NOT call any
tool more than once for the same metric. Return only the final summary
text in your last message.
"""

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "run_srm_check",
        "description": (
            "Chi-square Sample Ratio Mismatch test. Detects whether the "
            "observed variant allocation differs from expected at p < 0.001."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "variant_column": {"type": "string"},
                "expected_ratio": {
                    "type": "object",
                    "description": (
                        "Map of variant label -> expected proportion. "
                        "Omit for uniform across observed levels."
                    ),
                    "additionalProperties": {"type": "number"},
                },
            },
            "required": ["variant_column"],
        },
    },
    {
        "name": "run_metric_test",
        "description": (
            "Run the appropriate metric-shift test (chi-square + Newcombe "
            "CI for binary; Welch's t-test + Mann-Whitney U for continuous) "
            "on a single metric column."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string"},
                "variant_column": {"type": "string"},
                "control_label": {"type": "string"},
                "treatment_label": {"type": "string"},
            },
            "required": ["metric", "variant_column", "control_label", "treatment_label"],
        },
    },
    {
        "name": "run_propensity_score_match",
        "description": (
            "Estimate the Average Treatment Effect on the Treated (ATT) via "
            "1-NN propensity-score matching with caliper, common-support "
            "trimming, bootstrap SE, and Rosenbaum sensitivity bounds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string"},
                "variant_column": {"type": "string"},
                "control_label": {"type": "string"},
                "treatment_label": {"type": "string"},
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": [
                "metric",
                "variant_column",
                "control_label",
                "treatment_label",
                "covariates",
            ],
        },
    },
]


def _execute_tool(
    name: str, args: dict[str, Any], df: pd.DataFrame, run: dict[str, Any]
) -> dict[str, Any]:
    """Dispatch a tool call to the corresponding deterministic function."""
    if name == "run_srm_check":
        result = srm_check(
            df,
            variant_column=args["variant_column"],
            expected_ratio=args.get("expected_ratio"),
        )
        run["srm"] = result
        return result.to_dict()

    if name == "run_metric_test":
        result = metric_test(
            df,
            metric=args["metric"],
            variant_column=args["variant_column"],
            control_label=args["control_label"],
            treatment_label=args["treatment_label"],
        )
        run["metric_results"].append(result)
        return result.to_dict()

    if name == "run_propensity_score_match":
        result = propensity_score_match(
            df,
            metric=args["metric"],
            variant_column=args["variant_column"],
            control_label=args["control_label"],
            treatment_label=args["treatment_label"],
            covariates=args["covariates"],
        )
        run["psm_results"].append(result)
        return result.to_dict()

    raise AgentError(f"Unknown tool: {name!r}")


def run_tool_agent(
    df: pd.DataFrame,
    profile: DatasetProfile,
    *,
    max_turns: int = 12,
    progress: Callable[[str], None] | None = None,
) -> AgentRun:
    """Run the tool-use agent. Returns the structured results + narrative.

    Falls through to :func:`AgentError` if the API key is missing or the
    SDK is not installed — the caller is responsible for catching and
    degrading to ``simple`` or ``none`` mode.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AgentError("ANTHROPIC_API_KEY not set; cannot use --agent tools.")
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise AgentError(f"anthropic SDK not installed: {exc}") from exc

    client = Anthropic(api_key=api_key)
    user_payload = {
        "n_rows": profile.n_rows,
        "columns": profile.columns,
        "dtypes": profile.dtypes,
        "numeric_columns": profile.numeric_columns,
        "binary_candidates": profile.binary_candidates,
        "sample_rows": profile.sample_rows,
    }
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": json.dumps(user_payload, default=str)}
    ]
    run: dict[str, Any] = {
        "srm": None,
        "metric_results": [],
        "psm_results": [],
    }
    tool_calls: list[dict[str, Any]] = []
    narrative: str = ""

    for _ in range(max_turns):
        try:
            resp = client.messages.create(
                model=AGENT_MODEL,
                max_tokens=2048,
                system=_AGENT_SYSTEM,
                tools=_TOOLS,
                messages=messages,
            )
        except Exception as exc:
            raise AgentError(f"Tool-agent API call failed: {exc}") from exc

        if resp.stop_reason == "end_turn":
            # Final narrative is whatever text is in the last assistant message.
            narrative = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
            break

        if resp.stop_reason != "tool_use":
            # Unexpected; surface whatever text is there as the narrative.
            narrative = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
            break

        # Echo the assistant turn (must include tool_use blocks).
        messages.append({"role": "assistant", "content": resp.content})
        tool_results_blocks: list[dict[str, Any]] = []
        for block in resp.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            name = block.name
            args = block.input or {}
            tool_calls.append({"tool": name, "args": args})
            if progress:
                progress(f"tool: {name}({json.dumps(args, default=str)})")
            try:
                out = _execute_tool(name, args, df, run)
                tool_results_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(out, default=str),
                    }
                )
            except Exception as exc:
                tool_results_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"ERROR: {exc}",
                        "is_error": True,
                    }
                )
        messages.append({"role": "user", "content": tool_results_blocks})
    else:
        raise AgentError(f"Tool agent did not converge in {max_turns} turns.")

    # Synthesise a SchemaPlan from the observed tool calls so the report
    # and verdict logic don't have to know which mode produced the results.
    plan = _plan_from_tool_calls(tool_calls, profile, run)

    if run["srm"] is None:
        raise AgentError("Agent never called run_srm_check.")

    return AgentRun(
        plan=plan,
        srm=run["srm"],
        metric_results=run["metric_results"],
        psm_results=run["psm_results"],
        narrative=narrative or deterministic_narrative(_payload_from_run(run, plan)),
        tool_calls=tool_calls,
    )


def _plan_from_tool_calls(
    tool_calls: list[dict[str, Any]], profile: DatasetProfile, run: dict[str, Any]
) -> SchemaPlan:
    """Reverse-engineer a SchemaPlan from the agent's actual tool calls."""
    srm_call = next((c for c in tool_calls if c["tool"] == "run_srm_check"), None)
    metric_calls = [c for c in tool_calls if c["tool"] == "run_metric_test"]
    psm_calls = [c for c in tool_calls if c["tool"] == "run_propensity_score_match"]
    if srm_call is None or not metric_calls:
        raise AgentError("Agent did not call the required tools.")

    variant_column = srm_call["args"]["variant_column"]
    first_metric = metric_calls[0]["args"]
    control_label = first_metric["control_label"]
    treatment_label = first_metric["treatment_label"]
    primary_metric = first_metric["metric"]
    secondary_metrics = [c["args"]["metric"] for c in metric_calls[1:]]
    covariates: list[str] = []
    for c in psm_calls:
        for cov in c["args"]["covariates"]:
            if cov not in covariates:
                covariates.append(cov)

    return SchemaPlan(
        variant_column=variant_column,
        control_label=control_label,
        treatment_label=treatment_label,
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
        covariates=covariates,
        rationale=f"Tool-use agent orchestrated {len(tool_calls)} tool calls.",
    )


def _payload_from_run(run: dict[str, Any], plan: SchemaPlan) -> dict[str, Any]:
    return {
        "verdict": "PENDING",
        "schema_rationale": plan.rationale,
        "srm": run["srm"].to_dict() if run["srm"] else {},
        "metric_tests": [m.to_dict() for m in run["metric_results"]],
        "psm": [p.to_dict() for p in run["psm_results"]],
    }


# ===========================================================================
# helpers
# ===========================================================================

def _extract_json(text: str) -> dict[str, Any]:
    """Tolerate code fences or trailing prose from the LLM."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fence.group(1) if fence else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not m:
            raise AgentError(f"No JSON object found in LLM response: {text[:200]!r}")
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise AgentError(f"LLM JSON did not parse: {exc}; text={text[:200]!r}") from exc


def _validate_plan(plan: dict[str, Any], profile: DatasetProfile) -> SchemaPlan:
    required = {
        "variant_column", "control_label", "treatment_label",
        "primary_metric", "secondary_metrics", "covariates",
    }
    missing = required - plan.keys()
    if missing:
        raise AgentError(f"LLM schema plan missing fields: {sorted(missing)}")
    if plan["variant_column"] not in profile.binary_candidates:
        raise AgentError(
            f"LLM chose variant_column={plan['variant_column']!r} "
            f"which is not a binary candidate ({profile.binary_candidates})."
        )
    if plan["primary_metric"] not in profile.numeric_columns:
        raise AgentError(
            f"LLM chose primary_metric={plan['primary_metric']!r} "
            f"which is not numeric."
        )
    for sm in plan["secondary_metrics"]:
        if sm not in profile.numeric_columns:
            raise AgentError(f"secondary metric {sm!r} is not numeric.")
    for cv in plan["covariates"]:
        if cv not in profile.columns:
            raise AgentError(f"covariate {cv!r} not in DataFrame.")
    return SchemaPlan(
        variant_column=plan["variant_column"],
        control_label=str(plan["control_label"]),
        treatment_label=str(plan["treatment_label"]),
        primary_metric=plan["primary_metric"],
        secondary_metrics=list(plan["secondary_metrics"]),
        covariates=list(plan["covariates"]),
        rationale=str(plan.get("rationale", "")),
    )


def deterministic_narrative(payload: dict[str, Any]) -> str:
    """No-LLM fallback so the CLI is always usable."""
    verdict = payload.get("verdict", "UNKNOWN")
    srm = payload.get("srm", {})
    metrics = payload.get("metric_tests", [])
    psm_blocks = payload.get("psm", [])

    parts: list[str] = []
    if srm.get("srm_detected"):
        parts.append(
            f"Sample Ratio Mismatch detected (chi-square p={srm['p_value']:.2g}; "
            f"observed split {srm['observed_ratio']}, expected {srm['expected_ratio']}). "
            "The experiment is unsafe to interpret until the randomiser is fixed."
        )
    elif srm:
        parts.append(
            f"Allocation is consistent with the planned split "
            f"(chi-square p={srm['p_value']:.2g})."
        )

    for m in metrics:
        sign = "uplift" if m["absolute_diff"] > 0 else "drop"
        parts.append(
            f"{m['metric']}: {m['treatment_mean']:.4f} vs {m['control_mean']:.4f} "
            f"({sign} of {m['absolute_diff']:+.4f}, "
            f"{m['primary_test']} p={m['primary_p_value']:.2g})."
        )

    for p in psm_blocks:
        gamma = p.get("rosenbaum", {}).get("gamma_critical") if p.get("rosenbaum") else None
        gamma_str = (
            f"; robust to hidden bias up to Γ ≈ {gamma:.2f}" if gamma else ""
        )
        parts.append(
            f"After propensity matching on {', '.join(p['covariates'])}, "
            f"the ATT for {p['metric']} is {p['psm_att']:+.4f} "
            f"(naive estimate was {p['naive_effect']:+.4f}; "
            f"{p['n_matched_pairs']:,} matched pairs{gamma_str})."
        )

    parts.append(f"Final verdict: {verdict}.")
    return " ".join(parts)
