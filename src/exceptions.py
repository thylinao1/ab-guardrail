"""Typed exceptions for the experimentation-guardrail agent.

Using a small hierarchy of custom errors makes the CLI's failure modes
explicit. The top-level `GuardrailError` is what callers should generally
catch when running the pipeline.
"""

from __future__ import annotations


class GuardrailError(Exception):
    """Base class for all errors raised by this package."""


class DataValidationError(GuardrailError):
    """The input CSV is missing required columns, has wrong dtypes,
    or contains structurally invalid values (e.g. fewer than two variants)."""


class StatisticalCheckError(GuardrailError):
    """A statistical routine could not be executed — e.g. degenerate
    sample sizes, zero variance, or a singular design matrix in PSM."""


class AgentError(GuardrailError):
    """The LLM agent failed to produce a usable response —
    network error, schema-violating JSON, or missing API key."""
