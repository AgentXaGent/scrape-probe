"""Output validation for Stage 4 responses.

Implements the cross-reference check documented in
``docs/how-it-works.md`` under the **hallucinated endpoint** failure mode:

    The validation layer is explicit: Pydantic enforces schema shape on
    the response, and a cross-reference check confirms that any endpoint
    URL Claude recommends appears verbatim somewhere in the stage 1-3
    findings dictionary. Endpoints that fail the cross-reference get
    stripped from the output before display and logged.

Pydantic shape validation lives in ``cartograph_ai.schema`` and runs
inside ``stages.claude_classify.classify()``. This module handles the
second guard: every URL-looking value in
``extraction_strategy.specifics`` must appear verbatim in the JSON
serialisation of the Stage 1-3 payload, or it is treated as
hallucinated and stripped.

The check is conservative on purpose. A recommendation that cannot be
traced back to evidence the probe actually observed does not ship.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from cartograph_ai.schema import ClaudeResponse

# Prefixes we consider "URL-shaped". Root-relative paths (/api/...) count
# because they are the most common endpoint shape the prompt emits for
# REST and WordPress recommendations.
_URL_LIKE_PREFIXES: tuple[str, ...] = ("http://", "https://", "//", "/")


@dataclass
class ValidationReport:
    """Result of cross-referencing a ``ClaudeResponse`` against the probe payload."""

    response: ClaudeResponse
    """The response, with any unsupported URLs stripped from
    ``extraction_strategy.specifics``."""

    stripped_endpoints: list[str] = field(default_factory=list)
    """URLs that were removed because they did not appear in the probe
    payload. Logged at the orchestrator level."""

    @property
    def had_stripped_endpoints(self) -> bool:
        return bool(self.stripped_endpoints)


def cross_reference_endpoints(
    response: ClaudeResponse,
    *,
    probe_payload: dict[str, Any],
) -> ValidationReport:
    """Strip URLs from ``response.extraction_strategy.specifics`` that are
    not present in the probe payload.

    Args:
        response: The parsed Stage 4 response.
        probe_payload: The structured findings dict from Stages 1-3 that
            was sent to the model (the same dict ``build_prompt`` ran
            against).

    Returns:
        A ``ValidationReport`` carrying the cleaned response and the list
        of stripped URLs. If nothing was stripped the cleaned response
        is the original instance.
    """
    haystack = _payload_text(probe_payload)
    stripped: list[str] = []
    new_specifics: dict[str, Any] = {}

    for key, value in response.extraction_strategy.specifics.items():
        if _is_url_like(value):
            if _verbatim_match(value, haystack):
                new_specifics[key] = value
            else:
                stripped.append(value)
            continue

        if isinstance(value, list):
            cleaned_list: list[Any] = []
            for item in value:
                if _is_url_like(item) and not _verbatim_match(item, haystack):
                    stripped.append(item)
                else:
                    cleaned_list.append(item)
            new_specifics[key] = cleaned_list
            continue

        if isinstance(value, dict):
            cleaned_dict, nested_stripped = _strip_dict_urls(value, haystack)
            new_specifics[key] = cleaned_dict
            stripped.extend(nested_stripped)
            continue

        new_specifics[key] = value

    if not stripped:
        return ValidationReport(response=response, stripped_endpoints=[])

    cleaned_strategy = response.extraction_strategy.model_copy(
        update={"specifics": new_specifics}
    )
    cleaned_response = response.model_copy(
        update={"extraction_strategy": cleaned_strategy}
    )
    return ValidationReport(response=cleaned_response, stripped_endpoints=stripped)


# ---------------- Helpers ---------------------------------------------


def _payload_text(payload: dict[str, Any]) -> str:
    """Flatten the probe payload to a searchable string.

    JSON serialisation is used so URL-looking string values appear with
    the same quoting and escaping the search will encounter.
    """
    return json.dumps(payload, sort_keys=True, default=str)


def _is_url_like(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith(_URL_LIKE_PREFIXES)


def _verbatim_match(needle: str, haystack: str) -> bool:
    """Conservative match: needle must appear verbatim in haystack.

    A surrounding quote/brace check would catch a few extra cases but
    would also miss legitimate hits where the URL appears inside another
    JSON value. Substring containment is intentional.
    """
    return needle in haystack


def _strip_dict_urls(
    obj: dict[str, Any], haystack: str
) -> tuple[dict[str, Any], list[str]]:
    """Recurse one level into a nested dict, stripping unsupported URLs."""
    cleaned: dict[str, Any] = {}
    stripped: list[str] = []
    for k, v in obj.items():
        if _is_url_like(v):
            if _verbatim_match(v, haystack):
                cleaned[k] = v
            else:
                stripped.append(v)
            continue
        if isinstance(v, list):
            sub: list[Any] = []
            for item in v:
                if _is_url_like(item) and not _verbatim_match(item, haystack):
                    stripped.append(item)
                else:
                    sub.append(item)
            cleaned[k] = sub
            continue
        cleaned[k] = v
    return cleaned, stripped
