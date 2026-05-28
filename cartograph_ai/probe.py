"""The cartograph probe orchestrator.

Public entry point for the library. Wires the four stages together,
applies validation, and assembles the final ``ProbeResult``.

Stages 1 (HTTP), 2 (HTML analysis), and 4 (Claude classification) always
run in Phase 1. Stage 3 (JS execution) is skipped and reported in
``probe_stages_skipped``; users opt in by installing the ``browser``
extra (Phase 2).
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from cartograph_ai.exceptions import (
    AuthWalledError,
    CartographError,
    HTMLAnalysisError,
    HTTPProbeError,
    LowConfidenceError,
)
from cartograph_ai.schema import (
    Classification,
    EndpointDescriptor,
    ExtractionStrategy,
    ProbeResult,
)
from cartograph_ai.stages.claude_classify import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    ClassificationResult,
    classify,
)
from cartograph_ai.stages.html_analysis import analyze_html
from cartograph_ai.stages.http_probe import DEFAULT_USER_AGENT, probe_http
from cartograph_ai.validation import cross_reference_endpoints

log = logging.getLogger("cartograph_ai")

LOW_CONFIDENCE_THRESHOLD = 0.7
"""Confidence below this triggers the low-confidence warning in default
mode and a hard refusal in ``--strict`` mode. Defined here so callers can
override on a per-probe basis if their workflow needs a different bar."""


@dataclass
class ProbeOptions:
    """Configuration for a single probe call.

    Attributes:
        strict: If True, raise ``LowConfidenceError`` when the model's
            confidence falls below ``LOW_CONFIDENCE_THRESHOLD``. Default
            behaviour returns the result with ``low_confidence_warning``
            set instead.
        debug: If True, log the assembled Stage 4 payload at DEBUG level
            so ``--debug`` can route it to stderr.
        model: The Claude model to use at Stage 4.
        max_tokens: Output token cap for the Stage 4 call.
        timeout: Per-request timeout (seconds) for Stage 1 fetches.
        user_agent: User-Agent header for Stage 1 fetches.
        retry_on_stage1_failure: If True, Stage 1 transient errors get
            one retry with a half-second backoff before the orchestrator
            raises.
    """

    strict: bool = False
    debug: bool = False
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = 10.0
    user_agent: str = DEFAULT_USER_AGENT
    retry_on_stage1_failure: bool = True


def probe(
    url: str,
    *,
    anthropic_client: Any = None,
    http_client: Optional[httpx.Client] = None,
    options: Optional[ProbeOptions] = None,
) -> ProbeResult:
    """Run the full probe pipeline against ``url``.

    Args:
        url: The URL to probe.
        anthropic_client: An Anthropic-compatible client. If ``None``, a
            default ``anthropic.Anthropic()`` is constructed (which picks
            up ``ANTHROPIC_API_KEY`` from the environment).
        http_client: An optional pre-configured ``httpx.Client`` to share
            across stages. If ``None``, a transient client is used.
        options: A ``ProbeOptions`` instance. Defaults if omitted.

    Returns:
        A ``ProbeResult`` matching the schema in
        ``docs/how-it-works.md``.

    Raises:
        HTTPProbeError: Stage 1 could not reach the target across retries.
        AuthWalledError: Target requires authentication (401).
        HTMLAnalysisError: Stage 2 had no HTML to walk.
        ClassificationError: Stage 4 failed to parse the model response.
        LowConfidenceError: ``strict`` was True and confidence dropped
            below threshold.
    """
    opts = options or ProbeOptions()
    if anthropic_client is None:
        anthropic_client = _default_anthropic_client()

    stages_completed: list[str] = []
    stages_skipped: list[str] = ["js_execution"]
    skip_reason = "Phase 1 does not run Stage 3 (install [browser] extra to enable)"
    extra_limitations: list[str] = []

    # ---- Stage 1: HTTP probe -----------------------------------------
    stage1 = _run_stage1(url, http_client=http_client, opts=opts)
    stages_completed.append("http")

    if stage1["error"]:
        raise HTTPProbeError(f"Stage 1 failed for {url}: {stage1['error']}")

    status = stage1["status"]
    if status == 401:
        raise AuthWalledError(
            f"Target {url} requires authentication (HTTP 401). "
            "Phase 3 may add authenticated-probe support."
        )

    body = stage1.get("body") or ""
    if not body:
        raise HTMLAnalysisError(
            f"Target {url} returned no HTML body (status {status}); "
            "cannot run Stage 2 analysis."
        )

    # ---- Stage 2: HTML analysis --------------------------------------
    stage2 = analyze_html(body, stage1.get("final_url") or url)
    stages_completed.append("html_analysis")

    # ---- Stage 4: Claude classification ------------------------------
    probe_payload = {
        "url": url,
        "stage1": _stage1_for_payload(stage1),
        "stage2": stage2,
    }
    if opts.debug:
        log.debug("cartograph probe payload assembled for %s", url)

    classify_result = classify(
        probe_payload=probe_payload,
        client=anthropic_client,
        model=opts.model,
        max_tokens=opts.max_tokens,
    )
    stages_completed.append("claude_classify")

    # ---- Validation: strip hallucinated endpoints --------------------
    report = cross_reference_endpoints(
        classify_result.response, probe_payload=probe_payload
    )
    if report.stripped_endpoints:
        log.warning(
            "cartograph stripped %d hallucinated endpoint(s) from response: %s",
            len(report.stripped_endpoints),
            report.stripped_endpoints,
        )
        extra_limitations.append(
            "cartograph stripped "
            f"{len(report.stripped_endpoints)} endpoint(s) the model "
            "recommended that did not appear in the probe input."
        )

    cleaned_response = report.response

    # ---- Confidence handling -----------------------------------------
    low_confidence = cleaned_response.confidence < LOW_CONFIDENCE_THRESHOLD
    if low_confidence and opts.strict:
        raise LowConfidenceError(
            f"Confidence {cleaned_response.confidence:.2f} is below threshold "
            f"{LOW_CONFIDENCE_THRESHOLD} and strict mode was requested. "
            f"Limitations: {cleaned_response.limitations or 'none reported'}"
        )

    # ---- Assemble public output --------------------------------------
    return ProbeResult(
        url=url,
        probe_timestamp=_dt.datetime.now(_dt.timezone.utc),
        model=classify_result.model,
        classification=Classification(
            category=cleaned_response.classification,
            subcategory=cleaned_response.extraction_strategy.method or None,
            confidence=cleaned_response.confidence,
            reasoning=cleaned_response.reasoning,
        ),
        endpoints_discovered=_build_endpoints_discovered(stage2),
        extraction_strategy=cleaned_response.extraction_strategy,
        probe_stages_completed=stages_completed,
        probe_stages_skipped=stages_skipped,
        skip_reason=skip_reason,
        limitations=list(cleaned_response.limitations) + extra_limitations,
        low_confidence_warning=low_confidence,
    )


# ---------------- Helpers ---------------------------------------------


def _default_anthropic_client() -> Any:
    """Lazy import of the anthropic SDK so the library imports cleanly
    even when the SDK is not installed (e.g., during ``pip install -e .``
    without the API key configured)."""
    try:
        from anthropic import Anthropic  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import guard
        raise CartographError(
            "The anthropic package is required to run the probe. "
            "Install with 'pip install cartograph-ai' or pass a custom client."
        ) from exc
    return Anthropic()


def _run_stage1(
    url: str,
    *,
    http_client: Optional[httpx.Client],
    opts: ProbeOptions,
) -> dict[str, Any]:
    """Run probe_http with an optional single retry on transient error."""
    stage1 = probe_http(
        url,
        client=http_client,
        timeout=opts.timeout,
        user_agent=opts.user_agent,
    )
    if stage1["error"] and opts.retry_on_stage1_failure:
        log.info("cartograph Stage 1 transient error; retrying once: %s", stage1["error"])
        time.sleep(0.5)
        stage1 = probe_http(
            url,
            client=http_client,
            timeout=opts.timeout,
            user_agent=opts.user_agent,
        )
    return stage1


def _stage1_for_payload(stage1: dict[str, Any]) -> dict[str, Any]:
    """Drop the raw HTML body before serialising Stage 1 into the prompt.

    Stage 2's structured findings already represent the body. Sending
    the raw HTML on top would multiply token cost without adding signal.
    """
    summary = dict(stage1)
    summary.pop("body", None)
    return summary


def _build_endpoints_discovered(stage2: dict[str, Any]) -> list[EndpointDescriptor]:
    """Convert Stage 2 ``api_endpoints`` entries into EndpointDescriptors."""
    out: list[EndpointDescriptor] = []
    seen: set[str] = set()
    for endpoint in stage2.get("api_endpoints", []):
        url = endpoint.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(
            EndpointDescriptor(
                url=url,
                type=endpoint.get("type", "unknown"),
            )
        )
    return out
