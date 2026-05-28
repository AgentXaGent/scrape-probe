"""End-to-end tests for the probe orchestrator with mocked Stage 1 + Stage 4."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
import pytest
import respx

from cartograph_ai import (
    AuthWalledError,
    HTMLAnalysisError,
    HTTPProbeError,
    LowConfidenceError,
    ProbeOptions,
    ProbeResult,
    probe,
)


# ---------------- Stubs --------------------------------------------------


@dataclass
class StubUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class StubTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class StubMessage:
    content: list
    model: str = "claude-sonnet-4-6"
    usage: StubUsage = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = StubUsage(input_tokens=1500, output_tokens=300)


class StubMessagesEndpoint:
    def __init__(self, response: StubMessage):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class StubAnthropic:
    def __init__(self, response: StubMessage):
        self.messages = StubMessagesEndpoint(response)


def _claude_response_text(
    *,
    classification: str = "direct_api",
    confidence: float = 0.94,
    reasoning: str = "Algolia endpoint discovered in inline script.",
    method: str = "algolia_search",
    specifics: dict | None = None,
    limitations: list | None = None,
) -> str:
    return json.dumps(
        {
            "classification": classification,
            "confidence": confidence,
            "reasoning": reasoning,
            "extraction_strategy": {
                "method": method,
                "requires_browser": False,
                "estimated_requests": 2,
                "recommended_tool": "requests",
                "specifics": specifics or {"app_id": "AHNZ21XTZ6"},
            },
            "limitations": limitations or [],
        }
    )


SASAKI_HTML = """\
<!doctype html>
<html>
<head><title>Sasaki Projects</title></head>
<body>
  <h1>Projects</h1>
  <p>Real visible content describing 90 portfolio projects.</p>
  <script>
    fetch("https://AHNZ21XTZ6-dsn.algolia.net/1/indexes/prod_projects/query");
  </script>
</body>
</html>
"""


# ---------------- Happy path -------------------------------------------


@respx.mock
def test_probe_end_to_end_happy_path():
    respx.get("https://sasaki.com/projects").mock(
        return_value=httpx.Response(200, content=SASAKI_HTML, headers={"server": "nginx"})
    )
    respx.get("https://sasaki.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://sasaki.com/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://sasaki.com/sitemap_index.xml").mock(return_value=httpx.Response(404))

    client = StubAnthropic(
        StubMessage(content=[StubTextBlock(text=_claude_response_text())])
    )
    http_client = httpx.Client(follow_redirects=True, timeout=5.0)
    try:
        result = probe(
            "https://sasaki.com/projects",
            anthropic_client=client,
            http_client=http_client,
        )
    finally:
        http_client.close()

    assert isinstance(result, ProbeResult)
    assert result.url == "https://sasaki.com/projects"
    assert result.model == "claude-sonnet-4-6"
    assert result.classification.category == "direct_api"
    assert result.classification.subcategory == "algolia_search"
    assert result.classification.confidence == 0.94
    assert result.probe_stages_completed == ["http", "html_analysis", "claude_classify"]
    assert "js_execution" in result.probe_stages_skipped
    assert result.low_confidence_warning is False
    # Algolia URL passed validation
    assert result.extraction_strategy.specifics == {"app_id": "AHNZ21XTZ6"}


@respx.mock
def test_probe_endpoints_discovered_populated_from_stage2():
    respx.get("https://sasaki.com/").mock(
        return_value=httpx.Response(200, content=SASAKI_HTML)
    )
    respx.get("https://sasaki.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://sasaki.com/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://sasaki.com/sitemap_index.xml").mock(return_value=httpx.Response(404))

    client = StubAnthropic(StubMessage(content=[StubTextBlock(text=_claude_response_text())]))
    http_client = httpx.Client(follow_redirects=True, timeout=5.0)
    try:
        result = probe(
            "https://sasaki.com/",
            anthropic_client=client,
            http_client=http_client,
        )
    finally:
        http_client.close()

    urls = [e.url for e in result.endpoints_discovered]
    assert any("algolia.net" in u for u in urls)


# ---------------- Stage 1 failure modes --------------------------------


@respx.mock
def test_probe_raises_http_probe_error_on_unreachable():
    respx.get("https://nope.invalid/").mock(side_effect=httpx.ConnectError("nope"))
    client = StubAnthropic(StubMessage(content=[StubTextBlock(text="ignored")]))
    http_client = httpx.Client(timeout=5.0)
    try:
        with pytest.raises(HTTPProbeError):
            probe(
                "https://nope.invalid/",
                anthropic_client=client,
                http_client=http_client,
                options=ProbeOptions(retry_on_stage1_failure=False),
            )
    finally:
        http_client.close()


@respx.mock
def test_probe_raises_auth_walled_on_401():
    respx.get("https://gated.example/").mock(return_value=httpx.Response(401, content=""))
    respx.get("https://gated.example/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://gated.example/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://gated.example/sitemap_index.xml").mock(return_value=httpx.Response(404))
    client = StubAnthropic(StubMessage(content=[StubTextBlock(text="ignored")]))
    http_client = httpx.Client(timeout=5.0)
    try:
        with pytest.raises(AuthWalledError):
            probe(
                "https://gated.example/",
                anthropic_client=client,
                http_client=http_client,
            )
    finally:
        http_client.close()


@respx.mock
def test_probe_raises_html_analysis_error_on_empty_body():
    """If Stage 1 succeeds but returns no body, we cannot run Stage 2."""
    respx.get("https://empty.example/").mock(
        return_value=httpx.Response(204, content=b"")
    )
    respx.get("https://empty.example/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://empty.example/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://empty.example/sitemap_index.xml").mock(return_value=httpx.Response(404))
    client = StubAnthropic(StubMessage(content=[StubTextBlock(text="ignored")]))
    http_client = httpx.Client(timeout=5.0)
    try:
        with pytest.raises(HTMLAnalysisError):
            probe(
                "https://empty.example/",
                anthropic_client=client,
                http_client=http_client,
            )
    finally:
        http_client.close()


# ---------------- Confidence handling ----------------------------------


@respx.mock
def test_low_confidence_default_returns_with_warning():
    respx.get("https://unclear.example/").mock(
        return_value=httpx.Response(200, content=SASAKI_HTML)
    )
    respx.get("https://unclear.example/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://unclear.example/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://unclear.example/sitemap_index.xml").mock(return_value=httpx.Response(404))

    body = _claude_response_text(
        classification="unknown",
        confidence=0.4,
        reasoning="Could not narrow down between Algolia and a custom backend.",
        method="manual_review",
        specifics={},
        limitations=["No app ID could be extracted from inline scripts."],
    )
    client = StubAnthropic(StubMessage(content=[StubTextBlock(text=body)]))
    http_client = httpx.Client(timeout=5.0)
    try:
        result = probe(
            "https://unclear.example/",
            anthropic_client=client,
            http_client=http_client,
        )
    finally:
        http_client.close()

    assert result.low_confidence_warning is True
    assert result.classification.confidence == 0.4
    assert "No app ID" in " ".join(result.limitations)


@respx.mock
def test_low_confidence_strict_raises():
    respx.get("https://unclear.example/").mock(
        return_value=httpx.Response(200, content=SASAKI_HTML)
    )
    respx.get("https://unclear.example/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://unclear.example/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://unclear.example/sitemap_index.xml").mock(return_value=httpx.Response(404))

    body = _claude_response_text(
        classification="unknown",
        confidence=0.4,
        reasoning="Ambiguous evidence.",
        method="manual_review",
        specifics={},
        limitations=["Insufficient signal in the served page."],
    )
    client = StubAnthropic(StubMessage(content=[StubTextBlock(text=body)]))
    http_client = httpx.Client(timeout=5.0)
    try:
        with pytest.raises(LowConfidenceError):
            probe(
                "https://unclear.example/",
                anthropic_client=client,
                http_client=http_client,
                options=ProbeOptions(strict=True),
            )
    finally:
        http_client.close()


# ---------------- Validation behaviour --------------------------------


@respx.mock
def test_hallucinated_endpoint_is_stripped_and_recorded():
    respx.get("https://sasaki.com/").mock(
        return_value=httpx.Response(200, content=SASAKI_HTML)
    )
    respx.get("https://sasaki.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://sasaki.com/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://sasaki.com/sitemap_index.xml").mock(return_value=httpx.Response(404))

    body = _claude_response_text(
        specifics={
            "app_id": "AHNZ21XTZ6",
            "fake_endpoint": "https://hallucinated.invalid/api/x",
        }
    )
    client = StubAnthropic(StubMessage(content=[StubTextBlock(text=body)]))
    http_client = httpx.Client(follow_redirects=True, timeout=5.0)
    try:
        result = probe(
            "https://sasaki.com/",
            anthropic_client=client,
            http_client=http_client,
        )
    finally:
        http_client.close()

    assert result.extraction_strategy.specifics == {"app_id": "AHNZ21XTZ6"}
    # The strip is surfaced in limitations.
    assert any("stripped" in lim for lim in result.limitations)


# ---------------- Retry behaviour --------------------------------------


@respx.mock
def test_stage1_retry_succeeds_after_transient_error():
    """First request raises ConnectError; second returns a real response."""
    route = respx.get("https://retry.example/")
    route.side_effect = [
        httpx.ConnectError("transient"),
        httpx.Response(200, content=SASAKI_HTML),
    ]
    respx.get("https://retry.example/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://retry.example/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://retry.example/sitemap_index.xml").mock(return_value=httpx.Response(404))

    client = StubAnthropic(StubMessage(content=[StubTextBlock(text=_claude_response_text())]))
    http_client = httpx.Client(timeout=5.0)
    try:
        result = probe(
            "https://retry.example/",
            anthropic_client=client,
            http_client=http_client,
            options=ProbeOptions(retry_on_stage1_failure=True),
        )
    finally:
        http_client.close()

    assert result.probe_stages_completed == ["http", "html_analysis", "claude_classify"]
