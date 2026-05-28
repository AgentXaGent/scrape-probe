"""Tests for ``cartograph_ai.validation``.

Specifically the cross-reference check that strips hallucinated URLs
from Stage 4 responses.
"""

from __future__ import annotations

import pytest

from cartograph_ai.schema import (
    ClaudeResponse,
    ExtractionStrategy,
)
from cartograph_ai.validation import (
    ValidationReport,
    cross_reference_endpoints,
)


def _response(specifics: dict) -> ClaudeResponse:
    return ClaudeResponse(
        classification="direct_api",
        confidence=0.94,
        reasoning="Found endpoint evidence in inline script.",
        extraction_strategy=ExtractionStrategy(
            method="algolia_search",
            requires_browser=False,
            estimated_requests=2,
            recommended_tool="requests",
            specifics=specifics,
        ),
        limitations=[],
    )


SASAKI_PAYLOAD = {
    "stage1": {
        "final_url": "https://www.sasaki.com/projects",
        "status": 200,
    },
    "stage2": {
        "api_endpoints": [
            {
                "url": "https://AHNZ21XTZ6-dsn.algolia.net/1/indexes/prod_projects/query",
                "type": "algolia_search_api",
            },
        ],
        "fingerprints": [{"id": "algolia", "category": "search", "evidence": "Found algolia.net hostname"}],
    },
}


# ---------------- Happy paths -----------------------------------------


def test_no_stripping_when_all_urls_match():
    r = _response({
        "endpoint": "https://AHNZ21XTZ6-dsn.algolia.net/1/indexes/prod_projects/query",
        "app_id": "AHNZ21XTZ6",
    })
    report = cross_reference_endpoints(r, probe_payload=SASAKI_PAYLOAD)
    assert isinstance(report, ValidationReport)
    assert report.stripped_endpoints == []
    assert report.had_stripped_endpoints is False
    assert report.response is r  # exact same instance when nothing was changed


def test_non_url_values_pass_through():
    r = _response({
        "app_id": "AHNZ21XTZ6",
        "index": "prod_projects",
        "page_size_max": 1000,
        "use_post": True,
    })
    report = cross_reference_endpoints(r, probe_payload=SASAKI_PAYLOAD)
    assert report.stripped_endpoints == []
    assert report.response.extraction_strategy.specifics == {
        "app_id": "AHNZ21XTZ6",
        "index": "prod_projects",
        "page_size_max": 1000,
        "use_post": True,
    }


# ---------------- Stripping -------------------------------------------


def test_strips_hallucinated_https_url():
    r = _response({
        "endpoint": "https://hallucinated.example.com/api/v1/products",
        "app_id": "AHNZ21XTZ6",
    })
    report = cross_reference_endpoints(r, probe_payload=SASAKI_PAYLOAD)
    assert report.had_stripped_endpoints is True
    assert "https://hallucinated.example.com/api/v1/products" in report.stripped_endpoints
    # The supported value remains.
    assert report.response.extraction_strategy.specifics == {"app_id": "AHNZ21XTZ6"}


def test_strips_hallucinated_root_relative_path():
    r = _response({
        "endpoint": "/api/v99/imaginary",
    })
    report = cross_reference_endpoints(r, probe_payload={"stage2": {"api_endpoints": []}})
    assert report.stripped_endpoints == ["/api/v99/imaginary"]
    assert "endpoint" not in report.response.extraction_strategy.specifics


def test_preserves_supported_root_relative_path():
    payload = {"stage2": {"api_endpoints": [{"url": "/api/v1/products"}]}}
    r = _response({"endpoint": "/api/v1/products"})
    report = cross_reference_endpoints(r, probe_payload=payload)
    assert report.stripped_endpoints == []
    assert report.response.extraction_strategy.specifics["endpoint"] == "/api/v1/products"


# ---------------- List handling ---------------------------------------


def test_strips_individual_urls_in_list():
    payload = {"stage2": {"endpoints": ["/api/v1/good", "/api/v1/other"]}}
    r = _response({
        "endpoints": [
            "/api/v1/good",
            "/api/v99/hallucinated",
            "/api/v1/other",
        ],
    })
    report = cross_reference_endpoints(r, probe_payload=payload)
    assert report.stripped_endpoints == ["/api/v99/hallucinated"]
    assert report.response.extraction_strategy.specifics["endpoints"] == [
        "/api/v1/good",
        "/api/v1/other",
    ]


def test_list_with_non_url_items_passes_through():
    r = _response({"sizes": [10, 100, 1000]})
    report = cross_reference_endpoints(r, probe_payload={})
    assert report.stripped_endpoints == []
    assert report.response.extraction_strategy.specifics["sizes"] == [10, 100, 1000]


# ---------------- Nested dict handling --------------------------------


def test_strips_url_inside_nested_dict():
    payload = {"stage2": {"endpoints": [{"url": "https://x.algolia.net/1/idx/q"}]}}
    r = _response({
        "config": {
            "endpoint": "https://hallucinated.example.com/x",
            "app_id": "X",
        },
    })
    report = cross_reference_endpoints(r, probe_payload=payload)
    assert report.stripped_endpoints == ["https://hallucinated.example.com/x"]
    assert report.response.extraction_strategy.specifics["config"] == {"app_id": "X"}


def test_preserves_supported_url_inside_nested_dict():
    url = "https://x.algolia.net/1/idx/q"
    payload = {"stage2": {"endpoints": [{"url": url}]}}
    r = _response({"config": {"endpoint": url}})
    report = cross_reference_endpoints(r, probe_payload=payload)
    assert report.stripped_endpoints == []
    assert report.response.extraction_strategy.specifics["config"]["endpoint"] == url


# ---------------- Reasoning prose is untouched ------------------------


def test_reasoning_prose_is_not_modified_even_if_it_mentions_urls():
    """We strip URLs from actionable fields, not from prose explanations."""
    r = ClaudeResponse(
        classification="direct_api",
        confidence=0.9,
        reasoning="See https://docs.example.com/scraping for context.",
        extraction_strategy=ExtractionStrategy(
            method="algolia_search",
            requires_browser=False,
            estimated_requests=1,
            recommended_tool="requests",
            specifics={"app_id": "X"},
        ),
        limitations=[],
    )
    report = cross_reference_endpoints(r, probe_payload={})
    assert report.stripped_endpoints == []
    assert "https://docs.example.com/scraping" in report.response.reasoning


# ---------------- Object identity --------------------------------------


def test_clean_path_returns_original_response_instance():
    r = _response({"app_id": "X"})
    report = cross_reference_endpoints(r, probe_payload=SASAKI_PAYLOAD)
    assert report.response is r


def test_stripped_path_returns_new_instance():
    r = _response({"endpoint": "https://nope.example/x", "app_id": "X"})
    report = cross_reference_endpoints(r, probe_payload=SASAKI_PAYLOAD)
    assert report.response is not r
    assert r.extraction_strategy.specifics == {
        "endpoint": "https://nope.example/x",
        "app_id": "X",
    }  # original unchanged
