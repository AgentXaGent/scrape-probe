"""Tests for ``cartograph_ai.stages.http_probe``.

Uses respx to mock httpx at the transport layer so the real probe code
runs against synthetic responses. No network calls leave the sandbox.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from cartograph_ai.stages.http_probe import (
    DEFAULT_USER_AGENT,
    probe_http,
)


SASAKI_HTML = """\
<!doctype html><html><head><title>Sasaki Projects</title></head>
<body><div id="root"></div></body></html>
"""

ROBOTS_WITH_SITEMAP = """\
User-agent: *
Disallow: /admin
Disallow: /private

Sitemap: https://www.sasaki.com/sitemap.xml
"""

SITEMAP_BODY = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://www.sasaki.com/projects</loc></url>
  <url><loc>https://www.sasaki.com/about</loc></url>
  <url><loc>https://www.sasaki.com/contact</loc></url>
</urlset>
"""

SITEMAP_INDEX_BODY = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://www.sasaki.com/sitemap-1.xml</loc></sitemap>
  <sitemap><loc>https://www.sasaki.com/sitemap-2.xml</loc></sitemap>
</sitemapindex>
"""


@pytest.fixture
def client() -> httpx.Client:
    """A real httpx.Client routed entirely through respx mocks."""
    with httpx.Client(
        follow_redirects=True,
        timeout=5.0,
        headers={"User-Agent": "test-runner"},
    ) as c:
        yield c


# ---------------- Happy path -------------------------------------------


@respx.mock
def test_probe_happy_path_captures_status_and_headers(client):
    respx.get("https://www.sasaki.com/projects").mock(
        return_value=httpx.Response(
            200,
            content=SASAKI_HTML,
            headers={
                "server": "nginx",
                "x-powered-by": "Next.js",
                "content-type": "text/html; charset=utf-8",
                "set-cookie": "session=abc",  # not in the allowlist
            },
        )
    )
    respx.get("https://www.sasaki.com/robots.txt").mock(
        return_value=httpx.Response(200, text=ROBOTS_WITH_SITEMAP)
    )
    respx.get("https://www.sasaki.com/sitemap.xml").mock(
        return_value=httpx.Response(200, text=SITEMAP_BODY)
    )

    out = probe_http("https://www.sasaki.com/projects", client=client)
    assert out["error"] is None
    assert out["status"] == 200
    assert out["final_url"] == "https://www.sasaki.com/projects"
    assert out["headers"]["server"] == "nginx"
    assert out["headers"]["x-powered-by"] == "Next.js"
    assert "set-cookie" not in out["headers"], "Non-allowlisted headers should be filtered"


@respx.mock
def test_probe_captures_redirect_chain(client):
    respx.get("https://sasaki.com/projects").mock(
        return_value=httpx.Response(
            301, headers={"location": "https://www.sasaki.com/projects"}
        )
    )
    respx.get("https://www.sasaki.com/projects").mock(
        return_value=httpx.Response(200, content=SASAKI_HTML)
    )
    respx.get("https://www.sasaki.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://www.sasaki.com/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://www.sasaki.com/sitemap_index.xml").mock(
        return_value=httpx.Response(404)
    )

    out = probe_http("https://sasaki.com/projects", client=client)
    assert out["status"] == 200
    assert out["final_url"] == "https://www.sasaki.com/projects"
    assert len(out["redirect_chain"]) == 1
    assert out["redirect_chain"][0]["url"] == "https://sasaki.com/projects"
    assert out["redirect_chain"][0]["status"] == 301
    assert out["redirect_chain"][0]["to"] == "https://www.sasaki.com/projects"


# ---------------- robots.txt -------------------------------------------


_ROBOTS_X_COM = """\
User-agent: *
Disallow: /admin
Disallow: /private

Sitemap: https://x.com/sitemap.xml
"""


@respx.mock
def test_probe_parses_robots_sitemap_directive(client):
    respx.get("https://x.com/").mock(return_value=httpx.Response(200, content="<html/>"))
    respx.get("https://x.com/robots.txt").mock(
        return_value=httpx.Response(200, text=_ROBOTS_X_COM)
    )
    respx.get("https://x.com/sitemap.xml").mock(
        return_value=httpx.Response(200, text=SITEMAP_BODY)
    )

    out = probe_http("https://x.com/", client=client)
    assert out["robots_txt"]["present"] is True
    assert out["robots_txt"]["disallow_count"] == 2
    assert out["robots_txt"]["user_agent_blocks"] == 1
    assert out["robots_txt"]["sitemap_urls"] == ["https://x.com/sitemap.xml"]


@respx.mock
def test_probe_handles_missing_robots(client):
    respx.get("https://x.com/").mock(return_value=httpx.Response(200, content="<html/>"))
    respx.get("https://x.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://x.com/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://x.com/sitemap_index.xml").mock(return_value=httpx.Response(404))

    out = probe_http("https://x.com/", client=client)
    assert out["robots_txt"]["present"] is False
    assert out["robots_txt"]["status"] == 404
    assert out["robots_txt"]["sitemap_urls"] == []


@respx.mock
def test_probe_handles_empty_robots(client):
    respx.get("https://x.com/").mock(return_value=httpx.Response(200, content="<html/>"))
    respx.get("https://x.com/robots.txt").mock(return_value=httpx.Response(200, text="   \n"))
    respx.get("https://x.com/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://x.com/sitemap_index.xml").mock(return_value=httpx.Response(404))

    out = probe_http("https://x.com/", client=client)
    assert out["robots_txt"]["present"] is False


# ---------------- Sitemaps ---------------------------------------------


@respx.mock
def test_probe_counts_sitemap_urls(client):
    respx.get("https://x.com/").mock(return_value=httpx.Response(200, content="<html/>"))
    respx.get("https://x.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://x.com/sitemap.xml").mock(
        return_value=httpx.Response(200, text=SITEMAP_BODY)
    )
    respx.get("https://x.com/sitemap_index.xml").mock(return_value=httpx.Response(404))

    out = probe_http("https://x.com/", client=client)
    sitemap_xml = next(s for s in out["sitemaps"] if s["url"].endswith("sitemap.xml"))
    assert sitemap_xml["status"] == 200
    assert sitemap_xml["url_count"] == 3
    assert sitemap_xml["child_sitemap_count"] == 0


@respx.mock
def test_probe_recognizes_sitemap_index(client):
    respx.get("https://x.com/").mock(return_value=httpx.Response(200, content="<html/>"))
    respx.get("https://x.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://x.com/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://x.com/sitemap_index.xml").mock(
        return_value=httpx.Response(200, text=SITEMAP_INDEX_BODY)
    )

    out = probe_http("https://x.com/", client=client)
    index = next(s for s in out["sitemaps"] if s["url"].endswith("sitemap_index.xml"))
    assert index["status"] == 200
    assert index["url_count"] == 0
    assert index["child_sitemap_count"] == 2


@respx.mock
def test_probe_caps_sitemap_fetches(client):
    """If robots.txt declares many sitemaps, Stage 1 only touches the first few."""
    many_sitemaps = "\n".join(
        f"Sitemap: https://x.com/sitemap-{i}.xml" for i in range(10)
    )
    respx.get("https://x.com/").mock(return_value=httpx.Response(200, content="<html/>"))
    respx.get("https://x.com/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\n" + many_sitemaps)
    )
    for i in range(10):
        respx.get(f"https://x.com/sitemap-{i}.xml").mock(
            return_value=httpx.Response(200, text=SITEMAP_BODY)
        )

    out = probe_http("https://x.com/", client=client)
    # Cap is 3 per the module-level constant.
    assert len(out["sitemaps"]) == 3


# ---------------- Failure modes ----------------------------------------


@respx.mock
def test_probe_returns_error_on_connection_failure(client):
    respx.get("https://nope.invalid/").mock(side_effect=httpx.ConnectError("nope"))

    out = probe_http("https://nope.invalid/", client=client)
    assert out["error"] is not None
    assert "ConnectError" in out["error"]
    assert out["status"] is None
    assert out["final_url"] is None


@respx.mock
def test_probe_returns_error_on_timeout(client):
    respx.get("https://slow.invalid/").mock(side_effect=httpx.ReadTimeout("timed out"))

    out = probe_http("https://slow.invalid/", client=client)
    assert out["error"] is not None
    assert "Timeout" in out["error"]


@respx.mock
def test_probe_captures_4xx(client):
    respx.get("https://gated.example/").mock(return_value=httpx.Response(403))
    respx.get("https://gated.example/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://gated.example/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://gated.example/sitemap_index.xml").mock(return_value=httpx.Response(404))

    out = probe_http("https://gated.example/", client=client)
    assert out["error"] is None  # not a probe failure; just a status code to record
    assert out["status"] == 403


# ---------------- Misc -------------------------------------------------


def test_default_user_agent_identifies_cartograph():
    assert "cartograph-ai" in DEFAULT_USER_AGENT
    assert "github.com/AgentXaGent/cartograph-ai" in DEFAULT_USER_AGENT


# ---------------- Body capture -----------------------------------------


@respx.mock
def test_probe_captures_body_on_200(client):
    body = "<html><body><h1>Sasaki</h1></body></html>"
    respx.get("https://x.com/").mock(return_value=httpx.Response(200, content=body))
    respx.get("https://x.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://x.com/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://x.com/sitemap_index.xml").mock(return_value=httpx.Response(404))

    out = probe_http("https://x.com/", client=client)
    assert out["body"] == body
    assert out["body_size_bytes"] == len(body)
    assert out["body_truncated"] is False


@respx.mock
def test_probe_truncates_large_body(client):
    from cartograph_ai.stages.http_probe import _BODY_MAX_BYTES
    big = "<html>" + ("x" * (_BODY_MAX_BYTES + 1000)) + "</html>"
    respx.get("https://x.com/").mock(return_value=httpx.Response(200, content=big))
    respx.get("https://x.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://x.com/sitemap.xml").mock(return_value=httpx.Response(404))
    respx.get("https://x.com/sitemap_index.xml").mock(return_value=httpx.Response(404))

    out = probe_http("https://x.com/", client=client)
    assert out["body_truncated"] is True
    assert len(out["body"]) == _BODY_MAX_BYTES
    assert out["body_size_bytes"] == len(big)
