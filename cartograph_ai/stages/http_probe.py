"""Stage 1: HTTP probe.

Cheapest possible signal collection. No HTML parsing, no LLM call.
Fetches the URL with redirect following, the response headers and HTTP
version, ``/robots.txt`` if present, and any sitemaps declared by
``robots.txt`` or living at the conventional paths.

The output is a structured ``dict`` (not a Pydantic model) so it
serialises cleanly into the Stage 4 prompt payload.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import httpx

from cartograph_ai._version import __version__

DEFAULT_USER_AGENT = (
    f"cartograph-ai/{__version__} (+https://github.com/AgentXaGent/cartograph-ai)"
)
"""User-Agent string. Identifies the tool honestly so site owners can
distinguish probes from generic scrapers."""

# Headers worth surfacing to Stage 4. Server / X-Powered-By / generator
# headers are the highest-signal ones; we also keep Content-Type for
# basic sanity and cache headers because they sometimes leak the CDN /
# platform in front of the origin.
_INTERESTING_HEADERS: tuple[str, ...] = (
    "server",
    "x-powered-by",
    "x-generator",
    "x-drupal-cache",
    "x-aem-instance",
    "x-host",
    "x-vercel-cache",
    "x-vercel-id",
    "x-amz-cf-pop",
    "x-cache",
    "via",
    "cf-ray",
    "content-type",
    "content-encoding",
    "cache-control",
    "etag",
    "last-modified",
)

_SITEMAP_DECL_RE = re.compile(r"(?im)^\s*sitemap:\s*(\S+)\s*$")
"""Match a Sitemap directive in robots.txt (case-insensitive, per-line)."""

_LOC_RE = re.compile(r"<loc>\s*([^<]+?)\s*</loc>", re.IGNORECASE)
"""Lightweight <loc> extractor; we count URLs, we do not deeply validate."""

# Sitemap fetch hard caps so a 100 MB sitemap does not blow up the probe.
_SITEMAP_MAX_BYTES = 2_000_000  # 2 MB per sitemap fetched
_SITEMAP_MAX_FETCH = 3  # at most 3 sitemap URLs touched in Stage 1

# Cap on raw HTML body captured for Stage 2. 5 MB is generous for
# real-world pages; runaway responses get truncated.
_BODY_MAX_BYTES = 5_000_000


def probe_http(
    url: str,
    *,
    client: Optional[httpx.Client] = None,
    timeout: float = 10.0,
    max_redirects: int = 5,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict[str, Any]:
    """Run Stage 1 against ``url`` and return a structured findings dict.

    Args:
        url: The URL to probe.
        client: Optional pre-configured ``httpx.Client``. Injected for
            testability; if ``None`` a transient client is created and
            disposed inside the call.
        timeout: Per-request timeout in seconds.
        max_redirects: Maximum redirect hops to follow.
        user_agent: User-Agent header value.

    Returns:
        A dict with keys ``url``, ``final_url``, ``status``,
        ``redirect_chain``, ``headers``, ``http_version``, ``robots_txt``,
        ``sitemaps``, and ``error``. The ``error`` field is ``None`` on
        success and carries a short string when the probe could not
        reach the target at all.
    """
    owns_client = client is None
    if client is None:
        client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 5.0)),
            follow_redirects=True,
            max_redirects=max_redirects,
            headers={"User-Agent": user_agent},
        )

    findings: dict[str, Any] = {
        "url": url,
        "final_url": None,
        "status": None,
        "redirect_chain": [],
        "headers": {},
        "http_version": None,
        "body": None,
        "body_size_bytes": 0,
        "body_truncated": False,
        "robots_txt": {"present": False},
        "sitemaps": [],
        "error": None,
    }

    try:
        try:
            response = client.get(url)
        except httpx.HTTPError as exc:
            findings["error"] = f"{type(exc).__name__}: {exc}"
            return findings

        findings["final_url"] = str(response.url)
        findings["status"] = response.status_code
        findings["http_version"] = response.http_version
        raw = response.content or b""
        findings["body_size_bytes"] = len(raw)
        if len(raw) > _BODY_MAX_BYTES:
            findings["body"] = raw[:_BODY_MAX_BYTES].decode("utf-8", errors="ignore")
            findings["body_truncated"] = True
        else:
            findings["body"] = raw.decode("utf-8", errors="ignore")
        findings["redirect_chain"] = [
            {"url": str(h.url), "status": h.status_code, "to": h.headers.get("location")}
            for h in response.history
        ]
        findings["headers"] = _collect_interesting_headers(response.headers)

        base = _origin(str(response.url))
        findings["robots_txt"] = _fetch_robots(client, base)

        sitemap_candidates = list(findings["robots_txt"].get("sitemap_urls") or [])
        if not sitemap_candidates:
            sitemap_candidates = [
                f"{base}/sitemap.xml",
                f"{base}/sitemap_index.xml",
            ]

        findings["sitemaps"] = _fetch_sitemaps(client, sitemap_candidates)

    finally:
        if owns_client:
            client.close()

    return findings


# --- helpers -------------------------------------------------------------


def _origin(url: str) -> str:
    """Return ``scheme://host`` portion of a URL."""
    parsed = httpx.URL(url)
    return f"{parsed.scheme}://{parsed.host}" + (f":{parsed.port}" if parsed.port else "")


def _collect_interesting_headers(headers: httpx.Headers) -> dict[str, str]:
    """Subset header dict to high-signal entries, lowercased keys."""
    out: dict[str, str] = {}
    for name in _INTERESTING_HEADERS:
        if name in headers:
            out[name] = headers[name]
    return out


def _fetch_robots(client: httpx.Client, origin: str) -> dict[str, Any]:
    """Fetch /robots.txt and parse out Sitemap directives + a few counts.

    Returns a structured dict regardless of fetch outcome so the Stage 4
    payload has a stable shape.
    """
    record: dict[str, Any] = {
        "present": False,
        "status": None,
        "size_bytes": 0,
        "sitemap_urls": [],
        "user_agent_blocks": 0,
        "disallow_count": 0,
    }
    try:
        r = client.get(f"{origin}/robots.txt")
    except httpx.HTTPError as exc:
        record["fetch_error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["status"] = r.status_code
    if r.status_code != 200 or not r.text.strip():
        return record

    body = r.text
    record["present"] = True
    record["size_bytes"] = len(body)
    record["sitemap_urls"] = _SITEMAP_DECL_RE.findall(body)
    record["user_agent_blocks"] = sum(
        1 for line in body.splitlines() if line.strip().lower().startswith("user-agent:")
    )
    record["disallow_count"] = sum(
        1 for line in body.splitlines() if line.strip().lower().startswith("disallow:")
    )
    return record


def _fetch_sitemaps(client: httpx.Client, urls: list[str]) -> list[dict[str, Any]]:
    """Fetch up to ``_SITEMAP_MAX_FETCH`` sitemaps and summarize each.

    For nested sitemap indexes we record the child sitemap URLs but do
    not recurse; the Stage 4 prompt only needs to know roughly how many
    URLs the site exposes and where to find them.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for raw in urls:
        if len(out) >= _SITEMAP_MAX_FETCH:
            break
        if raw in seen:
            continue
        seen.add(raw)

        record: dict[str, Any] = {"url": raw, "status": None, "url_count": 0, "child_sitemap_count": 0}
        try:
            r = client.get(raw)
        except httpx.HTTPError as exc:
            record["fetch_error"] = f"{type(exc).__name__}: {exc}"
            out.append(record)
            continue

        record["status"] = r.status_code
        if r.status_code != 200:
            out.append(record)
            continue

        body = r.content[:_SITEMAP_MAX_BYTES].decode("utf-8", errors="ignore")
        record["size_bytes"] = len(r.content)
        record["truncated"] = len(r.content) > _SITEMAP_MAX_BYTES

        # Count <loc> occurrences. A sitemap index also uses <loc> but
        # nests <sitemap> wrappers; we distinguish via the <sitemapindex>
        # root element marker.
        loc_count = len(_LOC_RE.findall(body))
        if "<sitemapindex" in body.lower():
            record["child_sitemap_count"] = loc_count
        else:
            record["url_count"] = loc_count

        out.append(record)

    return out
