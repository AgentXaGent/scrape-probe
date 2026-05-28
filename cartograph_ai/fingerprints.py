"""25 framework, API, and data-layer fingerprints for Stage 2 HTML analysis.

Each fingerprint is a small detector ``(html, soup) -> Optional[str]`` that
returns a short evidence string when its pattern is present and ``None``
otherwise. ``detect_all`` runs every fingerprint against a parsed page
and returns the hits, which feed into the Stage 4 prompt as part of the
structured probe payload.

The v0.1 set has 25 fingerprints covering:

* **framework signatures (8)**: Next.js, Nuxt, WordPress, Adobe Experience
  Manager, Webflow, Squarespace, Shopify, Gatsby.
* **search-as-a-service (2)**: Algolia, Elasticsearch.
* **API conventions (4)**: generic REST ``/api/``, GraphQL, WordPress
  ``/wp-json/``, generic ``/_api/``.
* **embedded data signals (5)**: JSON-LD, Open Graph, Schema.org
  microdata, ``__NEXT_DATA__`` blob, Apollo/initial-state blob.
* **structural patterns (6)**: form-gated dataset, CSV / Excel / ZIP /
  JSON bulk download links, SPA empty shell.

Adding a fingerprint:

1. Write a detector that takes ``(html, soup)`` and returns
   ``Optional[str]`` (evidence on hit, ``None`` otherwise).
2. Append a ``Fingerprint(...)`` to ``FINGERPRINTS`` below.
3. Add a positive (and where helpful, negative) test to
   ``tests/test_fingerprints.py``.
4. If the count moves off 25, update ``docs/how-it-works.md`` and
   ``README.md`` so the published count matches reality (honest count
   beats hedged "around 25").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from bs4 import BeautifulSoup

FingerprintCategory = Literal[
    "framework",
    "search",
    "api",
    "embedded_data",
    "structural",
]


@dataclass(frozen=True)
class Fingerprint:
    """One detector plus its metadata."""

    id: str
    category: FingerprintCategory
    description: str
    detect: Callable[[str, BeautifulSoup], Optional[str]]


@dataclass(frozen=True)
class FingerprintHit:
    """A successful detection, suitable for inclusion in the Stage 4 payload."""

    id: str
    category: FingerprintCategory
    description: str
    evidence: str


# ===================== Framework signatures (8) =========================


def _detect_nextjs(html: str, soup: BeautifulSoup) -> Optional[str]:
    if soup.find("script", id="__NEXT_DATA__"):
        return "Found <script id=\"__NEXT_DATA__\"> tag"
    if "/_next/" in html:
        return "Found /_next/ asset path reference"
    return None


def _detect_nuxtjs(html: str, soup: BeautifulSoup) -> Optional[str]:
    if "/_nuxt/" in html:
        return "Found /_nuxt/ asset path reference"
    if "window.__NUXT__" in html:
        return "Found window.__NUXT__ hydration blob"
    return None


def _detect_wordpress(html: str, soup: BeautifulSoup) -> Optional[str]:
    if "/wp-content/" in html:
        return "Found /wp-content/ path reference"
    if "/wp-includes/" in html:
        return "Found /wp-includes/ path reference"
    gen = soup.find("meta", attrs={"name": "generator"})
    if gen and isinstance(gen.get("content"), str) and gen["content"].startswith("WordPress"):
        return f"<meta name=generator content={gen['content']!r}>"
    return None


def _detect_adobe_experience_manager(html: str, soup: BeautifulSoup) -> Optional[str]:
    if "/content/dam/" in html:
        return "Found /content/dam/ asset path (AEM DAM)"
    if "urn:aaid:aem:" in html:
        return "Found urn:aaid:aem: identifier"
    if "/etc.clientlibs/" in html:
        return "Found /etc.clientlibs/ asset path"
    return None


def _detect_webflow(html: str, soup: BeautifulSoup) -> Optional[str]:
    htmltag = soup.find("html")
    if htmltag is not None:
        for attr in htmltag.attrs:
            if attr.startswith("data-wf-"):
                return f"Found data-wf-* attribute on <html>: {attr}"
    if "webflow.com/" in html or "assets.webflow.com" in html:
        return "Found webflow.com asset reference"
    return None


def _detect_squarespace(html: str, soup: BeautifulSoup) -> Optional[str]:
    if "static.squarespace.com" in html:
        return "Found static.squarespace.com asset reference"
    if "Static.SQUARESPACE_CONTEXT" in html:
        return "Found Static.SQUARESPACE_CONTEXT JS variable"
    return None


def _detect_shopify(html: str, soup: BeautifulSoup) -> Optional[str]:
    if "cdn.shopify.com" in html:
        return "Found cdn.shopify.com asset reference"
    if "Shopify.shop" in html or "Shopify.theme" in html:
        return "Found Shopify.* JS variable"
    if soup.find("meta", attrs={"name": lambda x: bool(x) and x.startswith("shopify-")}):
        return "Found <meta name=shopify-*> tag"
    return None


def _detect_gatsby(html: str, soup: BeautifulSoup) -> Optional[str]:
    if soup.find(id="___gatsby"):
        return "Found <#___gatsby> mount point"
    if "/page-data/" in html and "app-data.json" in html:
        return "Found /page-data/ and app-data.json references"
    return None


# ===================== Search-as-a-service (2) ==========================


def _detect_algolia(html: str, soup: BeautifulSoup) -> Optional[str]:
    if "algolia.net" in html or "algolianet.com" in html:
        return "Found algolia.net hostname"
    if "ALGOLIA_API_KEY" in html or "algoliasearch" in html.lower():
        return "Found Algolia client identifier in inline script"
    return None


def _detect_elasticsearch(html: str, soup: BeautifulSoup) -> Optional[str]:
    lowered = html.lower()
    if "elastic.co/" in lowered or "/_search?" in html or "elasticsearch.js" in lowered:
        return "Found Elasticsearch reference"
    return None


# ===================== API conventions (4) ==============================

_API_PATH_RE = re.compile(r"[\"'`]/api/(?:v\d+/)?[a-zA-Z0-9_\-]+")


def _detect_rest_api(html: str, soup: BeautifulSoup) -> Optional[str]:
    hits = _API_PATH_RE.findall(html)
    if hits:
        sample = sorted(set(hits))[:3]
        return f"Found {len(hits)} reference(s) to /api/ paths, e.g. {sample}"
    return None


def _detect_graphql(html: str, soup: BeautifulSoup) -> Optional[str]:
    if "/graphql" in html:
        return "Found /graphql endpoint reference"
    if "ApolloClient" in html or "@apollo/client" in html:
        return "Found Apollo Client reference"
    return None


def _detect_wp_json_api(html: str, soup: BeautifulSoup) -> Optional[str]:
    if "/wp-json/" in html:
        return "Found /wp-json/ REST endpoint reference"
    link = soup.find("link", attrs={"rel": "https://api.w.org/"})
    if link is not None:
        return "Found <link rel=https://api.w.org/> (WP REST discovery)"
    return None


def _detect_underscore_api(html: str, soup: BeautifulSoup) -> Optional[str]:
    if "/_api/" in html:
        return "Found /_api/ path reference"
    return None


# ===================== Embedded data signals (5) ========================


def _detect_json_ld(html: str, soup: BeautifulSoup) -> Optional[str]:
    scripts = soup.find_all("script", type="application/ld+json")
    if scripts:
        return f"Found {len(scripts)} <script type=application/ld+json> block(s)"
    return None


def _detect_open_graph(html: str, soup: BeautifulSoup) -> Optional[str]:
    og = soup.find_all(
        "meta", attrs={"property": lambda x: bool(x) and x.startswith("og:")}
    )
    if og:
        return f"Found {len(og)} Open Graph <meta property=og:*> tag(s)"
    return None


def _detect_schema_org_microdata(html: str, soup: BeautifulSoup) -> Optional[str]:
    items = soup.find_all(
        attrs={"itemtype": lambda x: bool(x) and "schema.org" in x}
    )
    if items:
        return f"Found {len(items)} element(s) with itemtype=schema.org/*"
    return None


def _detect_next_data_blob(html: str, soup: BeautifulSoup) -> Optional[str]:
    tag = soup.find("script", id="__NEXT_DATA__")
    if tag is not None and tag.string:
        return f"Found __NEXT_DATA__ blob ({len(tag.string)} chars)"
    return None


def _detect_apollo_state_blob(html: str, soup: BeautifulSoup) -> Optional[str]:
    if "window.__APOLLO_STATE__" in html:
        return "Found window.__APOLLO_STATE__ hydration blob"
    if "window.__INITIAL_STATE__" in html:
        return "Found window.__INITIAL_STATE__ hydration blob"
    return None


# ===================== Structural patterns (6) ==========================

_FORM_API_SEGMENTS = ("/api/", "/_api/", "/search", "/export", "/download")


def _detect_form_gated_dataset(html: str, soup: BeautifulSoup) -> Optional[str]:
    for form in soup.find_all("form"):
        action = form.get("action") or ""
        if not action:
            continue
        lowered = action.lower()
        if any(seg in lowered for seg in _FORM_API_SEGMENTS):
            method = (form.get("method") or "get").lower()
            return f"<form method={method!r} action={action!r}> looks API/export-bound"
    return None


def _find_bulk_download_link(
    soup: BeautifulSoup, suffixes: tuple[str, ...], *, label: str
) -> Optional[str]:
    hits = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        stem = href.split("?", 1)[0].split("#", 1)[0].lower()
        if any(stem.endswith(s) for s in suffixes):
            hits.append(href)
            if len(hits) >= 5:
                break
    if hits:
        return f"Found {len(hits)} {label} download link(s), first: {hits[0]!r}"
    return None


def _detect_bulk_download_csv(html: str, soup: BeautifulSoup) -> Optional[str]:
    return _find_bulk_download_link(soup, (".csv",), label="CSV")


def _detect_bulk_download_xlsx(html: str, soup: BeautifulSoup) -> Optional[str]:
    return _find_bulk_download_link(soup, (".xlsx", ".xls"), label="Excel")


def _detect_bulk_download_zip(html: str, soup: BeautifulSoup) -> Optional[str]:
    return _find_bulk_download_link(soup, (".zip",), label="ZIP")


def _detect_bulk_download_json(html: str, soup: BeautifulSoup) -> Optional[str]:
    return _find_bulk_download_link(soup, (".json",), label="JSON")


_SPA_ROOT_IDS = ("root", "app", "__next", "___gatsby", "nuxt", "app-root")
_SCRIPT_LIKE = {"script", "style", "noscript", "template"}


def _detect_spa_empty_shell(html: str, soup: BeautifulSoup) -> Optional[str]:
    """Body has very little visible text and a known SPA mount point.

    Conservative threshold: under 200 chars of non-script visible text.
    Filters out scripts, styles, noscript, and template content so a
    server-rendered page with a large hydration blob is not mistaken for
    an empty shell.
    """
    body = soup.find("body")
    if body is None:
        return None
    text_segments = [
        s.strip()
        for s in body.find_all(string=True)
        if s.parent is not None and s.parent.name not in _SCRIPT_LIKE
    ]
    text = " ".join(t for t in text_segments if t)
    if len(text) > 200:
        return None
    for marker_id in _SPA_ROOT_IDS:
        if body.find(id=marker_id) is not None:
            return (
                f"Body has {len(text)} chars of visible text + "
                f"<#{marker_id}> mount point (likely SPA shell)"
            )
    return None


# ===================== Registry =========================================

FINGERPRINTS: tuple[Fingerprint, ...] = (
    # Framework signatures (8)
    Fingerprint("nextjs", "framework", "Next.js framework signals", _detect_nextjs),
    Fingerprint("nuxtjs", "framework", "Nuxt framework signals", _detect_nuxtjs),
    Fingerprint("wordpress", "framework", "WordPress signals", _detect_wordpress),
    Fingerprint(
        "adobe_experience_manager",
        "framework",
        "Adobe Experience Manager (AEM)",
        _detect_adobe_experience_manager,
    ),
    Fingerprint("webflow", "framework", "Webflow", _detect_webflow),
    Fingerprint("squarespace", "framework", "Squarespace", _detect_squarespace),
    Fingerprint("shopify", "framework", "Shopify storefront", _detect_shopify),
    Fingerprint("gatsby", "framework", "Gatsby", _detect_gatsby),
    # Search-as-a-service (2)
    Fingerprint("algolia", "search", "Algolia search backend", _detect_algolia),
    Fingerprint("elasticsearch", "search", "Elasticsearch backend", _detect_elasticsearch),
    # API conventions (4)
    Fingerprint("rest_api", "api", "Generic REST /api/ convention", _detect_rest_api),
    Fingerprint("graphql", "api", "GraphQL API", _detect_graphql),
    Fingerprint("wp_json_api", "api", "WordPress REST /wp-json/", _detect_wp_json_api),
    Fingerprint("underscore_api", "api", "Generic /_api/ convention", _detect_underscore_api),
    # Embedded data signals (5)
    Fingerprint("json_ld", "embedded_data", "JSON-LD structured data", _detect_json_ld),
    Fingerprint("open_graph", "embedded_data", "Open Graph metadata", _detect_open_graph),
    Fingerprint(
        "schema_org_microdata",
        "embedded_data",
        "Schema.org microdata",
        _detect_schema_org_microdata,
    ),
    Fingerprint(
        "next_data_blob",
        "embedded_data",
        "__NEXT_DATA__ hydration blob",
        _detect_next_data_blob,
    ),
    Fingerprint(
        "apollo_state_blob",
        "embedded_data",
        "Apollo / __INITIAL_STATE__ hydration blob",
        _detect_apollo_state_blob,
    ),
    # Structural patterns (6)
    Fingerprint(
        "form_gated_dataset",
        "structural",
        "Form posting to API/export endpoint",
        _detect_form_gated_dataset,
    ),
    Fingerprint("bulk_download_csv", "structural", "CSV bulk download link", _detect_bulk_download_csv),
    Fingerprint("bulk_download_xlsx", "structural", "Excel bulk download link", _detect_bulk_download_xlsx),
    Fingerprint("bulk_download_zip", "structural", "ZIP bulk download link", _detect_bulk_download_zip),
    Fingerprint("bulk_download_json", "structural", "JSON bulk download link", _detect_bulk_download_json),
    Fingerprint(
        "spa_empty_shell",
        "structural",
        "Empty body shell with SPA mount point",
        _detect_spa_empty_shell,
    ),
)

assert len(FINGERPRINTS) == 25, (
    f"Expected exactly 25 fingerprints to match docs/how-it-works.md; got {len(FINGERPRINTS)}. "
    "Update the docs and README to the real count."
)


def detect_all(html: str, soup: BeautifulSoup) -> list[FingerprintHit]:
    """Run every fingerprint and return the hits in registry order."""
    hits: list[FingerprintHit] = []
    for fp in FINGERPRINTS:
        evidence = fp.detect(html, soup)
        if evidence:
            hits.append(FingerprintHit(fp.id, fp.category, fp.description, evidence))
    return hits


def get_fingerprint(fp_id: str) -> Fingerprint:
    """Look up a fingerprint by ID. Raises ``KeyError`` if not found."""
    for fp in FINGERPRINTS:
        if fp.id == fp_id:
            return fp
    raise KeyError(f"Unknown fingerprint: {fp_id!r}")
