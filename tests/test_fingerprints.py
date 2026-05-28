"""Tests for ``cartograph_ai.fingerprints``.

Every fingerprint gets at least one positive synthetic HTML sample.
Negative tests cover the obvious false-positive risks (empty body,
content-rich page without SPA mount, etc.).
"""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from cartograph_ai.fingerprints import (
    FINGERPRINTS,
    FingerprintHit,
    detect_all,
    get_fingerprint,
)


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


# ---------------- Registry shape ----------------------------------------


def test_registry_has_exactly_25_fingerprints():
    assert len(FINGERPRINTS) == 25


def test_registry_ids_are_unique():
    ids = [fp.id for fp in FINGERPRINTS]
    assert len(ids) == len(set(ids))


def test_registry_categories_are_valid():
    valid = {"framework", "search", "api", "embedded_data", "structural"}
    for fp in FINGERPRINTS:
        assert fp.category in valid, f"{fp.id} has invalid category {fp.category!r}"


def test_registry_distribution_matches_docs():
    """8 framework + 2 search + 4 api + 5 embedded_data + 6 structural = 25."""
    counts: dict[str, int] = {}
    for fp in FINGERPRINTS:
        counts[fp.category] = counts.get(fp.category, 0) + 1
    assert counts == {
        "framework": 8,
        "search": 2,
        "api": 4,
        "embedded_data": 5,
        "structural": 6,
    }


def test_get_fingerprint_known():
    assert get_fingerprint("nextjs").id == "nextjs"


def test_get_fingerprint_unknown_raises():
    with pytest.raises(KeyError):
        get_fingerprint("not_a_real_fingerprint")


# ---------------- Framework signatures (8) -----------------------------


def test_nextjs_via_next_data_tag():
    html = '<html><body><script id="__NEXT_DATA__">{"a":1}</script></body></html>'
    assert get_fingerprint("nextjs").detect(html, _soup(html)) is not None


def test_nextjs_via_asset_path():
    html = '<html><body><link rel="preload" href="/_next/static/chunks/x.js"></body></html>'
    assert get_fingerprint("nextjs").detect(html, _soup(html)) is not None


def test_nextjs_no_signal_returns_none():
    html = "<html><body><p>nothing here</p></body></html>"
    assert get_fingerprint("nextjs").detect(html, _soup(html)) is None


def test_nuxtjs_via_asset_path():
    html = '<script src="/_nuxt/chunks/main.js"></script>'
    assert get_fingerprint("nuxtjs").detect(html, _soup(html)) is not None


def test_nuxtjs_via_window_blob():
    html = "<script>window.__NUXT__ = {a: 1}</script>"
    assert get_fingerprint("nuxtjs").detect(html, _soup(html)) is not None


def test_wordpress_via_wp_content():
    html = '<link rel="stylesheet" href="/wp-content/themes/twentytwentyfour/style.css">'
    assert get_fingerprint("wordpress").detect(html, _soup(html)) is not None


def test_wordpress_via_wp_includes():
    html = '<script src="/wp-includes/js/jquery/jquery.min.js"></script>'
    assert get_fingerprint("wordpress").detect(html, _soup(html)) is not None


def test_wordpress_via_generator_meta():
    html = '<head><meta name="generator" content="WordPress 6.4.2"></head>'
    assert get_fingerprint("wordpress").detect(html, _soup(html)) is not None


def test_aem_via_content_dam():
    html = '<img src="/content/dam/ford/2026/mach-e.jpg">'
    assert get_fingerprint("adobe_experience_manager").detect(html, _soup(html)) is not None


def test_aem_via_urn():
    html = '<meta name="aem-uuid" content="urn:aaid:aem:abc-def-123">'
    assert get_fingerprint("adobe_experience_manager").detect(html, _soup(html)) is not None


def test_aem_via_clientlibs():
    html = '<link rel="stylesheet" href="/etc.clientlibs/ford/global.css">'
    assert get_fingerprint("adobe_experience_manager").detect(html, _soup(html)) is not None


def test_webflow_via_data_wf_attr():
    html = '<html data-wf-domain="example.com" data-wf-page="abc"><body></body></html>'
    assert get_fingerprint("webflow").detect(html, _soup(html)) is not None


def test_webflow_via_asset_host():
    html = '<script src="https://assets.webflow.com/12345/main.js"></script>'
    assert get_fingerprint("webflow").detect(html, _soup(html)) is not None


def test_squarespace_via_static_cdn():
    html = '<script src="//static.squarespace.com/static/sqs.js"></script>'
    assert get_fingerprint("squarespace").detect(html, _soup(html)) is not None


def test_squarespace_via_context_var():
    html = "<script>Static.SQUARESPACE_CONTEXT = {}</script>"
    assert get_fingerprint("squarespace").detect(html, _soup(html)) is not None


def test_shopify_via_cdn():
    html = '<link rel="stylesheet" href="https://cdn.shopify.com/s/files/x.css">'
    assert get_fingerprint("shopify").detect(html, _soup(html)) is not None


def test_shopify_via_js_var():
    html = "<script>Shopify.shop = 'mystore.myshopify.com';</script>"
    assert get_fingerprint("shopify").detect(html, _soup(html)) is not None


def test_shopify_via_meta_tag():
    html = '<head><meta name="shopify-digital-wallet" content="x"></head>'
    assert get_fingerprint("shopify").detect(html, _soup(html)) is not None


def test_gatsby_via_div_id():
    html = '<body><div id="___gatsby"><div></div></div></body>'
    assert get_fingerprint("gatsby").detect(html, _soup(html)) is not None


def test_gatsby_via_page_data():
    html = '<script>fetch("/page-data/index/page-data.json")</script>fetch("/page-data/app-data.json")'
    assert get_fingerprint("gatsby").detect(html, _soup(html)) is not None


# ---------------- Search-as-a-service (2) -------------------------------


def test_algolia_via_hostname():
    html = '<script src="https://AHNZ21XTZ6-dsn.algolia.net/1/indexes/x/query"></script>'
    assert get_fingerprint("algolia").detect(html, _soup(html)) is not None


def test_algolia_via_api_key_var():
    html = "<script>const ALGOLIA_API_KEY = 'abc123';</script>"
    assert get_fingerprint("algolia").detect(html, _soup(html)) is not None


def test_algolia_via_client_library_name():
    html = "<script>import algoliasearch from 'algoliasearch';</script>"
    assert get_fingerprint("algolia").detect(html, _soup(html)) is not None


def test_elasticsearch_via_search_endpoint():
    html = '<script>fetch("/_search?q=fooditem")</script>'
    assert get_fingerprint("elasticsearch").detect(html, _soup(html)) is not None


def test_elasticsearch_via_host_reference():
    html = '<a href="https://www.elastic.co/">Powered by</a>'
    assert get_fingerprint("elasticsearch").detect(html, _soup(html)) is not None


# ---------------- API conventions (4) -----------------------------------


def test_rest_api_via_v1_path():
    html = '<script>fetch("/api/v1/products")</script>'
    assert get_fingerprint("rest_api").detect(html, _soup(html)) is not None


def test_rest_api_via_plain_path():
    html = '<script>const r = fetch("/api/inventory", {});</script>'
    assert get_fingerprint("rest_api").detect(html, _soup(html)) is not None


def test_graphql_via_endpoint():
    html = '<script>fetch("/graphql", {body: q})</script>'
    assert get_fingerprint("graphql").detect(html, _soup(html)) is not None


def test_graphql_via_apollo_client():
    html = "<script>import { ApolloClient } from '@apollo/client';</script>"
    assert get_fingerprint("graphql").detect(html, _soup(html)) is not None


def test_wp_json_api_via_path():
    html = '<link rel="https://api.w.org/" href="https://x.com/wp-json/">'
    assert get_fingerprint("wp_json_api").detect(html, _soup(html)) is not None


def test_underscore_api():
    html = '<script>fetch("/_api/inventory")</script>'
    assert get_fingerprint("underscore_api").detect(html, _soup(html)) is not None


# ---------------- Embedded data signals (5) -----------------------------


def test_json_ld_block():
    html = (
        '<script type="application/ld+json">'
        '{"@context":"https://schema.org","@type":"Organization"}</script>'
    )
    assert get_fingerprint("json_ld").detect(html, _soup(html)) is not None


def test_open_graph_meta():
    html = (
        '<head><meta property="og:title" content="Hello">'
        '<meta property="og:url" content="https://x.com"></head>'
    )
    assert get_fingerprint("open_graph").detect(html, _soup(html)) is not None


def test_schema_org_microdata():
    html = '<div itemscope itemtype="https://schema.org/Product"><span>Hello</span></div>'
    assert get_fingerprint("schema_org_microdata").detect(html, _soup(html)) is not None


def test_next_data_blob():
    html = '<script id="__NEXT_DATA__">{"page":"projects","query":{}}</script>'
    assert get_fingerprint("next_data_blob").detect(html, _soup(html)) is not None


def test_apollo_state_blob():
    html = "<script>window.__APOLLO_STATE__ = {x: 1};</script>"
    assert get_fingerprint("apollo_state_blob").detect(html, _soup(html)) is not None


def test_apollo_state_blob_via_initial_state():
    html = "<script>window.__INITIAL_STATE__ = {a: 1};</script>"
    assert get_fingerprint("apollo_state_blob").detect(html, _soup(html)) is not None


# ---------------- Structural patterns (6) -------------------------------


def test_form_gated_dataset_via_api_action():
    html = '<form method="post" action="/api/export"><input name="q"></form>'
    assert get_fingerprint("form_gated_dataset").detect(html, _soup(html)) is not None


def test_form_gated_dataset_via_download_action():
    html = '<form method="get" action="/download/inventory"></form>'
    assert get_fingerprint("form_gated_dataset").detect(html, _soup(html)) is not None


def test_form_gated_dataset_ignores_plain_form():
    html = '<form method="post" action="/contact"></form>'
    assert get_fingerprint("form_gated_dataset").detect(html, _soup(html)) is None


def test_bulk_download_csv():
    html = '<a href="/data/inventory.csv">Download CSV</a>'
    assert get_fingerprint("bulk_download_csv").detect(html, _soup(html)) is not None


def test_bulk_download_csv_with_query_string():
    html = '<a href="/data/x.csv?year=2026">CSV</a>'
    assert get_fingerprint("bulk_download_csv").detect(html, _soup(html)) is not None


def test_bulk_download_xlsx():
    html = '<a href="/data.xlsx">Excel</a>'
    assert get_fingerprint("bulk_download_xlsx").detect(html, _soup(html)) is not None


def test_bulk_download_xls():
    html = '<a href="/old/data.xls">Old Excel</a>'
    assert get_fingerprint("bulk_download_xlsx").detect(html, _soup(html)) is not None


def test_bulk_download_zip():
    html = '<a href="/dump.zip">Archive</a>'
    assert get_fingerprint("bulk_download_zip").detect(html, _soup(html)) is not None


def test_bulk_download_json():
    html = '<a href="/feed.json">Feed</a>'
    assert get_fingerprint("bulk_download_json").detect(html, _soup(html)) is not None


def test_bulk_download_no_match_for_html_links():
    html = '<a href="/data.html">HTML page</a>'
    assert get_fingerprint("bulk_download_csv").detect(html, _soup(html)) is None
    assert get_fingerprint("bulk_download_json").detect(html, _soup(html)) is None


def test_spa_empty_shell_with_root_div():
    html = '<html><body><div id="root"></div><script>app.start()</script></body></html>'
    assert get_fingerprint("spa_empty_shell").detect(html, _soup(html)) is not None


def test_spa_empty_shell_with_app_div():
    html = '<html><body><div id="app"></div></body></html>'
    assert get_fingerprint("spa_empty_shell").detect(html, _soup(html)) is not None


def test_spa_not_flagged_when_body_has_real_text():
    body = "<p>" + ("Real content. " * 50) + "</p>"
    html = f"<html><body><div id='root'>{body}</div></body></html>"
    assert get_fingerprint("spa_empty_shell").detect(html, _soup(html)) is None


def test_spa_not_flagged_when_no_mount_point():
    html = "<html><body><p>just a quiet page</p></body></html>"
    assert get_fingerprint("spa_empty_shell").detect(html, _soup(html)) is None


# ---------------- detect_all aggregation -------------------------------


def test_detect_all_finds_multiple_fingerprints():
    html = """
    <html data-wf-domain="example.com">
    <head>
      <meta property="og:title" content="Sasaki Projects">
      <link rel="preload" href="/_next/static/chunks/main.js">
    </head>
    <body>
      <script id="__NEXT_DATA__">{"page":"projects"}</script>
      <script type="application/ld+json">{"@context":"https://schema.org"}</script>
      <script>fetch("https://x.algolia.net/1/indexes/x/query")</script>
      <a href="/exports/projects.csv">Download</a>
    </body>
    </html>
    """
    hits = detect_all(html, _soup(html))
    ids = {h.id for h in hits}
    assert {
        "nextjs",
        "next_data_blob",
        "json_ld",
        "open_graph",
        "algolia",
        "bulk_download_csv",
        "webflow",  # data-wf-domain on the html tag
    }.issubset(ids)


def test_detect_all_returns_empty_on_plain_content():
    html = "<html><body><p>" + ("quiet text. " * 80) + "</p></body></html>"
    hits = detect_all(html, _soup(html))
    assert hits == []


def test_fingerprint_hit_carries_metadata():
    html = '<script id="__NEXT_DATA__">{"a":1}</script>'
    hits = detect_all(html, _soup(html))
    nextjs_hits = [h for h in hits if h.id == "nextjs"]
    assert len(nextjs_hits) == 1
    hit = nextjs_hits[0]
    assert isinstance(hit, FingerprintHit)
    assert hit.category == "framework"
    assert hit.description == "Next.js framework signals"
    assert "__NEXT_DATA__" in hit.evidence


def test_detect_all_preserves_registry_order():
    """When multiple fingerprints hit, results come in registry order."""
    html = """
    <html>
    <body>
      <script id="__NEXT_DATA__">{"x":1}</script>
      <a href="/data.csv">CSV</a>
    </body>
    </html>
    """
    hits = detect_all(html, _soup(html))
    ids = [h.id for h in hits]
    # nextjs comes before next_data_blob comes before bulk_download_csv in registry
    assert ids.index("nextjs") < ids.index("next_data_blob") < ids.index("bulk_download_csv")
