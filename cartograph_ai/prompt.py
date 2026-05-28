"""The Stage 4 prompt published in ``docs/how-it-works.md``.

The prompt is checked into source verbatim. Anyone reading the package
can see exactly what the model is asked to do.  The accompanying test
(``tests/test_prompt.py``) re-extracts the prompt from ``how-it-works.md``
at test time and asserts byte equality with ``PROMPT_TEMPLATE`` below.
That guards against silent drift between docs and code.

Bumping the prompt requires three things in the same commit:

1. Update the fenced code block under ``### The prompt`` in ``how-it-works.md``.
2. Update ``PROMPT_TEMPLATE`` below to match.
3. Bump ``PROMPT_VERSION`` and add a CHANGELOG entry per the pinning policy.
"""

from __future__ import annotations

import json
from typing import Any

# Bump on any change to PROMPT_TEMPLATE. CHANGELOG entry required.
PROMPT_VERSION = "v1.0"

# The substitution sentinel inside PROMPT_TEMPLATE that gets replaced with
# the JSON-serialized probe results at runtime.
PROBE_RESULTS_SENTINEL = "{probe_results}"


PROMPT_TEMPLATE = """\
You are the intelligence layer of cartograph, a tool that classifies how
websites serve data and recommends extraction strategies. You receive
structured probe results from earlier stages and return a classification
plus a recommended approach.

Probe results (JSON):
{probe_results}

Apply these heuristics in order. Stop at the first one that fits.

1. Direct API discovered (REST, GraphQL, Algolia, Elasticsearch).
   Almost always the cleanest path. Recommend it.

2. Embedded data carries the target content (__NEXT_DATA__,
   window.__INITIAL_STATE__, hydration JSON, large inline JSON blobs).
   Extract from HTML; no API call required.

3. Structured static HTML carries the data (product cards, article
   listings, table rows with consistent selectors).
   Recommend HTML parsing with explicit selectors.

4. Form-gated bulk data (the NHTSA pattern: search form POSTs to an
   endpoint that returns CSV or JSON; or a downloads index page links
   to bulk files).
   Recommend form-POST or direct bulk download. Do not recommend
   scraping the search interface itself.

5. JS-rendered SPA with no accessible data layer in the HTML.
   If the browser extra is available, recommend re-probing with it.
   Otherwise, report honestly: this site needs the browser extra.

6. None of the above. Classify as "unknown" and explain what's
   missing or contradictory. Do not invent a strategy.

Return JSON matching this exact schema:

{
  "classification": "direct_api" | "embedded_data" | "static_html"
                  | "form_gated_bulk" | "js_rendered_spa" | "unknown",
  "confidence": float between 0.0 and 1.0,
  "reasoning": "one to three sentences explaining the call",
  "extraction_strategy": {
    "method": short label, e.g., "algolia_search", "wp_rest_api",
              "html_parsing", "form_post_bulk", "browser_render",
    "requires_browser": boolean,
    "estimated_requests": integer,
    "recommended_tool": "requests" | "httpx" | "playwright"
                       | "firecrawl" | "manual",
    "specifics": object with method-specific parameters
                 (endpoint URLs, selectors, query params, etc.)
  },
  "limitations": list of strings describing anything you could not
                determine. Populate when confidence is below 0.7.
}

If confidence is below 0.7, the limitations field MUST list specific
unknowns. "Insufficient information" alone is not acceptable; say what
information would change the classification.

Do not invent endpoints, app IDs, selectors, or parameters that were
not in the probe input. If you would need to guess a value, omit it
and list the gap in limitations."""


def build_prompt(probe_results: dict[str, Any]) -> str:
    """Substitute the probe results into the prompt template.

    Uses ``str.replace`` rather than ``str.format`` because the schema
    example in the prompt contains literal curly braces that would
    otherwise need escaping.  Replace operates only on the documented
    sentinel and runs exactly once.

    Args:
        probe_results: The structured findings dictionary assembled from
            Stages 1-3.  Serialized with ``json.dumps(..., indent=2,
            sort_keys=True)`` so two probes against the same site produce
            byte-identical prompts (reproducibility requirement from
            ``docs/how-it-works.md``).

    Returns:
        The full prompt with ``{probe_results}`` replaced by the JSON
        payload.

    Raises:
        ValueError: if ``PROBE_RESULTS_SENTINEL`` is not present in
            ``PROMPT_TEMPLATE`` (indicates the template was edited
            improperly).
    """
    if PROBE_RESULTS_SENTINEL not in PROMPT_TEMPLATE:
        raise ValueError(
            "PROMPT_TEMPLATE is missing the {probe_results} sentinel; "
            "fix prompt.py before calling build_prompt()."
        )
    payload = json.dumps(probe_results, indent=2, sort_keys=True)
    return PROMPT_TEMPLATE.replace(PROBE_RESULTS_SENTINEL, payload, 1)
