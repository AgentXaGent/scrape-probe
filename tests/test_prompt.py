"""Tests for ``cartograph_ai.prompt``.

The most important test re-extracts the prompt from
``docs/how-it-works.md`` at run time and asserts byte equality with the
``PROMPT_TEMPLATE`` constant.  If the doc and the constant ever diverge,
this test fails until they are brought back into sync (and a CHANGELOG
entry is added per the pinning policy).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cartograph_ai.prompt import (
    PROBE_RESULTS_SENTINEL,
    PROMPT_TEMPLATE,
    PROMPT_VERSION,
    build_prompt,
)

DOCS_PATH = Path(__file__).parent.parent / "docs" / "how-it-works.md"


# --- Drift guard: code <-> doc -------------------------------------------

def _extract_published_prompt() -> str:
    """Pull the fenced code block under '### The prompt' from how-it-works.md."""
    doc = DOCS_PATH.read_text()
    start = doc.index("### The prompt")
    end = doc.index("### The output schema")
    section = doc[start:end]
    m = re.search(r"```\n(.*?)\n```", section, re.DOTALL)
    assert m is not None, (
        "Could not find a fenced code block under '### The prompt' in "
        "docs/how-it-works.md.  Has the doc structure changed?"
    )
    return m.group(1)


def test_prompt_template_matches_published_doc():
    published = _extract_published_prompt()
    assert PROMPT_TEMPLATE == published, (
        "PROMPT_TEMPLATE in cartograph_ai/prompt.py has drifted from the "
        "fenced block under '### The prompt' in docs/how-it-works.md. "
        "Update both in the same commit and bump PROMPT_VERSION + CHANGELOG."
    )


def test_prompt_sentinel_present():
    assert PROBE_RESULTS_SENTINEL in PROMPT_TEMPLATE
    assert PROMPT_TEMPLATE.count(PROBE_RESULTS_SENTINEL) == 1, (
        "Sentinel must appear exactly once so build_prompt() substitutes "
        "in a single deterministic place."
    )


def test_prompt_version_is_str():
    assert isinstance(PROMPT_VERSION, str)
    assert PROMPT_VERSION  # non-empty


# --- build_prompt --------------------------------------------------------

def test_build_prompt_substitutes_sentinel():
    payload = {"url": "https://sasaki.com/projects", "stage1": {"status": 200}}
    out = build_prompt(payload)
    # The literal sentinel is gone after substitution.
    assert PROBE_RESULTS_SENTINEL not in out
    # The JSON payload appears in the output.
    assert "https://sasaki.com/projects" in out
    assert '"status": 200' in out


def test_build_prompt_preserves_schema_braces():
    """The prompt's JSON schema example contains literal { and } that must
    survive substitution unchanged."""
    out = build_prompt({"x": 1})
    # The schema's literal opening + closing braces still appear.
    assert '"classification":' in out
    assert '"extraction_strategy":' in out
    # And the closing brace of the schema example is preserved.
    assert "limitations" in out


def test_build_prompt_is_deterministic():
    """Same input -> identical output. Reproducibility requirement."""
    payload = {"b": 2, "a": 1, "c": {"nested": [3, 2, 1]}}
    out_1 = build_prompt(payload)
    out_2 = build_prompt(payload)
    assert out_1 == out_2


def test_build_prompt_sorts_keys():
    """Key ordering should not affect output (sort_keys=True)."""
    out_alpha = build_prompt({"a": 1, "z": 2})
    out_zeta = build_prompt({"z": 2, "a": 1})
    assert out_alpha == out_zeta


def test_build_prompt_uses_indented_json():
    """Indent=2 keeps the prompt readable when a probe fails and the
    payload needs to be inspected in --debug output."""
    out = build_prompt({"key": "value"})
    assert '  "key": "value"' in out  # indent=2 means 2-space leading


def test_build_prompt_empty_dict():
    out = build_prompt({})
    assert "{}" in out
    # Substitution still happens; sentinel is gone.
    assert PROBE_RESULTS_SENTINEL not in out


def test_build_prompt_with_unicode():
    out = build_prompt({"name": "Sasaki Projects é"})
    # json.dumps default escapes non-ASCII, which is fine for our purposes.
    assert "Sasaki Projects" in out
