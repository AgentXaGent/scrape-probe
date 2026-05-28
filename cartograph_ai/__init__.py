"""cartograph-ai: probe before extract.

Given a URL, classify how a site serves data and recommend the optimal
extraction strategy. Claude is the intelligence layer, not the scraper.

Public API::

    from cartograph_ai import probe

See https://github.com/AgentXaGent/cartograph-ai for full docs.
"""

from cartograph_ai._version import __version__
from cartograph_ai.exceptions import (
    AntiBotDetectedError,
    AuthWalledError,
    CartographError,
    ClassificationError,
    HTMLAnalysisError,
    HTTPProbeError,
    LowConfidenceError,
    OutputValidationError,
    ProbeTimeoutError,
)
from cartograph_ai.probe import (
    LOW_CONFIDENCE_THRESHOLD,
    ProbeOptions,
    probe,
)
from cartograph_ai.schema import (
    Classification,
    ClaudeResponse,
    EndpointDescriptor,
    ExtractionStrategy,
    ProbeResult,
)

__all__ = [
    "__version__",
    # Probe entry point
    "probe",
    "ProbeOptions",
    "LOW_CONFIDENCE_THRESHOLD",
    # Schema models (re-exported for typing convenience)
    "Classification",
    "ClaudeResponse",
    "EndpointDescriptor",
    "ExtractionStrategy",
    "ProbeResult",
    # Exception types
    "AntiBotDetectedError",
    "AuthWalledError",
    "CartographError",
    "ClassificationError",
    "HTMLAnalysisError",
    "HTTPProbeError",
    "LowConfidenceError",
    "OutputValidationError",
    "ProbeTimeoutError",
]
