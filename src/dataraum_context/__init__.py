"""DataRaum Context Engine.

A rich metadata context engine for AI-driven data analytics.
"""

__version__ = "0.1.0"

from dataraum_context.core.models import (
    ContextDocument,
    Result,
)

__all__ = [
    "ContextDocument",
    "Result",
    "__version__",
]
