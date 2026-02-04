"""DataRaum Context Engine.

A rich metadata context engine for AI-driven data analytics.

Example:
    from dataraum import Context

    ctx = Context("./pipeline_output")
    ctx.tables
    ctx.entropy.summary()
    result = ctx.query("What is the total revenue?")
"""

__version__ = "0.1.0"

from dataraum.context import Context
from dataraum.core.models.base import Result

__all__ = [
    "Context",
    "Result",
    "__version__",
]
