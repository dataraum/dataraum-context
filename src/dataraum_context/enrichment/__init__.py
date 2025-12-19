"""Enrichment layer - semantic and topological metadata extraction.

NOTE: Semantic enrichment has been moved to analysis/semantic/.
NOTE: Temporal enrichment has been moved to analysis/temporal/.
NOTE: Topology enrichment has been removed (was using deleted TDA code).

This module re-exports for backward compatibility.
"""

# Re-export from canonical locations
from dataraum_context.analysis.semantic import SemanticAgent, enrich_semantic

__all__ = [
    "enrich_semantic",
    "SemanticAgent",
]
