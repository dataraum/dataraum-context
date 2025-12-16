"""Enrichment layer - semantic, topological, and temporal metadata extraction.

NOTE: Semantic analysis has been moved to analysis/semantic/.
Import from dataraum_context.analysis.semantic for new code.

This module keeps local copies for backwards compatibility and to avoid
circular imports during the transition.
"""

# Keep local semantic imports to avoid circular imports
# New code should import from dataraum_context.analysis.semantic
from dataraum_context.enrichment.agent import SemanticAgent
from dataraum_context.enrichment.semantic import enrich_semantic
from dataraum_context.enrichment.temporal import enrich_temporal
from dataraum_context.enrichment.topology import enrich_topology

__all__ = [
    "enrich_semantic",
    "enrich_temporal",
    "enrich_topology",
    "SemanticAgent",
]
