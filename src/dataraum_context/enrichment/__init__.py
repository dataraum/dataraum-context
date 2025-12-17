"""Enrichment layer - semantic and topological metadata extraction.

NOTE: Temporal enrichment has been moved to analysis/temporal/.
Use profile_temporal() from dataraum_context.analysis.temporal instead.
"""

from dataraum_context.enrichment.agent import SemanticAgent
from dataraum_context.enrichment.semantic import enrich_semantic
from dataraum_context.enrichment.topology import enrich_topology

__all__ = [
    "enrich_semantic",
    "enrich_topology",
    "SemanticAgent",
]
