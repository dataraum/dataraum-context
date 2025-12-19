"""Relationship utilities package.

NOTE: This module is a re-export shim for backward compatibility.
The canonical location is now dataraum_context.domains.financial.cycles.

For relationship gathering and graph analysis, use:
    from dataraum_context.domains.financial.cycles import (
        gather_relationships,
        analyze_relationship_graph,
        EnrichedRelationship,
    )
"""

# Re-export from canonical location for backward compatibility
from dataraum_context.domains.financial.cycles.relationships import (  # noqa: F401
    CONFIDENCE_THRESHOLDS,
    EnrichedRelationship,
    GraphAnalysisResult,
    analyze_relationship_graph,
    analyze_relationship_graph_detailed,
    build_relationship_graph,
    gather_relationships,
)

__all__ = [
    "gather_relationships",
    "analyze_relationship_graph",
    "analyze_relationship_graph_detailed",
    "build_relationship_graph",
    "EnrichedRelationship",
    "GraphAnalysisResult",
    "CONFIDENCE_THRESHOLDS",
]
