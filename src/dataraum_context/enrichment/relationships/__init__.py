"""Relationship utilities package.

This package consolidates all relationship-related functionality:
- Detection: TDA-based relationship discovery between tables
- Gathering: Loading relationships from database with filtering
- Graph Analysis: Cycle detection, connectivity analysis

Public API:
    - gather_relationships: Load enriched relationships from database
    - analyze_relationship_graph: Detect cycles and connectivity in relationship graph
    - EnrichedRelationship: Relationship model with column/table metadata

Example:
    from dataraum_context.enrichment.relationships import (
        gather_relationships,
        analyze_relationship_graph,
        EnrichedRelationship,
    )

    # Gather relationships from database
    relationships = await gather_relationships(table_ids, session)

    # Analyze graph structure
    graph_result = analyze_relationship_graph(table_ids, relationships)
    cycles = graph_result["cycles"]
    connected_components = graph_result["betti_0"]
"""

# Core models
# Gathering (database loading)
from dataraum_context.enrichment.relationships.gathering import (
    CONFIDENCE_THRESHOLDS,
    gather_relationships,
)

# Graph analysis (cycles, connectivity)
from dataraum_context.enrichment.relationships.graph_analysis import (
    analyze_relationship_graph,
    build_relationship_graph,
)
from dataraum_context.enrichment.relationships.models import (
    EnrichedRelationship,
    GraphAnalysisResult,
)

__all__ = [
    # Models
    "EnrichedRelationship",
    "GraphAnalysisResult",
    # Gathering
    "gather_relationships",
    "CONFIDENCE_THRESHOLDS",
    # Graph analysis
    "analyze_relationship_graph",
    "build_relationship_graph",
]
