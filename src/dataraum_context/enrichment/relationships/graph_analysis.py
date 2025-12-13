"""Relationship graph analysis.

Analyze the graph structure of table relationships to detect cycles,
connectivity, and other topological properties.
"""

from collections.abc import Sequence
from typing import Any, Protocol

import networkx as nx

from dataraum_context.enrichment.relationships.models import GraphAnalysisResult


class RelationshipLike(Protocol):
    """Protocol for relationship objects (duck typing)."""

    from_table_id: str
    to_table_id: str
    confidence: float
    relationship_type: str | Any
    cardinality: str | Any | None


def build_relationship_graph(
    table_ids: list[str],
    relationships: Sequence[RelationshipLike],
) -> nx.DiGraph:  # type: ignore[type-arg]
    """Build a directed graph from table relationships.

    Args:
        table_ids: List of table IDs (nodes)
        relationships: List of relationships (edges)

    Returns:
        NetworkX directed graph with tables as nodes and relationships as edges
    """
    G: nx.DiGraph = nx.DiGraph()  # type: ignore[type-arg]

    # Add all tables as nodes
    for table_id in table_ids:
        G.add_node(table_id)

    # Add relationships as edges
    for rel in relationships:
        G.add_edge(
            rel.from_table_id,
            rel.to_table_id,
            confidence=rel.confidence,
            relationship_type=rel.relationship_type,
            cardinality=rel.cardinality,
        )

    return G


def analyze_relationship_graph(
    table_ids: list[str],
    relationships: Sequence[RelationshipLike],
) -> dict[str, list[list[str]] | int]:
    """Analyze the graph of table relationships to detect cycles.

    Uses NetworkX to find cycles in the directed graph of tables.
    These cycles represent business process flows (e.g., AR cycle, AP cycle).

    Args:
        table_ids: List of table IDs
        relationships: List of relationship objects (duck-typed)

    Returns:
        Dict containing:
        - cycles: List of cycles (each cycle is a list of table_ids)
        - betti_0: Number of connected components in undirected graph
        - cycle_count: Total number of cycles detected
    """
    G = build_relationship_graph(table_ids, relationships)

    # Find all simple cycles (paths that return to starting node)
    try:
        cycles = list(nx.simple_cycles(G))
    except Exception:
        cycles = []  # Graph might have issues

    # Compute Betti-0 (connected components) using undirected version
    G_undirected = G.to_undirected()
    betti_0 = nx.number_connected_components(G_undirected)

    return {
        "cycles": cycles,
        "betti_0": betti_0,
        "cycle_count": len(cycles),
    }


def analyze_relationship_graph_detailed(
    table_ids: list[str],
    relationships: Sequence[RelationshipLike],
) -> GraphAnalysisResult:
    """Analyze relationship graph with detailed results.

    Extended version of analyze_relationship_graph that returns a Pydantic model
    with additional metrics.

    Args:
        table_ids: List of table IDs
        relationships: List of relationship objects

    Returns:
        GraphAnalysisResult with cycles, connectivity, and graph metrics
    """
    G = build_relationship_graph(table_ids, relationships)

    # Find cycles
    try:
        cycles = list(nx.simple_cycles(G))
    except Exception:
        cycles = []

    # Compute connectivity
    G_undirected = G.to_undirected()
    betti_0 = nx.number_connected_components(G_undirected)

    return GraphAnalysisResult(
        cycles=cycles,
        betti_0=betti_0,
        cycle_count=len(cycles),
        node_count=G.number_of_nodes(),
        edge_count=G.number_of_edges(),
    )
