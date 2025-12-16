"""Enrichment layer models.

Defines data structures for semantic and topological relationship discovery.

NOTE: Semantic models have been moved to analysis/semantic/models.py.
This module re-exports them for backwards compatibility.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Import semantic models from their new canonical location
# Re-exported for backwards compatibility
from dataraum_context.analysis.semantic.models import (  # noqa: F401
    EntityDetection,
    Relationship,
    SemanticAnnotation,
    SemanticEnrichmentResult,
)

# Import quality models that enrichment references
from dataraum_context.quality.models import TemporalQualityResult


class JoinStep(BaseModel):
    """A single step in a join path."""

    from_column: str
    to_table: str
    to_column: str
    confidence: float


class JoinPath(BaseModel):
    """A computed join path between tables."""

    from_table: str
    to_table: str
    steps: list[JoinStep]
    total_confidence: float


# === Enrichment Result Models ===
# SemanticEnrichmentResult is imported from analysis/semantic/models.py above


class TopologyEnrichmentResult(BaseModel):
    """Result of topology enrichment operation."""

    relationships: list[Relationship] = Field(default_factory=list)
    join_paths: list[JoinPath] = Field(default_factory=list)


class TemporalEnrichmentResult(BaseModel):
    """Result of temporal enrichment operation."""

    profiles: list[TemporalQualityResult] = Field(default_factory=list)


__all__ = [
    # Re-exported from analysis/semantic/models.py
    "EntityDetection",
    "Relationship",
    "SemanticAnnotation",
    "SemanticEnrichmentResult",
    # Local models
    "JoinStep",
    "JoinPath",
    "TopologyEnrichmentResult",
    "TemporalEnrichmentResult",
]
