"""
Core storage models.

This module contains the core storage models and shared infrastructure.
Domain-specific models have been co-located with their business logic:
- Profiling models: dataraum_context.profiling.db_models
- Enrichment models: dataraum_context.enrichment.db_models
- Quality models: dataraum_context.quality.db_models
- Domain quality models: dataraum_context.quality.domains.db_models
- LLM models: dataraum_context.llm.db_models
- Graphs models: dataraum_context.graphs.db_models

Pruned (unused):
- Filtering models (FilteringRecommendationRecord, FilteringExecutionRecord)
- Ontology models (ontologies, ontology_applications) - loaded from config files instead
- Schema version tracking (DBSchemaVersion)

Core modules (remain here):
- core.py: Core entities (sources, tables, columns)

Created: 2025-11-28
"""

# Import all model modules to ensure SQLAlchemy relationships are resolved
# ruff: noqa: F401
from dataraum_context.storage.models_v2 import core
from dataraum_context.storage.models_v2.base import Base, metadata_obj

# Core entities
from dataraum_context.storage.models_v2.core import Column, Source, Table

__all__ = [
    # Base
    "Base",
    "metadata_obj",
    # Core entities
    "Source",
    "Table",
    "Column",
]
