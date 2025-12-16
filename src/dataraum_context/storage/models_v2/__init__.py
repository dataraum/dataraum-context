"""
Core storage models.

This module contains the core storage models and shared infrastructure.
Domain-specific models have been co-located with their business logic:
- Profiling models: dataraum_context.profiling.db_models

Remaining models (to be co-located in future phases):
- Pillar 2: Topological Context (topological_context.py, relationship.py)
- Pillar 3: Semantic Context (semantic_context.py)
- Pillar 4: Temporal Context (temporal_context.py)
- Pillar 5: Quality Context (domain_quality.py, quality_rules.py)

Core modules (will remain here):
- core.py: Core entities (sources, tables, columns)
- ontology.py: Ontology definitions and applications
- llm.py: LLM response caching (to be moved to llm module)

Created: 2025-11-28
"""

# Import all model modules to ensure SQLAlchemy relationships are resolved
# ruff: noqa: F401
from dataraum_context.storage.models_v2 import (
    core,
    domain_quality,
    filtering,
    llm,
    ontology,
    quality_rules,
    relationship,
    schema,
    semantic_context,
    temporal_context,
    topological_context,
)
from dataraum_context.storage.models_v2.base import Base, metadata_obj

# Core entities
from dataraum_context.storage.models_v2.core import Column, Source, Table

# Pillar 5: Quality Context (domain-specific)
from dataraum_context.storage.models_v2.domain_quality import (
    DomainQualityMetrics,
    DoubleEntryCheck,
    FinancialQualityMetrics,
    FiscalPeriodIntegrity,
    SignConventionViolation,
    TrialBalanceCheck,
)

# Filtering persistence
from dataraum_context.storage.models_v2.filtering import (
    FilteringExecutionRecord,
    FilteringRecommendationRecord,
)

# Graph execution persistence (unified transformation graphs)
from dataraum_context.storage.models_v2.graphs import (
    GeneratedCodeRecord,
    GraphExecutionRecord,
    StepResultRecord,
)

# LLM integration
from dataraum_context.storage.models_v2.llm import LLMCache

# Ontology system
from dataraum_context.storage.models_v2.ontology import Ontology, OntologyApplication

# Pillar 5: Quality Context (rules-based)
from dataraum_context.storage.models_v2.quality_rules import (
    QualityResult,
    QualityRule,
    QualityScore,
)
from dataraum_context.storage.models_v2.relationship import JoinPath, Relationship

# Schema version tracking
from dataraum_context.storage.models_v2.schema import DBSchemaVersion

# Pillar 3: Semantic Context
from dataraum_context.storage.models_v2.semantic_context import (
    SemanticAnnotation,
    TableEntity,
)

# Pillar 4: Temporal Context
from dataraum_context.storage.models_v2.temporal_context import (
    TemporalQualityMetrics,
)

# Pillar 2: Topological Context
from dataraum_context.storage.models_v2.topological_context import (
    BusinessCycleClassification,
    MultiTableTopologyMetrics,
    TopologicalQualityMetrics,
)

__all__ = [
    # Base
    "Base",
    "metadata_obj",
    # Core entities
    "Source",
    "Table",
    "Column",
    # Pillar 2: Topological Context
    "TopologicalQualityMetrics",
    "MultiTableTopologyMetrics",
    "BusinessCycleClassification",
    "Relationship",
    "JoinPath",
    # Pillar 3: Semantic Context
    "SemanticAnnotation",
    "TableEntity",
    # Pillar 4: Temporal Context
    "TemporalQualityMetrics",
    # Pillar 5: Domain Quality
    "DomainQualityMetrics",
    "FinancialQualityMetrics",
    "DoubleEntryCheck",
    "TrialBalanceCheck",
    "SignConventionViolation",
    "FiscalPeriodIntegrity",
    # Pillar 5: Quality Rules
    "QualityRule",
    "QualityResult",
    "QualityScore",
    # Ontology System
    "Ontology",
    "OntologyApplication",
    # LLM Integration
    "LLMCache",
    # Schema Version
    "DBSchemaVersion",
    # Filtering Persistence
    "FilteringRecommendationRecord",
    "FilteringExecutionRecord",
    # Graph Execution Persistence
    "GeneratedCodeRecord",
    "GraphExecutionRecord",
    "StepResultRecord",
]
