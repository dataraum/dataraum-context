"""Context assembly models.

Defines the final context document structure for AI consumption."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from dataraum_context.core.models.base import ColumnRef, SemanticRole


class ColumnContext(BaseModel):
    """Context for a single column."""

    name: str
    data_type: str
    description: str | None = None

    null_ratio: float
    cardinality_ratio: float

    semantic_role: SemanticRole
    business_name: str | None = None
    detected_unit: str | None = None

    quality_score: float


class TableContext(BaseModel):
    """Context for a single table."""

    name: str
    description: str | None = None

    row_count: int
    columns: list[ColumnContext]

    entity_type: str | None = None
    grain_columns: list[str] = Field(default_factory=list)
    is_fact_table: bool = False
    is_dimension_table: bool = False

    time_columns: list[str] = Field(default_factory=list)
    granularity: str | None = None
    date_range: tuple[datetime, datetime] | None = None

    quality_score: float


class MetricDefinition(BaseModel):
    """A metric that can be computed."""

    name: str
    formula: str
    description: str
    required_columns: list[str] = Field(default_factory=list)
    applicable_tables: list[str] = Field(default_factory=list)
    output_type: str = "number"


class DomainConcept(BaseModel):
    """A domain concept from the ontology."""

    name: str
    description: str
    mapped_columns: list[ColumnRef] = Field(default_factory=list)


class QualitySummary(BaseModel):
    """Summary of data quality."""

    overall_score: float
    tables_assessed: int
    rules_executed: int
    issues_found: int
    critical_issues: list[str] = Field(default_factory=list)


class SuggestedQuery(BaseModel):
    """A suggested SQL query for the context (LLM-generated)."""

    name: str
    description: str
    category: str = "overview"  # 'overview', 'metrics', 'trends', 'segments', 'quality'
    sql: str
    complexity: str = "simple"  # 'simple', 'moderate', 'complex'


class ContextSummary(BaseModel):
    """Natural language summary of the data context (LLM-generated)."""

    summary: str
    key_facts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ContextDocument(BaseModel):
    """The main context document for AI consumption.

    This is the primary output of the context engine - a comprehensive
    document that gives AI everything it needs to understand and query
    the data effectively.

    Several fields are LLM-generated when enabled in config/llm.yaml.
    """

    # Data inventory
    tables: list[TableContext]
    relationships: list[Relationship]

    # Ontology interpretation
    ontology: str
    relevant_metrics: list[MetricDefinition]
    domain_concepts: list[DomainConcept]

    # Quality summary
    quality_summary: QualitySummary

    # LLM-generated content
    suggested_queries: list[SuggestedQuery] = Field(default_factory=list)
    context_summary: ContextSummary | None = None

    # Metadata
    llm_features_used: list[str] = Field(default_factory=list)  # which LLM features contributed


# Import Relationship after all models are defined to avoid circular import
from dataraum_context.enrichment.models import Relationship  # noqa: E402

# Rebuild ContextDocument model now that Relationship is available
ContextDocument.model_rebuild()
