# Entropy Models Specification

This document defines the data models for the entropy layer. All models follow the original specification in [entropy-management-framework.md](../entropy-management-framework.md).

**Related Documentation:**
- [ENTROPY_IMPLEMENTATION_PLAN.md](./ENTROPY_IMPLEMENTATION_PLAN.md) - Implementation roadmap
- [ENTROPY_CONTRACTS.md](./ENTROPY_CONTRACTS.md) - Data readiness thresholds
- [ENTROPY_QUERY_BEHAVIOR.md](./ENTROPY_QUERY_BEHAVIOR.md) - Agent response policies

---

## Core Models

### EntropyObject

The core entropy measurement object. Each EntropyObject represents one entropy measurement for one target (column, table, or relationship).

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EntropyObject:
    """Core entropy measurement object.

    Represents a single entropy measurement for a specific dimension/sub-dimension
    applied to a specific target (column, table, or relationship).
    """

    # Identity
    object_id: str  # UUID
    layer: str  # structural, semantic, value, computational
    dimension: str  # schema, types, relations, business_meaning, units, etc.
    sub_dimension: str  # naming_clarity, type_fidelity, etc.
    target: str  # column:{table}.{column}, table:{table_name}, relationship:{t1}-{t2}

    # Measurement
    score: float  # 0.0 = deterministic, 1.0 = maximum uncertainty
    confidence: float  # How confident are we in this score (0.0-1.0)

    # Evidence (dimension-specific)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    # Resolution options
    resolution_options: list["ResolutionOption"] = field(default_factory=list)

    # Context for different consumers
    llm_context: "LLMContext" = field(default_factory=lambda: LLMContext())
    human_context: "HumanContext" = field(default_factory=lambda: HumanContext())

    # Metadata
    computed_at: datetime = field(default_factory=datetime.utcnow)
    source_analysis_ids: list[str] = field(default_factory=list)  # Links to source analyses
    detector_id: str = ""  # Which detector produced this

    def is_high_entropy(self, threshold: float = 0.6) -> bool:
        """Check if entropy exceeds threshold."""
        return self.score >= threshold

    def is_critical(self, threshold: float = 0.8) -> bool:
        """Check if entropy is at critical level."""
        return self.score >= threshold
```

### ResolutionOption

An actionable fix that can reduce entropy.

```python
@dataclass
class ResolutionOption:
    """An actionable resolution that can reduce entropy.

    Each resolution has an action type, parameters, expected impact,
    and effort level for prioritization.
    """

    action: str  # e.g., "add_column_alias", "declare_unit", "add_definition"
    parameters: dict[str, Any]  # Action-specific parameters
    expected_entropy_reduction: float  # Expected reduction (0.0-1.0)
    effort: str  # "low", "medium", "high"
    description: str = ""  # Human-readable description

    # Cascade tracking - which other dimensions this resolution affects
    cascade_dimensions: list[str] = field(default_factory=list)

    def priority_score(self) -> float:
        """Calculate priority based on reduction vs effort.

        Higher score = higher priority.
        """
        effort_factor = {"low": 1.0, "medium": 2.0, "high": 4.0}.get(self.effort, 2.0)
        return self.expected_entropy_reduction / effort_factor
```

### LLMContext

Context for query agents to understand and work with entropy.

```python
@dataclass
class LLMContext:
    """Context for LLM/query agents.

    Provides information the agent needs to:
    - Understand the uncertainty
    - Decide how to handle it (answer, ask, refuse)
    - Generate appropriate caveats or assumptions
    """

    description: str = ""  # Plain language description of the entropy
    query_impact: str = ""  # How this affects query generation/results

    # For answering with assumptions
    best_guess: str | None = None  # Best guess if forced to assume
    best_guess_confidence: float = 0.0  # Confidence in the guess
    assumption_if_unresolved: str | None = None  # Assumption to state

    # For query generation
    filter_recommendation: str | None = None  # Suggested WHERE clause
    aggregation_recommendation: str | None = None  # Suggested aggregation
    join_recommendation: str | None = None  # Suggested join approach

    # Warnings and caveats
    warning: str | None = None  # Warning to include in response
    caveat_template: str | None = None  # Template for caveat in answer
```

### HumanContext

Context for human users and administrators.

```python
@dataclass
class HumanContext:
    """Context for human users and administrators.

    Provides information for UI display, alerting, and manual resolution.
    """

    severity: str = "medium"  # none, low, medium, high, critical
    category: str = ""  # e.g., "Data Types", "Business Definitions", "Schema Design"
    message: str = ""  # Short human-readable message
    recommendation: str = ""  # Actionable recommendation

    # For UI display
    icon: str | None = None  # Suggested icon
    color: str | None = None  # Suggested color (for severity visualization)
```

---

## Aggregation Models

### ColumnEntropyProfile

Aggregated entropy for a single column across all dimensions.

```python
@dataclass
class ColumnEntropyProfile:
    """Aggregated entropy profile for a single column."""

    column_id: str
    column_name: str
    table_name: str

    # Per-layer scores (0.0-1.0)
    structural_entropy: float = 0.0
    semantic_entropy: float = 0.0
    value_entropy: float = 0.0
    computational_entropy: float = 0.0

    # Composite score
    composite_score: float = 0.0

    # Detailed breakdown by dimension
    dimension_scores: dict[str, float] = field(default_factory=dict)
    # e.g., {"structural.types.type_fidelity": 0.3, "semantic.units.unit_declared": 0.8}

    # High-entropy dimensions (score > 0.5)
    high_entropy_dimensions: list[str] = field(default_factory=list)

    # Resolution hints (top 3 by priority)
    top_resolution_hints: list[ResolutionOption] = field(default_factory=list)

    # Compound risks affecting this column
    compound_risks: list["CompoundRisk"] = field(default_factory=list)

    # Readiness classification
    readiness: str = "investigate"  # ready, investigate, blocked

    def calculate_composite(self, weights: dict[str, float] | None = None) -> float:
        """Calculate composite score from layer scores.

        Default weights:
        - structural: 0.25
        - semantic: 0.30
        - value: 0.30
        - computational: 0.15
        """
        weights = weights or {
            "structural": 0.25,
            "semantic": 0.30,
            "value": 0.30,
            "computational": 0.15,
        }
        self.composite_score = (
            self.structural_entropy * weights["structural"]
            + self.semantic_entropy * weights["semantic"]
            + self.value_entropy * weights["value"]
            + self.computational_entropy * weights["computational"]
        )
        return self.composite_score
```

### TableEntropyProfile

Aggregated entropy for a table.

```python
@dataclass
class TableEntropyProfile:
    """Aggregated entropy profile for a table."""

    table_id: str
    table_name: str

    # Column profiles
    column_profiles: list[ColumnEntropyProfile] = field(default_factory=list)

    # Table-level scores (averages across columns)
    avg_structural_entropy: float = 0.0
    avg_semantic_entropy: float = 0.0
    avg_value_entropy: float = 0.0
    avg_computational_entropy: float = 0.0
    avg_composite_score: float = 0.0

    # Worst scores (max across columns)
    max_structural_entropy: float = 0.0
    max_semantic_entropy: float = 0.0
    max_value_entropy: float = 0.0
    max_computational_entropy: float = 0.0
    max_composite_score: float = 0.0

    # High-entropy columns (composite > 0.5)
    high_entropy_columns: list[str] = field(default_factory=list)

    # Blocked columns (composite > 0.8)
    blocked_columns: list[str] = field(default_factory=list)

    # Table-level compound risks
    compound_risks: list["CompoundRisk"] = field(default_factory=list)

    # Readiness classification
    readiness: str = "investigate"  # ready, investigate, blocked
    readiness_blockers: list[str] = field(default_factory=list)
```

### RelationshipEntropyProfile

Entropy for a relationship between tables.

```python
@dataclass
class RelationshipEntropyProfile:
    """Entropy profile for a relationship between tables."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str

    # Relationship-specific entropy
    cardinality_entropy: float = 0.0  # Uncertainty in cardinality
    join_path_entropy: float = 0.0  # Ambiguity in join path
    referential_integrity_entropy: float = 0.0  # Orphan ratio
    semantic_clarity_entropy: float = 0.0  # Is relationship meaning clear?

    # Composite
    composite_score: float = 0.0

    # Is this join deterministic?
    is_deterministic: bool = True

    # Warning for graph agent
    join_warning: str | None = None
```

---

## Compound Risk Models

### CompoundRisk

A dangerous combination of high-entropy dimensions.

```python
@dataclass
class CompoundRisk:
    """A dangerous combination of entropy dimensions.

    Some dimension pairs create multiplicative risk that exceeds
    the sum of individual scores.
    """

    risk_id: str  # UUID
    target: str  # column, table, or relationship

    # Which dimensions are combined
    dimensions: list[str]  # e.g., ["semantic.units", "computational.aggregations"]
    dimension_scores: dict[str, float]  # Individual scores for each dimension

    # Risk assessment
    risk_level: str  # "critical", "high", "medium"
    impact: str  # Description of what can go wrong
    multiplier: float  # Risk multiplier (1.0-3.0)

    # Combined score after multiplier
    combined_score: float = 0.0

    # Mitigation
    mitigation_options: list[ResolutionOption] = field(default_factory=list)

    @classmethod
    def from_scores(
        cls,
        target: str,
        dimensions: list[str],
        scores: dict[str, float],
        risk_level: str,
        impact: str,
        multiplier: float,
    ) -> "CompoundRisk":
        """Create a CompoundRisk from dimension scores."""
        from uuid import uuid4

        dimension_scores = {d: scores.get(d, 0.0) for d in dimensions}
        avg_score = sum(dimension_scores.values()) / len(dimension_scores)
        combined = min(1.0, avg_score * multiplier)

        return cls(
            risk_id=str(uuid4()),
            target=target,
            dimensions=dimensions,
            dimension_scores=dimension_scores,
            risk_level=risk_level,
            impact=impact,
            multiplier=multiplier,
            combined_score=combined,
        )
```

### CompoundRiskDefinition

Configuration for detecting compound risks.

```python
@dataclass
class CompoundRiskDefinition:
    """Definition of a compound risk pattern.

    Loaded from config/entropy/compound_risks.yaml.
    """

    risk_type: str  # e.g., "units_aggregations"
    dimensions: list[str]  # Dimensions that must both be high
    threshold: float  # Score threshold for each dimension (default 0.5)
    risk_level: str  # Resulting risk level
    impact_template: str  # Template for impact description
    multiplier: float  # Risk multiplier
```

---

## Resolution Cascade Models

### ResolutionCascade

A single resolution that improves multiple entropy dimensions.

```python
@dataclass
class ResolutionCascade:
    """A single resolution that improves multiple entropy dimensions.

    Used to prioritize resolutions that have broad impact.
    """

    cascade_id: str  # UUID
    action: str  # e.g., "rename_column", "add_fk_constraint"
    parameters: dict[str, Any]

    # What gets fixed
    affected_targets: list[str]  # Columns/tables affected
    entropy_reductions: dict[str, float]  # dimension -> expected reduction

    # Aggregated impact
    total_reduction: float = 0.0  # Sum of all reductions
    dimensions_improved: int = 0  # Count of dimensions improved

    # Effort and priority
    effort: str = "medium"  # low, medium, high
    priority_score: float = 0.0  # total_reduction / effort_factor

    def calculate_priority(self) -> float:
        """Calculate priority score."""
        effort_factor = {"low": 1.0, "medium": 2.0, "high": 4.0}.get(self.effort, 2.0)
        self.total_reduction = sum(self.entropy_reductions.values())
        self.dimensions_improved = len(self.entropy_reductions)
        self.priority_score = self.total_reduction / effort_factor
        return self.priority_score
```

---

## Context Models (for Graph Agent)

### EntropyContext

The complete entropy context passed to the graph agent.

```python
@dataclass
class EntropyContext:
    """Complete entropy context for graph agent consumption.

    This is the primary interface between the entropy layer
    and the graph execution context.
    """

    # Per-column entropy
    column_profiles: dict[str, ColumnEntropyProfile] = field(default_factory=dict)
    # Key: "{table_name}.{column_name}"

    # Per-table entropy
    table_profiles: dict[str, TableEntropyProfile] = field(default_factory=dict)
    # Key: table_name

    # Per-relationship entropy
    relationship_profiles: dict[str, RelationshipEntropyProfile] = field(
        default_factory=dict
    )
    # Key: "{from_table}.{from_col}->{to_table}.{to_col}"

    # Global compound risks
    compound_risks: list[CompoundRisk] = field(default_factory=list)

    # Top resolution hints (across all targets)
    top_resolution_hints: list[ResolutionCascade] = field(default_factory=list)

    # Summary statistics
    total_entropy_objects: int = 0
    high_entropy_count: int = 0  # score > 0.5
    critical_entropy_count: int = 0  # score > 0.8
    compound_risk_count: int = 0

    # Readiness assessment
    overall_readiness: str = "investigate"  # ready, investigate, blocked
    readiness_blockers: list[str] = field(default_factory=list)

    # Contract compliance (if evaluated)
    contract_compliance: dict[str, bool] = field(default_factory=dict)
    # Key: use_case name, Value: whether compliant

    # Computed timestamp
    computed_at: datetime = field(default_factory=datetime.utcnow)

    def get_column_entropy(self, table: str, column: str) -> ColumnEntropyProfile | None:
        """Get entropy profile for a specific column."""
        key = f"{table}.{column}"
        return self.column_profiles.get(key)

    def get_high_entropy_columns(self, threshold: float = 0.5) -> list[str]:
        """Get list of columns with entropy above threshold."""
        return [
            key
            for key, profile in self.column_profiles.items()
            if profile.composite_score >= threshold
        ]

    def has_critical_risks(self) -> bool:
        """Check if any critical compound risks exist."""
        return any(r.risk_level == "critical" for r in self.compound_risks)
```

---

## Database Models

### EntropyObjectRecord (SQLAlchemy)

```python
from sqlalchemy import Column, String, Float, DateTime, JSON, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class EntropyObjectRecord(Base):
    """SQLAlchemy model for storing entropy objects."""

    __tablename__ = "entropy_objects"

    object_id = Column(String, primary_key=True)
    layer = Column(String, nullable=False, index=True)
    dimension = Column(String, nullable=False, index=True)
    sub_dimension = Column(String, nullable=False)
    target = Column(String, nullable=False, index=True)

    score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)

    # JSON fields for flexibility
    evidence = Column(JSON, default=list)
    resolution_options = Column(JSON, default=list)
    llm_context = Column(JSON, default=dict)
    human_context = Column(JSON, default=dict)

    computed_at = Column(DateTime, nullable=False)
    source_analysis_ids = Column(JSON, default=list)
    detector_id = Column(String)

    __table_args__ = (
        Index("ix_entropy_layer_dimension", "layer", "dimension"),
        Index("ix_entropy_target_dimension", "target", "dimension"),
    )


class CompoundRiskRecord(Base):
    """SQLAlchemy model for storing compound risks."""

    __tablename__ = "entropy_compound_risks"

    risk_id = Column(String, primary_key=True)
    target = Column(String, nullable=False, index=True)
    dimensions = Column(JSON, nullable=False)
    dimension_scores = Column(JSON, nullable=False)
    risk_level = Column(String, nullable=False, index=True)
    impact = Column(String)
    multiplier = Column(Float, nullable=False)
    combined_score = Column(Float, nullable=False)
    mitigation_options = Column(JSON, default=list)
    computed_at = Column(DateTime, nullable=False)
```

---

## Evidence Schemas

Evidence objects are dimension-specific. Here are examples for common dimensions:

### Type Fidelity Evidence

```python
{
    "type": "type_mismatch",
    "declared_type": "VARCHAR(50)",
    "detected_content_type": "date",
    "format_variance": [
        {"format": "YYYY-MM-DD", "frequency": 0.65},
        {"format": "DD.MM.YYYY", "frequency": 0.25},
        {"format": "Mon DD, YYYY", "frequency": 0.08},
        {"format": "empty_string", "frequency": 0.02}
    ],
    "parse_success_rate": 0.72,
    "ambiguous_values": [
        {"value": "01.02.2024", "interpretations": ["Jan 2, 2024", "Feb 1, 2024"]}
    ]
}
```

### Null Semantics Evidence

```python
{
    "type": "null_analysis",
    "null_count": 4521,
    "total_count": 15000,
    "null_rate": 0.30,
    "null_representations": [
        {"value": None, "count": 4200},
        {"value": "", "count": 250},
        {"value": "N/A", "count": 71}
    ],
    "possible_meanings": [
        {"meaning": "not_applicable", "evidence": "Column is optional per schema"},
        {"meaning": "unknown", "evidence": "Some rows have this filled, others don't"}
    ]
}
```

### Join Path Evidence

```python
{
    "type": "multiple_paths_detected",
    "from_table": "orders",
    "to_table": "products",
    "paths": [
        {
            "path": ["orders", "order_items", "products"],
            "semantics": "products actually purchased",
            "join_keys": ["orders.id = order_items.order_id", "order_items.product_id = products.id"]
        },
        {
            "path": ["orders", "recommendations", "products"],
            "semantics": "products recommended at order time",
            "join_keys": ["orders.id = recommendations.order_id", "recommendations.product_id = products.id"]
        }
    ],
    "has_canonical_declaration": False
}
```

### Business Meaning Evidence

```python
{
    "type": "definition_analysis",
    "has_description": True,
    "description_length": 15,
    "description_text": "Customer margin",
    "quality_issues": ["too_brief", "ambiguous_term"],
    "term_ambiguity": {
        "term": "margin",
        "possible_meanings": [
            {"meaning": "gross_margin", "formula": "(revenue - cogs) / revenue"},
            {"meaning": "net_margin", "formula": "(revenue - all_costs) / revenue"},
            {"meaning": "contribution_margin", "formula": "(revenue - variable_costs) / revenue"}
        ]
    }
}
```

---

## Usage Example

```python
from dataraum_context.entropy.models import (
    EntropyObject,
    ResolutionOption,
    LLMContext,
    HumanContext,
)
from datetime import datetime
from uuid import uuid4


# Create an entropy object for type fidelity
entropy = EntropyObject(
    object_id=str(uuid4()),
    layer="structural",
    dimension="types",
    sub_dimension="type_fidelity",
    target="column:events.event_date",
    score=0.72,
    confidence=0.95,
    evidence=[
        {
            "type": "type_mismatch",
            "declared_type": "VARCHAR(50)",
            "detected_content_type": "date",
            "parse_success_rate": 0.28,
        }
    ],
    resolution_options=[
        ResolutionOption(
            action="convert_to_proper_type",
            parameters={"target_type": "DATE", "parsing_format": "auto_detect"},
            expected_entropy_reduction=0.65,
            effort="medium",
            description="Convert column to DATE type with format detection",
            cascade_dimensions=["value.patterns.format_consistency"],
        )
    ],
    llm_context=LLMContext(
        description="Column 'event_date' is stored as VARCHAR but contains date values in multiple formats.",
        query_impact="Date filtering and arithmetic will fail or produce incorrect results.",
        warning="Date comparisons as strings will sort incorrectly.",
    ),
    human_context=HumanContext(
        severity="high",
        category="Data Types",
        message="Date stored as string with multiple formats",
        recommendation="Convert to DATE type with explicit parsing rules",
    ),
    computed_at=datetime.utcnow(),
    detector_id="TypeFidelityDetector",
)
```
