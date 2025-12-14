"""Schema mapping: Abstract fields ↔ Concrete columns.

Maps abstract calculation fields (revenue, accounts_receivable) to concrete
columns in the actual dataset schema. This enables:

1. Understanding which columns feed into which calculations
2. Knowing what aggregations to apply (SUM, END_OF_PERIOD, etc.)
3. Downstream impact analysis for quality issues

The mapping has two parts:
- Origin column mapping: Which concrete columns provide data for an abstract field
- Aggregation definition: How to aggregate origin columns into the target field

Usage:
    from dataraum_context.quality.formatting.schema_mapping import (
        SchemaMapping,
        ColumnMapping,
        AggregationDefinition,
    )

    # Create mapping manually
    mapping = SchemaMapping(
        abstract_field="revenue",
        origin_mappings=[
            ColumnMapping(
                table="transactions",
                column="amount",
                filter_condition="type = 'sale'",
                confidence=0.95,
            )
        ],
        aggregation=AggregationDefinition(
            method="sum",
            group_by=["period"],
        ),
    )

    # Or load from stored mappings
    store = SchemaMappingStore(session)
    mappings = store.get_mappings_for_dataset(dataset_id)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ColumnMapping:
    """Mapping from a concrete column to an abstract field.

    Represents one source column that contributes to an abstract field.
    Multiple columns can map to the same abstract field (e.g., multiple
    revenue account columns all map to 'revenue').
    """

    table: str  # Origin table name
    column: str  # Origin column name
    confidence: float  # Mapping confidence (0.0 to 1.0)
    reasoning: str | None = None  # Why this mapping was chosen
    filter_condition: str | None = None  # Optional WHERE clause (e.g., "type = 'sale'")
    sign_adjustment: int = 1  # Multiply by -1 for contra accounts


@dataclass
class AggregationDefinition:
    """How to aggregate origin columns into the target field.

    Defines the transformation from raw origin data to the abstract field value.
    """

    method: str  # sum, end_of_period, average, count, min, max
    group_by: list[str] = field(default_factory=list)  # Columns to group by
    time_column: str | None = None  # For end_of_period: which column has dates
    period_type: str | None = None  # month, quarter, year
    additional_filters: str | None = None  # Extra WHERE conditions


@dataclass
class SchemaMapping:
    """Complete mapping from abstract field to concrete schema.

    Combines origin column mappings with aggregation logic.
    """

    abstract_field: str  # Target field ID (e.g., "revenue")
    description: str | None = None  # Human description of what this maps
    origin_mappings: list[ColumnMapping] = field(default_factory=list)
    aggregation: AggregationDefinition | None = None
    is_required: bool = True  # Is this field required for calculations?
    is_nullable: bool = False  # Can this field be NULL?

    # Metadata
    created_at: datetime | None = None
    created_by: str | None = None  # "llm", "manual", "inferred"
    confidence: float = 0.0  # Overall mapping confidence

    def get_origin_columns(self) -> list[str]:
        """Get list of origin column references (table.column)."""
        return [f"{m.table}.{m.column}" for m in self.origin_mappings]

    def get_origin_tables(self) -> list[str]:
        """Get unique origin tables."""
        return list({m.table for m in self.origin_mappings})

    def to_sql_expression(self) -> str | None:
        """Generate SQL expression for this mapping.

        Returns None if no aggregation defined.
        """
        if not self.aggregation or not self.origin_mappings:
            return None

        # Build column expressions with sign adjustments and filters
        col_exprs = []
        for mapping in self.origin_mappings:
            col_ref = f"{mapping.table}.{mapping.column}"
            if mapping.sign_adjustment == -1:
                col_ref = f"-{col_ref}"

            if mapping.filter_condition:
                col_ref = f"CASE WHEN {mapping.filter_condition} THEN {col_ref} ELSE 0 END"

            col_exprs.append(col_ref)

        # Combine based on aggregation method
        method = self.aggregation.method.upper()

        if len(col_exprs) == 1:
            inner = col_exprs[0]
        else:
            # Sum multiple columns
            inner = " + ".join(f"({expr})" for expr in col_exprs)

        if method == "SUM":
            return f"SUM({inner})"
        elif method == "END_OF_PERIOD":
            # For end_of_period, we need the last value
            time_col = self.aggregation.time_column or "date"
            return f"LAST_VALUE({inner}) OVER (ORDER BY {time_col})"
        elif method == "AVERAGE":
            return f"AVG({inner})"
        elif method == "COUNT":
            return f"COUNT({inner})"
        elif method == "MIN":
            return f"MIN({inner})"
        elif method == "MAX":
            return f"MAX({inner})"
        else:
            return f"{method}({inner})"


@dataclass
class DatasetSchemaMapping:
    """All schema mappings for a dataset.

    Contains mappings for all abstract fields that could be computed
    from the dataset's columns.
    """

    dataset_id: str
    dataset_name: str | None = None
    mappings: dict[str, SchemaMapping] = field(default_factory=dict)

    # Metadata
    created_at: datetime | None = None
    last_updated: datetime | None = None
    mapping_source: str | None = None  # "llm", "manual", "imported"

    def get_mapping(self, abstract_field: str) -> SchemaMapping | None:
        """Get mapping for a specific abstract field."""
        return self.mappings.get(abstract_field)

    def get_all_origin_columns(self) -> set[str]:
        """Get all origin columns used across all mappings."""
        columns: set[str] = set()
        for mapping in self.mappings.values():
            columns.update(mapping.get_origin_columns())
        return columns

    def get_mappings_for_column(self, table: str, column: str) -> list[SchemaMapping]:
        """Find all abstract fields that use a specific column.

        Useful for downstream impact analysis.
        """
        col_ref = f"{table}.{column}"
        return [m for m in self.mappings.values() if col_ref in m.get_origin_columns()]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "mapping_source": self.mapping_source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "mappings": {
                field_id: {
                    "abstract_field": m.abstract_field,
                    "description": m.description,
                    "is_required": m.is_required,
                    "is_nullable": m.is_nullable,
                    "confidence": m.confidence,
                    "origin_columns": [
                        {
                            "table": cm.table,
                            "column": cm.column,
                            "confidence": cm.confidence,
                            "reasoning": cm.reasoning,
                            "filter_condition": cm.filter_condition,
                            "sign_adjustment": cm.sign_adjustment,
                        }
                        for cm in m.origin_mappings
                    ],
                    "aggregation": (
                        {
                            "method": m.aggregation.method,
                            "group_by": m.aggregation.group_by,
                            "time_column": m.aggregation.time_column,
                            "period_type": m.aggregation.period_type,
                        }
                        if m.aggregation
                        else None
                    ),
                }
                for field_id, m in self.mappings.items()
            },
        }


def get_downstream_impact(
    column: str,
    table: str,
    dataset_mapping: DatasetSchemaMapping,
) -> dict[str, Any]:
    """Analyze downstream impact of quality issues in a column.

    Args:
        column: Column name
        table: Table name
        dataset_mapping: The schema mapping for the dataset

    Returns:
        Impact analysis including affected calculations and severity
    """
    affected_mappings = dataset_mapping.get_mappings_for_column(table, column)

    if not affected_mappings:
        return {
            "column": f"{table}.{column}",
            "has_downstream_impact": False,
            "affected_fields": [],
            "affected_calculations": [],
            "impact_severity": "none",
        }

    affected_fields = []
    for mapping in affected_mappings:
        affected_fields.append(
            {
                "abstract_field": mapping.abstract_field,
                "is_required": mapping.is_required,
                "is_nullable": mapping.is_nullable,
                "aggregation": mapping.aggregation.method if mapping.aggregation else None,
            }
        )

    # Determine impact severity
    has_required = any(m.is_required for m in affected_mappings)
    has_non_nullable = any(not m.is_nullable for m in affected_mappings)

    if has_required and has_non_nullable:
        severity = "critical"
    elif has_required:
        severity = "high"
    elif has_non_nullable:
        severity = "moderate"
    else:
        severity = "low"

    return {
        "column": f"{table}.{column}",
        "has_downstream_impact": True,
        "affected_fields": affected_fields,
        "num_affected": len(affected_fields),
        "impact_severity": severity,
        "explanation": _generate_impact_explanation(affected_fields, severity),
    }


def _generate_impact_explanation(affected_fields: list[dict[str, Any]], severity: str) -> str:
    """Generate human-readable impact explanation."""
    field_names = [f["abstract_field"] for f in affected_fields]

    if severity == "critical":
        return (
            f"Quality issues in this column will directly affect {len(field_names)} "
            f"required, non-nullable calculation field(s): {', '.join(field_names)}. "
            "Missing or invalid values will cause calculation failures."
        )
    elif severity == "high":
        return (
            f"This column feeds into {len(field_names)} required field(s): "
            f"{', '.join(field_names)}. Quality issues may significantly "
            "impact calculation accuracy."
        )
    elif severity == "moderate":
        return (
            f"This column contributes to {len(field_names)} calculation field(s): "
            f"{', '.join(field_names)}. Quality issues may affect results."
        )
    else:
        return (
            f"This column is used by {len(field_names)} optional field(s): "
            f"{', '.join(field_names)}. Impact is limited."
        )
