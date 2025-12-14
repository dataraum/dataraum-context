"""Calculation graphs and schema mapping.

Provides tools for understanding financial calculations and mapping
concrete database columns to abstract calculation fields.

Usage:
    from dataraum_context.calculations import (
        GraphLoader,
        SchemaMatcherLLM,
        get_downstream_impact,
    )

    # Load calculation graphs
    loader = GraphLoader()
    loader.load_all_graphs()

    # Map columns to abstract fields
    matcher = SchemaMatcherLLM(llm_provider)
    mapping = await matcher.map_columns(columns, dataset_id="my_dataset")

    # Analyze downstream impact
    impact = get_downstream_impact("amount", "transactions", mapping)
"""

from dataraum_context.calculations.graphs import (
    AbstractField,
    CalculationGraph,
    GraphLoader,
)
from dataraum_context.calculations.mapping import (
    AggregationDefinition,
    ColumnMapping,
    DatasetSchemaMapping,
    SchemaMapping,
    get_downstream_impact,
)
from dataraum_context.calculations.matcher import (
    ColumnInfo,
    SchemaMatcherLLM,
    columns_from_profiles,
)

__all__ = [
    # Graph loading
    "GraphLoader",
    "CalculationGraph",
    "AbstractField",
    # Schema mapping
    "SchemaMapping",
    "ColumnMapping",
    "AggregationDefinition",
    "DatasetSchemaMapping",
    "get_downstream_impact",
    # LLM matcher
    "SchemaMatcherLLM",
    "ColumnInfo",
    "columns_from_profiles",
]
