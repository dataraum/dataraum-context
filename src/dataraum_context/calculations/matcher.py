"""LLM-powered schema matcher.

Maps concrete database columns to abstract calculation fields using LLM
for semantic understanding.

Usage:
    from dataraum_context.calculations.matcher import SchemaMatcherLLM

    matcher = SchemaMatcherLLM(llm_provider)
    mapping = await matcher.map_columns(
        columns=column_info_list,
        abstract_fields=abstract_fields,
    )
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from dataraum_context.calculations.graphs import AbstractField, GraphLoader
from dataraum_context.calculations.mapping import (
    AggregationDefinition,
    ColumnMapping,
    DatasetSchemaMapping,
    SchemaMapping,
)
from dataraum_context.core.models.base import Result
from dataraum_context.llm.providers.base import LLMProvider, LLMRequest

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Information about a column to be mapped.

    Includes both technical and semantic metadata.
    """

    table: str
    column: str
    data_type: str
    # Semantic info (from SemanticAnnotation if available)
    semantic_role: str | None = None  # measure, dimension, identifier, etc.
    entity_type: str | None = None  # currency, date, percentage, etc.
    business_name: str | None = None  # Human-readable name
    description: str | None = None
    # Statistical info
    null_ratio: float | None = None
    sample_values: list[Any] | None = None


SYSTEM_PROMPT = """You are a financial data analyst expert specializing in mapping database schemas
to standardized financial calculation fields.

Your task is to analyze column metadata and determine which abstract financial
fields each column contributes to. You understand:

- Financial statement structure (balance sheet, income statement, cash flow)
- Common accounting terminology in multiple languages (English, German, etc.)
- Data aggregation patterns (SUM for flows, END_OF_PERIOD for stocks)
- Sign conventions (assets positive, liabilities positive, expenses as positive costs)

Return your mappings as a JSON array."""

USER_PROMPT_TEMPLATE = """## Task: Map Columns to Abstract Financial Fields

I need you to analyze the following database columns and determine which
abstract financial fields they map to.

### Available Abstract Fields

{abstract_fields_section}

### Columns to Map

{columns_section}

### Instructions

For EACH column, determine:
1. Which abstract field it maps to (use "unmapped" if none match)
2. The aggregation method: "sum" for flows, "end_of_period" for balances, "average" for rates
3. Any filter conditions needed (e.g., only certain transaction types)
4. Your confidence (0.0-1.0) and reasoning

Return a JSON array with one object per column:
```json
[
  {{
    "table": "transactions",
    "column": "amount",
    "abstract_field": "revenue",
    "aggregation": "sum",
    "filter_condition": "type = 'sale'",
    "sign_adjustment": 1,
    "confidence": 0.9,
    "reasoning": "Column name and sample values suggest sales revenue"
  }}
]
```

If a column doesn't map to any abstract field, use "abstract_field": "unmapped".

Return ONLY the JSON array, no other text."""


class SchemaMatcherLLM:
    """LLM-powered schema matcher.

    Uses an LLM to semantically match database columns to abstract
    calculation fields defined in calculation graphs.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        graph_loader: GraphLoader | None = None,
    ):
        """Initialize matcher.

        Args:
            llm_provider: LLM provider for semantic matching
            graph_loader: Optional pre-loaded graph loader
        """
        self.llm = llm_provider
        self.graph_loader = graph_loader or GraphLoader()

        # Load graphs if not already loaded
        if not self.graph_loader.graphs:
            self.graph_loader.load_all_graphs()

    async def map_columns(
        self,
        columns: list[ColumnInfo],
        dataset_id: str,
        dataset_name: str | None = None,
        abstract_fields: dict[str, AbstractField] | None = None,
    ) -> Result[DatasetSchemaMapping]:
        """Map columns to abstract fields using LLM.

        Args:
            columns: List of columns with metadata
            dataset_id: ID for the dataset
            dataset_name: Optional human-readable name
            abstract_fields: Optional override for abstract fields.
                            If not provided, uses all fields from loaded graphs.

        Returns:
            Result containing DatasetSchemaMapping or error
        """
        if not columns:
            return Result.fail("No columns provided for mapping")

        # Get abstract fields
        if abstract_fields is None:
            abstract_fields = self.graph_loader.get_all_abstract_fields()

        if not abstract_fields:
            return Result.fail("No abstract fields available for mapping")

        # Build prompt
        prompt = self._build_prompt(columns, abstract_fields)

        # Call LLM
        request = LLMRequest(
            prompt=prompt,
            max_tokens=4000,
            temperature=0.0,
            response_format="json",
        )

        result = await self.llm.complete(request)
        if not result.success or result.value is None:
            return Result.fail(f"LLM call failed: {result.error}")

        # Parse response
        try:
            mappings = self._parse_response(result.value.content, abstract_fields)
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return Result.fail(f"Failed to parse LLM response: {e}")

        # Build DatasetSchemaMapping
        dataset_mapping = DatasetSchemaMapping(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            mappings=mappings,
            mapping_source="llm",
        )

        return Result.ok(dataset_mapping)

    def _build_prompt(
        self,
        columns: list[ColumnInfo],
        abstract_fields: dict[str, AbstractField],
    ) -> str:
        """Build the LLM prompt."""
        # Format abstract fields
        fields_lines = []
        for field_id, field in abstract_fields.items():
            agg = "unknown"
            statement = "unknown"
            if field.source:
                agg = field.source.aggregation
                statement = field.source.statement

            fields_lines.append(
                f"- **{field_id}** ({statement})\n"
                f"  Description: {field.description}\n"
                f"  Aggregation: {agg}\n"
                f"  Required: {field.required}"
            )

        abstract_fields_section = "\n".join(fields_lines)

        # Format columns
        columns_lines = []
        for i, col in enumerate(columns, 1):
            parts = [f"{i}. **{col.table}.{col.column}**"]
            parts.append(f"   Type: {col.data_type}")

            if col.business_name:
                parts.append(f"   Business name: {col.business_name}")
            if col.semantic_role:
                parts.append(f"   Semantic role: {col.semantic_role}")
            if col.entity_type:
                parts.append(f"   Entity type: {col.entity_type}")
            if col.description:
                parts.append(f"   Description: {col.description}")
            if col.null_ratio is not None:
                parts.append(f"   Null ratio: {col.null_ratio:.1%}")
            if col.sample_values:
                samples_str = ", ".join(str(v) for v in col.sample_values[:5])
                parts.append(f"   Samples: {samples_str}")

            columns_lines.append("\n".join(parts))

        columns_section = "\n\n".join(columns_lines)

        return (
            SYSTEM_PROMPT
            + "\n\n"
            + USER_PROMPT_TEMPLATE.format(
                abstract_fields_section=abstract_fields_section,
                columns_section=columns_section,
            )
        )

    def _parse_response(
        self,
        response_content: str,
        abstract_fields: dict[str, AbstractField],
    ) -> dict[str, SchemaMapping]:
        """Parse LLM response into SchemaMapping objects."""
        # Extract JSON from response
        content = response_content.strip()

        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (```json and ```)
            content = "\n".join(lines[1:-1])

        mappings_data = json.loads(content)

        # Group by abstract field
        field_to_columns: dict[str, list[dict[str, Any]]] = {}
        for item in mappings_data:
            field_id = item.get("abstract_field", "unmapped")
            if field_id == "unmapped":
                continue

            if field_id not in field_to_columns:
                field_to_columns[field_id] = []
            field_to_columns[field_id].append(item)

        # Build SchemaMapping for each abstract field
        result: dict[str, SchemaMapping] = {}

        for field_id, column_items in field_to_columns.items():
            # Get field info
            field_def = abstract_fields.get(field_id)

            origin_mappings = []
            total_confidence = 0.0

            for item in column_items:
                origin_mappings.append(
                    ColumnMapping(
                        table=item["table"],
                        column=item["column"],
                        confidence=item.get("confidence", 0.5),
                        reasoning=item.get("reasoning"),
                        filter_condition=item.get("filter_condition"),
                        sign_adjustment=item.get("sign_adjustment", 1),
                    )
                )
                total_confidence += item.get("confidence", 0.5)

            # Determine aggregation
            agg_method = "sum"  # Default
            if column_items and column_items[0].get("aggregation"):
                agg_method = column_items[0]["aggregation"]
            elif field_def and field_def.source:
                agg_method = field_def.source.aggregation

            result[field_id] = SchemaMapping(
                abstract_field=field_id,
                description=field_def.description if field_def else None,
                origin_mappings=origin_mappings,
                aggregation=AggregationDefinition(method=agg_method),
                is_required=field_def.required if field_def else True,
                is_nullable=field_def.nullable if field_def else True,
                confidence=total_confidence / len(column_items) if column_items else 0.0,
                created_by="llm",
            )

        return result


def columns_from_profiles(
    profiles: list[Any],
    semantic_annotations: dict[str, Any] | None = None,
) -> list[ColumnInfo]:
    """Convert statistical profiles to ColumnInfo for mapping.

    Helper function to create ColumnInfo from existing profile data.

    Args:
        profiles: List of StatisticalProfile or similar objects
        semantic_annotations: Optional dict mapping column_id to semantic info

    Returns:
        List of ColumnInfo ready for mapping
    """
    columns = []
    semantic_annotations = semantic_annotations or {}

    for profile in profiles:
        # Get semantic info if available
        semantic = semantic_annotations.get(profile.column_id, {})

        columns.append(
            ColumnInfo(
                table=profile.table_name if hasattr(profile, "table_name") else "unknown",
                column=profile.column_name
                if hasattr(profile, "column_name")
                else str(profile.column_id),
                data_type=profile.data_type if hasattr(profile, "data_type") else "unknown",
                semantic_role=semantic.get("semantic_role"),
                entity_type=semantic.get("entity_type"),
                business_name=semantic.get("business_name"),
                description=semantic.get("description"),
                null_ratio=profile.null_ratio if hasattr(profile, "null_ratio") else None,
                sample_values=None,  # Would need to query for samples
            )
        )

    return columns
