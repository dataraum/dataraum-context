"""Calculation graph loader for downstream impact analysis.

Loads calculation graphs from YAML files to understand which abstract fields
(revenue, accounts_receivable, etc.) depend on which origin columns.

This enables:
1. Understanding downstream impact of quality issues
2. Prioritizing which columns need strictest quality controls
3. Explaining why a quality issue matters in business terms

Usage:
    from dataraum_context.quality.formatting.calculation_graphs import (
        GraphLoader,
        CalculationGraph,
        AbstractField,
    )

    loader = GraphLoader()
    loader.load_all_graphs()

    # Get all abstract fields across all graphs
    fields = loader.get_all_abstract_fields()

    # Find which calculations depend on a field
    dependents = loader.get_dependent_calculations("revenue")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FieldSource:
    """Source definition for an abstract field.

    Describes where the data comes from and how to aggregate it.
    """

    statement: str  # balance_sheet, income_statement, etc.
    standard_field: str  # Abstract field name
    aggregation: str  # sum, end_of_period, average, etc.
    period: str | None = None  # current, prior, ytd, etc.


@dataclass
class FieldValidation:
    """Validation rule for a field."""

    check: str  # SQL-like condition
    severity: str  # error, warning
    message: str


@dataclass
class AbstractField:
    """Abstract field definition from a calculation graph.

    Represents a standardized field like 'revenue' or 'accounts_receivable'
    that can be mapped to concrete columns in a dataset.
    """

    field_id: str
    description: str
    level: int  # Dependency level (1=source, 2=param, 3=derived, etc.)
    field_type: str  # direct, derived, parameter
    source: FieldSource | None
    required: bool
    nullable: bool
    validations: list[FieldValidation]
    calculation: dict[str, Any] | None = None  # For derived fields
    default_value: Any = None
    accounting_notes: str | None = None


@dataclass
class CalculationGraph:
    """Complete calculation graph loaded from YAML.

    Represents a financial calculation like DSO, Cash Runway, or OCF
    with all its dependencies and validation rules.
    """

    graph_id: str
    version: str
    category: str
    complexity: str
    description: str
    formula_human: str
    formula_math: str
    output_metric: str
    output_unit: str
    fields: dict[str, AbstractField]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_source_fields(self) -> list[AbstractField]:
        """Get fields that come directly from data (level 1)."""
        return [f for f in self.fields.values() if f.level == 1]

    def get_required_fields(self) -> list[AbstractField]:
        """Get all required fields."""
        return [f for f in self.fields.values() if f.required]

    def get_field_ids(self) -> list[str]:
        """Get all field IDs in this graph."""
        return list(self.fields.keys())


class GraphLoader:
    """Load calculation graphs from YAML files.

    Graphs define abstract fields and their dependencies for
    financial calculations. This loader extracts the field
    definitions needed for schema mapping.
    """

    def __init__(self, graphs_dir: Path | None = None):
        """Initialize loader.

        Args:
            graphs_dir: Directory containing *_graph.yaml files.
                        Defaults to config/calculation_graphs/
        """
        if graphs_dir is None:
            # Default to config directory (4 levels up from calculations/)
            graphs_dir = (
                Path(__file__).parent.parent.parent.parent / "config" / "calculation_graphs"
            )
        self.graphs_dir = graphs_dir
        self.graphs: dict[str, CalculationGraph] = {}

    def load_all_graphs(self) -> dict[str, CalculationGraph]:
        """Load all calculation graphs from the directory.

        Returns:
            Dict mapping graph_id to CalculationGraph
        """
        if not self.graphs_dir.exists():
            return {}

        for yaml_file in self.graphs_dir.glob("*_graph.yaml"):
            try:
                graph = self.load_graph(yaml_file)
                self.graphs[graph.graph_id] = graph
            except Exception:
                # Skip files that can't be loaded
                continue

        return self.graphs

    def load_graph(self, yaml_path: Path) -> CalculationGraph:
        """Load a single calculation graph from YAML.

        Args:
            yaml_path: Path to the YAML file

        Returns:
            CalculationGraph instance
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Parse fields from dependencies
        fields: dict[str, AbstractField] = {}
        for field_id, field_data in data.get("dependencies", {}).items():
            fields[field_id] = self._parse_field(field_id, field_data)

        # Extract formula info
        formula = data.get("formula", {})
        output = data.get("output", {})

        return CalculationGraph(
            graph_id=data["graph_id"],
            version=data.get("version", "1.0"),
            category=data.get("category", "unknown"),
            complexity=data.get("complexity", "unknown"),
            description=output.get("description", ""),
            formula_human=formula.get("human_readable", ""),
            formula_math=formula.get("mathematical", ""),
            output_metric=output.get("metric_id", data["graph_id"]),
            output_unit=output.get("unit", ""),
            fields=fields,
            metadata=data.get("metadata", {}),
        )

    def _parse_field(self, field_id: str, data: dict[str, Any]) -> AbstractField:
        """Parse a single field definition."""
        # Parse source
        source = None
        source_data = data.get("source")
        if source_data:
            source = FieldSource(
                statement=source_data.get("statement", ""),
                standard_field=source_data.get("standard_field", field_id),
                aggregation=source_data.get("aggregation", "sum"),
                period=source_data.get("period"),
            )

        # Parse validations
        validations = []
        for val_data in data.get("validation", []):
            validations.append(
                FieldValidation(
                    check=val_data.get("check", ""),
                    severity=val_data.get("severity", "warning"),
                    message=val_data.get("message", ""),
                )
            )

        return AbstractField(
            field_id=field_id,
            description=data.get("description", ""),
            level=data.get("level", 1),
            field_type=data.get("type", "direct"),
            source=source,
            required=data.get("required", True),
            nullable=data.get("nullable", True),
            validations=validations,
            calculation=data.get("calculation"),
            default_value=data.get("default_value") or data.get("default_if_missing"),
            accounting_notes=data.get("accounting_notes"),
        )

    def get_graph(self, graph_id: str) -> CalculationGraph | None:
        """Get a specific graph by ID."""
        return self.graphs.get(graph_id)

    def get_all_abstract_fields(self) -> dict[str, AbstractField]:
        """Get all unique abstract fields across all graphs.

        Returns:
            Dict mapping field_id to AbstractField.
            If same field appears in multiple graphs, uses first occurrence.
        """
        all_fields: dict[str, AbstractField] = {}
        for graph in self.graphs.values():
            for field_id, field_def in graph.fields.items():
                if field_id not in all_fields:
                    all_fields[field_id] = field_def
        return all_fields

    def get_dependent_calculations(self, field_id: str) -> list[CalculationGraph]:
        """Find all calculations that depend on a given field.

        Args:
            field_id: The abstract field ID (e.g., "revenue")

        Returns:
            List of CalculationGraphs that use this field
        """
        dependents = []
        for graph in self.graphs.values():
            if field_id in graph.fields:
                dependents.append(graph)
        return dependents

    def get_field_usage_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary of how each field is used across calculations.

        Returns:
            Dict mapping field_id to usage info:
            {
                "revenue": {
                    "used_in": ["dso", "cash_runway"],
                    "required_in": ["dso"],
                    "aggregations": ["sum"],
                    "statements": ["income_statement"]
                }
            }
        """
        usage: dict[str, dict[str, Any]] = {}

        for graph in self.graphs.values():
            for field_id, field_def in graph.fields.items():
                if field_id not in usage:
                    usage[field_id] = {
                        "used_in": [],
                        "required_in": [],
                        "aggregations": set(),
                        "statements": set(),
                    }

                usage[field_id]["used_in"].append(graph.graph_id)

                if field_def.required:
                    usage[field_id]["required_in"].append(graph.graph_id)

                if field_def.source:
                    usage[field_id]["aggregations"].add(field_def.source.aggregation)
                    usage[field_id]["statements"].add(field_def.source.statement)

        # Convert sets to lists for JSON serialization
        for field_id in usage:
            usage[field_id]["aggregations"] = list(usage[field_id]["aggregations"])
            usage[field_id]["statements"] = list(usage[field_id]["statements"])

        return usage
