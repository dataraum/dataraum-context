"""Tests for entropy interpretation module.

Tests the LLM-powered entropy interpretation feature including
models and prompt context building.
"""

import json

import pytest
from pydantic import ValidationError

from dataraum.entropy.interpretation import (
    Assumption,
    EntropyInterpretation,
    EntropyInterpretationOutput,
    InterpretationInput,
    ResolutionAction,
    ResolutionActionOutput,
    TableEntropyInterpretationOutput,
    TableInterpretationInput,
)


class TestAssumption:
    """Tests for Assumption model."""

    def test_create_assumption(self):
        """Test creating an assumption."""
        assumption = Assumption(
            dimension="semantic.units",
            assumption_text="Assuming currency is EUR",
            confidence="medium",
            impact="Results may be incorrect if currency is not EUR",
            basis="inferred",
        )

        assert assumption.dimension == "semantic.units"
        assert assumption.assumption_text == "Assuming currency is EUR"
        assert assumption.confidence == "medium"
        assert assumption.impact == "Results may be incorrect if currency is not EUR"
        assert assumption.basis == "inferred"

    def test_assumption_default_basis(self):
        """Test assumption has default basis."""
        assumption = Assumption(
            dimension="value.nulls",
            assumption_text="Nulls will be excluded",
            confidence="high",
            impact="Missing data may affect totals",
        )

        assert assumption.basis == "inferred"


class TestResolutionAction:
    """Tests for ResolutionAction model."""

    def test_create_resolution_action(self):
        """Test creating a resolution action."""
        action = ResolutionAction(
            action="add_unit_declaration",
            description="Declare the currency unit for this column",
            effort="low",
            expected_impact="Reduces semantic.units entropy",
            parameters={"column": "amount", "suggested_unit": "EUR"},
        )

        assert action.action == "add_unit_declaration"
        assert action.description == "Declare the currency unit for this column"
        assert action.effort == "low"
        assert action.expected_impact == "Reduces semantic.units entropy"
        assert action.parameters == {"column": "amount", "suggested_unit": "EUR"}

    def test_resolution_action_default_parameters(self):
        """Test resolution action has default empty parameters."""
        action = ResolutionAction(
            action="investigate",
            description="Review data quality",
            effort="high",
            expected_impact="Understanding of issues",
        )

        assert action.parameters == {}


class TestEntropyInterpretation:
    """Tests for EntropyInterpretation model."""

    def test_create_interpretation(self):
        """Test creating an interpretation."""
        assumptions = [
            Assumption(
                dimension="value.nulls",
                assumption_text="Nulls excluded from aggregations",
                confidence="high",
                impact="May undercount totals",
            )
        ]
        resolution_actions = [
            ResolutionAction(
                action="document_null_semantics",
                description="Document what null values represent",
                effort="low",
                expected_impact="Clearer data semantics",
            )
        ]

        interpretation = EntropyInterpretation(
            column_name="amount",
            table_name="orders",
            assumptions=assumptions,
            resolution_actions=resolution_actions,
            explanation="Column has moderate entropy due to null values.",
        )

        assert interpretation.column_name == "amount"
        assert interpretation.table_name == "orders"
        assert len(interpretation.assumptions) == 1
        assert len(interpretation.resolution_actions) == 1
        assert interpretation.explanation == "Column has moderate entropy due to null values."


class TestInterpretationInput:
    """Tests for InterpretationInput model."""

    def test_create_input(self):
        """Test creating interpretation input with minimal fields."""
        input_data = InterpretationInput(
            table_name="orders",
            column_name="amount",
            detected_type="DECIMAL",
            business_description="Order total amount",
        )

        assert input_data.table_name == "orders"
        assert input_data.column_name == "amount"
        assert input_data.detected_type == "DECIMAL"
        assert input_data.business_description == "Order total amount"
        assert input_data.network_analysis is None
        assert input_data.quality_grade is None
        assert input_data.quality_findings is None

    def test_create_input_with_quality_context(self):
        """Test creating input with quality context."""
        input_data = InterpretationInput(
            table_name="orders",
            column_name="amount",
            detected_type="DECIMAL",
            business_description="Order total amount",
            quality_grade="D",
            quality_findings=["15% null values", "Outliers detected"],
        )

        assert input_data.quality_grade == "D"
        assert input_data.quality_findings == ["15% null values", "Outliers detected"]

    def test_create_input_with_network_analysis(self):
        """Test creating input with network analysis."""
        network_analysis = {
            "readiness": "investigate",
            "intents": {
                "query_intent": {"p_high": 0.10, "readiness": "ready"},
                "aggregation_intent": {"p_high": 0.65, "readiness": "blocked"},
            },
            "high_impact_nodes": [
                {"node": "unit_declaration", "state": "high", "impact_delta": 0.25},
            ],
            "top_fix": {"node": "unit_declaration", "impact_delta": 0.25},
        }
        input_data = InterpretationInput(
            table_name="orders",
            column_name="amount",
            detected_type="DECIMAL",
            business_description="Total order amount in USD",
            network_analysis=network_analysis,
        )

        assert input_data.network_analysis is not None
        assert input_data.network_analysis["readiness"] == "investigate"
        assert input_data.network_analysis["intents"]["aggregation_intent"]["readiness"] == "blocked"


class TestInterpretationInputFields:
    """Tests for InterpretationInput field access and serialization."""

    def test_input_fields_accessible_for_batch_prompt(self):
        """Test that InterpretationInput fields can be serialized for batch prompt."""
        network_analysis = {
            "readiness": "investigate",
            "intents": {
                "query_intent": {"p_high": 0.10, "readiness": "ready"},
                "aggregation_intent": {"p_high": 0.65, "readiness": "blocked"},
            },
            "high_impact_nodes": [
                {"node": "unit_declaration", "state": "high", "impact_delta": 0.25},
            ],
            "top_fix": {"node": "unit_declaration", "impact_delta": 0.25},
        }
        input_data = InterpretationInput(
            table_name="orders",
            column_name="amount",
            detected_type="DECIMAL",
            business_description="Total order amount in USD",
            network_analysis=network_analysis,
        )

        # Build the batch column data structure (as done in interpret_batch)
        column_data = {
            "key": f"{input_data.table_name}.{input_data.column_name}",
            "table_name": input_data.table_name,
            "column_name": input_data.column_name,
            "detected_type": input_data.detected_type,
            "business_description": input_data.business_description or "Not documented",
            "network_analysis": input_data.network_analysis,
        }

        assert column_data["key"] == "orders.amount"
        assert column_data["table_name"] == "orders"
        assert column_data["column_name"] == "amount"
        assert column_data["detected_type"] == "DECIMAL"
        assert column_data["business_description"] == "Total order amount in USD"
        assert column_data["network_analysis"]["readiness"] == "investigate"
        assert column_data["network_analysis"]["intents"]["aggregation_intent"]["readiness"] == "blocked"

        # Verify it can be JSON serialized
        json_str = json.dumps(column_data)
        assert "orders.amount" in json_str
        assert "network_analysis" in json_str

    def test_network_analysis_defaults_to_none(self):
        """Test that network_analysis defaults to None."""
        input_data = InterpretationInput(
            table_name="orders",
            column_name="amount",
            detected_type="DECIMAL",
            business_description=None,
        )
        assert input_data.network_analysis is None


class TestResolutionActionOutputSchema:
    """Tests for ResolutionActionOutput Pydantic model."""

    def test_parameters_field_in_schema(self):
        """Parameters field exists in the JSON schema for LLM tool use."""
        schema = ResolutionActionOutput.model_json_schema()
        assert "parameters" in schema["properties"]
        assert schema["properties"]["parameters"]["type"] == "object"

    def test_parameters_default_empty(self):
        """Parameters defaults to empty dict when not provided."""
        output = ResolutionActionOutput(
            action="document_unit",
            description="Declare unit",
            effort="low",
            expected_impact="Reduces semantic.units entropy",
        )
        assert output.parameters == {}

    def test_parameters_populated(self):
        """Parameters populated when provided."""
        output = ResolutionActionOutput(
            action="document_unit",
            description="Declare unit",
            effort="low",
            expected_impact="Reduces semantic.units entropy",
            parameters={"column_name": "amount", "unit": "EUR"},
        )
        assert output.parameters == {"column_name": "amount", "unit": "EUR"}


class TestEntropyInterpretationDashboardDict:
    """Tests for to_dashboard_dict method."""

    def test_column_level_key(self):
        """Column-level interpretation uses table.column key."""
        interp = EntropyInterpretation(
            column_name="amount",
            table_name="orders",
            assumptions=[],
            resolution_actions=[],
            explanation="Test",
        )
        result = interp.to_dashboard_dict()
        assert result["column_key"] == "orders.amount"

    def test_table_level_key(self):
        """Table-level interpretation (column_name=None) uses table name as key."""
        interp = EntropyInterpretation(
            column_name=None,
            table_name="orders",
            assumptions=[],
            resolution_actions=[],
            explanation="Table-level interpretation",
        )
        result = interp.to_dashboard_dict()
        assert result["column_key"] == "orders"
        assert result["column_name"] is None
        assert result["table_name"] == "orders"

    def test_dashboard_dict_has_no_composite_score_or_readiness(self):
        """Dashboard dict should not contain composite_score or readiness."""
        interp = EntropyInterpretation(
            column_name="amount",
            table_name="orders",
            assumptions=[],
            resolution_actions=[],
            explanation="Test",
        )
        result = interp.to_dashboard_dict()
        assert "composite_score" not in result
        assert "readiness" not in result


class TestTableInterpretationInput:
    """Tests for TableInterpretationInput model."""

    def test_create_with_network_analysis(self):
        """Create TableInterpretationInput with network analysis."""
        network_analysis = {
            "readiness": "investigate",
            "intents": {
                "query_intent": {
                    "worst_p_high": 0.40,
                    "mean_p_high": 0.25,
                    "columns_blocked": 0,
                    "columns_investigate": 1,
                    "columns_ready": 1,
                    "readiness": "investigate",
                },
            },
            "columns": [
                {"column": "amount", "readiness": "investigate", "worst_p_high": 0.40, "top_fix": "unit_declaration"},
                {"column": "name", "readiness": "ready", "worst_p_high": 0.10},
            ],
            "top_fix": {"node": "unit_declaration", "columns_affected": 1, "total_delta": 0.25},
        }
        result = TableInterpretationInput(
            table_name="orders",
            column_count=2,
            network_analysis=network_analysis,
        )

        assert result.table_name == "orders"
        assert result.column_count == 2
        assert result.network_analysis is not None
        assert result.network_analysis["readiness"] == "investigate"
        assert len(result.network_analysis["columns"]) == 2
        assert result.network_analysis["top_fix"]["node"] == "unit_declaration"

    def test_create_minimal(self):
        """Create TableInterpretationInput without optional fields."""
        result = TableInterpretationInput(
            table_name="empty_table",
            column_count=0,
        )

        assert result.table_name == "empty_table"
        assert result.column_count == 0
        assert result.network_analysis is None
        assert result.dimensional_patterns is None
        assert result.column_interpretations_summary is None
        assert result.quality_overview is None

    def test_enrichment_context(self):
        """Enrichment fields are set independently of network analysis."""
        result = TableInterpretationInput(
            table_name="orders",
            column_count=3,
            network_analysis={"readiness": "ready", "intents": {}, "columns": [], "top_fix": None},
            dimensional_patterns=[{"detector_id": "dim1", "score": 0.5}],
            column_interpretations_summary=[{"column": "amount", "top_action": "document_unit"}],
            quality_overview={"grade_counts": {"A": 2, "C": 1}, "total": 3},
        )

        assert len(result.dimensional_patterns) == 1
        assert len(result.column_interpretations_summary) == 1
        assert result.quality_overview["total"] == 3


class TestValidateOutput:
    """Tests for Pydantic model validation of LLM tool output."""

    def test_valid_with_columns_key(self):
        """Normal case: data has 'columns' wrapper."""
        data = {
            "columns": {
                "orders.amount": {
                    "explanation": "The amount column has moderate uncertainty.",
                    "assumptions": [],
                    "resolution_actions": [],
                }
            }
        }
        output = EntropyInterpretationOutput.model_validate(data)
        assert "orders.amount" in output.columns

    def test_rejects_missing_columns_key(self):
        """When 'columns' key is missing, validation should fail."""
        data = {
            "orders.amount": {
                "explanation": "The amount column has moderate uncertainty.",
                "assumptions": [],
                "resolution_actions": [],
            }
        }
        with pytest.raises(ValidationError):
            EntropyInterpretationOutput.model_validate(data)

    def test_rejects_missing_tables_key(self):
        """When 'tables' key is missing, validation should fail."""
        data = {
            "orders": {
                "explanation": "Table has moderate uncertainty.",
                "assumptions": [],
                "resolution_actions": [],
            }
        }
        with pytest.raises(ValidationError):
            TableEntropyInterpretationOutput.model_validate(data)
