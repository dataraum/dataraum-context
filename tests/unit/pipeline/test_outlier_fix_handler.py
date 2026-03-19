"""Tests for the accept-finding fix via bridge (replaces old handler tests)."""

from __future__ import annotations

from dataraum.entropy.fix_schemas import get_schemas_for_detector
from dataraum.pipeline.fixes import FixInput
from dataraum.pipeline.fixes.bridge import build_fix_documents


class TestOutlierDetectorFixSchemas:
    def test_has_fix_schema(self) -> None:
        schemas = get_schemas_for_detector("outlier_rate")
        assert len(schemas) == 1
        assert schemas[0].action == "document_accepted_outlier_rate"
        assert schemas[0].target == "metadata"
        assert schemas[0].routing == "postprocess"
        assert schemas[0].gate == "quality_review"


class TestAcceptFindingBridge:
    def _get_schema(self):
        return get_schemas_for_detector("outlier_rate")[0]

    def test_single_column(self) -> None:
        schema = self._get_schema()
        fix_input = FixInput(
            action_name="document_accepted_outlier_rate",
            affected_columns=["orders.amount"],
            parameters={"detector_id": "outlier_rate"},
            interpretation="Outliers are expected for this column",
        )
        docs = build_fix_documents(schema, fix_input, "orders", "amount", "quality_review")

        assert len(docs) == 1
        doc = docs[0]
        assert doc.target == "metadata"
        assert doc.action == "document_accepted_outlier_rate"
        assert doc.column_name == "amount"
        assert doc.payload["reason"] == "Outliers are expected for this column"

    def test_multiple_columns(self) -> None:
        schema = self._get_schema()
        fix_input = FixInput(
            action_name="document_accepted_outlier_rate",
            affected_columns=["orders.amount", "orders.quantity"],
            parameters={"detector_id": "outlier_rate"},
            interpretation="Outliers expected",
        )
        docs = build_fix_documents(schema, fix_input, "orders", "amount", "quality_review")

        assert len(docs) == 2
        col_names = [d.column_name for d in docs]
        assert "amount" in col_names
        assert "quantity" in col_names

    def test_no_affected_columns(self) -> None:
        schema = self._get_schema()
        fix_input = FixInput(
            action_name="document_accepted_outlier_rate",
            affected_columns=[],
            parameters={"detector_id": "outlier_rate"},
        )
        docs = build_fix_documents(schema, fix_input, "orders", "amount", "quality_review")
        assert docs == []

    def test_uses_interpretation_as_reason(self) -> None:
        schema = self._get_schema()
        fix_input = FixInput(
            action_name="document_accepted_outlier_rate",
            affected_columns=["orders.amount"],
            parameters={"detector_id": "outlier_rate"},
            interpretation="User confirmed outliers are valid business data",
        )
        docs = build_fix_documents(schema, fix_input, "orders", "amount", "quality_review")
        assert docs[0].payload["reason"] == "User confirmed outliers are valid business data"
