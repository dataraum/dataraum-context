"""Tests for entropy models."""

import pytest

from dataraum.entropy.analysis.aggregator import ColumnSummary, EntropyAggregator
from dataraum.entropy.compound_risk import CompoundRiskDetector
from dataraum.entropy.models import (
    CompoundRisk,
    CompoundRiskDefinition,
    EntropyObject,
    ResolutionCascade,
    ResolutionOption,
)


class TestEntropyObject:
    """Tests for EntropyObject."""

    def test_is_high_entropy(self, sample_entropy_object: EntropyObject):
        """Test high entropy detection."""
        assert sample_entropy_object.is_high_entropy(threshold=0.3) is True
        assert sample_entropy_object.is_high_entropy(threshold=0.5) is False

    def test_is_critical(self, sample_entropy_object: EntropyObject):
        """Test critical entropy detection."""
        assert sample_entropy_object.is_critical(threshold=0.8) is False

        critical_obj = EntropyObject(
            score=0.85, layer="value", dimension="nulls", sub_dimension="null_ratio", target="test"
        )
        assert critical_obj.is_critical() is True

    def test_dimension_path(self, sample_entropy_object: EntropyObject):
        """Test dimension path property."""
        assert sample_entropy_object.dimension_path == "structural.types.type_fidelity"


class TestResolutionOption:
    """Tests for ResolutionOption."""

    def test_priority_score_low_effort(self):
        """Test priority score calculation for low effort."""
        opt = ResolutionOption(
            action="test",
            parameters={},
            expected_entropy_reduction=0.5,
            effort="low",
        )
        assert opt.priority_score() == 0.5  # 0.5 / 1.0

    def test_priority_score_high_effort(self):
        """Test priority score calculation for high effort."""
        opt = ResolutionOption(
            action="test",
            parameters={},
            expected_entropy_reduction=0.8,
            effort="high",
        )
        assert opt.priority_score() == 0.2  # 0.8 / 4.0


class TestCompoundRisk:
    """Tests for CompoundRisk."""

    def test_from_scores(self):
        """Test creating compound risk from scores."""
        risk = CompoundRisk.from_scores(
            target="column:test.col",
            dimensions=["semantic.units", "computational.aggregations"],
            scores={
                "semantic.units": 0.7,
                "computational.aggregations": 0.6,
            },
            risk_level="critical",
            impact="Test impact",
            multiplier=2.0,
        )

        assert risk.risk_level == "critical"
        assert risk.multiplier == 2.0
        # Combined: avg(0.7, 0.6) * 2.0 = 0.65 * 2.0 = 1.3, capped at 1.0
        assert risk.combined_score == 1.0


class TestResolutionCascade:
    """Tests for ResolutionCascade."""

    def test_calculate_priority(self):
        """Test priority score calculation."""
        cascade = ResolutionCascade(
            action="declare_unit",
            parameters={"unit": "USD"},
            entropy_reductions={
                "semantic.units": 0.5,
                "computational.aggregations": 0.3,
            },
            effort="low",
        )
        priority = cascade.calculate_priority()

        assert cascade.total_reduction == 0.8
        assert cascade.dimensions_improved == 2
        assert priority == 0.8  # 0.8 / 1.0 (low effort factor)


class TestCompoundRiskDetector:
    """Tests for CompoundRiskDetector dimension score lookup."""

    def test_missing_dimension_returns_zero(self):
        """Missing dimension should return 0.0, not fall back to layer score."""
        detector = CompoundRiskDetector()
        summary = ColumnSummary(
            column_id="col1",
            column_name="amount",
            table_id="t1",
            table_name="orders",
            layer_scores={"semantic": 0.8},
            dimension_scores={
                "semantic.units.unit_declaration": 0.8,
            },
        )

        # semantic.temporal has no score — should be 0.0
        score = detector._get_dimension_score(summary, "semantic.temporal")
        assert score == 0.0

    def test_non_temporal_column_no_false_compound_risk(self):
        """Non-temporal column with high semantic entropy should not trigger temporal compound risk."""
        detector = CompoundRiskDetector()
        detector.risk_definitions = [
            CompoundRiskDefinition(
                risk_type="temporal_nulls",
                dimensions=["semantic.temporal", "value.nulls"],
                threshold=0.5,
                risk_level="high",
                impact_template="Timestamp columns have null values.",
                multiplier=1.5,
            ),
        ]
        detector.config_loaded = True

        # Measure column: high null entropy, high semantic (from units), NO temporal
        summary = ColumnSummary(
            column_id="col1",
            column_name="quantity",
            table_id="t1",
            table_name="orders",
            layer_scores={"semantic": 0.8, "value": 0.7},
            dimension_scores={
                "semantic.units.unit_declaration": 0.8,
                "value.nulls.null_ratio": 0.7,
            },
        )

        risks = detector.detect_risks(summary)
        assert len(risks) == 0

    def test_exact_dimension_match(self):
        """Exact dimension path match returns correct score."""
        detector = CompoundRiskDetector()
        summary = ColumnSummary(
            column_id="col1",
            column_name="amount",
            table_id="t1",
            table_name="orders",
            dimension_scores={"semantic.temporal.time_role": 0.6},
        )

        score = detector._get_dimension_score(summary, "semantic.temporal.time_role")
        assert score == 0.6

    def test_partial_dimension_match(self):
        """Partial prefix match returns correct score."""
        detector = CompoundRiskDetector()
        summary = ColumnSummary(
            column_id="col1",
            column_name="created_at",
            table_id="t1",
            table_name="orders",
            dimension_scores={"semantic.temporal.time_role": 0.6},
        )

        score = detector._get_dimension_score(summary, "semantic.temporal")
        assert score == 0.6

    def test_compound_risk_gradient_preservation(self):
        """Boost-above-threshold preserves gradient instead of clamping to 1.0."""
        detector = CompoundRiskDetector()
        detector.risk_definitions = [
            CompoundRiskDefinition(
                risk_type="test_risk",
                dimensions=["semantic.units", "computational.derived_values"],
                threshold=0.5,
                risk_level="critical",
                impact_template="Test impact",
                multiplier=2.0,
            ),
        ]
        detector.config_loaded = True

        # Test with scores that average to 0.6 (above threshold 0.5)
        # excess = 0.6 - 0.5 = 0.1; boosted = 0.5 + 0.1 * 2.0 = 0.7
        summary = ColumnSummary(
            column_id="col1",
            column_name="amount",
            table_id="t1",
            table_name="orders",
            dimension_scores={
                "semantic.units.unit_declaration": 0.6,
                "computational.derived_values.formula_match": 0.6,
            },
        )
        risks = detector.detect_risks(summary)
        assert len(risks) == 1
        assert risks[0].combined_score == pytest.approx(0.7, abs=0.01)

    def test_compound_risk_at_threshold(self):
        """At exactly the threshold, combined_score equals threshold."""
        detector = CompoundRiskDetector()
        detector.risk_definitions = [
            CompoundRiskDefinition(
                risk_type="test_risk",
                dimensions=["semantic.units", "computational.derived_values"],
                threshold=0.5,
                risk_level="critical",
                impact_template="Test impact",
                multiplier=2.0,
            ),
        ]
        detector.config_loaded = True

        summary = ColumnSummary(
            column_id="col1",
            column_name="amount",
            table_id="t1",
            table_name="orders",
            dimension_scores={
                "semantic.units.unit_declaration": 0.5,
                "computational.derived_values.formula_match": 0.5,
            },
        )
        risks = detector.detect_risks(summary)
        assert len(risks) == 1
        # excess = 0.5 - 0.5 = 0; boosted = 0.5 + 0 * 2.0 = 0.5
        assert risks[0].combined_score == pytest.approx(0.5, abs=0.01)


class TestAggregatorEmptyLayerNormalization:
    """Tests for EntropyAggregator empty layer normalization."""

    def test_only_structural_data(self):
        """Column with only structural data should normalize to structural score."""
        aggregator = EntropyAggregator()
        objects = [
            EntropyObject(
                layer="structural",
                dimension="types",
                sub_dimension="type_fidelity",
                target="column:test.col",
                score=0.5,
                detector_id="type_fidelity",
            ),
        ]
        summary = aggregator.summarize_column(
            column_id="col1",
            column_name="col",
            table_id="t1",
            table_name="test",
            entropy_objects=objects,
        )
        # Only structural populated (weight 0.25), so composite = 0.5 * 0.25 / 0.25 = 0.5
        assert summary.composite_score == pytest.approx(0.5, abs=0.01)
        assert "semantic" in summary.empty_layers
        assert "value" in summary.empty_layers
        assert "computational" in summary.empty_layers
        assert "structural" not in summary.empty_layers

    def test_two_layers_populated(self):
        """Two populated layers normalize weights to those two."""
        aggregator = EntropyAggregator()
        objects = [
            EntropyObject(
                layer="structural",
                dimension="types",
                sub_dimension="type_fidelity",
                target="column:test.col",
                score=0.4,
                detector_id="type_fidelity",
            ),
            EntropyObject(
                layer="value",
                dimension="nulls",
                sub_dimension="null_ratio",
                target="column:test.col",
                score=0.6,
                detector_id="null_ratio",
            ),
        ]
        summary = aggregator.summarize_column(
            column_id="col1",
            column_name="col",
            table_id="t1",
            table_name="test",
            entropy_objects=objects,
        )
        # structural weight=0.25, value weight=0.30
        # normalized: structural = 0.25/0.55, value = 0.30/0.55
        # composite = 0.4 * (0.25/0.55) + 0.6 * (0.30/0.55)
        expected = 0.4 * (0.25 / 0.55) + 0.6 * (0.30 / 0.55)
        assert summary.composite_score == pytest.approx(expected, abs=0.01)

    def test_all_layers_populated(self):
        """All four layers populated uses standard weights."""
        aggregator = EntropyAggregator()
        objects = [
            EntropyObject(
                layer="structural",
                dimension="types",
                sub_dimension="tf",
                target="t",
                score=0.2,
                detector_id="d1",
            ),
            EntropyObject(
                layer="semantic",
                dimension="bm",
                sub_dimension="nc",
                target="t",
                score=0.3,
                detector_id="d2",
            ),
            EntropyObject(
                layer="value",
                dimension="nulls",
                sub_dimension="nr",
                target="t",
                score=0.4,
                detector_id="d3",
            ),
            EntropyObject(
                layer="computational",
                dimension="dv",
                sub_dimension="fm",
                target="t",
                score=0.1,
                detector_id="d4",
            ),
        ]
        summary = aggregator.summarize_column(
            column_id="col1",
            column_name="col",
            table_id="t1",
            table_name="test",
            entropy_objects=objects,
        )
        # All weights sum to 1.0, no normalization needed
        expected = 0.2 * 0.25 + 0.3 * 0.30 + 0.4 * 0.30 + 0.1 * 0.15
        assert summary.composite_score == pytest.approx(expected, abs=0.01)
        assert summary.empty_layers == []

    def test_no_entropy_objects(self):
        """No entropy objects should give 0 composite and 'ready' status."""
        aggregator = EntropyAggregator()
        summary = aggregator.summarize_column(
            column_id="col1",
            column_name="col",
            table_id="t1",
            table_name="test",
            entropy_objects=[],
        )
        assert summary.composite_score == 0.0
        assert summary.readiness == "ready"
