"""Tests for entropy contracts module."""

import pytest

from dataraum.entropy.contracts import (
    ConfidenceLevel,
    ContractProfile,
    Violation,
    _calculate_confidence_level,
    clear_contracts_cache,
    evaluate_contract,
    find_best_contract,
    get_contract,
    get_contracts,
    list_contracts,
)
from dataraum.entropy.models import (
    ColumnEntropyProfile,
    CompoundRisk,
    EntropyContext,
    TableEntropyProfile,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear contracts cache before each test."""
    clear_contracts_cache()
    yield
    clear_contracts_cache()


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_emoji(self):
        """Each level has correct emoji."""
        assert ConfidenceLevel.GREEN.emoji == "🟢"
        assert ConfidenceLevel.YELLOW.emoji == "🟡"
        assert ConfidenceLevel.ORANGE.emoji == "🟠"
        assert ConfidenceLevel.RED.emoji == "🔴"

    def test_label(self):
        """Each level has correct label."""
        assert ConfidenceLevel.GREEN.label == "GOOD"
        assert ConfidenceLevel.YELLOW.label == "MARGINAL"
        assert ConfidenceLevel.ORANGE.label == "ISSUES"
        assert ConfidenceLevel.RED.label == "BLOCKED"

    def test_value(self):
        """Each level has correct string value."""
        assert ConfidenceLevel.GREEN.value == "green"
        assert ConfidenceLevel.YELLOW.value == "yellow"
        assert ConfidenceLevel.ORANGE.value == "orange"
        assert ConfidenceLevel.RED.value == "red"


class TestContractsFromYaml:
    """Tests for contract definitions loaded from YAML."""

    def test_contracts_exist(self):
        """Contracts are available from YAML config."""
        contracts = get_contracts()
        assert len(contracts) >= 5

        expected = [
            "exploratory_analysis",
            "operational_analytics",
            "executive_dashboard",
            "regulatory_reporting",
            "data_science",
        ]
        for name in expected:
            assert name in contracts

    def test_contracts_ordered_by_strictness(self):
        """Contracts have expected strictness ordering."""
        contracts = get_contracts()

        # Strictness is measured by overall_threshold (lower = stricter)
        assert (
            contracts["regulatory_reporting"].overall_threshold
            < contracts["executive_dashboard"].overall_threshold
        )
        assert (
            contracts["executive_dashboard"].overall_threshold
            < contracts["operational_analytics"].overall_threshold
        )
        assert (
            contracts["operational_analytics"].overall_threshold
            < contracts["exploratory_analysis"].overall_threshold
        )

    def test_contract_has_dimension_thresholds(self):
        """Each contract has dimension thresholds defined."""
        contracts = get_contracts()

        for name, contract in contracts.items():
            assert len(contract.dimension_thresholds) > 0, f"{name} has no dimension thresholds"
            # Should have at least structural and semantic dimensions
            assert any("structural" in d for d in contract.dimension_thresholds), (
                f"{name} missing structural dimensions"
            )
            assert any("semantic" in d for d in contract.dimension_thresholds), (
                f"{name} missing semantic dimensions"
            )


class TestContractAccess:
    """Tests for contract access functions."""

    def test_list_contracts(self):
        """list_contracts returns summary of all contracts."""
        contracts = list_contracts()
        assert len(contracts) >= 5

        # Each entry has required fields
        for c in contracts:
            assert "name" in c
            assert "display_name" in c
            assert "description" in c
            assert "overall_threshold" in c

    def test_get_contract(self):
        """get_contract returns specific contract."""
        contract = get_contract("executive_dashboard")
        assert contract is not None
        assert contract.name == "executive_dashboard"
        assert contract.overall_threshold == 0.25

    def test_get_contract_not_found(self):
        """get_contract returns None for unknown contract."""
        contract = get_contract("nonexistent_contract")
        assert contract is None

    def test_get_contracts_cached(self):
        """get_contracts caches results."""
        contracts1 = get_contracts()
        contracts2 = get_contracts()
        assert contracts1 is contracts2


class TestContractEvaluation:
    """Tests for contract evaluation."""

    @pytest.fixture
    def low_entropy_context(self):
        """Context with low entropy across all dimensions."""
        column_profile = ColumnEntropyProfile(
            column_id="col1",
            column_name="test_column",
            table_name="test_table",
            structural_entropy=0.1,
            semantic_entropy=0.1,
            value_entropy=0.1,
            computational_entropy=0.1,
            composite_score=0.1,
            dimension_scores={
                "structural.types": 0.1,
                "structural.relations": 0.1,
                "semantic.business_meaning": 0.1,
                "semantic.units": 0.1,
                "semantic.temporal": 0.1,
                "value.nulls": 0.1,
                "value.outliers": 0.1,
                "computational.derived_values": 0.1,
                "computational.aggregations": 0.1,
            },
            readiness="ready",
        )

        table_profile = TableEntropyProfile(
            table_id="t1",
            table_name="test_table",
            column_profiles=[column_profile],
            avg_composite_score=0.1,
            readiness="ready",
        )

        context = EntropyContext(
            column_profiles={"test_table.test_column": column_profile},
            table_profiles={"test_table": table_profile},
            overall_readiness="ready",
        )
        return context

    @pytest.fixture
    def high_entropy_context(self):
        """Context with high entropy across dimensions."""
        column_profile = ColumnEntropyProfile(
            column_id="col1",
            column_name="test_column",
            table_name="test_table",
            structural_entropy=0.7,
            semantic_entropy=0.8,
            value_entropy=0.6,
            computational_entropy=0.5,
            composite_score=0.65,
            dimension_scores={
                "structural.types": 0.7,
                "structural.relations": 0.6,
                "semantic.business_meaning": 0.8,
                "semantic.units": 0.7,
                "semantic.temporal": 0.6,
                "value.nulls": 0.5,
                "value.outliers": 0.6,
                "computational.derived_values": 0.4,
                "computational.aggregations": 0.5,
            },
            readiness="investigate",
        )

        table_profile = TableEntropyProfile(
            table_id="t1",
            table_name="test_table",
            column_profiles=[column_profile],
            avg_composite_score=0.65,
            readiness="investigate",
        )

        context = EntropyContext(
            column_profiles={"test_table.test_column": column_profile},
            table_profiles={"test_table": table_profile},
            overall_readiness="investigate",
        )
        return context

    @pytest.fixture
    def context_with_critical_risk(self):
        """Context with critical compound risk."""
        column_profile = ColumnEntropyProfile(
            column_id="col1",
            column_name="test_column",
            table_name="test_table",
            structural_entropy=0.3,
            semantic_entropy=0.3,
            value_entropy=0.3,
            computational_entropy=0.3,
            composite_score=0.3,
            dimension_scores={
                "structural.types": 0.2,
                "semantic.units": 0.3,
            },
            readiness="investigate",
        )

        table_profile = TableEntropyProfile(
            table_id="t1",
            table_name="test_table",
            column_profiles=[column_profile],
            avg_composite_score=0.3,
        )

        critical_risk = CompoundRisk(
            target="test_table.test_column",
            dimensions=["semantic.units", "computational.aggregations"],
            risk_level="critical",
            impact="Unknown currencies being summed",
            combined_score=0.9,
        )

        context = EntropyContext(
            column_profiles={"test_table.test_column": column_profile},
            table_profiles={"test_table": table_profile},
            compound_risks=[critical_risk],
            overall_readiness="blocked",
        )
        return context

    def test_evaluate_exploratory_low_entropy(self, low_entropy_context):
        """Low entropy passes exploratory contract."""
        evaluation = evaluate_contract(low_entropy_context, "exploratory_analysis")

        assert evaluation.is_compliant is True
        assert evaluation.confidence_level == ConfidenceLevel.GREEN
        assert len(evaluation.violations) == 0

    def test_evaluate_regulatory_low_entropy(self, low_entropy_context):
        """Low entropy passes regulatory contract."""
        evaluation = evaluate_contract(low_entropy_context, "regulatory_reporting")

        assert evaluation.is_compliant is True
        # Score of 0.1 equals the threshold, so may be YELLOW (approaching threshold)
        assert evaluation.confidence_level in (ConfidenceLevel.GREEN, ConfidenceLevel.YELLOW)

    def test_evaluate_exploratory_high_entropy(self, high_entropy_context):
        """High entropy may fail exploratory contract."""
        evaluation = evaluate_contract(high_entropy_context, "exploratory_analysis")

        # Exploratory is lenient but 0.8+ scores will trigger blocking
        # Our fixture has 0.8 for semantic.business_meaning
        # Should have violations but may still pass depending on exact scores
        assert len(evaluation.violations) > 0 or len(evaluation.warnings) > 0

    def test_evaluate_regulatory_high_entropy(self, high_entropy_context):
        """High entropy fails regulatory contract."""
        evaluation = evaluate_contract(high_entropy_context, "regulatory_reporting")

        assert evaluation.is_compliant is False
        assert evaluation.confidence_level in (ConfidenceLevel.ORANGE, ConfidenceLevel.RED)
        assert len(evaluation.violations) > 0

    def test_critical_risk_blocks_strict_contracts(self, context_with_critical_risk):
        """Critical compound risk blocks strict contracts."""
        # Regulatory should be blocked by critical risk
        reg_eval = evaluate_contract(context_with_critical_risk, "regulatory_reporting")
        assert reg_eval.is_compliant is False

        # Executive should also be blocked
        exec_eval = evaluate_contract(context_with_critical_risk, "executive_dashboard")
        assert exec_eval.is_compliant is False

    def test_evaluation_has_dimension_scores(self, low_entropy_context):
        """Evaluation includes dimension scores."""
        evaluation = evaluate_contract(low_entropy_context, "exploratory_analysis")

        assert len(evaluation.dimension_scores) > 0
        # Should have scores for dimensions in contract
        assert "structural.types" in evaluation.dimension_scores or any(
            "structural" in k for k in evaluation.dimension_scores
        )

    def test_evaluation_to_dict(self, low_entropy_context):
        """Evaluation can be serialized to dict."""
        evaluation = evaluate_contract(low_entropy_context, "exploratory_analysis")
        d = evaluation.to_dict()

        assert d["contract_name"] == "exploratory_analysis"
        assert d["is_compliant"] is True
        assert "confidence_level" in d
        assert "dimension_scores" in d
        assert "evaluated_at" in d

    def test_unknown_contract_raises(self, low_entropy_context):
        """Evaluating unknown contract raises ValueError."""
        with pytest.raises(ValueError, match="Contract not found"):
            evaluate_contract(low_entropy_context, "nonexistent_contract")


class TestFindBestContract:
    """Tests for find_best_contract function."""

    def test_find_best_with_low_entropy(self):
        """Low entropy data should pass strict contracts."""
        column_profile = ColumnEntropyProfile(
            column_id="col1",
            column_name="test_column",
            table_name="test_table",
            structural_entropy=0.05,
            semantic_entropy=0.05,
            value_entropy=0.05,
            computational_entropy=0.05,
            composite_score=0.05,
            dimension_scores={
                "structural.types": 0.05,
                "structural.relations": 0.05,
                "semantic.business_meaning": 0.05,
                "semantic.units": 0.05,
                "semantic.temporal": 0.05,
                "value.nulls": 0.05,
                "value.outliers": 0.05,
                "computational.derived_values": 0.05,
                "computational.aggregations": 0.05,
            },
            readiness="ready",
        )

        context = EntropyContext(
            column_profiles={"test_table.test_column": column_profile},
            table_profiles={
                "test_table": TableEntropyProfile(
                    table_id="t1",
                    table_name="test_table",
                    column_profiles=[column_profile],
                    avg_composite_score=0.05,
                )
            },
            overall_readiness="ready",
        )

        name, evaluation = find_best_contract(context)

        # Should find regulatory_reporting as strictest passing
        assert name == "regulatory_reporting"
        assert evaluation.is_compliant is True

    def test_find_best_with_medium_entropy(self):
        """Medium entropy should find appropriate contract."""
        column_profile = ColumnEntropyProfile(
            column_id="col1",
            column_name="test_column",
            table_name="test_table",
            structural_entropy=0.3,
            semantic_entropy=0.35,
            value_entropy=0.3,
            computational_entropy=0.25,
            composite_score=0.3,
            dimension_scores={
                "structural.types": 0.25,
                "structural.relations": 0.3,
                "semantic.business_meaning": 0.35,
                "semantic.units": 0.3,
                "semantic.temporal": 0.35,
                "value.nulls": 0.3,
                "value.outliers": 0.25,
                "computational.derived_values": 0.25,
                "computational.aggregations": 0.25,
            },
            readiness="investigate",
        )

        context = EntropyContext(
            column_profiles={"test_table.test_column": column_profile},
            table_profiles={
                "test_table": TableEntropyProfile(
                    table_id="t1",
                    table_name="test_table",
                    column_profiles=[column_profile],
                    avg_composite_score=0.3,
                )
            },
            overall_readiness="investigate",
        )

        name, evaluation = find_best_contract(context)

        # Should find something less strict than regulatory
        assert name in ("operational_analytics", "data_science", "exploratory_analysis")
        assert evaluation.is_compliant is True

    def test_find_best_with_very_high_entropy(self):
        """Very high entropy data should return None when no contracts pass."""
        column_profile = ColumnEntropyProfile(
            column_id="col1",
            column_name="test_column",
            table_name="test_table",
            structural_entropy=0.9,
            semantic_entropy=0.9,
            value_entropy=0.9,
            computational_entropy=0.9,
            composite_score=0.9,
            dimension_scores={
                "structural.types": 0.9,
                "structural.relations": 0.9,
                "semantic.business_meaning": 0.9,
                "semantic.units": 0.9,
                "semantic.temporal": 0.9,
                "value.nulls": 0.9,
                "value.outliers": 0.9,
                "computational.derived_values": 0.9,
                "computational.aggregations": 0.9,
            },
            readiness="blocked",
        )

        context = EntropyContext(
            column_profiles={"test_table.test_column": column_profile},
            table_profiles={
                "test_table": TableEntropyProfile(
                    table_id="t1",
                    table_name="test_table",
                    column_profiles=[column_profile],
                    avg_composite_score=0.9,
                )
            },
            overall_readiness="blocked",
        )

        name, evaluation = find_best_contract(context)

        # Should return None when no contracts pass
        assert name is None
        assert evaluation is None


class TestConfidenceLevelCalculation:
    """Tests for confidence level calculation."""

    def test_green_when_compliant_no_warnings(self):
        """GREEN when compliant with no warnings."""
        contract = ContractProfile(
            name="test",
            display_name="Test",
            description="Test contract",
            overall_threshold=0.5,
        )

        level = _calculate_confidence_level(
            is_compliant=True,
            violations=[],
            warnings=[],
            dimension_scores={},
            contract=contract,
        )

        assert level == ConfidenceLevel.GREEN

    def test_yellow_when_compliant_with_warnings(self):
        """YELLOW when compliant but has warnings."""
        contract = ContractProfile(
            name="test",
            display_name="Test",
            description="Test contract",
            overall_threshold=0.5,
        )

        warning = Violation(
            violation_type="dimension",
            severity="warning",
            dimension="test.dim",
            max_allowed=0.3,
            actual=0.28,
            details="Approaching threshold",
        )

        level = _calculate_confidence_level(
            is_compliant=True,
            violations=[],
            warnings=[warning],
            dimension_scores={},
            contract=contract,
        )

        assert level == ConfidenceLevel.YELLOW

    def test_orange_when_non_compliant_few_blocking(self):
        """ORANGE when non-compliant with few blocking violations."""
        contract = ContractProfile(
            name="test",
            display_name="Test",
            description="Test contract",
            overall_threshold=0.5,
        )

        violation = Violation(
            violation_type="dimension",
            severity="blocking",
            dimension="test.dim",
            max_allowed=0.3,
            actual=0.5,
            details="Exceeds threshold",
        )

        level = _calculate_confidence_level(
            is_compliant=False,
            violations=[violation],
            warnings=[],
            dimension_scores={},
            contract=contract,
        )

        assert level == ConfidenceLevel.ORANGE

    def test_red_when_many_blocking_violations(self):
        """RED when many blocking violations."""
        contract = ContractProfile(
            name="test",
            display_name="Test",
            description="Test contract",
            overall_threshold=0.5,
        )

        violations = [
            Violation(
                violation_type="dimension",
                severity="blocking",
                dimension=f"test.dim{i}",
                max_allowed=0.3,
                actual=0.5,
                details="Exceeds threshold",
            )
            for i in range(4)
        ]

        level = _calculate_confidence_level(
            is_compliant=False,
            violations=violations,
            warnings=[],
            dimension_scores={},
            contract=contract,
        )

        assert level == ConfidenceLevel.RED

    def test_red_when_blocking_condition_triggered(self):
        """RED when blocking condition is triggered."""
        contract = ContractProfile(
            name="test",
            display_name="Test",
            description="Test contract",
            overall_threshold=0.5,
        )

        violation = Violation(
            violation_type="blocking_condition",
            severity="blocking",
            condition="has_critical_compound_risk",
            details="Critical compound risk exists",
        )

        level = _calculate_confidence_level(
            is_compliant=False,
            violations=[violation],
            warnings=[],
            dimension_scores={},
            contract=contract,
        )

        assert level == ConfidenceLevel.RED
