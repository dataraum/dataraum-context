"""Tests for quality synthesis module.

This module tests the complete quality synthesis pipeline including:
- Dimensional scoring functions
- Issue aggregation from all pillars
- Column-level quality assessment
- Table-level quality assessment
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from dataraum_context.core.models.quality_synthesis import (
    QualityDimension,
    QualitySeverity,
)
from dataraum_context.quality.synthesis import (
    _aggregate_correlation_issues,
    _aggregate_domain_quality_issues,
    _aggregate_statistical_issues,
    _aggregate_temporal_issues,
    _aggregate_topological_issues,
    _compute_accuracy_score,
    _compute_completeness_score,
    _compute_consistency_score,
    _compute_timeliness_score,
    _compute_uniqueness_score,
    _compute_validity_score,
    assess_column_quality,
    assess_table_quality,
)
from dataraum_context.storage.models_v2.base import Base
from dataraum_context.storage.models_v2.core import Column, Source, Table
from dataraum_context.storage.models_v2.correlation_context import (
    ColumnCorrelation,
    FunctionalDependency,
)
from dataraum_context.storage.models_v2.domain_quality import DomainQualityMetrics
from dataraum_context.storage.models_v2.statistical_context import (
    StatisticalProfile,
    StatisticalQualityMetrics,
)
from dataraum_context.storage.models_v2.temporal_context import TemporalQualityMetrics
from dataraum_context.storage.models_v2.topological_context import TopologicalQualityMetrics

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def async_session():
    """Create an in-memory SQLite session for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session_maker() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def sample_source(async_session: AsyncSession):
    """Create a sample data source for testing."""
    source = Source(
        source_id=str(uuid4()),
        name="test_source",
        source_type="csv",
        created_at=datetime.now(UTC),
    )
    async_session.add(source)
    await async_session.commit()
    return source


@pytest.fixture
async def sample_table(async_session: AsyncSession, sample_source: Source):
    """Create a sample table for testing."""
    table = Table(
        table_id=str(uuid4()),
        source_id=sample_source.source_id,
        table_name="test_table",
        layer="typed",
        duckdb_path="typed_test_table",
        created_at=datetime.now(UTC),
    )
    async_session.add(table)
    await async_session.commit()
    return table


@pytest.fixture
async def sample_column(async_session: AsyncSession, sample_table: Table):
    """Create a sample column for testing."""
    column = Column(
        column_id=str(uuid4()),
        table_id=sample_table.table_id,
        column_name="test_column",
        column_position=0,
        resolved_type="INTEGER",
    )
    async_session.add(column)
    await async_session.commit()
    return column


# ============================================================================
# Test Dimensional Scoring Functions
# ============================================================================


class TestCompletenessScore:
    """Test completeness score computation."""

    def test_perfect_completeness(self):
        """Test perfect completeness (no nulls, no gaps)."""
        score, explanation = _compute_completeness_score(
            null_ratio=0.0,
            temporal_completeness=1.0,
        )
        assert score == 1.0
        assert "100.0% non-null" in explanation
        assert "100.0% temporally complete" in explanation

    def test_partial_nulls(self):
        """Test with some null values."""
        score, explanation = _compute_completeness_score(
            null_ratio=0.2,  # 20% nulls
            temporal_completeness=None,
        )
        assert score == 0.8
        assert "80.0% non-null" in explanation

    def test_temporal_gaps(self):
        """Test with temporal gaps."""
        score, explanation = _compute_completeness_score(
            null_ratio=None,
            temporal_completeness=0.7,  # 70% complete
        )
        assert score == 0.7
        assert "70.0% temporally complete" in explanation

    def test_combined_incompleteness(self):
        """Test combined null ratio and temporal gaps."""
        score, explanation = _compute_completeness_score(
            null_ratio=0.1,  # 10% nulls
            temporal_completeness=0.8,  # 80% temporally complete
        )
        # (1 - 0.1) * 0.8 = 0.72
        assert score == pytest.approx(0.72)

    def test_no_metrics(self):
        """Test when no metrics are available."""
        score, explanation = _compute_completeness_score(None, None)
        assert score == 1.0
        assert "No completeness metrics available" in explanation


class TestValidityScore:
    """Test validity score computation."""

    def test_perfect_validity(self):
        """Test perfect validity (all values parse, no outliers)."""
        score, explanation = _compute_validity_score(
            parse_success_rate=1.0,
            outlier_ratio=0.0,
        )
        assert score == 1.0
        assert "100.0% parse successfully" in explanation

    def test_parse_failures(self):
        """Test with parse failures."""
        score, explanation = _compute_validity_score(
            parse_success_rate=0.9,  # 90% parse
            outlier_ratio=None,
        )
        assert score == 0.9

    def test_outliers_below_threshold(self):
        """Test with outliers below 5% threshold (no penalty)."""
        score, explanation = _compute_validity_score(
            parse_success_rate=None,
            outlier_ratio=0.03,  # 3% outliers
        )
        assert score == 1.0

    def test_outliers_above_threshold(self):
        """Test with outliers above 5% threshold."""
        score, explanation = _compute_validity_score(
            parse_success_rate=None,
            outlier_ratio=0.15,  # 15% outliers
        )
        # Penalty = min((0.15 - 0.05) / 0.10, 0.5) = min(1.0, 0.5) = 0.5
        # Score = 1.0 * (1 - 0.5) = 0.5
        assert score == 0.5
        assert "15.0% outliers" in explanation

    def test_combined_validity_issues(self):
        """Test combined parse failures and outliers."""
        score, explanation = _compute_validity_score(
            parse_success_rate=0.95,
            outlier_ratio=0.10,  # 10% outliers, penalty = 0.5
        )
        # 0.95 * (1 - 0.5) = 0.475
        assert score == pytest.approx(0.475)


class TestConsistencyScore:
    """Test consistency score computation."""

    def test_perfect_consistency(self):
        """Test perfect consistency."""
        score, explanation = _compute_consistency_score(
            vif_score=1.0,
            functional_dep_violations=0,
            orphaned_components=0,
            anomalous_cycles_count=0,
            high_correlations_count=0,
        )
        assert score == 1.0
        assert "No consistency metrics available" in explanation

    def test_high_vif(self):
        """Test with high VIF (multicollinearity)."""
        score, explanation = _compute_consistency_score(
            vif_score=20.0,  # VIF of 20
            functional_dep_violations=None,
            orphaned_components=None,
            anomalous_cycles_count=None,
            high_correlations_count=None,
        )
        # Penalty = min((20 - 10) / 20, 0.5) = 0.5
        # Score = 1.0 * (1 - 0.5) = 0.5
        assert score == 0.5
        assert "VIF=20.0" in explanation

    def test_fd_violations(self):
        """Test with functional dependency violations."""
        score, explanation = _compute_consistency_score(
            vif_score=None,
            functional_dep_violations=3,
            orphaned_components=None,
            anomalous_cycles_count=None,
            high_correlations_count=None,
        )
        # Penalty = min(3 * 0.1, 0.5) = 0.3
        # Score = 1.0 * (1 - 0.3) = 0.7
        assert score == pytest.approx(0.7)
        assert "3 FD violations" in explanation

    def test_orphaned_components(self):
        """Test with orphaned structural components."""
        score, explanation = _compute_consistency_score(
            vif_score=None,
            functional_dep_violations=None,
            orphaned_components=2,
            anomalous_cycles_count=None,
            high_correlations_count=None,
        )
        # Penalty = min(2 * 0.15, 0.4) = 0.3
        # Score = 1.0 * (1 - 0.3) = 0.7
        assert score == pytest.approx(0.7)
        assert "2 disconnected components" in explanation

    def test_anomalous_cycles(self):
        """Test with anomalous cycles."""
        score, explanation = _compute_consistency_score(
            vif_score=None,
            functional_dep_violations=None,
            orphaned_components=None,
            anomalous_cycles_count=2,
            high_correlations_count=None,
        )
        # Penalty = min(2 * 0.1, 0.3) = 0.2
        # Score = 1.0 * (1 - 0.2) = 0.8
        assert score == pytest.approx(0.8)
        assert "2 anomalous cycles" in explanation

    def test_high_correlations(self):
        """Test with high correlations."""
        score, explanation = _compute_consistency_score(
            vif_score=None,
            functional_dep_violations=None,
            orphaned_components=None,
            anomalous_cycles_count=None,
            high_correlations_count=4,
        )
        # Penalty = min(4 * 0.05, 0.3) = 0.2
        # Score = 1.0 * (1 - 0.2) = 0.8
        assert score == pytest.approx(0.8)
        assert "4 high correlations" in explanation

    def test_combined_consistency_issues(self):
        """Test multiple consistency issues."""
        score, explanation = _compute_consistency_score(
            vif_score=15.0,  # Penalty = 0.25
            functional_dep_violations=1,  # Penalty = 0.1
            orphaned_components=1,  # Penalty = 0.15
            anomalous_cycles_count=1,  # Penalty = 0.1
            high_correlations_count=2,  # Penalty = 0.1
        )
        # Total penalties applied multiplicatively:
        # 1.0 * (1-0.25) * (1-0.1) * (1-0.15) * (1-0.1) * (1-0.1)
        # = 0.75 * 0.9 * 0.85 * 0.9 * 0.9 = 0.4385...
        assert score < 0.5
        assert "VIF=" in explanation
        assert "FD violations" in explanation


class TestUniquenessScore:
    """Test uniqueness score computation."""

    def test_perfect_uniqueness(self):
        """Test perfect uniqueness (all distinct)."""
        score, explanation = _compute_uniqueness_score(
            cardinality_ratio=1.0,
            duplicate_count=0,
            total_count=100,
        )
        assert score == 1.0
        assert "100.0% unique values" in explanation

    def test_low_cardinality(self):
        """Test low cardinality."""
        score, explanation = _compute_uniqueness_score(
            cardinality_ratio=0.5,  # 50% distinct
            duplicate_count=None,
            total_count=None,
        )
        assert score == 0.5
        assert "50.0% unique values" in explanation

    def test_with_duplicates(self):
        """Test with duplicate count."""
        score, explanation = _compute_uniqueness_score(
            cardinality_ratio=0.8,
            duplicate_count=20,
            total_count=100,
        )
        assert score == 0.8
        assert "20.0% duplicates" in explanation


class TestTimelinessScore:
    """Test timeliness score computation."""

    def test_fresh_data(self):
        """Test fresh data (not stale)."""
        score, explanation = _compute_timeliness_score(
            is_stale=False,
            data_freshness_days=3,
        )
        assert score == 1.0
        assert "Data is fresh" in explanation

    def test_stale_data(self):
        """Test stale data."""
        score, explanation = _compute_timeliness_score(
            is_stale=True,
            data_freshness_days=None,
        )
        assert score == 0.5
        assert "Data is stale" in explanation

    def test_old_data(self):
        """Test old data (90 days)."""
        score, explanation = _compute_timeliness_score(
            is_stale=None,
            data_freshness_days=90,
        )
        # freshness_score = max(0.25, 1.0 - (90 - 7) / 180)
        # = max(0.25, 1.0 - 83/180) = max(0.25, 0.539) = 0.539
        assert score == pytest.approx(0.539, abs=0.01)
        assert "90 days old" in explanation

    def test_very_old_data(self):
        """Test very old data (180+ days)."""
        score, explanation = _compute_timeliness_score(
            is_stale=None,
            data_freshness_days=200,
        )
        # Should hit minimum of 0.25
        assert score == pytest.approx(0.25)


class TestAccuracyScore:
    """Test accuracy score computation."""

    def test_perfect_accuracy(self):
        """Test perfect accuracy (Benford compliant, domain compliant)."""
        score, explanation = _compute_accuracy_score(
            benford_compliant=True,
            domain_compliance_score=1.0,
        )
        assert score == 1.0
        assert "Benford's Law satisfied" in explanation
        assert "100.0% domain compliant" in explanation

    def test_benford_violation(self):
        """Test Benford's Law violation."""
        score, explanation = _compute_accuracy_score(
            benford_compliant=False,
            domain_compliance_score=None,
        )
        assert score == 0.7  # 30% penalty
        assert "Benford's Law violated" in explanation

    def test_domain_compliance(self):
        """Test domain compliance."""
        score, explanation = _compute_accuracy_score(
            benford_compliant=None,
            domain_compliance_score=0.8,
        )
        assert score == 0.8
        assert "80.0% domain compliant" in explanation

    def test_combined_accuracy_issues(self):
        """Test both Benford and domain issues."""
        score, explanation = _compute_accuracy_score(
            benford_compliant=False,
            domain_compliance_score=0.9,
        )
        # 0.7 * 0.9 = 0.63
        assert score == pytest.approx(0.63)


# ============================================================================
# Test Issue Aggregation
# ============================================================================


class TestStatisticalIssueAggregation:
    """Test statistical issue aggregation."""

    def test_no_issues(self):
        """Test when no statistical issues exist."""
        stat_quality = StatisticalQualityMetrics(
            metric_id=str(uuid4()),
            column_id=str(uuid4()),
            computed_at=datetime.now(UTC),
            quality_issues=None,
        )

        issues = _aggregate_statistical_issues(stat_quality, "col1", "test_col")
        assert len(issues) == 0

    def test_benford_violation(self):
        """Test Benford violation issue."""
        stat_quality = StatisticalQualityMetrics(
            metric_id=str(uuid4()),
            column_id=str(uuid4()),
            computed_at=datetime.now(UTC),
            quality_issues={
                "issues": [
                    {
                        "issue_type": "benford_violation",
                        "severity": "warning",
                        "description": "First digit distribution deviates from Benford",
                        "evidence": {"chi_square_p": 0.01},
                    }
                ]
            },
        )

        issues = _aggregate_statistical_issues(stat_quality, "col1", "test_col")
        assert len(issues) == 1
        assert issues[0].issue_type == "benford_violation"
        assert issues[0].dimension == QualityDimension.ACCURACY
        assert issues[0].severity == QualitySeverity.WARNING
        assert issues[0].source_pillar == 1


class TestTemporalIssueAggregation:
    """Test temporal issue aggregation."""

    def test_low_completeness(self):
        """Test low completeness issue."""
        temp_quality = TemporalQualityMetrics(
            metric_id=str(uuid4()),
            column_id=str(uuid4()),
            computed_at=datetime.now(UTC),
            min_timestamp=datetime.now(UTC),
            max_timestamp=datetime.now(UTC),
            span_days=30,
            detected_granularity="day",
            granularity_confidence=0.9,
            quality_issues={
                "issues": [
                    {
                        "issue_type": "low_completeness",
                        "severity": "medium",
                        "description": "Only 60% of expected data points present",
                        "evidence": {"completeness_ratio": 0.6},
                    }
                ]
            },
        )

        issues = _aggregate_temporal_issues(temp_quality, "col1", "test_col")
        assert len(issues) == 1
        assert issues[0].issue_type == "low_completeness"
        assert issues[0].dimension == QualityDimension.COMPLETENESS
        assert issues[0].source_pillar == 4


class TestTopologicalIssueAggregation:
    """Test topological issue aggregation."""

    def test_orphaned_components(self):
        """Test orphaned components issue."""
        topo_quality = TopologicalQualityMetrics(
            metric_id=str(uuid4()),
            table_id=str(uuid4()),
            computed_at=datetime.now(UTC),
            orphaned_components=3,
        )

        issues = _aggregate_topological_issues(topo_quality, "table1", "test_table")
        assert len(issues) >= 1
        orphaned_issue = [i for i in issues if i.issue_type == "orphaned_components"][0]
        assert orphaned_issue.dimension == QualityDimension.CONSISTENCY
        assert orphaned_issue.source_pillar == 2
        assert "3 disconnected" in orphaned_issue.description


class TestCorrelationIssueAggregation:
    """Test correlation issue aggregation."""

    def test_high_correlation(self):
        """Test high correlation issue."""
        correlation = ColumnCorrelation(
            correlation_id=str(uuid4()),
            table_id=str(uuid4()),
            column1_id="col1",
            column2_id="col2",
            pearson_r=0.95,
            sample_size=100,
            computed_at=datetime.now(UTC),
        )

        issues = _aggregate_correlation_issues("col1", "test_col", [correlation], [])
        assert len(issues) == 1
        assert issues[0].issue_type == "high_correlation"
        assert issues[0].dimension == QualityDimension.CONSISTENCY
        assert "0.95" in issues[0].description

    def test_fd_violation(self):
        """Test functional dependency violation."""
        fd = FunctionalDependency(
            dependency_id=str(uuid4()),
            table_id=str(uuid4()),
            determinant_column_ids=["col1"],
            dependent_column_id="col2",
            confidence=0.95,
            violation_count=5,
            unique_determinant_values=100,
            computed_at=datetime.now(UTC),
        )

        issues = _aggregate_correlation_issues("col1", "test_col", [], [fd])
        assert len(issues) == 1
        assert issues[0].issue_type == "fd_violation"
        assert issues[0].severity == QualitySeverity.WARNING


class TestDomainQualityIssueAggregation:
    """Test domain quality issue aggregation."""

    def test_domain_violations(self):
        """Test domain quality violations."""
        domain_quality = DomainQualityMetrics(
            metric_id=uuid4(),
            table_id=uuid4(),
            domain="financial",
            computed_at=datetime.now(UTC),
            domain_compliance_score=0.8,
            metrics={},
            violations=[
                {
                    "severity": "high",
                    "description": "Negative revenue values detected",
                    "recommendation": "Verify data source",
                    "evidence": {"negative_count": 5},
                }
            ],
        )

        issues = _aggregate_domain_quality_issues(domain_quality, "col1", "revenue")
        assert len(issues) == 1
        assert issues[0].issue_type == "domain_rule_violation"
        assert issues[0].dimension == QualityDimension.ACCURACY
        assert issues[0].severity == QualitySeverity.ERROR
        assert issues[0].source_pillar == 5


# ============================================================================
# Test Column Quality Assessment
# ============================================================================


@pytest.mark.asyncio
async def test_assess_column_quality_no_metrics(async_session: AsyncSession, sample_column: Column):
    """Test column assessment with no metrics available."""
    result = await assess_column_quality(sample_column, async_session)

    assert result.success
    assessment = result.value
    assert assessment.column_id == sample_column.column_id
    assert assessment.overall_score == 1.0  # Default when no metrics
    assert len(assessment.dimension_scores) == 6
    assert len(assessment.issues) == 0


@pytest.mark.asyncio
async def test_assess_column_quality_with_statistical_metrics(
    async_session: AsyncSession, sample_column: Column
):
    """Test column assessment with statistical metrics."""
    # Add statistical profile
    stat_profile = StatisticalProfile(
        profile_id=str(uuid4()),
        column_id=sample_column.column_id,
        profiled_at=datetime.now(UTC),
        total_count=100,
        null_count=10,
        null_ratio=0.1,
        distinct_count=80,
        cardinality_ratio=0.8,
        duplicate_count=20,
    )
    async_session.add(stat_profile)

    # Add statistical quality metrics
    stat_quality = StatisticalQualityMetrics(
        metric_id=str(uuid4()),
        column_id=sample_column.column_id,
        computed_at=datetime.now(UTC),
        benford_compliant=True,
        iqr_outlier_ratio=0.02,
    )
    async_session.add(stat_quality)
    await async_session.commit()

    result = await assess_column_quality(sample_column, async_session)

    assert result.success
    assessment = result.value
    assert assessment.has_statistical_quality is True

    # Check completeness score
    completeness = [
        d for d in assessment.dimension_scores if d.dimension == QualityDimension.COMPLETENESS
    ][0]
    assert completeness.score == pytest.approx(0.9)  # 1 - 0.1 null_ratio

    # Check uniqueness score
    uniqueness = [
        d for d in assessment.dimension_scores if d.dimension == QualityDimension.UNIQUENESS
    ][0]
    assert uniqueness.score == pytest.approx(0.8)  # cardinality_ratio


@pytest.mark.asyncio
async def test_assess_column_quality_with_temporal_metrics(
    async_session: AsyncSession, sample_column: Column
):
    """Test column assessment with temporal metrics."""
    temp_quality = TemporalQualityMetrics(
        metric_id=str(uuid4()),
        column_id=sample_column.column_id,
        computed_at=datetime.now(UTC),
        min_timestamp=datetime.now(UTC),
        max_timestamp=datetime.now(UTC),
        span_days=30,
        detected_granularity="day",
        granularity_confidence=0.9,
        completeness_ratio=0.95,
        is_stale=False,
        data_freshness_days=2.0,
    )
    async_session.add(temp_quality)
    await async_session.commit()

    result = await assess_column_quality(sample_column, async_session)

    assert result.success
    assessment = result.value
    assert assessment.has_temporal_quality is True

    # Check timeliness score
    timeliness = [
        d for d in assessment.dimension_scores if d.dimension == QualityDimension.TIMELINESS
    ][0]
    assert timeliness.score == 1.0  # Fresh data


# ============================================================================
# Test Table Quality Assessment
# ============================================================================


@pytest.mark.asyncio
async def test_assess_table_quality_empty_table(async_session: AsyncSession, sample_table: Table):
    """Test table assessment with no columns."""
    result = await assess_table_quality(sample_table.table_id, async_session)

    assert result.success
    synthesis = result.value
    assert synthesis.table_id == sample_table.table_id
    assert synthesis.total_columns == 0
    assert synthesis.columns_assessed == 0


@pytest.mark.asyncio
async def test_assess_table_quality_with_topological_metrics(
    async_session: AsyncSession, sample_table: Table
):
    """Test table assessment with topological metrics."""
    # Add topological metrics
    topo_quality = TopologicalQualityMetrics(
        metric_id=str(uuid4()),
        table_id=sample_table.table_id,
        computed_at=datetime.now(UTC),
        betti_0=1,
        orphaned_components=2,
        anomalous_cycles={"cycles": [{"id": "c1"}, {"id": "c2"}]},
    )
    async_session.add(topo_quality)
    await async_session.commit()

    result = await assess_table_quality(sample_table.table_id, async_session)

    assert result.success
    synthesis = result.value
    assert synthesis.table_assessment.has_topological_quality is True

    # Should have topological issues
    assert len(synthesis.table_assessment.issues) > 0
    assert any(i.issue_type == "orphaned_components" for i in synthesis.table_assessment.issues)


@pytest.mark.asyncio
async def test_assess_table_quality_integration(async_session: AsyncSession, sample_table: Table):
    """Test full table assessment with columns and all metric types."""
    # Create multiple columns
    columns = []
    for i in range(3):
        col = Column(
            column_id=str(uuid4()),
            table_id=sample_table.table_id,
            column_name=f"col_{i}",
            column_position=i,
            resolved_type="INTEGER",
        )
        columns.append(col)
        async_session.add(col)

    # Add metrics for first column
    stat_profile = StatisticalProfile(
        profile_id=str(uuid4()),
        column_id=columns[0].column_id,
        profiled_at=datetime.now(UTC),
        total_count=100,
        null_count=5,
        null_ratio=0.05,
        distinct_count=95,
        cardinality_ratio=0.95,
    )
    async_session.add(stat_profile)

    await async_session.commit()

    result = await assess_table_quality(sample_table.table_id, async_session)

    assert result.success
    synthesis = result.value
    assert synthesis.total_columns == 3
    assert synthesis.columns_assessed > 0
    assert synthesis.table_assessment.overall_score >= 0.0
    assert synthesis.table_assessment.overall_score <= 1.0

    # Check that dimension scores were computed
    assert len(synthesis.table_assessment.dimension_scores) == 6
