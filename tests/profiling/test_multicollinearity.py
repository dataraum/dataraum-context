"""Tests for multicollinearity detection (VIF, Tolerance, Condition Index)."""

import numpy as np
import pytest
from sqlalchemy import select

from dataraum_context.profiling.correlation import (
    _compute_condition_index,
    compute_multicollinearity_for_table,
)
from dataraum_context.profiling.db_models import MulticollinearityMetrics
from dataraum_context.storage.models_v2.core import Column, Source, Table


def create_column(col_id: str, table_id: str, name: str, position: int, col_type: str = "DOUBLE"):
    """Helper to create a column with all required fields."""
    return Column(
        column_id=col_id,
        table_id=table_id,
        column_name=name,
        column_position=position,
        resolved_type=col_type,
    )


@pytest.fixture
async def sample_source(async_session):
    """Create a sample source for testing."""
    source = Source(
        source_id="test-source",
        name="test_multicollinearity",
        source_type="test",
    )
    async_session.add(source)
    await async_session.commit()
    await async_session.refresh(source)
    return source


@pytest.fixture
async def sample_table(async_session, sample_source):
    """Create a sample table for testing."""
    table = Table(
        table_id="test-multicollinearity-table",
        source_id=sample_source.source_id,
        table_name="multicollinearity_test",
        layer="typed",
        duckdb_path="multicollinearity_test",
    )
    async_session.add(table)
    await async_session.commit()
    await async_session.refresh(table)
    return table


@pytest.mark.asyncio
async def test_multicollinearity_high_vif(engine, duckdb_conn, async_session, sample_table):
    """Test VIF detection with high multicollinearity."""
    # Create data: y = 0.9*x + 0.1*noise, z = 0.8*x + 0.2*noise (high correlation)
    duckdb_conn.execute(f"""
        CREATE TABLE {sample_table.duckdb_path} AS
        SELECT
            x,
            (0.9 * x + 0.1 * random())::DOUBLE as y,
            (0.8 * x + 0.2 * random())::DOUBLE as z
        FROM (SELECT unnest(range(1, 101)) as x)
    """)

    # Create column metadata
    for idx, col_name in enumerate(["x", "y", "z"]):
        column = create_column(
            f"{sample_table.table_id}-{col_name}",
            sample_table.table_id,
            col_name,
            idx,
            "DOUBLE",
        )
        async_session.add(column)
    await async_session.commit()

    # Compute multicollinearity
    result = await compute_multicollinearity_for_table(sample_table, duckdb_conn, async_session)

    assert result.success
    analysis = result.value

    # Should detect high VIF for y and z (both correlated with x)
    assert analysis.num_problematic_columns >= 1
    assert analysis.overall_severity in ["moderate", "severe"]

    # Check VIF values
    vifs = {vif.column_ref.column_name: vif.vif for vif in analysis.column_vifs}
    assert vifs.get("y", 0) > 5  # Should have elevated VIF
    assert vifs.get("z", 0) > 5  # Should have elevated VIF

    # Verify interpretation property
    for vif in analysis.column_vifs:
        interpretation = vif.interpretation
        assert "multicollinearity" in interpretation.lower()
        assert "VIF=" in interpretation


@pytest.mark.asyncio
async def test_multicollinearity_low_vif(engine, duckdb_conn, async_session, sample_table):
    """Test VIF with independent columns."""
    # Create independent data
    duckdb_conn.execute(f"""
        CREATE TABLE {sample_table.duckdb_path} AS
        SELECT
            random() as x,
            random() as y,
            random() as z
        FROM range(100)
    """)

    # Create column metadata
    for idx, col_name in enumerate(["x", "y", "z"]):
        column = create_column(
            f"{sample_table.table_id}-{col_name}",
            sample_table.table_id,
            col_name,
            idx,
            "DOUBLE",
        )
        async_session.add(column)
    await async_session.commit()

    # Compute multicollinearity
    result = await compute_multicollinearity_for_table(sample_table, duckdb_conn, async_session)

    assert result.success
    analysis = result.value

    # Should have low/no multicollinearity
    assert analysis.num_problematic_columns == 0
    assert analysis.overall_severity == "none"

    # All VIFs should be low
    for vif in analysis.column_vifs:
        assert vif.vif < 5
        assert vif.severity == "none"
        assert not vif.has_multicollinearity


def test_condition_index_computation():
    """Test Condition Index calculation."""
    # Create highly correlated data
    np.random.seed(42)
    x = np.random.randn(100)
    y = 0.95 * x + 0.05 * np.random.randn(100)
    z = 0.90 * x + 0.10 * np.random.randn(100)

    X = np.column_stack([x, y, z])

    ci_analysis = _compute_condition_index(X)

    assert ci_analysis is not None
    # High correlation should produce CI > 10 (at least moderate)
    assert ci_analysis.condition_index >= 10
    assert ci_analysis.has_multicollinearity
    assert ci_analysis.severity in ["moderate", "severe"]
    assert len(ci_analysis.eigenvalues) == 3

    # Verify interpretation property
    interpretation = ci_analysis.interpretation
    assert "multicollinearity" in interpretation.lower()
    assert "CI=" in interpretation


def test_condition_index_independent_data():
    """Test Condition Index with independent data."""
    # Create independent data
    np.random.seed(42)
    x = np.random.randn(100)
    y = np.random.randn(100)
    z = np.random.randn(100)

    X = np.column_stack([x, y, z])

    ci_analysis = _compute_condition_index(X)

    assert ci_analysis is not None
    # Independent data should have CI < 10 (no multicollinearity)
    assert ci_analysis.condition_index < 10
    assert not ci_analysis.has_multicollinearity
    assert ci_analysis.severity == "none"
    assert ci_analysis.problematic_dimensions == 0


@pytest.mark.asyncio
async def test_multicollinearity_derived_column(engine, duckdb_conn, async_session, sample_table):
    """Test VIF with derived column (z = x + y)."""
    # Create data with perfect linear combination
    duckdb_conn.execute(f"""
        CREATE TABLE {sample_table.duckdb_path} AS
        SELECT
            x,
            y,
            (x + y)::DOUBLE as z
        FROM (
            SELECT
                random() as x,
                random() as y
            FROM range(100)
        )
    """)

    # Create column metadata
    for idx, col_name in enumerate(["x", "y", "z"]):
        column = create_column(
            f"{sample_table.table_id}-{col_name}",
            sample_table.table_id,
            col_name,
            idx,
            "DOUBLE",
        )
        async_session.add(column)
    await async_session.commit()

    # Compute multicollinearity
    result = await compute_multicollinearity_for_table(sample_table, duckdb_conn, async_session)

    assert result.success
    analysis = result.value

    # z should have very high VIF (perfect linear combination)
    vifs = {vif.column_ref.column_name: vif.vif for vif in analysis.column_vifs}
    assert vifs.get("z", 0) > 50  # Should be extremely high

    assert analysis.num_problematic_columns >= 1
    assert analysis.has_severe_multicollinearity
    assert analysis.overall_severity == "severe"


@pytest.mark.asyncio
async def test_multicollinearity_too_few_columns(engine, duckdb_conn, async_session, sample_table):
    """Test multicollinearity with < 2 numeric columns."""
    # Create table with only one numeric column
    duckdb_conn.execute(f"""
        CREATE TABLE {sample_table.duckdb_path} AS
        SELECT
            x,
            'text_' || CAST(x AS VARCHAR) as text_col
        FROM (SELECT unnest(range(1, 101)) as x)
    """)

    # Create column metadata
    column = create_column(
        f"{sample_table.table_id}-x",
        sample_table.table_id,
        "x",
        0,
        "DOUBLE",
    )
    async_session.add(column)
    await async_session.commit()

    # Compute multicollinearity
    result = await compute_multicollinearity_for_table(sample_table, duckdb_conn, async_session)

    assert result.success
    analysis = result.value

    # Should return "none" severity with no VIFs
    assert analysis.overall_severity == "none"
    assert len(analysis.column_vifs) == 0
    assert analysis.num_problematic_columns == 0


@pytest.mark.asyncio
async def test_multicollinearity_persistence(engine, duckdb_conn, async_session, sample_table):
    """Test that multicollinearity results are persisted to database."""
    # Create data with moderate multicollinearity
    duckdb_conn.execute(f"""
        CREATE TABLE {sample_table.duckdb_path} AS
        SELECT
            x,
            (0.7 * x + 0.3 * random())::DOUBLE as y,
            (0.6 * x + 0.4 * random())::DOUBLE as z
        FROM (SELECT unnest(range(1, 101)) as x)
    """)

    # Create column metadata
    for idx, col_name in enumerate(["x", "y", "z"]):
        column = create_column(
            f"{sample_table.table_id}-{col_name}",
            sample_table.table_id,
            col_name,
            idx,
            "DOUBLE",
        )
        async_session.add(column)
    await async_session.commit()

    # Compute and persist via analyze_correlations (integration test)
    from dataraum_context.profiling.correlation import analyze_correlations

    result = await analyze_correlations(sample_table.table_id, duckdb_conn, async_session)

    assert result.success

    # Verify persistence
    stmt = (
        select(MulticollinearityMetrics)
        .where(MulticollinearityMetrics.table_id == sample_table.table_id)
        .order_by(MulticollinearityMetrics.computed_at.desc())
    )
    db_result = await async_session.execute(stmt)
    stored_metrics = db_result.scalar_one_or_none()

    assert stored_metrics is not None
    assert stored_metrics.analysis_data is not None

    # Verify structured fields
    assert stored_metrics.has_severe_multicollinearity is not None
    assert stored_metrics.num_problematic_columns is not None
    assert stored_metrics.max_vif is not None

    # Verify JSONB contains full analysis
    assert "column_vifs" in stored_metrics.analysis_data
    assert "overall_severity" in stored_metrics.analysis_data


@pytest.mark.asyncio
async def test_multicollinearity_quality_issues(engine, duckdb_conn, async_session, sample_table):
    """Test that quality issues are generated for high multicollinearity."""
    # Create data with severe multicollinearity
    duckdb_conn.execute(f"""
        CREATE TABLE {sample_table.duckdb_path} AS
        SELECT
            x,
            (0.95 * x + 0.05 * random())::DOUBLE as y,
            (0.92 * x + 0.08 * random())::DOUBLE as z,
            (0.90 * x + 0.10 * random())::DOUBLE as w
        FROM (SELECT unnest(range(1, 101)) as x)
    """)

    # Create column metadata
    for idx, col_name in enumerate(["x", "y", "z", "w"]):
        column = create_column(
            f"{sample_table.table_id}-{col_name}",
            sample_table.table_id,
            col_name,
            idx,
            "DOUBLE",
        )
        async_session.add(column)
    await async_session.commit()

    # Compute multicollinearity
    result = await compute_multicollinearity_for_table(sample_table, duckdb_conn, async_session)

    assert result.success
    analysis = result.value

    # Should have quality issues
    assert len(analysis.quality_issues) > 0

    # Check issue types
    issue_types = [issue["issue_type"] for issue in analysis.quality_issues]
    assert "high_multicollinearity" in issue_types or "table_multicollinearity" in issue_types

    # Verify issue structure
    for issue in analysis.quality_issues:
        assert "issue_type" in issue
        assert "severity" in issue
        assert "description" in issue
        assert "evidence" in issue


@pytest.mark.asyncio
async def test_multicollinearity_tolerance_calculation(
    engine, duckdb_conn, async_session, sample_table
):
    """Test that Tolerance (1/VIF) is correctly calculated."""
    # Create data with known multicollinearity
    duckdb_conn.execute(f"""
        CREATE TABLE {sample_table.duckdb_path} AS
        SELECT
            x,
            (0.8 * x + 0.2 * random())::DOUBLE as y
        FROM (SELECT unnest(range(1, 101)) as x)
    """)

    # Create column metadata
    for idx, col_name in enumerate(["x", "y"]):
        column = create_column(
            f"{sample_table.table_id}-{col_name}",
            sample_table.table_id,
            col_name,
            idx,
            "DOUBLE",
        )
        async_session.add(column)
    await async_session.commit()

    # Compute multicollinearity
    result = await compute_multicollinearity_for_table(sample_table, duckdb_conn, async_session)

    assert result.success
    analysis = result.value

    # Verify Tolerance = 1/VIF relationship
    for vif in analysis.column_vifs:
        expected_tolerance = 1.0 / vif.vif
        assert abs(vif.tolerance - expected_tolerance) < 0.01

        # Verify severity thresholds
        if vif.vif > 10:
            assert vif.has_multicollinearity
            assert vif.severity == "severe"
            assert vif.tolerance < 0.1
        elif vif.vif > 5:
            assert vif.severity == "moderate"
        else:
            assert vif.severity == "none"
            assert not vif.has_multicollinearity


@pytest.mark.asyncio
async def test_multicollinearity_correlated_with_list(
    engine, duckdb_conn, async_session, sample_table
):
    """Test that correlated_with list identifies highly correlated columns."""
    # Create data where y and z are both highly correlated with x
    duckdb_conn.execute(f"""
        CREATE TABLE {sample_table.duckdb_path} AS
        SELECT
            x,
            (0.85 * x + 0.15 * random())::DOUBLE as y,
            (0.80 * x + 0.20 * random())::DOUBLE as z
        FROM (SELECT unnest(range(1, 101)) as x)
    """)

    # Create column metadata
    for idx, col_name in enumerate(["x", "y", "z"]):
        column = create_column(
            f"{sample_table.table_id}-{col_name}",
            sample_table.table_id,
            col_name,
            idx,
            "DOUBLE",
        )
        async_session.add(column)
    await async_session.commit()

    # Compute multicollinearity
    result = await compute_multicollinearity_for_table(sample_table, duckdb_conn, async_session)

    assert result.success
    analysis = result.value

    # Check correlated_with lists
    for vif in analysis.column_vifs:
        if vif.column_ref.column_name in ["y", "z"]:
            # Should have identified correlation with x
            assert len(vif.correlated_with) > 0
            # Verify it contains valid column IDs
            stmt = select(Column).where(Column.column_id.in_(vif.correlated_with))
            db_result = await async_session.execute(stmt)
            correlated_cols = db_result.scalars().all()
            assert len(correlated_cols) > 0


@pytest.mark.asyncio
async def test_vdp_dependency_group_detection(engine, duckdb_conn, async_session, sample_table):
    """Test VDP identifies dependency groups correctly."""
    # Create data: z = x + y (perfect linear combination)
    duckdb_conn.execute(f"""
        CREATE TABLE {sample_table.duckdb_path} AS
        SELECT
            x,
            y,
            (x + y)::DOUBLE as z,
            random()::DOUBLE as w
        FROM (
            SELECT
                random() as x,
                random() as y
            FROM range(100)
        )
    """)

    # Create column metadata
    for idx, col_name in enumerate(["x", "y", "z", "w"]):
        column = create_column(
            f"{sample_table.table_id}-{col_name}",
            sample_table.table_id,
            col_name,
            idx,
            "DOUBLE",
        )
        async_session.add(column)
    await async_session.commit()

    # Compute multicollinearity
    result = await compute_multicollinearity_for_table(sample_table, duckdb_conn, async_session)

    assert result.success
    analysis = result.value

    # Should have detected dependency groups
    assert analysis.condition_index is not None
    assert len(analysis.condition_index.dependency_groups) > 0

    # Find the group containing x, y, z
    xyz_group = None
    for group in analysis.condition_index.dependency_groups:
        # Get column names from IDs
        stmt = select(Column).where(Column.column_id.in_(group.involved_column_ids))
        db_result = await async_session.execute(stmt)
        group_cols = {col.column_name for col in db_result.scalars().all()}

        if {"x", "y", "z"}.issubset(group_cols):
            xyz_group = group
            break

    assert xyz_group is not None, "Should detect dependency group for z = x + y"
    assert xyz_group.severity == "severe"
    assert xyz_group.condition_index > 30  # Should be very high
    assert len(xyz_group.involved_column_ids) >= 2  # At least 2 columns


@pytest.mark.asyncio
async def test_vdp_multiple_dependency_groups(engine, duckdb_conn, async_session, sample_table):
    """Test VDP can identify multiple independent dependency groups."""
    # Create two separate dependency groups:
    # Group 1: total = a + b
    # Group 2: ratio = c / d
    duckdb_conn.execute(f"""
        CREATE TABLE {sample_table.duckdb_path} AS
        SELECT
            a,
            b,
            (a + b)::DOUBLE as total,
            c,
            d,
            (c / NULLIF(d, 0))::DOUBLE as ratio
        FROM (
            SELECT
                random() * 100 as a,
                random() * 100 as b,
                random() * 100 as c,
                random() * 100 + 1 as d
            FROM range(100)
        )
    """)

    # Create column metadata
    for idx, col_name in enumerate(["a", "b", "total", "c", "d", "ratio"]):
        column = create_column(
            f"{sample_table.table_id}-{col_name}",
            sample_table.table_id,
            col_name,
            idx,
            "DOUBLE",
        )
        async_session.add(column)
    await async_session.commit()

    # Compute multicollinearity
    result = await compute_multicollinearity_for_table(sample_table, duckdb_conn, async_session)

    assert result.success
    analysis = result.value

    # Should have dependency groups
    assert analysis.condition_index is not None
    assert len(analysis.condition_index.dependency_groups) >= 1

    # Verify each group has interpretation
    for group in analysis.condition_index.dependency_groups:
        assert group.interpretation is not None
        assert len(group.interpretation) > 0


@pytest.mark.asyncio
async def test_vdp_threshold_sensitivity(engine, duckdb_conn, async_session, sample_table):
    """Test VDP only includes columns above threshold."""
    # Create data where only 2 of 4 columns are highly involved
    duckdb_conn.execute(f"""
        CREATE TABLE {sample_table.duckdb_path} AS
        SELECT
            x,
            (0.99 * x + 0.01 * random())::DOUBLE as y,
            (0.3 * x + 0.7 * random())::DOUBLE as z,
            random()::DOUBLE as w
        FROM (SELECT unnest(range(1, 101)) as x)
    """)

    # Create column metadata
    for idx, col_name in enumerate(["x", "y", "z", "w"]):
        column = create_column(
            f"{sample_table.table_id}-{col_name}",
            sample_table.table_id,
            col_name,
            idx,
            "DOUBLE",
        )
        async_session.add(column)
    await async_session.commit()

    # Compute multicollinearity
    result = await compute_multicollinearity_for_table(sample_table, duckdb_conn, async_session)

    assert result.success
    analysis = result.value

    # If dependency groups exist, they should be selective
    if analysis.condition_index and analysis.condition_index.dependency_groups:
        for group in analysis.condition_index.dependency_groups:
            # Each VDP should be > 0.5 (Belsley threshold)
            for vdp in group.variance_proportions:
                assert vdp > 0.5, f"VDP {vdp} should be above Belsley threshold"


@pytest.mark.asyncio
async def test_vdp_formatting_in_context(engine, duckdb_conn, async_session, sample_table):
    """Test dependency groups are properly formatted for LLM context."""
    # Create simple dependency: z = x + y
    duckdb_conn.execute(f"""
        CREATE TABLE {sample_table.duckdb_path} AS
        SELECT
            x,
            y,
            (x + y)::DOUBLE as z
        FROM (
            SELECT
                random() as x,
                random() as y
            FROM range(100)
        )
    """)

    # Create column metadata
    for idx, col_name in enumerate(["x", "y", "z"]):
        column = create_column(
            f"{sample_table.table_id}-{col_name}",
            sample_table.table_id,
            col_name,
            idx,
            "DOUBLE",
        )
        async_session.add(column)
    await async_session.commit()

    # Compute multicollinearity
    result = await compute_multicollinearity_for_table(sample_table, duckdb_conn, async_session)
    assert result.success
    analysis = result.value

    # Format for LLM
    from dataraum_context.quality.formatting import format_multicollinearity_for_llm

    formatted = format_multicollinearity_for_llm(analysis)

    # Check dependency groups are in output if they exist
    if analysis.condition_index and analysis.condition_index.dependency_groups:
        assert "table_level" in formatted["multicollinearity_assessment"]
        table_level = formatted["multicollinearity_assessment"]["table_level"]

        if "dependency_groups" in table_level:
            dep_groups = table_level["dependency_groups"]
            assert isinstance(dep_groups, list)
            assert len(dep_groups) > 0

            # Verify structure of each group
            for group in dep_groups:
                assert "group_id" in group
                assert "severity" in group
                assert "columns" in group
                assert "interpretation" in group
                assert "recommendation" in group
                assert isinstance(group["columns"], list)
