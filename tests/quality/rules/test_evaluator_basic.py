"""Tests for basic rule evaluators (not_null, unique, type-based rules).

Tests cover:
- not_null rule (role-based for keys)
- unique rule (role-based for keys)
- not_nan rule (type-based for DOUBLE)
- not_inf rule (type-based for DOUBLE)
- numeric_type rule (role-based for measures)
"""

from datetime import UTC, datetime
from uuid import uuid4

import duckdb
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from dataraum_context.quality.rules import (
    RuleEvaluator,
    load_rules_config,
)
from dataraum_context.storage.models_v2.base import Base
from dataraum_context.storage.models_v2.core import Column, Table
from dataraum_context.storage.models_v2.semantic_context import SemanticAnnotation


@pytest.fixture
def duckdb_conn():
    """Create an in-memory DuckDB connection."""
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
async def async_session():
    """Create an in-memory SQLite async session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_maker() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def test_table_not_null(duckdb_conn, async_session):
    """Create a test table with null values in a key column.

    Schema:
    - id (INTEGER, role=key) - has 2 nulls out of 10 rows
    - name (VARCHAR) - all non-null
    """
    # Create DuckDB table
    duckdb_conn.execute("""
        CREATE TABLE test_not_null (
            id INTEGER,
            name VARCHAR
        )
    """)

    # Insert test data: 2 nulls in id column
    duckdb_conn.execute("""
        INSERT INTO test_not_null VALUES
        (1, 'Alice'),
        (2, 'Bob'),
        (NULL, 'Charlie'),
        (4, 'Diana'),
        (5, 'Eve'),
        (6, 'Frank'),
        (NULL, 'Grace'),
        (8, 'Hank'),
        (9, 'Ivy'),
        (10, 'Jack')
    """)

    # Create metadata
    table_id = str(uuid4())
    col_id_id = str(uuid4())
    col_name_id = str(uuid4())

    # Table metadata
    table = Table(
        table_id=table_id,
        source_id=str(uuid4()),
        table_name="test_not_null",
        layer="typed",
        duckdb_path="test_not_null",
        row_count=10,
        created_at=datetime.now(UTC),
    )
    async_session.add(table)

    # Column metadata
    col_id = Column(
        column_id=col_id_id,
        table_id=table_id,
        column_name="id",
        column_position=0,
        raw_type="INTEGER",
        resolved_type="INTEGER",
    )
    col_name = Column(
        column_id=col_name_id,
        table_id=table_id,
        column_name="name",
        column_position=1,
        raw_type="VARCHAR",
        resolved_type="VARCHAR",
    )
    async_session.add(col_id)
    async_session.add(col_name)

    # Semantic annotation: id is a key
    annotation = SemanticAnnotation(
        annotation_id=str(uuid4()),
        column_id=col_id_id,
        semantic_role="key",
        confidence=1.0,
        annotation_source="test",
        annotated_at=datetime.now(UTC),
    )
    async_session.add(annotation)

    await async_session.commit()

    return table_id


@pytest.fixture
async def test_table_unique(duckdb_conn, async_session):
    """Create a test table with duplicate values in a key column.

    Schema:
    - id (INTEGER, role=key) - has 2 duplicates (value 5 appears twice)
    """
    # Create DuckDB table
    duckdb_conn.execute("""
        CREATE TABLE test_unique (
            id INTEGER,
            value VARCHAR
        )
    """)

    # Insert test data: value 5 appears twice
    duckdb_conn.execute("""
        INSERT INTO test_unique VALUES
        (1, 'A'),
        (2, 'B'),
        (3, 'C'),
        (4, 'D'),
        (5, 'E'),
        (5, 'F'),
        (7, 'G'),
        (8, 'H'),
        (9, 'I'),
        (10, 'J')
    """)

    # Create metadata
    table_id = str(uuid4())
    col_id_id = str(uuid4())

    table = Table(
        table_id=table_id,
        source_id=str(uuid4()),
        table_name="test_unique",
        layer="typed",
        duckdb_path="test_unique",
        row_count=10,
        created_at=datetime.now(UTC),
    )
    async_session.add(table)

    col_id = Column(
        column_id=col_id_id,
        table_id=table_id,
        column_name="id",
        column_position=0,
        raw_type="INTEGER",
        resolved_type="INTEGER",
    )
    async_session.add(col_id)

    annotation = SemanticAnnotation(
        annotation_id=str(uuid4()),
        column_id=col_id_id,
        semantic_role="key",
        confidence=1.0,
        annotation_source="test",
        annotated_at=datetime.now(UTC),
    )
    async_session.add(annotation)

    await async_session.commit()

    return table_id


@pytest.fixture
async def test_table_nan_inf(duckdb_conn, async_session):
    """Create a test table with NaN and Inf values in DOUBLE column.

    Schema:
    - id (INTEGER)
    - value (DOUBLE) - has 2 NaN and 1 Inf
    """
    # Create DuckDB table
    duckdb_conn.execute("""
        CREATE TABLE test_nan_inf (
            id INTEGER,
            value DOUBLE
        )
    """)

    # Insert test data: 2 NaN, 1 Inf
    duckdb_conn.execute("""
        INSERT INTO test_nan_inf VALUES
        (1, 1.0),
        (2, 2.0),
        (3, 'NaN'::DOUBLE),
        (4, 4.0),
        (5, 'NaN'::DOUBLE),
        (6, 6.0),
        (7, 'Infinity'::DOUBLE),
        (8, 8.0),
        (9, 9.0),
        (10, 10.0)
    """)

    # Create metadata
    table_id = str(uuid4())
    col_value_id = str(uuid4())

    table = Table(
        table_id=table_id,
        source_id=str(uuid4()),
        table_name="test_nan_inf",
        layer="typed",
        duckdb_path="test_nan_inf",
        row_count=10,
        created_at=datetime.now(UTC),
    )
    async_session.add(table)

    col_value = Column(
        column_id=col_value_id,
        table_id=table_id,
        column_name="value",
        column_position=1,
        raw_type="DOUBLE",
        resolved_type="DOUBLE",
    )
    async_session.add(col_value)

    await async_session.commit()

    return table_id


# =============================================================================
# Tests for NOT_NULL Rule
# =============================================================================


@pytest.mark.asyncio
async def test_not_null_rule_with_failures(test_table_not_null, duckdb_conn, async_session):
    """Test not_null rule detects null values in key column."""
    # Load default rules
    rules_result = load_rules_config("default")
    assert rules_result.success
    rules_config = rules_result.unwrap()

    # Create evaluator and run
    evaluator = RuleEvaluator(duckdb_conn, async_session)
    result = await evaluator.evaluate_table(test_table_not_null, rules_config)

    # Verify evaluation succeeded
    assert result.success, f"Evaluation failed: {result.error}"

    table_results = result.unwrap()

    # Should have evaluated not_null and unique rules for the key column
    assert table_results.total_rules_evaluated >= 1, "Should have evaluated at least not_null rule"

    # Find the not_null rule result
    not_null_result = None
    for rule_result in table_results.rule_results:
        if rule_result.rule_name == "not_null":
            not_null_result = rule_result
            break

    assert not_null_result is not None, "Should have not_null rule result"

    # Verify counts: 10 total, 8 passed, 2 failed
    assert not_null_result.total_records == 10
    assert not_null_result.passed_records == 8
    assert not_null_result.failed_records == 2
    assert not_null_result.pass_rate == 0.8

    # Verify severity
    assert not_null_result.severity == "error"

    # Verify it has failures
    assert not_null_result.has_failures
    assert abs(not_null_result.failure_rate - 0.2) < 0.001  # Floating point tolerance


@pytest.mark.asyncio
async def test_not_null_rule_all_pass(duckdb_conn, async_session):
    """Test not_null rule when all values are non-null."""
    # Create table with no nulls
    duckdb_conn.execute("""
        CREATE TABLE test_all_pass (
            id INTEGER
        )
    """)
    duckdb_conn.execute("""
        INSERT INTO test_all_pass VALUES (1), (2), (3), (4), (5)
    """)

    # Create metadata
    table_id = str(uuid4())
    col_id = str(uuid4())

    table = Table(
        table_id=table_id,
        source_id=str(uuid4()),
        table_name="test_all_pass",
        layer="typed",
        duckdb_path="test_all_pass",
        row_count=5,
        created_at=datetime.now(UTC),
    )
    async_session.add(table)

    col = Column(
        column_id=col_id,
        table_id=table_id,
        column_name="id",
        column_position=0,
        raw_type="INTEGER",
        resolved_type="INTEGER",
    )
    async_session.add(col)

    annotation = SemanticAnnotation(
        annotation_id=str(uuid4()),
        column_id=col_id,
        semantic_role="key",
        confidence=1.0,
        annotation_source="test",
        annotated_at=datetime.now(UTC),
    )
    async_session.add(annotation)

    await async_session.commit()

    # Load rules and evaluate
    rules_config = load_rules_config("default").unwrap()
    evaluator = RuleEvaluator(duckdb_conn, async_session)
    result = await evaluator.evaluate_table(table_id, rules_config)

    assert result.success
    table_results = result.unwrap()

    # Find not_null result
    not_null_result = next(
        (r for r in table_results.rule_results if r.rule_name == "not_null"), None
    )

    assert not_null_result is not None
    assert not_null_result.total_records == 5
    assert not_null_result.passed_records == 5
    assert not_null_result.failed_records == 0
    assert not_null_result.pass_rate == 1.0
    assert not not_null_result.has_failures


# =============================================================================
# Tests for UNIQUE Rule
# =============================================================================


@pytest.mark.asyncio
async def test_unique_rule_with_duplicates(test_table_unique, duckdb_conn, async_session):
    """Test unique rule detects duplicate values."""
    rules_config = load_rules_config("default").unwrap()
    evaluator = RuleEvaluator(duckdb_conn, async_session)
    result = await evaluator.evaluate_table(test_table_unique, rules_config)

    assert result.success
    table_results = result.unwrap()

    # Find unique rule result
    unique_result = next((r for r in table_results.rule_results if r.rule_name == "unique"), None)

    assert unique_result is not None

    # 10 total rows, 9 distinct values (5 appears twice), so 1 "failed" (duplicate count)
    assert unique_result.total_records == 10
    assert unique_result.failed_records == 1  # COUNT(*) - COUNT(DISTINCT)
    assert unique_result.passed_records == 9
    assert unique_result.pass_rate == 0.9


# =============================================================================
# Tests for NOT_NAN and NOT_INF Rules
# =============================================================================


@pytest.mark.asyncio
async def test_not_nan_rule(test_table_nan_inf, duckdb_conn, async_session):
    """Test not_nan rule detects NaN values in DOUBLE columns."""
    rules_config = load_rules_config("default").unwrap()
    evaluator = RuleEvaluator(duckdb_conn, async_session)
    result = await evaluator.evaluate_table(test_table_nan_inf, rules_config)

    assert result.success
    table_results = result.unwrap()

    # Find not_nan rule result
    not_nan_result = next((r for r in table_results.rule_results if r.rule_name == "not_nan"), None)

    assert not_nan_result is not None

    # 10 total, 2 NaN, 8 passed (including the Inf, which is not NaN)
    assert not_nan_result.total_records == 10
    assert not_nan_result.failed_records == 2
    assert not_nan_result.passed_records == 8
    assert not_nan_result.pass_rate == 0.8


@pytest.mark.asyncio
async def test_not_inf_rule(test_table_nan_inf, duckdb_conn, async_session):
    """Test not_inf rule detects Infinity values in DOUBLE columns."""
    rules_config = load_rules_config("default").unwrap()
    evaluator = RuleEvaluator(duckdb_conn, async_session)
    result = await evaluator.evaluate_table(test_table_nan_inf, rules_config)

    assert result.success
    table_results = result.unwrap()

    # Find not_inf rule result
    not_inf_result = next((r for r in table_results.rule_results if r.rule_name == "not_inf"), None)

    assert not_inf_result is not None

    # 10 total, 1 Inf, 9 passed (NaN is not Inf)
    assert not_inf_result.total_records == 10
    assert not_inf_result.failed_records == 1
    assert not_inf_result.passed_records == 9
    assert not_inf_result.pass_rate == 0.9


# =============================================================================
# Tests for Table Results Aggregation
# =============================================================================


@pytest.mark.asyncio
async def test_table_results_aggregation(test_table_not_null, duckdb_conn, async_session):
    """Test that TableRuleResults correctly aggregates multiple rules."""
    rules_config = load_rules_config("default").unwrap()
    evaluator = RuleEvaluator(duckdb_conn, async_session)
    result = await evaluator.evaluate_table(test_table_not_null, rules_config)

    assert result.success
    table_results = result.unwrap()

    # Verify table info
    assert table_results.table_id == test_table_not_null
    assert table_results.table_name == "test_not_null"

    # Should have evaluated at least 2 rules (not_null, unique for key column)
    assert table_results.total_rules_evaluated >= 2

    # Should have some failures (from not_null rule)
    assert table_results.rules_failed > 0

    # Error count should be > 0 (not_null is severity="error")
    assert table_results.error_count > 0

    # Total violations should match sum of all failed_records
    expected_violations = sum(r.failed_records for r in table_results.rule_results)
    assert table_results.total_violations == expected_violations

    # Average pass rate should be between 0 and 1
    assert 0 <= table_results.avg_pass_rate <= 1

    # Should have evaluation time
    assert table_results.evaluation_time_ms > 0

    # Should have timestamp
    assert table_results.evaluated_at is not None


@pytest.mark.asyncio
async def test_violation_samples_collected(test_table_not_null, duckdb_conn, async_session):
    """Test that violation samples are collected for failed rules."""
    rules_config = load_rules_config("default").unwrap()
    evaluator = RuleEvaluator(duckdb_conn, async_session)
    result = await evaluator.evaluate_table(test_table_not_null, rules_config)

    assert result.success
    table_results = result.unwrap()

    # Find not_null rule
    not_null_result = next(
        (r for r in table_results.rule_results if r.rule_name == "not_null"), None
    )

    assert not_null_result is not None

    # Should have collected samples (up to 10)
    assert len(not_null_result.failure_samples) > 0
    assert len(not_null_result.failure_samples) <= 10

    # Each sample should have required fields
    for sample in not_null_result.failure_samples:
        assert sample.column_name == "id"
        assert sample.row_number is not None  # Should have row number
        assert sample.expected is not None  # Should have description
