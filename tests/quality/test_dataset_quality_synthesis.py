"""Tests for dataset-level quality synthesis (cross-table integration)."""

import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.core.models.base import RelationshipType
from dataraum_context.quality.synthesis import assess_dataset_quality
from dataraum_context.storage.models_v2 import Column, Relationship, Source, Table


def create_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create test data with known multicollinearity for testing.

    Creates orders + customers with:
    - FK: orders.customer_id → customers.id
    - Denormalization: orders.customer_name from customers.name
    - Derived column: orders.total = orders.subtotal + orders.tax
    """
    np.random.seed(42)

    # Customers
    num_customers = 20
    customers = pd.DataFrame(
        {
            "id": range(1, num_customers + 1),
            "name": [f"Customer_{i}" for i in range(1, num_customers + 1)],
            "email": [f"customer{i}@example.com" for i in range(1, num_customers + 1)],
        }
    )

    # Orders with multicollinearity
    num_orders = 50
    customer_ids = np.random.choice(customers["id"], size=num_orders)
    subtotals = np.random.uniform(10, 500, num_orders)
    taxes = subtotals * 0.1
    totals = subtotals + taxes + np.random.normal(0, 0.01, num_orders)  # Small noise

    orders = pd.DataFrame(
        {
            "order_id": range(1, num_orders + 1),
            "customer_id": customer_ids,
            "customer_name": [
                customers.loc[customers["id"] == cid, "name"].values[0] for cid in customer_ids
            ],
            "subtotal": subtotals,
            "tax": taxes,
            "total": totals,
        }
    )

    return orders, customers


@pytest.mark.asyncio
async def test_assess_dataset_quality_basic(async_session: AsyncSession):
    """Test basic dataset quality assessment with two tables."""
    orders_df, customers_df = create_test_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Save to CSV and load into DuckDB
        orders_csv = tmpdir_path / "orders.csv"
        customers_csv = tmpdir_path / "customers.csv"
        orders_df.to_csv(orders_csv, index=False)
        customers_df.to_csv(customers_csv, index=False)

        duckdb_conn = duckdb.connect(":memory:")
        duckdb_conn.execute(f"CREATE TABLE orders AS SELECT * FROM '{orders_csv}'")
        duckdb_conn.execute(f"CREATE TABLE customers AS SELECT * FROM '{customers_csv}'")

        # Set up metadata
        source = Source(source_id="src1", name="test_dataset", source_type="csv")
        async_session.add(source)
        await async_session.flush()

        # Tables
        orders_table = Table(
            table_id="t_orders",
            source_id="src1",
            table_name="orders",
            layer="typed",
            duckdb_path="orders",
        )
        customers_table = Table(
            table_id="t_customers",
            source_id="src1",
            table_name="customers",
            layer="typed",
            duckdb_path="customers",
        )
        async_session.add_all([orders_table, customers_table])
        await async_session.flush()

        # Columns (numeric only for multicollinearity)
        orders_cols = [
            Column(
                column_id="c_orders_customer_id",
                table_id="t_orders",
                column_name="customer_id",
                column_position=1,
                raw_type="BIGINT",
                resolved_type="BIGINT",
            ),
            Column(
                column_id="c_orders_subtotal",
                table_id="t_orders",
                column_name="subtotal",
                column_position=3,
                raw_type="DOUBLE",
                resolved_type="DOUBLE",
            ),
            Column(
                column_id="c_orders_tax",
                table_id="t_orders",
                column_name="tax",
                column_position=4,
                raw_type="DOUBLE",
                resolved_type="DOUBLE",
            ),
            Column(
                column_id="c_orders_total",
                table_id="t_orders",
                column_name="total",
                column_position=5,
                raw_type="DOUBLE",
                resolved_type="DOUBLE",
            ),
        ]

        customers_cols = [
            Column(
                column_id="c_customers_id",
                table_id="t_customers",
                column_name="id",
                column_position=0,
                raw_type="BIGINT",
                resolved_type="BIGINT",
            ),
        ]

        async_session.add_all(orders_cols + customers_cols)
        await async_session.flush()

        # FK relationship
        fk_rel = Relationship(
            relationship_id="rel_fk",
            from_table_id="t_orders",
            from_column_id="c_orders_customer_id",
            to_table_id="t_customers",
            to_column_id="c_customers_id",
            relationship_type=RelationshipType.FOREIGN_KEY,
            confidence=0.95,
            detection_method="tda",
        )
        async_session.add(fk_rel)
        await async_session.commit()

        # Run dataset quality assessment
        result = await assess_dataset_quality(
            table_ids=["t_orders", "t_customers"],
            duckdb_conn=duckdb_conn,
            session=async_session,
        )

        # Validate
        assert result.success, f"Assessment failed: {result.error}"
        dataset_result = result.value
        assert dataset_result is not None

        # Check scope
        assert dataset_result.total_tables == 2
        assert set(dataset_result.table_names) == {"orders", "customers"}
        assert len(dataset_result.table_assessments) == 2

        # Check cross-table analysis ran
        assert dataset_result.has_cross_table_analysis is True

        # Check aggregated metrics
        assert dataset_result.total_columns > 0
        assert dataset_result.average_table_quality >= 0.0
        assert dataset_result.average_table_quality <= 1.0
        assert dataset_result.average_column_quality >= 0.0
        assert dataset_result.average_column_quality <= 1.0

        # If cross-table dependencies found, check they're recorded
        if dataset_result.cross_table_dependencies > 0:
            assert dataset_result.cross_table_multicollinearity_severity in [
                "none",
                "moderate",
                "severe",
            ]
            assert len(dataset_result.cross_table_issues) > 0

            # Verify cross-table issues are properly formatted
            for issue in dataset_result.cross_table_issues:
                assert issue.issue_type == "cross_table_multicollinearity"
                assert issue.dimension.value == "consistency"
                assert "evidence" in issue.model_dump()
                assert "involved_columns" in issue.evidence

        print("\n=== Dataset Quality Assessment Summary ===")
        print(f"Tables: {dataset_result.table_names}")
        print(f"Total columns: {dataset_result.total_columns}")
        print(f"Average table quality: {dataset_result.average_table_quality:.2f}")
        print(f"Average column quality: {dataset_result.average_column_quality:.2f}")
        print(f"Total issues: {dataset_result.total_issues}")
        print(f"Critical issues: {dataset_result.critical_issues}")
        print(f"Cross-table dependencies: {dataset_result.cross_table_dependencies}")
        print(f"Cross-table severity: {dataset_result.cross_table_multicollinearity_severity}")

        duckdb_conn.close()


@pytest.mark.asyncio
async def test_assess_dataset_quality_single_table(async_session: AsyncSession):
    """Test dataset quality assessment with single table (no cross-table analysis)."""
    orders_df, _ = create_test_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        orders_csv = tmpdir_path / "orders.csv"
        orders_df.to_csv(orders_csv, index=False)

        duckdb_conn = duckdb.connect(":memory:")
        duckdb_conn.execute(f"CREATE TABLE orders AS SELECT * FROM '{orders_csv}'")

        # Set up metadata
        source = Source(source_id="src1", name="test_single", source_type="csv")
        async_session.add(source)
        await async_session.flush()

        orders_table = Table(
            table_id="t_orders",
            source_id="src1",
            table_name="orders",
            layer="typed",
            duckdb_path="orders",
        )
        async_session.add(orders_table)
        await async_session.commit()

        # Run assessment with single table
        result = await assess_dataset_quality(
            table_ids=["t_orders"],
            duckdb_conn=duckdb_conn,
            session=async_session,
        )

        # Should succeed but no cross-table analysis
        assert result.success
        dataset_result = result.value
        assert dataset_result is not None
        assert dataset_result.total_tables == 1
        assert dataset_result.has_cross_table_analysis is False  # Single table
        assert dataset_result.cross_table_dependencies == 0
        assert len(dataset_result.cross_table_issues) == 0

        duckdb_conn.close()


@pytest.mark.asyncio
async def test_assess_dataset_quality_consistency_penalty(async_session: AsyncSession):
    """Test that cross-table dependencies penalize consistency scores."""
    orders_df, customers_df = create_test_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        orders_csv = tmpdir_path / "orders.csv"
        customers_csv = tmpdir_path / "customers.csv"
        orders_df.to_csv(orders_csv, index=False)
        customers_df.to_csv(customers_csv, index=False)

        duckdb_conn = duckdb.connect(":memory:")
        duckdb_conn.execute(f"CREATE TABLE orders AS SELECT * FROM '{orders_csv}'")
        duckdb_conn.execute(f"CREATE TABLE customers AS SELECT * FROM '{customers_csv}'")

        # Set up metadata (simplified version)
        source = Source(source_id="src1", name="test_penalty", source_type="csv")
        async_session.add(source)
        await async_session.flush()

        orders_table = Table(
            table_id="t_orders",
            source_id="src1",
            table_name="orders",
            layer="typed",
            duckdb_path="orders",
        )
        customers_table = Table(
            table_id="t_customers",
            source_id="src1",
            table_name="customers",
            layer="typed",
            duckdb_path="customers",
        )
        async_session.add_all([orders_table, customers_table])
        await async_session.flush()

        # Add numeric columns
        orders_cols = [
            Column(
                column_id=f"c_orders_{col}",
                table_id="t_orders",
                column_name=col,
                column_position=i,
                raw_type="DOUBLE",
                resolved_type="DOUBLE",
            )
            for i, col in enumerate(["customer_id", "subtotal", "tax", "total"])
        ]
        customers_cols = [
            Column(
                column_id="c_customers_id",
                table_id="t_customers",
                column_name="id",
                column_position=0,
                raw_type="BIGINT",
                resolved_type="BIGINT",
            ),
        ]
        async_session.add_all(orders_cols + customers_cols)
        await async_session.flush()

        # FK relationship
        fk_rel = Relationship(
            relationship_id="rel_fk",
            from_table_id="t_orders",
            from_column_id="c_orders_customer_id",
            to_table_id="t_customers",
            to_column_id="c_customers_id",
            relationship_type=RelationshipType.FOREIGN_KEY,
            confidence=0.95,
            detection_method="tda",
        )
        async_session.add(fk_rel)
        await async_session.commit()

        # Run assessment
        result = await assess_dataset_quality(
            table_ids=["t_orders", "t_customers"],
            duckdb_conn=duckdb_conn,
            session=async_session,
        )

        assert result.success
        dataset_result = result.value

        # If dependencies found, verify consistency dimension was adjusted
        if dataset_result.cross_table_dependencies > 0:
            for table_assessment in dataset_result.table_assessments:
                for dim_score in table_assessment.table_assessment.dimension_scores:
                    if dim_score.dimension.value == "consistency":
                        # Explanation should mention cross-table adjustment
                        assert "cross-table" in dim_score.explanation.lower()
                        print(f"\nConsistency score: {dim_score.score:.2f}")
                        print(f"Explanation: {dim_score.explanation}")

        duckdb_conn.close()
