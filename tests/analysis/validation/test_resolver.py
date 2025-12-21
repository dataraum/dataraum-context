"""Tests for validation schema resolver."""

import pytest

from dataraum_context.analysis.validation.resolver import (
    format_schema_for_prompt,
    get_table_schema_for_llm,
)
from dataraum_context.storage import Column, Source, Table


@pytest.fixture
async def table_with_columns(async_session):
    """Create a test table with columns and semantic annotations."""
    from dataraum_context.analysis.semantic.db_models import (
        SemanticAnnotation as SemanticAnnotationDB,
    )

    # Create source and table
    source = Source(name="test_source", source_type="csv")
    async_session.add(source)
    await async_session.flush()

    table = Table(
        source_id=source.source_id,
        table_name="transactions",
        layer="typed",
        row_count=1000,
        duckdb_path="typed_transactions",
    )
    async_session.add(table)
    await async_session.flush()

    # Create columns
    col1 = Column(
        table_id=table.table_id,
        column_name="transaction_id",
        column_position=0,
        raw_type="VARCHAR",
        resolved_type="VARCHAR",
    )
    col2 = Column(
        table_id=table.table_id,
        column_name="amount",
        column_position=1,
        raw_type="VARCHAR",
        resolved_type="DECIMAL(18,2)",
    )
    col3 = Column(
        table_id=table.table_id,
        column_name="account_type",
        column_position=2,
        raw_type="VARCHAR",
        resolved_type="VARCHAR",
    )
    async_session.add_all([col1, col2, col3])
    await async_session.flush()

    # Add semantic annotation to amount column
    annotation = SemanticAnnotationDB(
        column_id=col2.column_id,
        semantic_role="measure",
        entity_type="amount",
        business_name="Transaction Amount",
        business_domain="finance",
    )
    async_session.add(annotation)
    await async_session.commit()

    return table


@pytest.mark.asyncio
async def test_get_table_schema_for_llm(async_session, table_with_columns):
    """Test fetching table schema for LLM context."""
    table = table_with_columns

    schema = await get_table_schema_for_llm(async_session, table.table_id)

    assert "error" not in schema
    assert schema["table_name"] == "transactions"
    assert schema["table_id"] == table.table_id
    assert schema["duckdb_path"] == "typed_transactions"
    assert len(schema["columns"]) == 3

    # Check column info
    col_names = [c["column_name"] for c in schema["columns"]]
    assert "transaction_id" in col_names
    assert "amount" in col_names
    assert "account_type" in col_names

    # Check semantic annotation is included
    amount_col = next(c for c in schema["columns"] if c["column_name"] == "amount")
    assert "semantic" in amount_col
    assert amount_col["semantic"]["role"] == "measure"
    assert amount_col["semantic"]["entity_type"] == "amount"
    assert amount_col["semantic"]["business_name"] == "Transaction Amount"
    assert amount_col["semantic"]["domain"] == "finance"


@pytest.mark.asyncio
async def test_get_table_schema_for_llm_nonexistent_table(async_session):
    """Test that nonexistent table returns error."""
    schema = await get_table_schema_for_llm(async_session, "nonexistent-id")

    assert "error" in schema
    assert "not found" in schema["error"]


@pytest.mark.asyncio
async def test_get_table_schema_for_llm_table_without_duckdb_path(async_session):
    """Test that table without DuckDB path returns error."""
    source = Source(name="test_source", source_type="csv")
    async_session.add(source)
    await async_session.flush()

    table = Table(
        source_id=source.source_id,
        table_name="no_duckdb",
        layer="raw",
        row_count=100,
        duckdb_path=None,  # No DuckDB path
    )
    async_session.add(table)
    await async_session.commit()

    schema = await get_table_schema_for_llm(async_session, table.table_id)

    assert "error" in schema
    assert "no DuckDB path" in schema["error"]


class TestFormatSchemaForPrompt:
    """Tests for formatting schema as prompt text."""

    def test_format_basic_schema(self):
        """Test formatting a basic schema."""
        schema = {
            "table_name": "orders",
            "duckdb_path": "typed_orders",
            "columns": [
                {"column_name": "order_id", "data_type": "VARCHAR"},
                {"column_name": "total", "data_type": "DECIMAL(18,2)"},
            ],
        }

        result = format_schema_for_prompt(schema)

        assert "Table: orders" in result
        assert "DuckDB Path: typed_orders" in result
        assert "order_id (VARCHAR)" in result
        assert "total (DECIMAL(18,2))" in result

    def test_format_schema_with_semantic_annotations(self):
        """Test formatting schema with semantic annotations."""
        schema = {
            "table_name": "accounts",
            "duckdb_path": "typed_accounts",
            "columns": [
                {
                    "column_name": "balance",
                    "data_type": "DECIMAL(18,2)",
                    "semantic": {
                        "role": "measure",
                        "entity_type": "amount",
                        "business_name": "Account Balance",
                        "domain": "finance",
                    },
                },
            ],
        }

        result = format_schema_for_prompt(schema)

        assert "balance (DECIMAL(18,2))" in result
        assert "entity: amount" in result
        assert "role: measure" in result
        assert "business_name: Account Balance" in result
        assert "domain: finance" in result

    def test_format_schema_with_error(self):
        """Test formatting an error schema."""
        schema = {"error": "Table not found"}

        result = format_schema_for_prompt(schema)

        assert "Error: Table not found" in result

    def test_format_schema_with_partial_annotations(self):
        """Test formatting with only some annotation fields."""
        schema = {
            "table_name": "test",
            "duckdb_path": "test_path",
            "columns": [
                {
                    "column_name": "col1",
                    "data_type": "VARCHAR",
                    "semantic": {
                        "role": None,
                        "entity_type": "customer_id",
                        "business_name": None,
                        "domain": None,
                    },
                },
            ],
        }

        result = format_schema_for_prompt(schema)

        assert "entity: customer_id" in result
        # Should not include None values
        assert "role: None" not in result
