"""Tests for entropy phase."""

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.pipeline.base import PhaseContext, PhaseStatus
from dataraum_context.pipeline.phases import EntropyPhase
from dataraum_context.storage import Column, Source, Table

if TYPE_CHECKING:
    import duckdb


class TestEntropyPhase:
    """Tests for EntropyPhase."""

    def test_phase_properties(self):
        phase = EntropyPhase()
        assert phase.name == "entropy"
        assert phase.description == "Entropy detection across all dimensions"
        assert phase.dependencies == ["statistics", "semantic", "relationships", "correlations"]
        assert phase.outputs == ["entropy_profiles", "compound_risks"]
        assert phase.is_llm_phase is False

    @pytest.mark.asyncio
    async def test_skip_when_no_typed_tables(
        self, async_session: AsyncSession, duckdb_conn: duckdb.DuckDBPyConnection
    ):
        """Test skip when no typed tables exist."""
        phase = EntropyPhase()
        source_id = str(uuid4())

        ctx = PhaseContext(
            session=async_session,
            duckdb_conn=duckdb_conn,
            source_id=source_id,
            config={},
        )

        skip_reason = await phase.should_skip(ctx)
        assert skip_reason is not None
        assert "No typed tables" in skip_reason

    @pytest.mark.asyncio
    async def test_fails_when_no_typed_tables(
        self, async_session: AsyncSession, duckdb_conn: duckdb.DuckDBPyConnection
    ):
        """Test failure when run without typed tables."""
        phase = EntropyPhase()
        source_id = str(uuid4())

        ctx = PhaseContext(
            session=async_session,
            duckdb_conn=duckdb_conn,
            source_id=source_id,
            config={},
        )

        result = await phase.run(ctx)

        assert result.status == PhaseStatus.FAILED
        assert "No typed tables" in (result.error or "")

    @pytest.mark.asyncio
    async def test_skip_when_no_columns(
        self, async_session: AsyncSession, duckdb_conn: duckdb.DuckDBPyConnection
    ):
        """Test skip when typed tables have no columns."""
        phase = EntropyPhase()
        source_id = str(uuid4())

        # Create a source with a typed table but no columns
        source = Source(
            source_id=source_id,
            name="test_source",
            source_type="csv",
        )
        async_session.add(source)

        table = Table(
            table_id=str(uuid4()),
            source_id=source_id,
            table_name="test_table",
            layer="typed",
            duckdb_path="typed_test_table",
            row_count=10,
        )
        async_session.add(table)
        await async_session.commit()

        ctx = PhaseContext(
            session=async_session,
            duckdb_conn=duckdb_conn,
            source_id=source_id,
            config={},
        )

        skip_reason = await phase.should_skip(ctx)
        assert skip_reason is not None
        assert "No columns" in skip_reason

    @pytest.mark.asyncio
    async def test_does_not_skip_with_columns(
        self, async_session: AsyncSession, duckdb_conn: duckdb.DuckDBPyConnection
    ):
        """Test does not skip when columns exist without entropy profiles."""
        phase = EntropyPhase()
        source_id = str(uuid4())
        table_id = str(uuid4())

        # Create a source with a typed table and columns
        source = Source(
            source_id=source_id,
            name="test_source",
            source_type="csv",
        )
        async_session.add(source)

        table = Table(
            table_id=table_id,
            source_id=source_id,
            table_name="test_table",
            layer="typed",
            duckdb_path="typed_test_table",
            row_count=10,
        )
        async_session.add(table)

        column = Column(
            column_id=str(uuid4()),
            table_id=table_id,
            column_name="test_column",
            column_position=0,
            raw_type="VARCHAR",
        )
        async_session.add(column)
        await async_session.commit()

        ctx = PhaseContext(
            session=async_session,
            duckdb_conn=duckdb_conn,
            source_id=source_id,
            config={},
        )

        skip_reason = await phase.should_skip(ctx)
        # Should not skip - columns need entropy analysis
        assert skip_reason is None

    @pytest.mark.asyncio
    async def test_skip_when_all_have_entropy(
        self, async_session: AsyncSession, duckdb_conn: duckdb.DuckDBPyConnection
    ):
        """Test skip when all columns already have entropy profiles."""
        from dataraum_context.entropy.db_models import EntropyObjectRecord

        phase = EntropyPhase()
        source_id = str(uuid4())
        table_id = str(uuid4())
        column_id = str(uuid4())

        # Create a source with a typed table and columns
        source = Source(
            source_id=source_id,
            name="test_source",
            source_type="csv",
        )
        async_session.add(source)

        table = Table(
            table_id=table_id,
            source_id=source_id,
            table_name="test_table",
            layer="typed",
            duckdb_path="typed_test_table",
            row_count=10,
        )
        async_session.add(table)

        column = Column(
            column_id=column_id,
            table_id=table_id,
            column_name="test_column",
            column_position=0,
            raw_type="VARCHAR",
        )
        async_session.add(column)
        await async_session.commit()

        # Add entropy record for the column (after parent records exist)
        entropy_record = EntropyObjectRecord(
            source_id=source_id,
            table_id=table_id,
            column_id=column_id,
            target="column:test_table.test_column",
            layer="structural",
            dimension="schema",
            sub_dimension="naming",
            score=0.3,
            detector_id="test",
        )
        async_session.add(entropy_record)
        await async_session.commit()

        ctx = PhaseContext(
            session=async_session,
            duckdb_conn=duckdb_conn,
            source_id=source_id,
            config={},
        )

        skip_reason = await phase.should_skip(ctx)
        assert skip_reason is not None
        assert "already have entropy" in skip_reason
