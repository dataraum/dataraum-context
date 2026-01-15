"""Tests for entropy interpretation phase."""

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.pipeline.base import PhaseContext, PhaseStatus
from dataraum_context.pipeline.phases import EntropyInterpretationPhase
from dataraum_context.storage import Column, Source, Table

if TYPE_CHECKING:
    import duckdb


class TestEntropyInterpretationPhase:
    """Tests for EntropyInterpretationPhase."""

    def test_phase_properties(self):
        phase = EntropyInterpretationPhase()
        assert phase.name == "entropy_interpretation"
        assert phase.description == "LLM interpretation of entropy"
        assert phase.dependencies == ["entropy"]
        assert phase.outputs == ["interpretations", "assumptions", "resolution_actions"]
        assert phase.is_llm_phase is True

    @pytest.mark.asyncio
    async def test_skip_when_no_typed_tables(
        self, async_session: AsyncSession, duckdb_conn: duckdb.DuckDBPyConnection
    ):
        """Test skip when no typed tables exist."""
        phase = EntropyInterpretationPhase()
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
        phase = EntropyInterpretationPhase()
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
    async def test_skip_when_no_entropy_records(
        self, async_session: AsyncSession, duckdb_conn: duckdb.DuckDBPyConnection
    ):
        """Test skip when no entropy records exist."""
        phase = EntropyInterpretationPhase()
        source_id = str(uuid4())

        # Create a source with a typed table but no entropy records
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
        assert "No entropy records" in skip_reason

    @pytest.mark.asyncio
    async def test_does_not_skip_with_entropy_records(
        self, async_session: AsyncSession, duckdb_conn: duckdb.DuckDBPyConnection
    ):
        """Test does not skip when entropy records exist."""
        from dataraum_context.entropy.db_models import EntropyObjectRecord

        phase = EntropyInterpretationPhase()
        source_id = str(uuid4())
        table_id = str(uuid4())
        column_id = str(uuid4())

        # Create a source with entropy records
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

        # Add entropy record (after parent records exist)
        entropy_record = EntropyObjectRecord(
            source_id=source_id,
            table_id=table_id,
            column_id=column_id,
            target="column:test_table.test_column",
            layer="structural",
            dimension="schema",
            sub_dimension="naming",
            score=0.5,
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
        # Should not skip - entropy records need interpretation
        assert skip_reason is None

    @pytest.mark.asyncio
    async def test_success_no_entropy_records(
        self, async_session: AsyncSession, duckdb_conn: duckdb.DuckDBPyConnection
    ):
        """Test returns success when no entropy records to interpret."""
        phase = EntropyInterpretationPhase()
        source_id = str(uuid4())
        table_id = str(uuid4())
        column_id = str(uuid4())

        # Create a source with columns but no entropy records
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

        ctx = PhaseContext(
            session=async_session,
            duckdb_conn=duckdb_conn,
            source_id=source_id,
            config={},
        )

        result = await phase.run(ctx)

        assert result.status == PhaseStatus.COMPLETED
        assert result.outputs["interpretations"] == 0
        assert "No entropy records" in result.outputs.get("message", "")
