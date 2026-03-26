"""Tests for begin_session MCP tool and session wiring."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

from sqlalchemy.orm import Session

from dataraum.investigation.db_models import InvestigationSession, InvestigationStep
from dataraum.storage import Source


def _id() -> str:
    return str(uuid4())


def _insert_source(session: Session, name: str = "test_source") -> str:
    source_id = _id()
    session.add(Source(source_id=source_id, name=name, source_type="csv"))
    session.flush()
    return source_id


@contextmanager
def _mock_manager(session: Session):
    """Mock get_manager_for_directory to return a manager using the test session."""
    manager = MagicMock()
    manager.session_scope.return_value.__enter__ = lambda _: session
    manager.session_scope.return_value.__exit__ = lambda *_: None

    with patch("dataraum.core.connections.get_manager_for_directory", return_value=manager):
        yield manager


class TestBeginSession:
    def test_creates_session_returns_id(self, session: Session, tmp_path) -> None:
        """begin_session creates an InvestigationSession and returns session_id."""
        source_id = _insert_source(session)

        with (
            _mock_manager(session),
            patch(
                "dataraum.mcp.server._get_pipeline_source",
                return_value=session.get(Source, source_id),
            ),
        ):
            from dataraum.mcp.server import _begin_session

            result = _begin_session(tmp_path, intent="test investigation")

        assert "error" not in result
        assert "session_id" in result
        assert result["sources"] == ["test_source"]
        assert result["has_pipeline_data"] is False
        assert "hint" in result

        # Verify session was actually created in DB
        inv = session.get(InvestigationSession, result["session_id"])
        assert inv is not None
        assert inv.intent == "test investigation"
        assert inv.status == "active"

    def test_has_pipeline_data_when_entropy_exists(self, session: Session, tmp_path) -> None:
        """has_pipeline_data is True when entropy records exist."""
        from dataraum.entropy.db_models import EntropyObjectRecord

        source_id = _insert_source(session)
        session.add(
            EntropyObjectRecord(
                source_id=source_id,
                layer="semantic",
                dimension="business_meaning",
                sub_dimension="naming_clarity",
                target="column:orders.amount",
                score=0.5,
                detector_id="naming_clarity",
            )
        )
        session.flush()

        with (
            _mock_manager(session),
            patch(
                "dataraum.mcp.server._get_pipeline_source",
                return_value=session.get(Source, source_id),
            ),
        ):
            from dataraum.mcp.server import _begin_session

            result = _begin_session(tmp_path, intent="check quality")

        assert result["has_pipeline_data"] is True
        assert "look" in result["hint"]

    def test_no_source_returns_error(self, tmp_path) -> None:
        """When no source exists, returns error."""
        with (
            _mock_manager(MagicMock()),
            patch("dataraum.mcp.server._get_pipeline_source", return_value=None),
        ):
            from dataraum.mcp.server import _begin_session

            result = _begin_session(tmp_path, intent="test")

        assert "error" in result

    def test_with_contract(self, session: Session, tmp_path) -> None:
        """Contract is stored on the investigation session."""
        source_id = _insert_source(session)

        with (
            _mock_manager(session),
            patch(
                "dataraum.mcp.server._get_pipeline_source",
                return_value=session.get(Source, source_id),
            ),
        ):
            from dataraum.mcp.server import _begin_session

            result = _begin_session(
                tmp_path, intent="compliance check", contract="executive_dashboard"
            )

        assert "error" not in result
        inv = session.get(InvestigationSession, result["session_id"])
        assert inv is not None
        assert inv.contract == "executive_dashboard"

    def test_multiple_sources_listed(self, session: Session, tmp_path) -> None:
        """All registered sources are returned."""
        source_id = _insert_source(session, name="source_a")
        _insert_source(session, name="source_b")

        with (
            _mock_manager(session),
            patch(
                "dataraum.mcp.server._get_pipeline_source",
                return_value=session.get(Source, source_id),
            ),
        ):
            from dataraum.mcp.server import _begin_session

            result = _begin_session(tmp_path, intent="test")

        assert len(result["sources"]) == 2
        assert "source_a" in result["sources"]
        assert "source_b" in result["sources"]


class TestRecordToolStep:
    def test_records_step_on_success(self, session: Session, tmp_path) -> None:
        """_record_tool_step writes an InvestigationStep to the DB."""
        from dataraum.investigation.recorder import begin_session as create_session

        source_id = _insert_source(session)
        inv = create_session(session, source_id, intent="test")
        session.flush()

        with _mock_manager(session):
            from dataraum.mcp.server import _record_tool_step

            _record_tool_step(
                tmp_path,
                session_id=inv.session_id,
                tool_name="look",
                arguments={"target": "orders"},
                result={"tables": []},
                started_at=datetime.now(UTC),
            )

        steps = session.query(InvestigationStep).filter_by(session_id=inv.session_id).all()
        assert len(steps) == 1
        assert steps[0].tool_name == "look"
        assert steps[0].status == "success"
        assert steps[0].target == "orders"

    def test_records_error_status(self, session: Session, tmp_path) -> None:
        """Error results are recorded with status='error'."""
        from dataraum.investigation.recorder import begin_session as create_session

        source_id = _insert_source(session)
        inv = create_session(session, source_id, intent="test")
        session.flush()

        with _mock_manager(session):
            from dataraum.mcp.server import _record_tool_step

            _record_tool_step(
                tmp_path,
                session_id=inv.session_id,
                tool_name="look",
                arguments={"target": "nonexistent"},
                result={"error": "Table not found"},
                started_at=datetime.now(UTC),
            )

        steps = session.query(InvestigationStep).filter_by(session_id=inv.session_id).all()
        assert len(steps) == 1
        assert steps[0].status == "error"
        assert steps[0].error == "Table not found"

    def test_recording_failure_does_not_raise(self, tmp_path) -> None:
        """_record_tool_step never raises, even with bad session_id."""
        from dataraum.mcp.server import _record_tool_step

        _record_tool_step(
            tmp_path / "nonexistent",
            session_id="bad-session-id",
            tool_name="look",
            arguments={},
            result={},
            started_at=datetime.now(UTC),
        )
