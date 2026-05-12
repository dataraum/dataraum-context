"""Tests for the import phase and per-source fingerprint.

Phase 2 of DAT-290 collapsed multi-source semantics into single-source-per-
session. This file still carries TestColumnLimit + TestLoadRegisteredSources
from before; Phase 3 will rewrite or delete them when the import phase is
itself rewritten. The fingerprint tests below cover the new
``_compute_source_fingerprint`` (per single source).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from dataraum.pipeline.base import PhaseContext, PhaseResult, PhaseStatus
from dataraum.pipeline.phases.import_phase import ImportPhase
from dataraum.pipeline.setup import _compute_source_fingerprint


class TestColumnLimit:
    """Tests for the column limit enforcement."""

    def test_column_limit_check_under_limit(self):
        """_check_column_limit returns None when under limit."""
        phase = ImportPhase()
        mock_session = MagicMock()
        mock_session.execute.return_value.scalar_one.return_value = 50

        ctx = PhaseContext(
            session=mock_session,
            duckdb_conn=MagicMock(),
            source_id="test-source",
            config={},
        )

        with patch(
            "dataraum.pipeline.phases.import_phase.load_pipeline_config",
            return_value={"limits": {"max_columns": 500}},
        ):
            result = phase._check_column_limit(ctx)

        assert result is None

    def test_column_limit_check_over_limit(self):
        """_check_column_limit returns error when over limit."""
        phase = ImportPhase()
        mock_session = MagicMock()
        mock_session.execute.return_value.scalar_one.return_value = 600

        ctx = PhaseContext(
            session=mock_session,
            duckdb_conn=MagicMock(),
            source_id="test-source",
            config={},
        )

        with patch(
            "dataraum.pipeline.phases.import_phase.load_pipeline_config",
            return_value={"limits": {"max_columns": 500}},
        ):
            result = phase._check_column_limit(ctx)

        assert result is not None
        assert "600 > 500" in result
        assert "limits.max_columns" in result

    def test_column_limit_defaults_to_500(self):
        """When limits section missing, defaults to 500."""
        phase = ImportPhase()
        mock_session = MagicMock()
        mock_session.execute.return_value.scalar_one.return_value = 50

        ctx = PhaseContext(
            session=mock_session,
            duckdb_conn=MagicMock(),
            source_id="test-source",
            config={},
        )

        with patch(
            "dataraum.pipeline.phases.import_phase.load_pipeline_config",
            return_value={},
        ):
            result = phase._check_column_limit(ctx)

        assert result is None

    def test_run_fails_when_no_source(self):
        """_run fails when neither source_path nor registered_sources provided."""
        phase = ImportPhase()
        ctx = PhaseContext(
            session=MagicMock(),
            duckdb_conn=MagicMock(),
            source_id="test-source",
            config={},
        )

        result = phase._run(ctx)

        assert result.status == PhaseStatus.FAILED
        assert "No source_path or registered_sources" in (result.error or "")


class TestSourceFingerprint:
    """Tests for single-source fingerprint computation."""

    def _spec(self, name: str = "a", path: str = "/a.csv") -> dict[str, Any]:
        return {
            "name": name,
            "source_type": "csv",
            "connection_config": {"path": path},
        }

    def test_fingerprint_deterministic(self) -> None:
        """Same source produces same fingerprint."""
        spec = self._spec()
        assert _compute_source_fingerprint(spec) == _compute_source_fingerprint(spec)

    def test_fingerprint_changes_with_connection_config(self) -> None:
        """Changing the connection_config changes the fingerprint."""
        a = self._spec(path="/a.csv")
        b = self._spec(path="/a_v2.csv")
        assert _compute_source_fingerprint(a) != _compute_source_fingerprint(b)

    def test_fingerprint_changes_with_name(self) -> None:
        """Renaming the source changes the fingerprint."""
        a = self._spec(name="a")
        b = self._spec(name="b")
        assert _compute_source_fingerprint(a) != _compute_source_fingerprint(b)

    def test_fingerprint_handles_none_connection_config(self) -> None:
        """Sources without connection_config (edge case) hash without error."""
        spec: dict[str, Any] = {
            "name": "a",
            "source_type": "csv",
            "connection_config": None,
        }
        fp = _compute_source_fingerprint(spec)
        assert isinstance(fp, str)
        assert len(fp) == 16


class TestLoadRegisteredSources:
    """Tests for multi-source dispatch logic in _run."""

    def test_delegates_to_load_from_path_when_source_path(self):
        """When source_path is in config, uses legacy path."""
        phase = ImportPhase()

        ctx = PhaseContext(
            session=MagicMock(),
            duckdb_conn=MagicMock(),
            source_id="test-source",
            config={"source_path": "/nonexistent/path.csv"},
        )

        result = phase._run(ctx)

        assert result.status == PhaseStatus.FAILED
        assert "Source path not found" in (result.error or "")

    def test_prefers_registered_sources_when_no_path(self):
        """When source_path missing but registered_sources present, uses multi-source."""
        phase = ImportPhase()

        # Mock _load_registered_sources to avoid SQLAlchemy mapper issues
        mock_result = PhaseResult.failed("No tables were loaded from any registered source")
        with patch.object(phase, "_load_registered_sources", return_value=mock_result):
            ctx = PhaseContext(
                session=MagicMock(),
                duckdb_conn=MagicMock(),
                source_id="test-source",
                config={
                    "registered_sources": [
                        {"name": "weird", "source_type": "unknown"},
                    ],
                },
            )

            result = phase._run(ctx)

        assert result.status == PhaseStatus.FAILED
        assert "No tables were loaded" in (result.error or "")

    def test_source_path_takes_precedence(self):
        """When both source_path and registered_sources, source_path wins."""
        phase = ImportPhase()

        ctx = PhaseContext(
            session=MagicMock(),
            duckdb_conn=MagicMock(),
            source_id="test-source",
            config={
                "source_path": "/nonexistent/path.csv",
                "registered_sources": [
                    {"name": "a", "source_type": "csv", "path": "/a.csv"},
                ],
            },
        )

        result = phase._run(ctx)

        # Should use legacy path (source_path) and fail on missing file
        assert result.status == PhaseStatus.FAILED
        assert "Source path not found" in (result.error or "")
