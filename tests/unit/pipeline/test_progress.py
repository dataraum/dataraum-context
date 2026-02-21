"""Tests for pipeline progress callback notifications."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from dataraum.pipeline.base import PhaseResult
from dataraum.pipeline.orchestrator import Pipeline, PipelineConfig


class TestNotifyProgress:
    """Tests for Pipeline._notify_progress static method."""

    def test_calls_callback(self):
        cb = MagicMock()
        Pipeline._notify_progress(cb, 1, 10, "hello")
        cb.assert_called_once_with(1, 10, "hello")

    def test_none_callback_is_noop(self):
        # Should not raise
        Pipeline._notify_progress(None, 1, 10, "hello")

    def test_exception_in_callback_is_swallowed(self):
        cb = MagicMock(side_effect=RuntimeError("boom"))
        # Should not raise
        Pipeline._notify_progress(cb, 1, 10, "hello")
        cb.assert_called_once()


def _make_mock_manager():
    """Create a mock ConnectionManager that won't trigger SQLAlchemy."""
    mgr = MagicMock()

    @contextmanager
    def _session_scope():
        session = MagicMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        session.execute.return_value = result_mock
        yield session

    mgr.session_scope = _session_scope
    return mgr


class TestPipelineProgressCallback:
    """Tests that Pipeline.run() invokes progress_callback at the right points."""

    @pytest.fixture()
    def mock_phase(self):
        from tests.unit.pipeline.conftest import MockPhase

        return MockPhase

    @pytest.fixture()
    def manager(self):
        return _make_mock_manager()

    def _run_with_callback(self, pipeline, manager, cb=None):
        """Run pipeline with DB models and metrics patched out."""
        with (
            patch("dataraum.pipeline.orchestrator.PipelineRun"),
            patch("dataraum.pipeline.orchestrator.update", return_value=MagicMock()),
            patch("dataraum.pipeline.orchestrator.select", return_value=MagicMock()),
            patch("dataraum.pipeline.orchestrator.start_pipeline_metrics"),
            patch("dataraum.pipeline.orchestrator.end_pipeline_metrics", return_value=None),
        ):
            return pipeline.run(
                manager=manager,
                source_id="test-src",
                run_id=str(uuid4()),
                progress_callback=cb,
            )

    def test_callback_called_for_each_phase(self, mock_phase, manager):
        """Callback receives started + completed notifications for each phase."""
        cb = MagicMock()

        pipeline = Pipeline(config=PipelineConfig(max_parallel=1, skip_completed=False))
        pipeline.register(mock_phase("phase_a"))
        pipeline.register(mock_phase("phase_b", dependencies=["phase_a"]))

        with patch.object(Pipeline, "_execute_phase") as mock_exec:
            mock_exec.return_value = PhaseResult.success(outputs={})
            self._run_with_callback(pipeline, manager, cb)

        messages = [c[0][2] for c in cb.call_args_list]

        assert any("Running phase_a" in m for m in messages)
        assert any("Completed phase_a" in m for m in messages)
        assert any("Running phase_b" in m for m in messages)
        assert any("Completed phase_b" in m for m in messages)
        assert any("Pipeline complete" in m for m in messages)

    def test_callback_reports_correct_total(self, mock_phase, manager):
        """Total always equals number of phases to run."""
        cb = MagicMock()

        pipeline = Pipeline(config=PipelineConfig(max_parallel=1, skip_completed=False))
        pipeline.register(mock_phase("a"))
        pipeline.register(mock_phase("b", dependencies=["a"]))
        pipeline.register(mock_phase("c", dependencies=["b"]))

        with patch.object(Pipeline, "_execute_phase") as mock_exec:
            mock_exec.return_value = PhaseResult.success(outputs={})
            self._run_with_callback(pipeline, manager, cb)

        for call in cb.call_args_list:
            assert call[0][1] == 3

    def test_callback_on_skipped_phase(self, mock_phase, manager):
        """Skipped phases trigger progress notifications."""
        cb = MagicMock()

        pipeline = Pipeline(config=PipelineConfig(max_parallel=1, skip_completed=False))
        pipeline.register(mock_phase("skipper", skip_reason="already done"))

        with patch.object(Pipeline, "_execute_phase") as mock_exec:
            mock_exec.return_value = PhaseResult.skipped("already done")
            self._run_with_callback(pipeline, manager, cb)

        messages = [c[0][2] for c in cb.call_args_list]
        assert any("Skipped skipper" in m for m in messages)

    def test_callback_on_failed_phase(self, mock_phase, manager):
        """Failed phases trigger progress notifications."""
        cb = MagicMock()

        pipeline = Pipeline(
            config=PipelineConfig(max_parallel=1, skip_completed=False, fail_fast=True)
        )
        pipeline.register(mock_phase("bad", should_fail=True))

        with patch.object(Pipeline, "_execute_phase") as mock_exec:
            mock_exec.return_value = PhaseResult.failed("intentional")
            self._run_with_callback(pipeline, manager, cb)

        messages = [c[0][2] for c in cb.call_args_list]
        assert any("Failed bad" in m for m in messages)

    def test_pipeline_works_without_callback(self, mock_phase, manager):
        """Pipeline runs normally when progress_callback is None."""
        pipeline = Pipeline(config=PipelineConfig(max_parallel=1, skip_completed=False))
        pipeline.register(mock_phase("solo"))

        with patch.object(Pipeline, "_execute_phase") as mock_exec:
            mock_exec.return_value = PhaseResult.success(outputs={})
            results = self._run_with_callback(pipeline, manager, None)

        assert "solo" in results

    def test_callback_exception_does_not_crash_pipeline(self, mock_phase, manager):
        """If the callback raises, the pipeline still completes."""
        cb = MagicMock(side_effect=RuntimeError("notification failed"))

        pipeline = Pipeline(config=PipelineConfig(max_parallel=1, skip_completed=False))
        pipeline.register(mock_phase("resilient"))

        with patch.object(Pipeline, "_execute_phase") as mock_exec:
            mock_exec.return_value = PhaseResult.success(outputs={})
            results = self._run_with_callback(pipeline, manager, cb)

        assert "resilient" in results

    def test_step_counter_increments(self, mock_phase, manager):
        """The current step increments with each completed/skipped/failed phase."""
        cb = MagicMock()

        pipeline = Pipeline(config=PipelineConfig(max_parallel=1, skip_completed=False))
        pipeline.register(mock_phase("a"))
        pipeline.register(mock_phase("b", dependencies=["a"]))

        with patch.object(Pipeline, "_execute_phase") as mock_exec:
            mock_exec.return_value = PhaseResult.success(outputs={})
            self._run_with_callback(pipeline, manager, cb)

        # Extract (current, total) from completed/skipped/failed notifications
        completion_steps = [
            c[0][0]
            for c in cb.call_args_list
            if any(
                kw in c[0][2]
                for kw in ("Completed", "Skipped", "Failed", "Pipeline complete")
            )
        ]
        # Should be [1, 2, 2] — "Completed a" (1), "Completed b" (2), "Pipeline complete" (2)
        assert completion_steps == [1, 2, 2]
