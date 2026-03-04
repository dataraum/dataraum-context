"""Tests for CLI gate handler functions."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from rich.console import Console

from dataraum.cli.gate_handler import (
    _render_violations,
    handle_exit_check,
)
from dataraum.entropy.fix_executor import ActionDefinition, ActionRegistry
from dataraum.pipeline.events import EventType, PipelineEvent
from dataraum.pipeline.runner import GateMode
from dataraum.pipeline.scheduler import ResolutionAction


def _make_exit_check_event(
    violations: dict[str, tuple[float, float]] | None = None,
) -> PipelineEvent:
    """Create a minimal EXIT_CHECK event."""
    return PipelineEvent(
        event_type=EventType.EXIT_CHECK,
        step=5,
        total=10,
        violations=violations
        or {
            "structural.types.type_fidelity": (0.62, 0.50),
        },
    )


def _make_registry_with_action() -> ActionRegistry:
    """Create a registry with one action that improves type_fidelity."""
    registry = ActionRegistry()
    registry.register(
        ActionDefinition(
            action_type="override_type",
            category="transform",
            description="Override column type",
            hard_verifiable=True,
            parameters_schema={"target_type": "Target SQL type"},
            improves_dimensions=["structural.types.type_fidelity"],
        )
    )
    return registry


class TestSkipMode:
    def test_returns_defer(self):
        console = Console(file=StringIO())
        event = _make_exit_check_event()
        result = handle_exit_check(console, event, GateMode.SKIP)
        assert result.action == ResolutionAction.DEFER

    def test_prints_message(self):
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        event = _make_exit_check_event()
        handle_exit_check(console, event, GateMode.SKIP)
        assert "deferred" in output.getvalue().lower()


class TestFailMode:
    def test_returns_abort(self):
        console = Console(file=StringIO())
        event = _make_exit_check_event()
        result = handle_exit_check(console, event, GateMode.FAIL)
        assert result.action == ResolutionAction.ABORT


class TestInteractiveMode:
    def test_defer_choice(self):
        """PAUSE + user picks defer → Resolution(DEFER)."""
        console = Console(file=StringIO())
        event = _make_exit_check_event()

        # The defer option is at index 1 (no fix options without registry)
        with patch("dataraum.cli.gate_handler.Prompt") as mock_prompt:
            mock_prompt.ask.return_value = "1"
            result = handle_exit_check(console, event, GateMode.PAUSE)

        assert result.action == ResolutionAction.DEFER

    def test_abort_choice(self):
        """PAUSE + user picks abort → Resolution(ABORT)."""
        console = Console(file=StringIO())
        event = _make_exit_check_event()

        # Without registry: option 1=defer, option 2=abort
        with patch("dataraum.cli.gate_handler.Prompt") as mock_prompt:
            mock_prompt.ask.return_value = "2"
            result = handle_exit_check(console, event, GateMode.PAUSE)

        assert result.action == ResolutionAction.ABORT

    def test_fix_choice(self):
        """PAUSE + user picks fix → Resolution(FIX, fixes=[...])."""
        console = Console(file=StringIO())
        event = _make_exit_check_event()
        registry = _make_registry_with_action()

        # With registry: option 1=fix:override_type, option 2=defer, option 3=abort
        with patch("dataraum.cli.gate_handler.Prompt") as mock_prompt:
            # First call: choose fix (option 1)
            # Second call: target
            # Third call: target_type parameter
            mock_prompt.ask.side_effect = ["1", "column:orders.amount", "DECIMAL(10,2)"]
            result = handle_exit_check(console, event, GateMode.PAUSE, registry)

        assert result.action == ResolutionAction.FIX
        assert len(result.fixes) == 1
        assert result.fixes[0].action_type == "override_type"
        assert result.fixes[0].target == "column:orders.amount"

    def test_keyboard_interrupt_defers(self):
        """Ctrl+C during prompt → Resolution(DEFER)."""
        console = Console(file=StringIO())
        event = _make_exit_check_event()

        with patch("dataraum.cli.gate_handler.Prompt") as mock_prompt:
            mock_prompt.ask.side_effect = KeyboardInterrupt
            result = handle_exit_check(console, event, GateMode.PAUSE)

        assert result.action == ResolutionAction.DEFER

    def test_eof_defers(self):
        """EOFError during prompt → Resolution(DEFER)."""
        console = Console(file=StringIO())
        event = _make_exit_check_event()

        with patch("dataraum.cli.gate_handler.Prompt") as mock_prompt:
            mock_prompt.ask.side_effect = EOFError
            result = handle_exit_check(console, event, GateMode.PAUSE)

        assert result.action == ResolutionAction.DEFER


class TestRenderViolations:
    def test_renders_panel(self):
        """Violations panel renders without errors."""
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=100)
        violations = {
            "structural.types.type_fidelity": (0.62, 0.50),
            "semantic.units.unit_declaration": (0.45, 0.30),
        }
        _render_violations(console, violations)
        rendered = output.getvalue()
        assert "type_fidelity" in rendered
        assert "unit_declaration" in rendered
        assert "0.62" in rendered
