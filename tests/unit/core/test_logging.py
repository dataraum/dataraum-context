"""Tests for structlog proxy logger routing."""

from __future__ import annotations

import sys
from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console
from rich.text import Text

from dataraum.core.logging import (
    _ProxyLogger,
    _ProxyLoggerFactory,
    deactivate_console,
)


class TestProxyLogger:
    def test_msg_to_stderr_when_no_console(self, monkeypatch: object) -> None:
        """Output goes to stderr when no console is active."""
        deactivate_console()
        buf = StringIO()
        import dataraum.core.logging as log_mod

        monkeypatch.setattr(sys, "stderr", buf)  # type: ignore[attr-defined]
        # Ensure no active console
        monkeypatch.setattr(log_mod, "_active_console", None)  # type: ignore[attr-defined]

        proxy = _ProxyLogger()
        proxy.msg("hello from stderr")

        assert "hello from stderr" in buf.getvalue()

    def test_msg_to_console_when_active(self, monkeypatch: object) -> None:
        """console.print() is called with highlight=False when console is active."""
        import dataraum.core.logging as log_mod

        mock_console = MagicMock(spec=Console)
        monkeypatch.setattr(log_mod, "_active_console", mock_console)  # type: ignore[attr-defined]

        proxy = _ProxyLogger()
        proxy.msg("hello from console")

        mock_console.print.assert_called_once()
        args, kwargs = mock_console.print.call_args
        assert isinstance(args[0], Text)
        assert kwargs.get("highlight") is False

    def test_deactivate_restores_stderr(self, monkeypatch: object) -> None:
        """After deactivation, output goes back to stderr."""
        import dataraum.core.logging as log_mod

        mock_console = MagicMock(spec=Console)
        monkeypatch.setattr(log_mod, "_active_console", mock_console)  # type: ignore[attr-defined]

        deactivate_console()

        buf = StringIO()
        monkeypatch.setattr(sys, "stderr", buf)  # type: ignore[attr-defined]

        proxy = _ProxyLogger()
        proxy.msg("back to stderr")

        assert "back to stderr" in buf.getvalue()
        mock_console.print.assert_not_called()

    def test_aliases_match_msg(self) -> None:
        """All level aliases point to the same method as msg."""
        proxy = _ProxyLogger()
        aliases = [
            "log",
            "debug",
            "info",
            "warn",
            "warning",
            "fatal",
            "failure",
            "err",
            "error",
            "critical",
            "exception",
        ]
        for alias in aliases:
            assert getattr(proxy, alias) == proxy.msg, f"{alias} does not match msg"

    def test_ansi_conversion(self, monkeypatch: object) -> None:
        """ANSI codes are preserved through Text.from_ansi()."""
        import dataraum.core.logging as log_mod

        mock_console = MagicMock(spec=Console)
        monkeypatch.setattr(log_mod, "_active_console", mock_console)  # type: ignore[attr-defined]

        proxy = _ProxyLogger()
        ansi_msg = "\033[32mgreen text\033[0m"
        proxy.msg(ansi_msg)

        args, _ = mock_console.print.call_args
        text_obj: Text = args[0]
        assert "green text" in text_obj.plain

    def test_factory_returns_proxy(self) -> None:
        """Factory returns a _ProxyLogger instance."""
        factory = _ProxyLoggerFactory()
        logger = factory()
        assert isinstance(logger, _ProxyLogger)
