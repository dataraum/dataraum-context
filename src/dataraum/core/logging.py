"""Structured logging infrastructure for cloud-ready instrumentation.

This module provides extensible logging that works for:
- Local CLI development (rich console output)
- Cloud deployments (JSON structured logs)
- Future OpenTelemetry integration

Usage:
    from dataraum.core.logging import get_logger, configure_logging

    # Configure at startup
    configure_logging(log_level="INFO", log_format="console")

    # Get logger in any module
    logger = get_logger(__name__)

    # Log with structured context
    logger.info("phase_started", phase="import", source_id="abc123")

    # Use context managers for automatic context propagation
    with logger.context(run_id="run-123", phase="typing"):
        logger.info("processing_table", table="customers", rows=1000)
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

import structlog
from structlog.typing import FilteringBoundLogger

# Context variables for correlation
_run_context: ContextVar[dict[str, Any] | None] = ContextVar("run_context", default=None)


@dataclass
class LogConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "console"  # "console" or "json"
    show_timestamps: bool = True
    show_caller: bool = False  # Show file:line in console mode
    color: bool = True


def _add_run_context(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Processor to add run context to log events."""
    context = _run_context.get()
    if context:
        event_dict.update(context)
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "console",
    show_timestamps: bool = True,
    color: bool = True,
) -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Output format ("console" for development, "json" for production/cloud)
        show_timestamps: Whether to show timestamps in console mode
        color: Whether to use colors in console mode
    """
    # Shared processors for all formats
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_run_context,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if show_timestamps:
        shared_processors.insert(0, structlog.processors.TimeStamper(fmt="iso", utc=True))

    if log_format == "json":
        # JSON format for cloud/production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=color,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level.upper())),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging for libraries
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Get a structured logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Bound structlog logger
    """
    return cast(FilteringBoundLogger, structlog.get_logger(name))


class LogContext:
    """Context manager for adding context to logs within a scope."""

    def __init__(self, **context: Any):
        """Initialize with context key-value pairs."""
        self.context = context
        self.token: Any = None

    def __enter__(self) -> LogContext:
        """Enter context, adding values to log context."""
        current = _run_context.get() or {}
        new_context = {**current, **self.context}
        self.token = _run_context.set(new_context)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context, restoring previous values."""
        if self.token:
            _run_context.reset(self.token)


def log_context(**context: Any) -> LogContext:
    """Create a context manager for scoped logging context.

    Usage:
        with log_context(run_id="abc", phase="import"):
            logger.info("processing")  # Will include run_id and phase
    """
    return LogContext(**context)


# Initialize with default configuration
configure_logging()
