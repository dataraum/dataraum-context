"""Content negotiation utility for API endpoints."""

from __future__ import annotations

from fastapi import Request


def wants_html(request: Request) -> bool:
    """Check if the client prefers HTML over JSON.

    Args:
        request: FastAPI request object

    Returns:
        True if the Accept header prefers text/html
    """
    accept = request.headers.get("accept", "")
    return "text/html" in accept
