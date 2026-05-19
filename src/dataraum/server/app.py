"""Platform FastAPI control plane shell.

Substrate-only entrypoint: ``/health`` (DuckLake catalog + workspace Postgres
probes) and the DuckLake anchor opened at startup. Engine REST routes
(``/api/*``) land in step 3b of the Cockpit + Engine REST v1 plan; the chat
BFF moves to TanStack Start (TypeScript), not this app.

Run via ``uvicorn dataraum.server.app:app`` or ``docker compose up``.

No MCP transport, no bearer middleware. The MCP server in
``src/dataraum/mcp/server.py`` is no longer mounted — its engine logic is
extracted into FastAPI route handlers as the v1 plan progresses, and the
file itself retires once nothing imports it.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text

from dataraum.core.logging import get_logger
from dataraum.server.storage import bootstrap_lake, health_probe, teardown_lake

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the DuckLake substrate at startup; tear it down on shutdown.

    Refuses to start if either ``DUCKLAKE_CATALOG_URL`` or
    ``DUCKLAKE_DATA_PATH`` is unset — the container substrate (L1+L4)
    provides both.
    """
    catalog_url = os.environ.get("DUCKLAKE_CATALOG_URL")
    data_path = os.environ.get("DUCKLAKE_DATA_PATH")
    if not catalog_url:
        raise RuntimeError(
            "DUCKLAKE_CATALOG_URL is not set. The container substrate (L1) "
            "provides this; for local dev outside the container, export "
            "DUCKLAKE_CATALOG_URL=postgresql://<user>:<pass>@<host>:<port>/<db>."
        )
    if not data_path:
        raise RuntimeError(
            "DUCKLAKE_DATA_PATH is not set. The container substrate (L1) "
            "mounts the dataraum_lake named volume at /var/lib/dataraum/lake/."
        )

    bootstrap_lake(catalog_url, data_path)
    try:
        yield
    finally:
        teardown_lake()


app = FastAPI(title="DataRaum Control Plane", version="0.2.2", lifespan=lifespan)


def _postgres_probe() -> dict[str, str]:
    """Return a /health-shaped dict for the workspace Postgres engine.

    Uses a short-lived engine (no pool reuse) so the probe never wedges on a
    pool-exhausted main connection manager. ``DATABASE_URL`` is the same env
    var the rest of the engine reads at runtime.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        return {"status": "not_configured"}
    try:
        engine = create_engine(url, pool_pre_ping=True)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        finally:
            engine.dispose()
    except Exception as e:
        # Log the cause but don't expose it in the response body (CodeQL info-
        # exposure pattern). Misconfigured URL schemes (e.g. `postgresql://`
        # instead of `postgresql+psycopg://`) show up here as ModuleNotFoundError.
        logger.warning("postgres_health_probe_failed", error=str(e))
        return {"status": "unreachable"}
    return {"status": "ok"}


@app.get("/health")
def health() -> Response:
    """Substrate + DuckLake catalog + workspace Postgres health.

    Returns 200 with ``status: ok`` when both substrate components are
    reachable; 503 with ``status: degraded`` otherwise so k8s/ECS readiness
    probes that only inspect the status code route traffic away from the
    container instead of seeing a healthy 200 with a degraded body.
    """
    ducklake = health_probe()
    postgres = _postgres_probe()
    healthy = ducklake.get("status") == "ok" and postgres.get("status") == "ok"
    overall = "ok" if healthy else "degraded"
    return JSONResponse(
        {"status": overall, "ducklake": ducklake, "postgres": postgres},
        status_code=200 if healthy else 503,
    )
