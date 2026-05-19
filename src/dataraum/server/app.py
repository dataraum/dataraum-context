"""Platform FastAPI app.

L1 placeholder + L4 DuckLake bootstrap: exposes ``/health`` (substrate +
DuckLake catalog status) and opens the process-wide DuckLake anchor at
startup. Real subsystems (sessions, sources, REST API) are wired in by
later lanes.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from dataraum.server.storage import bootstrap_lake, health_probe, teardown_lake


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Open the DuckLake anchor at startup; close it at shutdown.

    Refuses to start if ``DUCKLAKE_CATALOG_URL`` or ``DUCKLAKE_DATA_PATH``
    is unset — the container substrate (L1) is responsible for both.
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


@app.get("/health")
def health() -> dict[str, object]:
    """Substrate + DuckLake catalog health.

    Returns ``status: ok`` overall only when both the process is up and the
    DuckLake catalog is reachable. Container orchestrators use this to
    decide readiness.
    """
    ducklake = health_probe()
    overall = "ok" if ducklake.get("status") == "ok" else "degraded"
    return {"status": overall, "ducklake": ducklake}
