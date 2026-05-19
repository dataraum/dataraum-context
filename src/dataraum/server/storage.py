"""DuckLake bootstrap and shared in-memory anchor for the FastAPI process.

DuckLake stores data as parquet files on a DATA_PATH backend (local-FS in the
spine; rustfs/S3 deferred to post-spine) with metadata in a Postgres catalog
database. DuckDB clients access DuckLake by ATTACHing the catalog as an
external database.

Connection model (post-DAT-323):

* One named in-memory DuckDB database, `:memory:dataraum_lake`, lives for the
  lifetime of the FastAPI process. We open one *anchor* connection at startup
  that is never used for queries — its sole purpose is to keep the named
  database alive (DuckDB tears down a named in-memory database once the last
  connection to it closes).
* Per-session ``ConnectionManager`` instances obtain their own fresh DuckDB
  connection to the same named database via :func:`connect_session`. Catalog
  state (the DuckLake ATTACH) is shared across connections to the same named
  in-memory database; connection-state (``USE``/search_path, prepared
  statements, transaction state) is per-connection. This gives per-session
  schema isolation without re-paying ATTACH cost.

Tests bootstrap the anchor against a testcontainer Postgres + a tmp_path
DATA_PATH; the FastAPI app does the same against the compose stack.
"""

from __future__ import annotations

import threading
from urllib.parse import unquote, urlparse

import duckdb

from dataraum.core.logging import get_logger

logger = get_logger(__name__)


LAKE_DB_NAME = ":memory:dataraum_lake"
"""Named in-memory DuckDB used as the process-wide shared catalog.

All connections opened with ``duckdb.connect(LAKE_DB_NAME)`` share catalog
state (schemas, ATTACHed DBs, tables). The anchor keeps the database alive.
"""

LAKE_CATALOG_ALIAS = "lake"
"""Alias under which the DuckLake catalog is ATTACHed.

Per-session schemas live as ``lake.session_<id>``; archived sessions as
``lake.archive_<id>`` (see :mod:`dataraum.mcp.server`).
"""


_anchor: duckdb.DuckDBPyConnection | None = None
_bootstrap_lock = threading.Lock()


def _pg_url_to_libpq(url: str) -> str:
    """Convert a ``postgresql://user:pass@host:port/db`` URL to libpq KV form.

    DuckLake's ATTACH string expects libpq keyword-value syntax
    (``dbname=... host=... user=... password=... port=...``), not the
    URL form.
    """
    p = urlparse(url)
    parts: list[str] = []
    if p.path and p.path != "/":
        parts.append(f"dbname={unquote(p.path.lstrip('/'))}")
    if p.hostname:
        parts.append(f"host={p.hostname}")
    if p.port:
        parts.append(f"port={p.port}")
    if p.username:
        parts.append(f"user={unquote(p.username)}")
    if p.password:
        # urlparse leaves percent-encoding in place; libpq wants the decoded
        # value. Then single-quote+escape if the decoded value contains
        # whitespace or quote characters; alphanumeric passes through bare.
        decoded = unquote(p.password)
        if any(c.isspace() or c in ("'", "\\") for c in decoded):
            escaped = decoded.replace("\\", "\\\\").replace("'", "\\'")
            parts.append(f"password='{escaped}'")
        else:
            parts.append(f"password={decoded}")
    return " ".join(parts)


def bootstrap_lake(catalog_url: str, data_path: str) -> None:
    """Open the process-wide DuckLake anchor.

    Idempotent: subsequent calls with the anchor already open are a no-op.

    Fails loud if the Postgres catalog is unreachable, ``data_path`` is not
    writable, or the ATTACH fails.

    Args:
        catalog_url: Postgres connection URL for the DuckLake catalog
            (``postgresql://...``).
        data_path: Filesystem path where DuckLake writes parquet files. The
            directory must exist and be writable.

    Raises:
        RuntimeError: If bootstrap fails. The original DuckDB exception is
            chained for inspection.
    """
    global _anchor

    with _bootstrap_lock:
        if _anchor is not None:
            return

        libpq = _pg_url_to_libpq(catalog_url)
        attach_sql = (
            f"ATTACH 'ducklake:postgres:{libpq}' AS {LAKE_CATALOG_ALIAS} "
            f"(DATA_PATH '{data_path}')"
        )

        try:
            conn = duckdb.connect(LAKE_DB_NAME)
            conn.execute("INSTALL ducklake")
            conn.execute("LOAD ducklake")
            conn.execute(attach_sql)
            # Smoke probe: ATTACH took effect (catalog reachable). DuckLake does
            # not expose ``information_schema``; ``duckdb_schemas()`` filtered by
            # database name exercises the catalog driver against Postgres.
            conn.execute(
                "SELECT 1 FROM duckdb_schemas() "
                f"WHERE database_name = '{LAKE_CATALOG_ALIAS}' LIMIT 1"
            )
        except Exception as e:
            raise RuntimeError(
                f"DuckLake bootstrap failed (catalog_url={catalog_url}, "
                f"data_path={data_path}): {e}"
            ) from e

        _anchor = conn
        logger.info(
            "ducklake_bootstrapped",
            catalog_url=catalog_url,
            data_path=data_path,
        )


def get_anchor() -> duckdb.DuckDBPyConnection:
    """Return the process-wide DuckLake anchor connection.

    Raises:
        RuntimeError: If :func:`bootstrap_lake` has not been called.
    """
    if _anchor is None:
        raise RuntimeError(
            "DuckLake not bootstrapped. Call bootstrap_lake(...) at server "
            "startup (or via the test fixture) before opening per-session "
            "connections."
        )
    return _anchor


def connect_session() -> duckdb.DuckDBPyConnection:
    """Open a fresh DuckDB connection to the named lake database.

    The returned connection shares catalog state with the anchor (so the
    DuckLake ATTACH is already visible) but has its own connection-state
    (``USE``, search_path, transaction). Callers own the connection's
    lifecycle and must ``.close()`` it.

    Raises:
        RuntimeError: If :func:`bootstrap_lake` has not been called.
    """
    if _anchor is None:
        raise RuntimeError(
            "DuckLake not bootstrapped. Call bootstrap_lake(...) at server "
            "startup before opening per-session connections."
        )
    return duckdb.connect(LAKE_DB_NAME)


def teardown_lake() -> None:
    """Close the anchor connection. Safe to call when not bootstrapped.

    The named in-memory database persists until the last connection to it
    closes; tests should ensure per-session managers are closed before this.
    """
    global _anchor
    with _bootstrap_lock:
        if _anchor is None:
            return
        try:
            _anchor.close()
        except Exception as e:
            logger.warning("ducklake_anchor_close_failed", error=str(e))
        _anchor = None


def health_probe() -> dict[str, str]:
    """Return a ``/health``-shaped dict for the DuckLake catalog.

    Status is ``ok`` when the anchor exists and the catalog is queryable,
    ``not_bootstrapped`` when the bootstrap hook hasn't run, or
    ``unreachable`` when the catalog query fails.
    """
    if _anchor is None:
        return {"status": "not_bootstrapped"}
    try:
        _anchor.execute(
            "SELECT 1 FROM duckdb_schemas() "
            f"WHERE database_name = '{LAKE_CATALOG_ALIAS}' LIMIT 1"
        )
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}
    return {"status": "ok"}


__all__ = [
    "LAKE_DB_NAME",
    "LAKE_CATALOG_ALIAS",
    "bootstrap_lake",
    "get_anchor",
    "connect_session",
    "teardown_lake",
    "health_probe",
]
