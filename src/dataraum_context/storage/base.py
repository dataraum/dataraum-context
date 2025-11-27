"""SQLAlchemy base configuration and session management."""

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import Text, event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Global engine and session factory (initialized by app)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(database_url: str | None = None) -> AsyncEngine:
    """
    Get or create the database engine.

    Args:
        database_url: Database connection string. If None, uses existing engine.
                     Supports:
                     - sqlite+aiosqlite:///path/to/db.sqlite
                     - postgresql+asyncpg://user:pass@host/db

    Returns:
        Async SQLAlchemy engine
    """
    global _engine

    if database_url is not None:
        _engine = create_async_engine(
            database_url,
            echo=False,
            future=True,
        )

        # Enable foreign keys for SQLite
        if database_url.startswith("sqlite"):

            @event.listens_for(_engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    if _engine is None:
        raise RuntimeError("Database engine not initialized. Call get_engine(database_url) first.")

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get the session factory."""
    global _session_factory

    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session.

    Usage:
        async with get_session() as session:
            result = await session.execute(select(Source))
            sources = result.scalars().all()
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def generate_uuid() -> str:
    """Generate a UUID string for primary keys."""
    return str(uuid.uuid4())


# Type alias for UUID columns (TEXT in SQLite, UUID in PostgreSQL)
def get_uuid_column_type():
    """Get the appropriate UUID column type based on the database."""
    # SQLAlchemy will handle this automatically with the Text type
    # and we'll use Python UUID objects that get converted to strings
    return Text
