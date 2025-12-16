"""Pytest fixtures for storage tests."""

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from dataraum_context.storage import init_database


@pytest.fixture(scope="function")
async def engine() -> AsyncEngine:
    """Create an in-memory SQLite engine for testing.

    Creates a fresh database for each test function.
    This bypasses the global engine to ensure test isolation.
    """
    # Create a new engine for each test (not using global get_engine)
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )

    # Enable foreign keys for SQLite
    @event.listens_for(test_engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    await init_database(test_engine)
    yield test_engine
    await test_engine.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncSession:
    """Create a test database session.

    Creates a session tied to the test's engine.
    """
    factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with factory() as session:
        yield session
        # Let the session close naturally
