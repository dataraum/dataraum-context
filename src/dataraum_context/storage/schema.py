"""Database schema initialization and management."""

from sqlalchemy.ext.asyncio import AsyncEngine

# Import all model modules to register them with SQLAlchemy Base metadata
# These imports ensure tables are created when init_database() is called
from dataraum_context.enrichment import db_models as _enrichment_models  # noqa: F401
from dataraum_context.graphs import db_models as _graphs_models  # noqa: F401
from dataraum_context.llm import db_models as _llm_models  # noqa: F401
from dataraum_context.profiling import db_models as _profiling_models  # noqa: F401
from dataraum_context.quality import db_models as _quality_models  # noqa: F401
from dataraum_context.quality.domains import db_models as _domain_quality_models  # noqa: F401
from dataraum_context.storage.base import Base
from dataraum_context.storage.models_v2.base import Base as BaseV2


async def init_database(engine: AsyncEngine) -> None:
    """
    Initialize database schema.

    Creates all tables defined in SQLAlchemy models.
    Safe to call multiple times - only creates missing tables.

    Args:
        engine: Async SQLAlchemy engine
    """
    async with engine.begin() as conn:
        # Create tables from both old and new Base classes
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(BaseV2.metadata.create_all)


async def reset_database(engine: AsyncEngine) -> None:
    """
    Drop and recreate all tables.

    WARNING: This destroys all data. Use only in development/testing.

    Args:
        engine: Async SQLAlchemy engine
    """
    async with engine.begin() as conn:
        # Drop and recreate tables from both old and new Base classes
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(BaseV2.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(BaseV2.metadata.create_all)
