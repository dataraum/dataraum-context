#!/usr/bin/env python3
"""Phase 3 verification script: analysis/statistics module.

This script verifies that the statistics module works correctly
by profiling real data from the finance_csv_example.
"""

import asyncio
from pathlib import Path

import duckdb
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from dataraum_context.analysis.statistics import profile_statistics
from dataraum_context.analysis.typing import infer_type_candidates, resolve_types
from dataraum_context.core.models import SourceConfig
from dataraum_context.sources.csv import CSVLoader
from dataraum_context.storage import Table, init_database


async def _get_typed_table_id(typed_table_name: str, session) -> str | None:
    """Get the table ID for a typed table by DuckDB path."""
    stmt = select(Table).where(Table.duckdb_path == typed_table_name)
    result = await session.execute(stmt)
    table = result.scalar_one_or_none()
    return table.table_id if table else None


async def main():
    """Run Phase 3 verification."""
    print("=" * 60)
    print("Phase 3 Verification: analysis/statistics module")
    print("=" * 60)

    # Setup in-memory databases
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    await init_database(engine)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    duckdb_conn = duckdb.connect(":memory:")

    # Find CSV file
    csv_path = Path("examples/finance_csv_example/Master_txn_table.csv")
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return

    print(f"\n1. Loading CSV: {csv_path}")

    async with async_session() as session:
        # Step 1: Load CSV
        loader = CSVLoader()
        config = SourceConfig(
            name="master_txn",
            source_type="csv",
            path=str(csv_path),
        )

        load_result = await loader.load(config, duckdb_conn, session)
        if not load_result.success:
            print(f"   ERROR: {load_result.error}")
            return

        staged_table = load_result.unwrap().tables[0]
        print(f"   Staged table ID: {staged_table.table_id}")

        # Get SQLAlchemy Table object
        raw_table = await session.get(Table, staged_table.table_id)
        if not raw_table:
            print("   ERROR: Could not find raw table")
            return
        print(f"   Raw table: {raw_table.duckdb_path}")

        # Step 2: Type inference
        print("\n2. Running type inference...")
        infer_result = await infer_type_candidates(raw_table, duckdb_conn, session)
        if not infer_result.success:
            print(f"   ERROR: {infer_result.error}")
            return
        print(f"   Generated {len(infer_result.unwrap())} type candidates")

        # Step 3: Type resolution
        print("\n3. Resolving types...")
        resolve_result = await resolve_types(staged_table.table_id, duckdb_conn, session)
        if not resolve_result.success:
            print(f"   ERROR: {resolve_result.error}")
            return

        resolution = resolve_result.unwrap()
        print(f"   Typed table: {resolution.typed_table_name}")
        print(f"   Total rows: {resolution.total_rows}")
        print(f"   Quarantined: {resolution.quarantined_rows}")

        # Get typed table ID
        typed_table_id = await _get_typed_table_id(resolution.typed_table_name, session)
        if not typed_table_id:
            print("   ERROR: Could not find typed table ID")
            return

        # Step 4: Statistics profiling
        print("\n4. Profiling statistics (new module)...")
        stats_result = await profile_statistics(typed_table_id, duckdb_conn, session)
        if not stats_result.success:
            print(f"   ERROR: {stats_result.error}")
            return

        stats = stats_result.unwrap()
        print(f"   Duration: {stats.duration_seconds:.2f}s")
        print(f"   Columns profiled: {len(stats.column_profiles)}")

        # Step 5: Display statistics
        print("\n5. Column Statistics:")
        print("-" * 80)

        for profile in stats.column_profiles:
            print(f"\n   {profile.column_ref.column_name}:")
            print(f"      Total: {profile.total_count}, Nulls: {profile.null_count}")
            print(f"      Distinct: {profile.distinct_count}")
            print(f"      Null ratio: {profile.null_ratio:.2%}")
            print(f"      Cardinality ratio: {profile.cardinality_ratio:.2%}")

            if profile.numeric_stats:
                ns = profile.numeric_stats
                print(f"      Numeric: min={ns.min_value:.2f}, max={ns.max_value:.2f}")
                print(f"               mean={ns.mean:.2f}, stddev={ns.stddev:.2f}")
                if ns.skewness is not None:
                    print(f"               skewness={ns.skewness:.2f}")

            if profile.string_stats:
                ss = profile.string_stats
                print(f"      String: min_len={ss.min_length}, max_len={ss.max_length}")
                print(f"              avg_len={ss.avg_length:.1f}")

            if profile.top_values:
                print(f"      Top values: {len(profile.top_values)}")
                for v in profile.top_values[:3]:
                    print(f"         {v.value}: {v.count} ({v.percentage:.1f}%)")

    duckdb_conn.close()
    await engine.dispose()

    print("\n" + "=" * 60)
    print("Phase 3 verification PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
