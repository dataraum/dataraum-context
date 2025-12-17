#!/usr/bin/env python3
"""Phase 7: Temporal Profiling with persistent storage.

This script analyzes temporal columns in typed tables for:
- Granularity detection (daily, weekly, monthly, etc.)
- Completeness and gap analysis
- Seasonality patterns
- Trend detection
- Change point detection
- Fiscal calendar alignment
- Distribution stability

Prerequisites:
    - Phase 2 must be completed (run_phase2_typing.py)
    - Phase 3 recommended (run_phase3_statistics.py)

Usage:
    uv run python scripts/run_phase7_temporal.py
"""

import asyncio
import sys
from typing import Any

from infra import (
    cleanup_connections,
    get_duckdb_conn,
    get_test_session,
    print_database_summary,
    print_phase_status,
)
from sqlalchemy import func, select


async def main() -> int:
    """Run Phase 7: Temporal Profiling."""
    print("=" * 70)
    print("Phase 7: Temporal Profiling (Persistent Storage)")
    print("=" * 70)

    # Get connections
    session_factory = await get_test_session()
    duckdb_conn = get_duckdb_conn()

    async with session_factory() as session:
        from dataraum_context.analysis.temporal import (
            TemporalColumnProfile,
            profile_temporal,
        )
        from dataraum_context.storage import Column, Table

        # Check prerequisites - need typed tables
        print("\n1. Checking prerequisites...")
        print("-" * 50)

        typed_tables_stmt = select(Table).where(Table.layer == "typed")
        typed_tables = (await session.execute(typed_tables_stmt)).scalars().all()

        if not typed_tables:
            print("   ERROR: No typed tables found!")
            print("   Please run run_phase2_typing.py first.")
            await cleanup_connections()
            return 1

        print(f"   Found {len(typed_tables)} typed tables")

        # Check for temporal columns
        temporal_types = ["DATE", "TIMESTAMP", "TIMESTAMPTZ"]
        typed_table_ids = [t.table_id for t in typed_tables]
        temporal_columns_stmt = select(Column).where(
            Column.table_id.in_(typed_table_ids),
            Column.resolved_type.in_(temporal_types),
        )
        temporal_columns = (await session.execute(temporal_columns_stmt)).scalars().all()

        if not temporal_columns:
            print("   No temporal columns found in typed tables.")
            print("   Nothing to profile.")
            await cleanup_connections()
            return 0

        print(f"   Found {len(temporal_columns)} temporal columns across tables")

        # Check existing profiles
        existing_profile_count = (
            await session.execute(select(func.count(TemporalColumnProfile.profile_id)))
        ).scalar() or 0

        if existing_profile_count > 0:
            print(f"\n   Already profiled {existing_profile_count} temporal columns!")
            print_phase_status("temporal", True)
            await _print_temporal_summary(session)
            await print_database_summary(session, duckdb_conn)
            await cleanup_connections()
            return 0

        print_phase_status("temporal", False)

        # Run temporal profiling
        print("\n2. Running temporal profiling...")
        print("-" * 50)

        profiled_count = 0
        total_profiles = 0

        for typed_table in typed_tables:
            print(f"      Processing: {typed_table.table_name}...")

            result = await profile_temporal(typed_table.table_id, duckdb_conn, session)
            if not result.success:
                print(f"         WARNING: {result.error}")
                continue

            profile_result = result.unwrap()
            num_profiles = len(profile_result.column_profiles)

            if num_profiles > 0:
                print(f"         Profiled {num_profiles} temporal columns")
                print(f"         Duration: {profile_result.duration_seconds:.2f}s")

                # Show summary of findings
                for profile in profile_result.column_profiles:
                    findings = []
                    if profile.seasonality and profile.seasonality.has_seasonality:
                        findings.append(f"seasonality ({profile.seasonality.period})")
                    if profile.trend and profile.trend.has_trend:
                        findings.append(f"trend ({profile.trend.direction})")
                    if profile.change_points:
                        findings.append(f"{len(profile.change_points)} change points")
                    if profile.update_frequency and profile.update_frequency.is_stale:
                        findings.append("stale")

                    if findings:
                        print(f"            {profile.column_name}: {', '.join(findings)}")
                    else:
                        print(
                            f"            {profile.column_name}: "
                            f"{profile.detected_granularity} granularity"
                        )

                total_profiles += num_profiles
                profiled_count += 1
            else:
                print("         No temporal columns in this table")

        print(f"\n   Profiled {total_profiles} columns across {profiled_count} tables")

        # Summary
        await _print_temporal_summary(session)
        await print_database_summary(session, duckdb_conn)

    await cleanup_connections()

    print("\n" + "=" * 70)
    print("Phase 7 COMPLETE")
    print("=" * 70)
    print("\nTemporal analysis complete. Review the temporal_column_profiles table.")
    return 0


async def _print_temporal_summary(session: Any) -> None:
    """Print summary of temporal profiling results."""
    from dataraum_context.analysis.temporal import TemporalColumnProfile

    # Count profiles
    total_count = (
        await session.execute(select(func.count(TemporalColumnProfile.profile_id)))
    ).scalar() or 0

    # Count by flags
    seasonality_count = (
        await session.execute(
            select(func.count(TemporalColumnProfile.profile_id)).where(
                TemporalColumnProfile.has_seasonality == True  # noqa: E712
            )
        )
    ).scalar() or 0

    trend_count = (
        await session.execute(
            select(func.count(TemporalColumnProfile.profile_id)).where(
                TemporalColumnProfile.has_trend == True  # noqa: E712
            )
        )
    ).scalar() or 0

    stale_count = (
        await session.execute(
            select(func.count(TemporalColumnProfile.profile_id)).where(
                TemporalColumnProfile.is_stale == True  # noqa: E712
            )
        )
    ).scalar() or 0

    print("\nTemporal Profiling Results:")
    print(f"  Total temporal columns profiled: {total_count}")
    print(f"  With seasonality: {seasonality_count}")
    print(f"  With trend: {trend_count}")
    print(f"  Stale data: {stale_count}")

    # Show granularity distribution
    granularities_stmt = (
        select(
            TemporalColumnProfile.detected_granularity,
            func.count(TemporalColumnProfile.profile_id),
        )
        .group_by(TemporalColumnProfile.detected_granularity)
        .order_by(func.count(TemporalColumnProfile.profile_id).desc())
    )
    granularities = (await session.execute(granularities_stmt)).all()

    if granularities:
        print("  By granularity:")
        for granularity, count in granularities:
            print(f"    {granularity}: {count}")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
