#!/usr/bin/env python3
"""Phase 6: Cross-Table Quality Analysis.

This script runs AFTER semantic analysis on LLM-confirmed relationships:
- Cross-table correlations between joined data
- Redundant/derived column detection across tables
- Multicollinearity (VDP) analysis

This complements the pre-semantic within-table analysis (Phase 4b)
by analyzing quality issues that can only be detected after tables are joined.

Prerequisites:
    - Phase 5 must be completed (run_phase5_semantic.py)
    - Requires LLM-confirmed relationships

Usage:
    uv run python scripts/run_phase6_correlation.py
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
    """Run Phase 6: Cross-Table Quality Analysis."""
    print("=" * 70)
    print("Phase 6: Cross-Table Quality Analysis")
    print("=" * 70)

    # Get connections
    session_factory = await get_test_session()
    duckdb_conn = get_duckdb_conn()

    async with session_factory() as session:
        from dataraum_context.analysis.correlation.db_models import (
            CorrelationAnalysisRun,
        )
        from dataraum_context.analysis.relationships.db_models import Relationship
        from dataraum_context.storage import Column, Table

        # Check prerequisites
        print("\n1. Checking prerequisites...")
        print("-" * 50)

        # Get LLM-confirmed relationships
        confirmed_rel_stmt = select(Relationship).where(Relationship.detection_method == "llm")
        confirmed_relationships = (await session.execute(confirmed_rel_stmt)).scalars().all()

        if not confirmed_relationships:
            print("   No LLM-confirmed relationships found.")
            print("   Please run run_phase5_semantic.py first.")
            print("   Cross-table analysis requires confirmed relationships.")
            await cleanup_connections()
            return 0  # Not an error, just nothing to do

        print(f"   Found {len(confirmed_relationships)} LLM-confirmed relationships")

        # Check which relationships have already been analyzed
        analyzed_rel_stmt = select(CorrelationAnalysisRun.target_id).where(
            CorrelationAnalysisRun.target_type == "relationship"
        )
        analyzed_rel_ids = set((await session.execute(analyzed_rel_stmt)).scalars().all())

        unanalyzed_rels = [
            r for r in confirmed_relationships if r.relationship_id not in analyzed_rel_ids
        ]

        # Show analysis status
        print("\n   Cross-table analysis status:")
        for rel in confirmed_relationships:
            rel_from_col = await session.get(Column, rel.from_column_id)
            rel_to_col = await session.get(Column, rel.to_column_id)
            rel_from_tbl = await session.get(Table, rel.from_table_id)
            rel_to_tbl = await session.get(Table, rel.to_table_id)

            if all([rel_from_col, rel_to_col, rel_from_tbl, rel_to_tbl]):
                assert rel_from_col is not None
                assert rel_to_col is not None
                assert rel_from_tbl is not None
                assert rel_to_tbl is not None
                rel_desc = f"{rel_from_tbl.table_name}.{rel_from_col.column_name} → {rel_to_tbl.table_name}.{rel_to_col.column_name}"
                is_analyzed = rel.relationship_id in analyzed_rel_ids
                print_phase_status(f"cross_{rel_desc[:40]}", is_analyzed)

        # Run cross-table quality analysis
        print("\n2. Running cross-table quality analysis...")
        print("-" * 50)

        from dataraum_context.analysis.correlation import analyze_cross_table_quality

        if not unanalyzed_rels:
            print("   All confirmed relationships already analyzed!")
        else:
            print(f"   Found {len(unanalyzed_rels)} unanalyzed relationships")

            cross_table_count = 0
            for rel in unanalyzed_rels:
                # Get table/column names for display
                rel_from_col = await session.get(Column, rel.from_column_id)
                rel_to_col = await session.get(Column, rel.to_column_id)
                rel_from_tbl = await session.get(Table, rel.from_table_id)
                rel_to_tbl = await session.get(Table, rel.to_table_id)

                if not all([rel_from_col, rel_to_col, rel_from_tbl, rel_to_tbl]):
                    continue

                # Type narrowing
                assert rel_from_col is not None
                assert rel_to_col is not None
                assert rel_from_tbl is not None
                assert rel_to_tbl is not None

                rel_desc = f"{rel_from_tbl.table_name}.{rel_from_col.column_name} → {rel_to_tbl.table_name}.{rel_to_col.column_name}"
                print(f"\n   Processing: {rel_desc}...")

                try:
                    ct_result = await analyze_cross_table_quality(rel, duckdb_conn, session)
                    if not ct_result.success:
                        print(f"      SKIPPED: {ct_result.error}")
                        await session.rollback()
                        continue

                    quality = ct_result.unwrap()
                    print(f"      Joined rows: {quality.joined_row_count}")
                    print(
                        f"      Cross-table correlations: {len(quality.cross_table_correlations)}"
                    )
                    print(f"      Multicollinearity: {quality.overall_severity}")
                    print(f"      Quality issues: {len(quality.issues)}")

                    # Show notable issues
                    if quality.issues:
                        print("      Notable issues:")
                        for issue in quality.issues[:3]:
                            print(f"         [{issue.severity}] {issue.message[:60]}...")

                    cross_table_count += 1
                except Exception as e:
                    print(f"      ERROR: {e}")
                    await session.rollback()

            print(f"\n   Analyzed {cross_table_count} relationships")

        # Summary
        await _print_cross_table_summary(session)
        await print_database_summary(session, duckdb_conn)

    await cleanup_connections()

    print("\n" + "=" * 70)
    print("Phase 6 COMPLETE")
    print("=" * 70)
    print("\nNext: Run run_phase7_temporal.py for temporal analysis")
    return 0


async def _print_cross_table_summary(session: Any) -> None:
    """Print summary of cross-table quality analysis results."""
    from dataraum_context.analysis.correlation.db_models import (
        CrossTableCorrelationDB,
        MulticollinearityGroup,
        QualityIssueDB,
    )

    # Cross-table counts
    cross_corr_count = (
        await session.execute(select(func.count(CrossTableCorrelationDB.correlation_id)))
    ).scalar()
    mc_group_count = (
        await session.execute(select(func.count(MulticollinearityGroup.group_id)))
    ).scalar()
    quality_issue_count = (
        await session.execute(select(func.count(QualityIssueDB.issue_id)))
    ).scalar()

    # Count by issue type
    issue_types = {}
    for issue_type in ["redundant_column", "unexpected_correlation", "multicollinearity"]:
        count = (
            await session.execute(
                select(func.count(QualityIssueDB.issue_id)).where(
                    QualityIssueDB.issue_type == issue_type
                )
            )
        ).scalar()
        issue_types[issue_type] = count

    print("\nCross-Table Quality Analysis Results:")
    print(f"   Cross-table correlations: {cross_corr_count}")
    print(f"   Multicollinearity groups: {mc_group_count}")
    print(f"   Quality issues: {quality_issue_count}")
    if any(issue_types.values()):
        print("   Issues by type:")
        for issue_type, count in issue_types.items():
            if count > 0:
                print(f"      {issue_type}: {count}")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
