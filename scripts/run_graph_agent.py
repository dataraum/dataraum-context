#!/usr/bin/env python3
"""Graph Agent Test Script.

This script demonstrates the graph agent executing business metrics
on the test finance data.

Usage:
    uv run python scripts/run_graph_agent.py

Prerequisites:
    - Run phase 1-5 first (import, typing, statistics, relationships, semantic)
"""

import asyncio
import sys

from infra import (
    cleanup_connections,
    get_duckdb_conn,
    get_test_session,
    print_database_summary,
)
from sqlalchemy import select


async def main() -> int:
    """Run graph agent demo."""
    print("\n" + "=" * 60)
    print("GRAPH AGENT TEST")
    print("=" * 60)

    # Connect to existing databases
    session_factory = await get_test_session()
    duckdb_conn = get_duckdb_conn()

    async with session_factory() as session:
        # Check prerequisites - query for any existing tables
        from dataraum_context.storage import Table

        tables_stmt = select(Table)
        tables_result = await session.execute(tables_stmt)
        tables = tables_result.scalars().all()

        if not tables:
            print("\nERROR: No tables found. Run phase 1 first.")
            return 1

        # Print current state
        await print_database_summary(session, duckdb_conn)

        print("\n" + "-" * 60)
        print("GRAPH LOADER STATUS")
        print("-" * 60)

        # Load graphs
        from dataraum_context.graphs import GraphLoader

        loader = GraphLoader()
        graphs = loader.load_all()

        print(f"\nLoaded {len(graphs)} graphs:")

        # Separate by type
        filters = loader.get_filter_graphs()
        metrics = loader.get_metric_graphs()

        print(f"\n  Filter Graphs ({len(filters)}):")
        for g in filters:
            applies_to = ""
            if g.metadata.applies_to:
                if g.metadata.applies_to.semantic_role:
                    applies_to = f" [role: {g.metadata.applies_to.semantic_role}]"
                elif g.metadata.applies_to.data_type:
                    applies_to = f" [type: {g.metadata.applies_to.data_type}]"
                elif g.metadata.applies_to.column_pattern:
                    applies_to = f" [pattern: {g.metadata.applies_to.column_pattern[:20]}...]"
            print(f"    - {g.graph_id}: {g.metadata.name}{applies_to}")

        print(f"\n  Metric Graphs ({len(metrics)}):")
        for g in metrics:
            scope = getattr(g, "scope", None)
            scope_str = f" [{scope.value}]" if scope else ""
            print(f"    - {g.graph_id}: {g.metadata.name}{scope_str}")

        # Show any load errors
        errors = loader.get_load_errors()
        if errors:
            print(f"\n  Load Errors ({len(errors)}):")
            for err in errors:
                print(f"    - {err}")

        print("\n" + "-" * 60)
        print("APPLICABLE FILTERS FOR DATASET")
        print("-" * 60)

        # Get column metadata from semantic analysis
        from dataraum_context.analysis.semantic.db_models import SemanticAnnotation
        from dataraum_context.analysis.typing.db_models import TypeDecision
        from dataraum_context.storage import Column

        # tables already loaded above

        # Build column metadata for filter matching
        columns_metadata = []
        for table in tables:
            cols_stmt = select(Column).where(Column.table_id == table.table_id)
            columns = (await session.execute(cols_stmt)).scalars().all()

            for col in columns:
                # Get type decision
                type_stmt = select(TypeDecision).where(TypeDecision.column_id == col.column_id)
                type_dec = (await session.execute(type_stmt)).scalar_one_or_none()

                # Get semantic annotation
                sem_stmt = select(SemanticAnnotation).where(
                    SemanticAnnotation.column_id == col.column_id
                )
                sem_ann = (await session.execute(sem_stmt)).scalar_one_or_none()

                columns_metadata.append(
                    {
                        "column_name": col.column_name,
                        "table_name": table.table_name,
                        "data_type": type_dec.resolved_type if type_dec else None,
                        "semantic_role": sem_ann.semantic_role if sem_ann else None,
                        "has_profile": True,  # Assume profiles exist after phase 3
                    }
                )

        # Get applicable filters
        summary = loader.get_quality_filter_summary(columns_metadata)

        print(f"\n  Total unique filters: {summary['total_filters']}")
        print(f"  Filter coverage: {summary['filter_coverage']:.1%}")
        print(f"  Filters applied: {', '.join(summary['filter_ids']) or 'none'}")

        # Show filter breakdown by table
        filters_by_col = loader.get_filters_for_dataset(columns_metadata)
        columns_with_filters = [
            (col["table_name"], col["column_name"], filters_by_col.get(col["column_name"], []))
            for col in columns_metadata
            if filters_by_col.get(col["column_name"])
        ]

        if columns_with_filters:
            print("\n  Columns with filters:")
            for table, col, col_filters in columns_with_filters[:15]:
                filter_names = [f.graph_id for f in col_filters]
                print(f"    - {table}.{col}: {', '.join(filter_names)}")
            if len(columns_with_filters) > 15:
                print(f"    ... and {len(columns_with_filters) - 15} more")

        print("\n" + "-" * 60)
        print("GRAPH EXECUTION CONTEXT (Sample)")
        print("-" * 60)

        # Build execution context for first table
        if tables:
            from dataraum_context.graphs.context import (
                build_execution_context,
                format_context_for_prompt,
            )

            table_ids = [t.table_id for t in tables[:3]]  # First 3 tables
            context = await build_execution_context(
                session=session,
                table_ids=table_ids,
                duckdb_conn=duckdb_conn,
            )

            print(f"\n  Tables in context: {context.total_tables}")
            print(f"  Columns in context: {context.total_columns}")
            print(f"  Relationships: {context.total_relationships}")
            print(f"  Graph pattern: {context.graph_pattern}")

            # Show formatted context (truncated)
            formatted = format_context_for_prompt(context)
            lines = formatted.split("\n")
            print("\n  Formatted context preview:")
            for line in lines[:20]:
                print(f"    {line}")
            if len(lines) > 20:
                print(f"    ... ({len(lines) - 20} more lines)")

        print("\n" + "=" * 60)
        print("GRAPH AGENT TEST COMPLETE")
        print("=" * 60 + "\n")

    await cleanup_connections()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
