"""MCP Server implementation for DataRaum.

Exposes high-level tools that call library functions directly (no HTTP).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from dataraum.mcp.formatters import (
    format_actions_report,
    format_context_for_llm,
    format_contract_evaluation,
    format_entropy_summary,
    format_query_result,
)


def create_server() -> Server:
    """Create and configure the MCP server with DataRaum tools."""
    server = Server("dataraum")

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="get_context",
                description=(
                    "Get the full data context document for AI analysis. "
                    "Returns schema, relationships, semantic annotations, and data quality info."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_dir": {
                            "type": "string",
                            "description": "Path to pipeline output directory",
                        },
                    },
                    "required": ["output_dir"],
                },
            ),
            Tool(
                name="get_entropy",
                description=(
                    "Get entropy analysis showing data uncertainty by dimension. "
                    "Helps understand what assumptions the system makes and what needs fixing."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_dir": {
                            "type": "string",
                            "description": "Path to pipeline output directory",
                        },
                        "table_name": {
                            "type": "string",
                            "description": "Optional: filter to a specific table",
                        },
                    },
                    "required": ["output_dir"],
                },
            ),
            Tool(
                name="evaluate_contract",
                description=(
                    "Evaluate data quality against a contract. "
                    "Returns compliance status, dimension scores, and violations."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_dir": {
                            "type": "string",
                            "description": "Path to pipeline output directory",
                        },
                        "contract_name": {
                            "type": "string",
                            "description": "Contract to evaluate (e.g., 'aggregation_safe', 'executive_dashboard')",
                        },
                    },
                    "required": ["output_dir", "contract_name"],
                },
            ),
            Tool(
                name="query",
                description=(
                    "Execute a natural language query against the data. "
                    "Returns answer, confidence level, generated SQL, and data."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_dir": {
                            "type": "string",
                            "description": "Path to pipeline output directory",
                        },
                        "question": {
                            "type": "string",
                            "description": "Natural language question about the data",
                        },
                        "contract_name": {
                            "type": "string",
                            "description": "Optional: contract to evaluate against",
                        },
                    },
                    "required": ["output_dir", "question"],
                },
            ),
            Tool(
                name="get_actions",
                description=(
                    "Get prioritized resolution actions to improve data quality. "
                    "Returns actionable steps with priority, effort, affected columns, and expected impact."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_dir": {
                            "type": "string",
                            "description": "Path to pipeline output directory",
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Optional: filter to a specific priority level",
                        },
                        "table_name": {
                            "type": "string",
                            "description": "Optional: filter to actions affecting a specific table",
                        },
                    },
                    "required": ["output_dir"],
                },
            ),
        ]

    @server.call_tool()  # type: ignore[no-untyped-call, untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute a tool and return results."""
        output_dir = Path(arguments.get("output_dir", "./pipeline_output"))

        if name == "get_context":
            result = _get_context(output_dir)
        elif name == "get_entropy":
            table_name = arguments.get("table_name")
            result = _get_entropy(output_dir, table_name)
        elif name == "evaluate_contract":
            contract_name = arguments["contract_name"]
            result = _evaluate_contract(output_dir, contract_name)
        elif name == "query":
            question = arguments["question"]
            contract_name = arguments.get("contract_name")
            result = _query(output_dir, question, contract_name)
        elif name == "get_actions":
            priority = arguments.get("priority")
            table_name = arguments.get("table_name")
            result = _get_actions(output_dir, priority, table_name)
        else:
            result = f"Unknown tool: {name}"

        return [TextContent(type="text", text=result)]

    return server


def _get_context(output_dir: Path) -> str:
    """Get formatted context document."""
    from sqlalchemy import select

    from dataraum.cli.common import get_manager
    from dataraum.graphs.context import build_execution_context, format_context_for_prompt
    from dataraum.storage import Source, Table

    manager = get_manager(output_dir)

    try:
        with manager.session_scope() as session:
            sources_result = session.execute(select(Source))
            sources = sources_result.scalars().all()

            if not sources:
                return "Error: No sources found in database"

            source = sources[0]

            tables_result = session.execute(
                select(Table).where(Table.source_id == source.source_id)
            )
            tables = tables_result.scalars().all()

            if not tables:
                return "Error: No tables found. Run pipeline first."

            table_ids = [t.table_id for t in tables]

            with manager.duckdb_cursor() as cursor:
                context = build_execution_context(
                    session=session,
                    table_ids=table_ids,
                    duckdb_conn=cursor,
                )

            formatted = format_context_for_prompt(context)
            return format_context_for_llm(source.name, formatted)
    finally:
        manager.close()


def _get_entropy(output_dir: Path, table_name: str | None = None) -> str:
    """Get entropy summary."""
    from sqlalchemy import select

    from dataraum.cli.common import get_manager
    from dataraum.entropy.db_models import (
        EntropySnapshotRecord,
    )
    from dataraum.entropy.interpretation_db_models import EntropyInterpretationRecord
    from dataraum.storage import Source

    manager = get_manager(output_dir)

    try:
        with manager.session_scope() as session:
            sources_result = session.execute(select(Source))
            sources = sources_result.scalars().all()

            if not sources:
                return "Error: No sources found"

            source = sources[0]

            # Get snapshot
            snapshot_result = session.execute(
                select(EntropySnapshotRecord)
                .where(EntropySnapshotRecord.source_id == source.source_id)
                .order_by(EntropySnapshotRecord.snapshot_at.desc())
                .limit(1)
            )
            snapshot = snapshot_result.scalar_one_or_none()

            if not snapshot:
                return "Error: No entropy data. Run entropy phase first."

            # Get interpretations
            interp_query = select(EntropyInterpretationRecord).where(
                EntropyInterpretationRecord.source_id == source.source_id
            )

            if table_name:
                interp_query = interp_query.where(
                    EntropyInterpretationRecord.table_name == table_name
                )

            interp_query = interp_query.order_by(EntropyInterpretationRecord.composite_score.desc())
            interp_result = session.execute(interp_query)
            interpretations = interp_result.scalars().all()

            return format_entropy_summary(source.name, snapshot, interpretations, table_name)
    finally:
        manager.close()


def _evaluate_contract(output_dir: Path, contract_name: str) -> str:
    """Evaluate a contract."""
    from sqlalchemy import select

    from dataraum.cli.common import get_manager
    from dataraum.entropy.analysis.aggregator import ColumnSummary, EntropyAggregator
    from dataraum.entropy.contracts import evaluate_contract, get_contract
    from dataraum.entropy.core.storage import EntropyRepository
    from dataraum.storage import Source, Table

    manager = get_manager(output_dir)

    try:
        with manager.session_scope() as session:
            sources_result = session.execute(select(Source))
            sources = sources_result.scalars().all()

            if not sources:
                return "Error: No sources found"

            source = sources[0]

            tables_result = session.execute(
                select(Table).where(Table.source_id == source.source_id)
            )
            tables = tables_result.scalars().all()

            if not tables:
                return "Error: No tables found"

            table_ids = [t.table_id for t in tables]

            # Build column summaries
            repo = EntropyRepository(session)
            aggregator = EntropyAggregator()

            typed_table_ids = repo.get_typed_table_ids(table_ids)
            column_summaries: dict[str, ColumnSummary] = {}
            compound_risks: list[Any] = []

            if typed_table_ids:
                table_map, column_map = repo.get_table_column_mapping(typed_table_ids)
                entropy_objects = repo.load_for_tables(typed_table_ids, enforce_typed=True)

                if entropy_objects:
                    column_summaries, _ = aggregator.summarize_columns_by_table(
                        entropy_objects=entropy_objects,
                        table_map=table_map,
                        column_map=column_map,
                    )
                    for summary in column_summaries.values():
                        compound_risks.extend(summary.compound_risks)

            profile = get_contract(contract_name)
            if profile is None:
                return f"Error: Contract not found: {contract_name}"

            evaluation = evaluate_contract(column_summaries, contract_name, compound_risks)
            return format_contract_evaluation(evaluation, profile)
    finally:
        manager.close()


def _query(output_dir: Path, question: str, contract_name: str | None = None) -> str:
    """Execute a natural language query."""
    from sqlalchemy import select

    from dataraum.cli.common import get_manager
    from dataraum.query import answer_question
    from dataraum.storage import Source

    manager = get_manager(output_dir)

    try:
        with manager.session_scope() as session:
            sources_result = session.execute(select(Source))
            sources = sources_result.scalars().all()

            if not sources:
                return "Error: No sources found"

            source = sources[0]

            with manager.duckdb_cursor() as cursor:
                result = answer_question(
                    question=question,
                    session=session,
                    duckdb_conn=cursor,
                    source_id=source.source_id,
                    contract=contract_name,
                    manager=manager,
                )

            if not result.success or not result.value:
                return f"Error: {result.error}"

            return format_query_result(result.value)
    finally:
        manager.close()


def _get_actions(
    output_dir: Path,
    priority: str | None = None,
    table_name: str | None = None,
) -> str:
    """Get prioritized resolution actions."""
    from collections import defaultdict

    from sqlalchemy import select

    from dataraum.cli.common import get_manager
    from dataraum.entropy.analysis.aggregator import EntropyAggregator
    from dataraum.entropy.contracts import evaluate_all_contracts
    from dataraum.entropy.core.storage import EntropyRepository
    from dataraum.entropy.db_models import EntropyObjectRecord
    from dataraum.entropy.interpretation_db_models import EntropyInterpretationRecord
    from dataraum.storage import Column, Source, Table

    manager = get_manager(output_dir)

    try:
        with manager.session_scope() as session:
            # Get source
            sources_result = session.execute(select(Source))
            sources = sources_result.scalars().all()

            if not sources:
                return "Error: No sources found"

            source = sources[0]

            # Get tables
            tables_result = session.execute(
                select(Table).where(Table.source_id == source.source_id)
            )
            tables = tables_result.scalars().all()

            if not tables:
                return "Error: No tables found. Run pipeline first."

            table_ids = [t.table_id for t in tables]

            # Build column_id -> column_key mapping
            col_id_to_key: dict[str, str] = {}
            for tbl in tables:
                cols_result = session.execute(select(Column).where(Column.table_id == tbl.table_id))
                for col in cols_result.scalars().all():
                    col_id_to_key[col.column_id] = f"{tbl.table_name}.{col.column_name}"

            # Get column summaries from entropy aggregator
            repo = EntropyRepository(session)
            aggregator = EntropyAggregator()

            typed_table_ids = repo.get_typed_table_ids(table_ids)
            column_summaries: dict[str, Any] = {}
            compound_risks: list[Any] = []

            if typed_table_ids:
                table_map, column_map = repo.get_table_column_mapping(typed_table_ids)
                entropy_objects = repo.load_for_tables(typed_table_ids, enforce_typed=True)

                if entropy_objects:
                    column_summaries, _ = aggregator.summarize_columns_by_table(
                        entropy_objects=entropy_objects,
                        table_map=table_map,
                        column_map=column_map,
                    )
                    for summary in column_summaries.values():
                        compound_risks.extend(summary.compound_risks)

            # Get LLM interpretations with resolution actions
            interp_result = session.execute(
                select(EntropyInterpretationRecord).where(
                    EntropyInterpretationRecord.source_id == source.source_id,
                    EntropyInterpretationRecord.column_name.isnot(None),
                )
            )
            interp_by_col: dict[str, Any] = {}
            for interp in interp_result.scalars().all():
                col_key = f"{interp.table_name}.{interp.column_name}"
                interp_by_col[col_key] = interp

            # Get entropy objects for evidence
            entropy_objects_by_col: dict[str, list[Any]] = defaultdict(list)
            if table_ids:
                eo_result = session.execute(
                    select(EntropyObjectRecord)
                    .where(EntropyObjectRecord.table_id.in_(table_ids))
                    .order_by(EntropyObjectRecord.score.desc())
                )
                for obj in eo_result.scalars().all():
                    col_key = col_id_to_key.get(obj.column_id, "") if obj.column_id else ""
                    if col_key:
                        entropy_objects_by_col[col_key].append(obj)

            # Get contract violations
            evaluations = evaluate_all_contracts(column_summaries, compound_risks)
            violation_dims: dict[str, list[str]] = {}
            for eval_result in evaluations.values():
                for v in eval_result.violations:
                    if v.dimension:
                        violation_dims.setdefault(v.dimension, []).extend(v.affected_columns)

            # Merge actions from all sources
            actions = _merge_actions(
                column_summaries=column_summaries,
                interp_by_col=interp_by_col,
                entropy_objects_by_col=entropy_objects_by_col,
                violation_dims=violation_dims,
            )

            # Apply filters
            if priority:
                actions = [a for a in actions if a["priority"] == priority]
            if table_name:
                actions = [
                    a
                    for a in actions
                    if any(col.startswith(f"{table_name}.") for col in a["affected_columns"])
                ]

            return format_actions_report(source.name, actions, priority, table_name)
    finally:
        manager.close()


def _merge_actions(
    column_summaries: dict[str, Any],
    interp_by_col: dict[str, Any],
    entropy_objects_by_col: dict[str, list[Any]],
    violation_dims: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Merge actions from all sources into a unified list."""
    actions_map: dict[str, dict[str, Any]] = {}

    # From ColumnSummary.top_resolution_hints (detector source)
    for col_key, summary in column_summaries.items():
        for hint in summary.top_resolution_hints:
            if hint.action not in actions_map:
                actions_map[hint.action] = {
                    "action": hint.action,
                    "priority": "medium",
                    "description": hint.description,
                    "effort": hint.effort,
                    "expected_impact": "",
                    "parameters": {},
                    "affected_columns": [],
                    "cascade_dimensions": list(hint.cascade_dimensions),
                    "max_reduction": hint.expected_entropy_reduction,
                    "total_reduction": 0.0,
                    "from_llm": False,
                    "from_detector": True,
                    "fixes_violations": [],
                    "evidence": [],
                }
            ma = actions_map[hint.action]
            if col_key not in ma["affected_columns"]:
                ma["affected_columns"].append(col_key)
            ma["max_reduction"] = max(ma["max_reduction"], hint.expected_entropy_reduction)
            ma["total_reduction"] += hint.expected_entropy_reduction

    # From LLM interpretation resolution_actions_json
    for col_key, interp in interp_by_col.items():
        actions = interp.resolution_actions_json
        if isinstance(actions, dict):
            actions = list(actions.values()) if actions else []
        elif not isinstance(actions, list):
            continue

        for action_dict in actions:
            if not isinstance(action_dict, dict):
                continue

            action_name = action_dict.get("action", "")
            if not action_name:
                continue

            if action_name not in actions_map:
                actions_map[action_name] = {
                    "action": action_name,
                    "priority": "medium",
                    "description": "",
                    "effort": "medium",
                    "expected_impact": "",
                    "parameters": {},
                    "affected_columns": [],
                    "cascade_dimensions": [],
                    "max_reduction": 0.0,
                    "total_reduction": 0.0,
                    "from_llm": True,
                    "from_detector": False,
                    "fixes_violations": [],
                    "evidence": [],
                }

            ma = actions_map[action_name]
            ma["from_llm"] = True

            # LLM provides richer metadata
            if not ma["description"]:
                ma["description"] = action_dict.get("description", "")
            if not ma["expected_impact"]:
                ma["expected_impact"] = action_dict.get("expected_impact", "")
            if not ma["parameters"]:
                ma["parameters"] = action_dict.get("parameters", {})

            # Priority from LLM
            llm_priority = action_dict.get("priority", "medium")
            ma["priority"] = str(llm_priority).lower()

            if action_dict.get("effort"):
                ma["effort"] = str(action_dict["effort"])

            if col_key not in ma["affected_columns"]:
                ma["affected_columns"].append(col_key)

    # Map contract violations to actions
    for dim, cols in violation_dims.items():
        for ma in actions_map.values():
            overlap = set(ma["affected_columns"]) & set(cols)
            if overlap and dim not in ma["fixes_violations"]:
                ma["fixes_violations"].append(dim)

    # Calculate priority scores
    effort_factors = {"low": 1.0, "medium": 2.0, "high": 4.0}
    for ma in actions_map.values():
        effort_factor = effort_factors.get(ma["effort"], 2.0)
        impact = ma["total_reduction"] + len(ma["affected_columns"]) * 0.1
        ma["priority_score"] = impact / effort_factor

    # Sort by priority bucket then by priority_score
    priority_order = {"high": 0, "medium": 1, "low": 2}
    result = sorted(
        actions_map.values(),
        key=lambda a: (priority_order.get(a["priority"], 1), -a["priority_score"]),
    )

    return result


async def run_server() -> None:
    """Run the MCP server using stdio transport."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Entry point for dataraum-mcp command."""
    import asyncio

    asyncio.run(run_server())


if __name__ == "__main__":
    main()
