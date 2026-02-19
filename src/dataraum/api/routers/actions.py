"""Resolution actions endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from dataraum.api.deps import SessionDep
from dataraum.api.schemas import ActionsResponse, ResolutionActionResponse
from dataraum.mcp.formatters import format_actions_report
from dataraum.mcp.server import _merge_actions
from dataraum.entropy.analysis.aggregator import EntropyAggregator
from dataraum.entropy.contracts import evaluate_all_contracts
from dataraum.entropy.core.storage import EntropyRepository
from dataraum.entropy.db_models import EntropyObjectRecord
from dataraum.entropy.interpretation_db_models import EntropyInterpretationRecord
from dataraum.storage import Column, Source, Table

router = APIRouter()


@router.get("/actions/{source_id}", response_model=ActionsResponse)
def get_actions(
    source_id: str,
    session: SessionDep,
    priority: str | None = Query(None, description="Filter by priority: high, medium, low"),
    table_name: str | None = Query(None, description="Filter by table name"),
) -> ActionsResponse:
    """Get prioritized resolution actions to improve data quality.

    Returns actionable steps with priority, effort, affected columns, and expected impact.
    Actions are sorted by priority bucket then by priority_score.
    """
    # Verify source exists
    stmt = select(Source).where(Source.source_id == source_id)
    result = session.execute(stmt)
    source = result.scalar_one_or_none()

    if source is None:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")

    # Get tables
    tables_result = session.execute(select(Table).where(Table.source_id == source_id))
    tables = tables_result.scalars().all()

    if not tables:
        return ActionsResponse(
            source_id=source_id,
            actions=[],
            summary={"high": 0, "medium": 0, "low": 0},
            prompt_text="No tables found. Run pipeline first.",
        )

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
            EntropyInterpretationRecord.source_id == source_id,
            EntropyInterpretationRecord.column_name.isnot(None),
        )
    )
    interp_by_col: dict[str, Any] = {}
    for interp in interp_result.scalars().all():
        col_key = f"{interp.table_name}.{interp.column_name}"
        interp_by_col[col_key] = interp

    # Get entropy objects for evidence
    from collections import defaultdict

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

    # Build response
    action_responses = [
        ResolutionActionResponse(
            action=a["action"],
            priority=a["priority"],
            description=a["description"],
            effort=a["effort"],
            expected_impact=a["expected_impact"],
            affected_columns=a["affected_columns"],
            parameters=a["parameters"],
            priority_score=a["priority_score"],
            from_llm=a["from_llm"],
            from_detector=a["from_detector"],
            fixes_violations=a["fixes_violations"],
        )
        for a in actions
    ]

    # Summary counts
    summary = {"high": 0, "medium": 0, "low": 0}
    for a in actions:
        p = a["priority"]
        if p in summary:
            summary[p] += 1

    # Format for LLM
    prompt_text = format_actions_report(source.name, actions, priority, table_name)

    return ActionsResponse(
        source_id=source_id,
        actions=action_responses,
        summary=summary,
        prompt_text=prompt_text,
    )
