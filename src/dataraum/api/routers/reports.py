"""Reports router - HTML report pages for query executions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select

from dataraum.api.deps import get_manager
from dataraum.core.logging import get_logger
from dataraum.query.db_models import QueryExecutionRecord, QueryLibraryEntry

logger = get_logger(__name__)

router = APIRouter()


def _badge_class(confidence_level: str) -> str:
    """Map confidence level to daisyUI badge class."""
    mapping = {
        "GREEN": "badge-success",
        "YELLOW": "badge-warning",
        "ORANGE": "badge-warning",
        "RED": "badge-error",
    }
    return mapping.get(confidence_level.upper(), "badge-neutral")


def _confidence_label(confidence_level: str) -> str:
    """Map confidence level to human-readable label."""
    mapping = {
        "GREEN": "High Confidence",
        "YELLOW": "Moderate Confidence",
        "ORANGE": "Low Confidence",
        "RED": "Insufficient Confidence",
    }
    return mapping.get(confidence_level.upper(), confidence_level)


def _action_label(action: str | None) -> str:
    """Map entropy action to human-readable label."""
    mapping = {
        "answer_confidently": "Data quality is high for this query",
        "answer_with_assumptions": "Some assumptions were needed due to data uncertainty",
        "ask_or_caveat": "Significant data uncertainty — review assumptions carefully",
        "refuse": "Data quality insufficient for reliable answer",
    }
    return mapping.get(action or "", "")


def _action_alert_class(action: str | None) -> str:
    """Map entropy action to daisyUI alert class."""
    mapping = {
        "answer_confidently": "alert-success",
        "answer_with_assumptions": "alert-warning",
        "ask_or_caveat": "alert-error",
        "refuse": "alert-error",
    }
    return mapping.get(action or "", "")


@router.get("/reports/query/{execution_id}", response_class=HTMLResponse)
async def query_report(execution_id: str, request: Request) -> HTMLResponse:
    """Render an HTML report for a query execution.

    Args:
        execution_id: The execution ID to display
        request: FastAPI request (for templates)

    Returns:
        Rendered HTML report
    """
    from jinja2 import Environment

    templates: Environment = request.app.state.templates

    manager = get_manager()

    try:
        with manager.session_scope() as session:
            # Load execution record
            exec_result = session.execute(
                select(QueryExecutionRecord).where(
                    QueryExecutionRecord.execution_id == execution_id
                )
            )
            execution = exec_result.scalar_one_or_none()

            if not execution:
                raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")

            # Load linked library entry for assumptions
            library_entry = None
            if execution.library_entry_id:
                entry_result = session.execute(
                    select(QueryLibraryEntry).where(
                        QueryLibraryEntry.query_id == execution.library_entry_id
                    )
                )
                library_entry = entry_result.scalar_one_or_none()

            # Build template context
            assumptions = library_entry.assumptions if library_entry else []
            # Normalize assumptions to dicts
            normalized_assumptions = []
            for a in assumptions:
                if isinstance(a, dict):
                    normalized_assumptions.append(a)

            # Re-execute SQL to get results (read-only)
            columns = None
            data = None
            if execution.sql_executed:
                try:
                    with manager.duckdb_cursor() as cursor:
                        result = cursor.execute(execution.sql_executed)
                        columns = [desc[0] for desc in result.description]
                        rows = result.fetchall()
                        data = [dict(zip(columns, row, strict=True)) for row in rows[:100]]
                except Exception as e:
                    logger.warning(f"Could not re-execute SQL for report: {e}")

            context = {
                "execution_id": execution.execution_id,
                "question": execution.question,
                "answer": "",  # Not stored in execution record
                "sql": execution.sql_executed,
                "confidence_level": execution.confidence_level,
                "confidence_label": _confidence_label(execution.confidence_level),
                "badge_class": _badge_class(execution.confidence_level),
                "executed_at": execution.executed_at.strftime("%Y-%m-%d %H:%M:%S")
                if execution.executed_at
                else "-",
                "entropy_score": None,
                "contract": execution.contract_name,
                "interpreted_question": "",
                "metric_type": "",
                "validation_notes": [],
                "assumptions": normalized_assumptions,
                "columns": columns,
                "data": data,
                "entropy_action": execution.entropy_action,
                "entropy_action_label": _action_label(execution.entropy_action),
                "entropy_action_class": _action_alert_class(execution.entropy_action),
            }

            # Enrich from library entry if available
            if library_entry:
                context["interpreted_question"] = library_entry.summary or ""

            template = templates.get_template("reports/query.html")
            html = template.render(**context)
            return HTMLResponse(content=html)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering query report: {e}")
        raise HTTPException(status_code=500, detail="Failed to render report") from e
