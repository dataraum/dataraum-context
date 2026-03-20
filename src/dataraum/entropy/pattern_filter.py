"""Business pattern filter for entropy gate scores.

Classifies detector findings as expected business behavior via LLM at gate
time.  Score discount: ``score *= (1 - filter_confidence)`` when confidence
>= 0.8.  DataFix records (``confirm_expected_pattern``) bypass LLM with
confidence 1.0.
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from sqlalchemy import select

from dataraum.core.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from dataraum.entropy.db_models import EntropyObjectRecord

logger = get_logger(__name__)

CONFIDENCE_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Pydantic output model for LLM structured response
# ---------------------------------------------------------------------------


class PatternClassification(BaseModel):
    """Classification of a single finding."""

    object_id: str
    is_expected_business_pattern: bool
    confidence: float = Field(ge=0.0, le=1.0)
    pattern_description: str = ""
    business_rule: str = ""


class PatternClassificationBatch(BaseModel):
    """Batch response from the LLM."""

    classifications: list[PatternClassification]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_pattern_filter(
    session: Session,
    source_id: str,
    records: list[EntropyObjectRecord],
) -> list[EntropyObjectRecord]:
    """Classify findings and discount scores for expected business patterns.

    Runs inside ``aggregate_at_gate()`` after loading records, before
    aggregation.  Modifies records in-place (score, filter fields) and
    persists changes.

    Args:
        session: SQLAlchemy session (caller manages commit).
        source_id: Source ID for DataFix lookup.
        records: Loaded ``EntropyObjectRecord`` rows.

    Returns:
        The same list with scores discounted where appropriate.
    """
    from dataraum.llm.config import load_llm_config

    # 1. Check feature config — fail hard if config is missing
    config = load_llm_config()

    feature = config.features.business_pattern_filter
    if feature is None or not feature.enabled:
        return records

    # 2. Filter to candidates: score > 0 and not already classified
    candidates = [r for r in records if r.score > 0 and r.filter_confidence is None]
    if not candidates:
        return records

    # 3. Apply DataFix overrides (confidence = 1.0)
    remaining = _apply_datafix_overrides(session, source_id, candidates)

    # 4. LLM classification for remaining candidates
    if remaining:
        _classify_with_llm(session, remaining, config)

    # 5. Discount scores for high-confidence classifications
    for record in records:
        if (
            record.filter_confidence is not None
            and record.filter_confidence >= CONFIDENCE_THRESHOLD
        ):
            record.score = round(record.score * (1 - record.filter_confidence), 4)

    return records


# ---------------------------------------------------------------------------
# DataFix override
# ---------------------------------------------------------------------------


def _apply_datafix_overrides(
    session: Session,
    source_id: str,
    candidates: list[EntropyObjectRecord],
) -> list[EntropyObjectRecord]:
    """Check DataFix for user-confirmed business patterns.

    ``confirm_expected_pattern`` sets confidence to 1.0 (full discount).

    Returns:
        Records that still need LLM classification.
    """
    from dataraum.pipeline.fixes.models import DataFix

    fixes = list(
        session.execute(
            select(DataFix).where(
                DataFix.source_id == source_id,
                DataFix.action == "confirm_expected_pattern",
                DataFix.status == "applied",
            )
        )
        .scalars()
        .all()
    )

    if not fixes:
        return candidates

    # Build lookup: (table_name, column_name | None) -> fix
    fix_lookup: dict[tuple[str, str | None], DataFix] = {}
    for fix in fixes:
        fix_lookup[(fix.table_name, fix.column_name)] = fix

    remaining: list[EntropyObjectRecord] = []
    for record in candidates:
        table_name, column_name = _parse_target(record.target)
        fix = fix_lookup.get((table_name, column_name)) or fix_lookup.get((table_name, None))
        if fix:
            record.filter_confidence = 1.0
            params = fix.payload.get("parameters", {}) if fix.payload else {}
            record.expected_business_pattern = params.get("pattern_type", fix.action)
            record.business_rule = params.get("description", fix.description or fix.action)
        else:
            remaining.append(record)

    return remaining


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------


def _classify_with_llm(
    session: Session,
    candidates: list[EntropyObjectRecord],
    config: Any,
) -> None:
    """Group candidates by table and run Haiku classification."""
    from dataraum.llm.prompts import PromptRenderer
    from dataraum.llm.providers import create_provider
    from dataraum.llm.providers.base import (
        ConversationRequest,
        Message,
        ToolDefinition,
    )
    from dataraum.storage import Column as ColumnModel
    from dataraum.storage import Table

    feature = config.features.business_pattern_filter
    provider_config = config.providers.get(config.active_provider)
    if not provider_config:
        raise RuntimeError(f"Pattern filter: provider '{config.active_provider}' not configured")

    provider = create_provider(config.active_provider, provider_config.model_dump())

    model = provider.get_model_for_tier(feature.model_tier)
    renderer = PromptRenderer()

    # Build table schema cache
    table_ids = {r.table_id for r in candidates if r.table_id}
    tables = (
        {
            t.table_id: t
            for t in session.execute(select(Table).where(Table.table_id.in_(table_ids)))
            .scalars()
            .all()
        }
        if table_ids
        else {}
    )

    table_schemas: dict[str, str] = {}
    for tid, t in tables.items():
        cols = list(
            session.execute(
                select(ColumnModel.column_name, ColumnModel.resolved_type).where(
                    ColumnModel.table_id == tid
                )
            ).all()
        )
        table_schemas[t.table_name] = ", ".join(f"{c[0]} ({c[1]})" for c in cols)

    # Group candidates by table
    by_table: dict[str, list[EntropyObjectRecord]] = defaultdict(list)
    for record in candidates:
        table_name, _ = _parse_target(record.target)
        by_table[table_name].append(record)

    # Tool definition
    tool = ToolDefinition(
        name="classify_patterns",
        description="Classify each finding as expected business pattern or genuine issue.",
        input_schema=PatternClassificationBatch.model_json_schema(),
    )

    def _classify_table(table_name: str, table_records: list[EntropyObjectRecord]) -> None:
        findings = [
            {
                "object_id": r.object_id,
                "detector_id": r.detector_id,
                "target": r.target,
                "score": r.score,
                "sub_dimension": r.sub_dimension,
                "evidence": r.evidence,
            }
            for r in table_records
        ]

        schema = table_schemas.get(table_name, "unknown")
        system_prompt, user_prompt, temperature = renderer.render_split(
            "business_pattern_filter",
            {
                "table_name": table_name,
                "table_schema": schema,
                "findings_json": json.dumps(findings, default=str),
            },
        )

        request = ConversationRequest(
            messages=[Message(role="user", content=user_prompt)],
            system=system_prompt,
            tools=[tool],
            tool_choice={"type": "tool", "name": "classify_patterns"},
            max_tokens=4096,
            temperature=temperature,
            model=model,
        )

        result = provider.converse(request)
        if not result.success:
            raise RuntimeError(f"Pattern filter LLM call failed for {table_name}: {result.error}")

        response = result.unwrap()
        if not response.tool_calls:
            raise RuntimeError(
                f"Pattern filter: LLM did not call classify_patterns tool for {table_name}"
            )

        batch = PatternClassificationBatch.model_validate(response.tool_calls[0].input)

        # Apply classifications to records
        record_map = {r.object_id: r for r in table_records}
        for cls in batch.classifications:
            rec = record_map.get(cls.object_id)
            if rec is None:
                continue
            if cls.is_expected_business_pattern:
                rec.filter_confidence = cls.confidence
                rec.expected_business_pattern = cls.pattern_description
                rec.business_rule = cls.business_rule
            else:
                # Mark as classified (idempotency) but no discount
                rec.filter_confidence = 0.0

    # Run classifications (parallel for multiple tables).
    # Thread safety: _classify_table only does LLM I/O and sets in-memory
    # attributes on ORM objects — no session operations inside threads.
    if len(by_table) > 1:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(_classify_table, tname, trecs) for tname, trecs in by_table.items()
            ]
            for f in futures:
                f.result()  # propagate exceptions
    else:
        for tname, trecs in by_table.items():
            _classify_table(tname, trecs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_target(target: str) -> tuple[str, str | None]:
    """Parse 'column:table.col' or 'table:name' into (table_name, column_name | None)."""
    if ":" not in target:
        return target, None
    scope, ref = target.split(":", 1)
    if scope == "column" and "." in ref:
        parts = ref.split(".", 1)
        return parts[0], parts[1]
    return ref, None
