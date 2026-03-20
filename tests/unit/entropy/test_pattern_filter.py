"""Tests for business pattern filter."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from dataraum.entropy.db_models import EntropyObjectRecord
from dataraum.entropy.pattern_filter import CONFIDENCE_THRESHOLD, apply_pattern_filter
from dataraum.storage import Source


def _ensure_source(session: Session, source_id: str = "src-1") -> None:
    if not session.get(Source, source_id):
        session.add(Source(source_id=source_id, name=source_id, source_type="csv"))
        session.flush()


def _make_record(
    source_id: str = "src-1",
    target: str = "column:orders.amount",
    score: float = 0.5,
    detector_id: str = "null_ratio",
    **kwargs,
) -> EntropyObjectRecord:
    return EntropyObjectRecord(
        source_id=source_id,
        target=target,
        layer="structural",
        dimension="schema",
        sub_dimension="null_ratio",
        score=score,
        detector_id=detector_id,
        **kwargs,
    )


def _disabled_config():
    """Mock load_llm_config returning disabled feature."""
    from dataraum.llm.config import FeatureConfig, LLMConfig, LLMFeatures, LLMLimits, LLMPrivacy

    return LLMConfig(
        providers={},
        active_provider="anthropic",
        features=LLMFeatures(
            semantic_analysis=FeatureConfig(enabled=True),
            business_pattern_filter=FeatureConfig(enabled=False),
        ),
        limits=LLMLimits(),
        privacy=LLMPrivacy(),
    )


def _enabled_config():
    """Mock load_llm_config returning enabled feature (no provider = LLM skipped)."""
    from dataraum.llm.config import FeatureConfig, LLMConfig, LLMFeatures, LLMLimits, LLMPrivacy

    return LLMConfig(
        providers={},
        active_provider="anthropic",
        features=LLMFeatures(
            semantic_analysis=FeatureConfig(enabled=True),
            business_pattern_filter=FeatureConfig(enabled=True, model_tier="fast"),
        ),
        limits=LLMLimits(),
        privacy=LLMPrivacy(),
    )


class TestPatternFilter:
    """Tests for apply_pattern_filter."""

    def test_filter_disabled_passthrough(self, session: Session):
        """When feature is disabled, records pass through unchanged."""
        _ensure_source(session)
        rec = _make_record(score=0.7)
        session.add(rec)
        session.flush()

        with patch(
            "dataraum.llm.config.load_llm_config",
            return_value=_disabled_config(),
        ):
            result = apply_pattern_filter(session, "src-1", [rec])

        assert len(result) == 1
        assert result[0].score == 0.7
        assert result[0].filter_confidence is None

    def test_filter_skips_zero_score_records(self, session: Session):
        """Records with score=0 are not candidates for classification."""
        _ensure_source(session)
        rec = _make_record(score=0.0)
        session.add(rec)
        session.flush()

        with patch(
            "dataraum.llm.config.load_llm_config",
            return_value=_enabled_config(),
        ):
            result = apply_pattern_filter(session, "src-1", [rec])

        assert result[0].score == 0.0
        assert result[0].filter_confidence is None

    def test_filter_idempotent(self, session: Session):
        """Already-classified records are not re-classified or re-discounted."""
        _ensure_source(session)
        # Simulate a record already classified in a prior call
        rec = _make_record(score=0.05, filter_confidence=0.9)
        session.add(rec)
        session.flush()

        with patch(
            "dataraum.llm.config.load_llm_config",
            return_value=_enabled_config(),
        ):
            result = apply_pattern_filter(session, "src-1", [rec])

        # Score unchanged — discount was already applied in prior call
        assert result[0].filter_confidence == 0.9
        assert result[0].score == 0.05

    def test_filter_discounts_high_confidence(self, session: Session):
        """DataFix with confidence=1.0 discounts score to 0."""
        from dataraum.pipeline.fixes.models import DataFix

        _ensure_source(session)
        rec = _make_record(score=0.6, target="column:orders.amount")
        session.add(rec)
        session.add(
            DataFix(
                source_id="src-1",
                action="confirm_expected_pattern",
                target="metadata",
                dimension="structural.schema.null_ratio",
                description="Expected pattern",
                table_name="orders",
                column_name="amount",
                payload={"parameters": {"pattern_type": "test_pattern"}},
                status="applied",
            )
        )
        session.flush()

        with patch(
            "dataraum.llm.config.load_llm_config",
            return_value=_enabled_config(),
        ):
            result = apply_pattern_filter(session, "src-1", [rec])

        # confidence=1.0 → score * (1-1.0) = 0
        assert result[0].filter_confidence == 1.0
        assert result[0].score == 0.0

    def test_filter_no_discount_below_threshold(self, session: Session):
        """Already-classified records below threshold keep their score."""
        _ensure_source(session)
        rec = _make_record(score=0.6, filter_confidence=0.5)
        session.add(rec)
        session.flush()

        with patch(
            "dataraum.llm.config.load_llm_config",
            return_value=_enabled_config(),
        ):
            result = apply_pattern_filter(session, "src-1", [rec])

        assert result[0].score == 0.6  # unchanged — below threshold

    def test_filter_user_confirmed_zeros_score(self, session: Session):
        """DataFix confirm_expected_pattern sets confidence=1.0 → score zeroed."""
        from dataraum.pipeline.fixes.models import DataFix

        _ensure_source(session)
        rec = _make_record(
            score=0.7,
            target="column:journal_lines.debit",
        )
        session.add(rec)

        # Add a DataFix record
        session.add(
            DataFix(
                source_id="src-1",
                action="confirm_expected_pattern",
                target="metadata",
                dimension="semantic.dimensional.dimensional_patterns",
                description="Double-entry bookkeeping",
                table_name="journal_lines",
                column_name="debit",
                payload={
                    "parameters": {
                        "pattern_type": "mutual_exclusivity",
                        "columns": "debit,credit",
                        "description": "Debit/credit mutual exclusivity",
                    }
                },
                status="applied",
            )
        )
        session.flush()

        with patch(
            "dataraum.llm.config.load_llm_config",
            return_value=_enabled_config(),
        ):
            result = apply_pattern_filter(session, "src-1", [rec])

        assert result[0].filter_confidence == 1.0
        assert result[0].expected_business_pattern == "mutual_exclusivity"
        assert result[0].business_rule == "Debit/credit mutual exclusivity"
        # score * (1 - 1.0) = 0
        assert result[0].score == 0.0

    def test_confidence_threshold_constant(self):
        """CONFIDENCE_THRESHOLD is 0.8."""
        assert CONFIDENCE_THRESHOLD == 0.8
