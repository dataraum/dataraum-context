"""Financial Quality Orchestrator - Complete Financial Domain Analysis.

This is the main entry point for financial quality analysis.

Architecture:
    Layer 1:   Compute metrics (financial.py + topological.py)
    Layer 1.5: Domain rules (fiscal stability, anomalies, quality score)
    Layer 2:   LLM classification (cycle interpretation)
    Layer 3:   LLM interpretation (holistic assessment)

The domain rules in Layer 1.5 provide deterministic, auditable outputs
that complement the LLM's interpretive capabilities.
"""

import logging
from pathlib import Path
from typing import Any

import duckdb
import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dataraum_context.core.models.base import Result
from dataraum_context.llm import LLMService
from dataraum_context.quality.domains.financial import analyze_financial_quality
from dataraum_context.quality.domains.financial_llm import (
    classify_financial_cycle_with_llm,
    interpret_financial_quality_with_llm,
)
from dataraum_context.quality.models import TopologicalAnomaly
from dataraum_context.quality.topological import analyze_topological_quality
from dataraum_context.storage.models_v2.core import Column, Table
from dataraum_context.storage.models_v2.semantic_context import SemanticAnnotation

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Loading
# =============================================================================

_FINANCIAL_CONFIG_CACHE: dict[str, Any] | None = None


def _load_financial_config() -> dict[str, Any]:
    """Load financial domain configuration from YAML."""
    global _FINANCIAL_CONFIG_CACHE
    if _FINANCIAL_CONFIG_CACHE is not None:
        return _FINANCIAL_CONFIG_CACHE

    config_paths = [
        Path("config/domains/financial.yaml"),
        Path(__file__).parent.parent.parent.parent.parent / "config/domains/financial.yaml",
    ]

    for path in config_paths:
        if path.exists():
            with open(path) as f:
                _FINANCIAL_CONFIG_CACHE = yaml.safe_load(f)
                return _FINANCIAL_CONFIG_CACHE

    logger.warning("Financial config not found, using empty config")
    return {}


# =============================================================================
# Layer 1.5: Domain Rules (Deterministic, Auditable)
# =============================================================================


def assess_fiscal_stability(
    stability: Any,  # StabilityAnalysis from topological analysis
    temporal_context: dict[str, Any],
) -> dict[str, Any]:
    """Enhance stability analysis with fiscal period awareness.

    Distinguishes between:
    - Fiscal period effects (expected, recurring changes)
    - Structural changes (unexpected, permanent topology shifts)

    Args:
        stability: StabilityAnalysis object from topological analysis
        temporal_context: Dict with fiscal calendar information

    Returns:
        Dict with enhanced stability assessment including fiscal context
    """
    if stability is None:
        return {
            "stability_level": "unknown",
            "fiscal_context": None,
            "is_fiscal_period_effect": False,
            "pattern_type": "unknown",
        }

    # Extract fiscal calendar info
    current_period = temporal_context.get("current_fiscal_period")
    is_period_end = temporal_context.get("is_period_end", False)
    is_quarter_end = temporal_context.get("is_quarter_end", False)
    is_year_end = temporal_context.get("is_year_end", False)

    # Determine if changes are fiscal period effects
    fiscal_context = None
    is_fiscal_period_effect = False
    pattern_type = "structural_change"  # Default assumption

    stability_level = getattr(stability, "stability_level", "unknown")

    if is_year_end and stability_level in ["significant_changes", "unstable"]:
        fiscal_context = "fiscal_year_end_close"
        is_fiscal_period_effect = True
        pattern_type = "recurring_spike"

    elif is_quarter_end and stability_level in ["minor_changes", "significant_changes"]:
        fiscal_context = "quarter_end_close"
        is_fiscal_period_effect = True
        pattern_type = "recurring_spike"

    elif is_period_end and stability_level == "minor_changes":
        fiscal_context = "month_end_close"
        is_fiscal_period_effect = True
        pattern_type = "recurring_spike"

    elif stability_level in ["significant_changes", "unstable"]:
        fiscal_context = "mid_period"
        is_fiscal_period_effect = False
        pattern_type = "structural_change"

    # Generate interpretation
    interpretation = _interpret_fiscal_stability(
        stability_level, is_fiscal_period_effect, fiscal_context
    )

    return {
        "original_stability_level": stability_level,
        "fiscal_context": fiscal_context,
        "is_fiscal_period_effect": is_fiscal_period_effect,
        "pattern_type": pattern_type,
        "affected_periods": [current_period] if current_period else [],
        "components_added": getattr(stability, "components_added", 0),
        "components_removed": getattr(stability, "components_removed", 0),
        "cycles_added": getattr(stability, "cycles_added", 0),
        "cycles_removed": getattr(stability, "cycles_removed", 0),
        "interpretation": interpretation,
    }


def _interpret_fiscal_stability(
    stability_level: str, is_fiscal_effect: bool, fiscal_context: str | None
) -> str:
    """Generate human-readable interpretation of stability assessment."""
    if is_fiscal_effect:
        if fiscal_context == "fiscal_year_end_close":
            return (
                "Expected topology changes due to fiscal year-end close. "
                "Increased activity and relationship complexity is normal."
            )
        elif fiscal_context == "quarter_end_close":
            return (
                "Recurring topology changes due to quarter-end close. "
                "Period-end spikes are expected."
            )
        elif fiscal_context == "month_end_close":
            return "Minor topology changes due to month-end close. Normal recurring pattern."
        else:
            return "Changes appear related to fiscal period effects."
    else:
        if stability_level == "unstable":
            return (
                "ALERT: Significant structural changes detected outside normal fiscal periods. "
                "Investigate data quality or business process changes."
            )
        elif stability_level == "significant_changes":
            return (
                "WARNING: Notable structural changes detected mid-period. "
                "May indicate data quality issues or business changes."
            )
        else:
            return "Topology is stable with minor expected variations."


def detect_financial_anomalies(
    topological_result: Any,  # TopologicalQualityResult
    classified_cycles: list[dict[str, Any]],
) -> list[TopologicalAnomaly]:
    """Detect financial-specific topological anomalies.

    Anomaly types:
    - excessive_financial_cycles: Too many cycles for financial data
    - unclassified_financial_cycles: Cycles couldn't be classified
    - financial_data_fragmentation: Disconnected components
    - missing_financial_cycles: Expected cycles not found
    - cost_center_isolation: Orphaned components

    Args:
        topological_result: TopologicalQualityResult from analysis
        classified_cycles: List of LLM-classified cycle dicts

    Returns:
        List of TopologicalAnomaly objects with financial context
    """
    anomalies: list[TopologicalAnomaly] = []

    if topological_result is None:
        return anomalies

    # Get table name for affected_tables
    table_name = getattr(topological_result, "table_name", "unknown")

    # Anomaly 1: Unusual cycle complexity
    cycle_count = len(classified_cycles)
    unclassified = [c for c in classified_cycles if c.get("cycle_type") in [None, "UNKNOWN"]]

    if cycle_count > 15:
        anomalies.append(
            TopologicalAnomaly(
                anomaly_type="excessive_financial_cycles",
                severity="high",
                description=f"Unusually high number of cycles ({cycle_count}) for financial data",
                evidence={"cycle_count": cycle_count, "expected_max": 10},
                affected_tables=[table_name],
                affected_columns=[],
            )
        )

    if len(unclassified) > 5:
        anomalies.append(
            TopologicalAnomaly(
                anomaly_type="unclassified_financial_cycles",
                severity="medium",
                description=f"{len(unclassified)} cycles could not be classified",
                evidence={"unclassified_count": len(unclassified), "total_count": cycle_count},
                affected_tables=[table_name],
                affected_columns=[],
            )
        )

    # Anomaly 2: Disconnected components
    if hasattr(topological_result, "betti_numbers"):
        betti_0 = topological_result.betti_numbers.betti_0
        if betti_0 > 3:
            anomalies.append(
                TopologicalAnomaly(
                    anomaly_type="financial_data_fragmentation",
                    severity="high",
                    description=f"Financial data has {betti_0} disconnected components. Expected: 1-2",
                    evidence={
                        "component_count": betti_0,
                        "expected_max": 2,
                        "interpretation": "Accounts or entities are not properly linked",
                    },
                    affected_tables=[table_name],
                    affected_columns=[],
                )
            )

    # Anomaly 3: Missing expected financial cycles
    cycle_types = {c.get("cycle_type") for c in classified_cycles if c.get("cycle_type")}
    expected_cycles = {"accounts_receivable_cycle", "expense_cycle", "revenue_cycle"}
    missing_cycles = expected_cycles - cycle_types

    if missing_cycles and cycle_count > 0:
        anomalies.append(
            TopologicalAnomaly(
                anomaly_type="missing_financial_cycles",
                severity="medium",
                description=f"Expected financial cycles not detected: {', '.join(missing_cycles)}",
                evidence={
                    "missing_cycles": list(missing_cycles),
                    "detected_cycles": list(cycle_types),
                },
                affected_tables=[table_name],
                affected_columns=[],
            )
        )

    # Anomaly 4: Cost center isolation
    orphaned = getattr(topological_result, "orphaned_components", 0)
    if orphaned > 0:
        anomalies.append(
            TopologicalAnomaly(
                anomaly_type="cost_center_isolation",
                severity="medium",
                description=f"{orphaned} isolated components detected",
                evidence={"orphaned_count": orphaned},
                affected_tables=[table_name],
                affected_columns=[],
            )
        )

    return anomalies


def compute_financial_quality_score(
    topological_result: Any,
    financial_anomalies: list[TopologicalAnomaly],
    classified_cycles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute domain-weighted quality score for financial data.

    Uses configuration-driven thresholds for penalties and bonuses.

    Args:
        topological_result: TopologicalQualityResult from analysis
        financial_anomalies: List of detected anomalies
        classified_cycles: List of LLM-classified cycles

    Returns:
        Dict with score and breakdown
    """
    config = _load_financial_config()
    thresholds = config.get("quality_thresholds", {})

    quality_score = 1.0
    penalties: list[dict[str, Any]] = []
    bonuses: list[dict[str, Any]] = []

    # Critical penalties (structural integrity)
    if topological_result and hasattr(topological_result, "betti_numbers"):
        betti_0 = topological_result.betti_numbers.betti_0
        critical_threshold = (
            thresholds.get("critical", {}).get("disconnected_components", {}).get("threshold", 3)
        )
        critical_penalty = (
            thresholds.get("critical", {}).get("disconnected_components", {}).get("penalty", 0.5)
        )
        moderate_penalty = (
            thresholds.get("critical", {}).get("moderate_disconnection", {}).get("penalty", 0.2)
        )

        if betti_0 > critical_threshold:
            quality_score -= critical_penalty
            penalties.append(
                {
                    "type": "critical_fragmentation",
                    "penalty": critical_penalty,
                    "reason": f"Data fragmentation ({betti_0} components)",
                }
            )
        elif betti_0 > 1:
            quality_score -= moderate_penalty
            penalties.append(
                {
                    "type": "moderate_fragmentation",
                    "penalty": moderate_penalty,
                    "reason": f"Multiple components ({betti_0})",
                }
            )

    # Medium penalties (anomalies)
    medium_thresholds = thresholds.get("medium", {})
    anomaly_penalties = {
        "excessive_financial_cycles": medium_thresholds.get("excessive_cycles", {}).get(
            "penalty", 0.3
        ),
        "missing_financial_cycles": medium_thresholds.get("missing_expected_cycles", {}).get(
            "penalty", 0.2
        ),
        "unclassified_financial_cycles": medium_thresholds.get("unclassified_cycles", {}).get(
            "penalty", 0.15
        ),
        "cost_center_isolation": medium_thresholds.get("cost_center_isolation", {}).get(
            "penalty", 0.2
        ),
        "financial_data_fragmentation": medium_thresholds.get("data_fragmentation", {}).get(
            "penalty", 0.3
        ),
    }

    for anomaly in financial_anomalies:
        penalty = anomaly_penalties.get(anomaly.anomaly_type, 0.1)
        quality_score -= penalty
        penalties.append(
            {
                "type": anomaly.anomaly_type,
                "penalty": penalty,
                "reason": anomaly.description,
            }
        )

    # Minor penalties (complexity)
    cycle_count = len(classified_cycles)
    minor_thresholds = thresholds.get("minor", {})
    very_high_threshold = minor_thresholds.get("very_high_complexity", {}).get("threshold", 20)
    high_threshold = minor_thresholds.get("high_complexity", {}).get("threshold", 15)

    if cycle_count > very_high_threshold:
        penalty = minor_thresholds.get("very_high_complexity", {}).get("penalty", 0.1)
        quality_score -= penalty
        penalties.append(
            {
                "type": "very_high_complexity",
                "penalty": penalty,
                "reason": f"Very high cycle count ({cycle_count})",
            }
        )
    elif cycle_count > high_threshold:
        penalty = minor_thresholds.get("high_complexity", {}).get("penalty", 0.05)
        quality_score -= penalty
        penalties.append(
            {
                "type": "high_complexity",
                "penalty": penalty,
                "reason": f"High cycle count ({cycle_count})",
            }
        )

    # Bonus: Well-classified cycles
    if cycle_count > 0:
        classified_count = sum(
            1 for c in classified_cycles if c.get("cycle_type") not in [None, "UNKNOWN"]
        )
        classification_rate = classified_count / cycle_count

        bonus_config = thresholds.get("bonuses", {}).get("well_classified_cycles", {})
        bonus_threshold = bonus_config.get("threshold", 0.8)
        bonus_amount = bonus_config.get("bonus", 0.05)

        if classification_rate > bonus_threshold:
            quality_score += bonus_amount
            bonuses.append(
                {
                    "type": "well_classified",
                    "bonus": bonus_amount,
                    "reason": f"{classification_rate:.0%} cycles classified",
                }
            )

    # Clamp to valid range
    quality_score = max(0.0, min(1.0, quality_score))

    return {
        "score": quality_score,
        "penalties": penalties,
        "bonuses": bonuses,
        "cycle_count": cycle_count,
        "anomaly_count": len(financial_anomalies),
    }


async def analyze_complete_financial_quality(
    table_id: str,
    duckdb_conn: duckdb.DuckDBPyConnection,
    session: AsyncSession,
    llm_service: LLMService | None = None,
) -> Result[dict[str, Any]]:
    """Complete financial quality analysis with LLM interpretation.

    Correct architecture:
    1. Python computes ALL metrics
    2. LLM interprets results with domain context
    3. Returns comprehensive assessment

    Args:
        table_id: Table to analyze
        duckdb_conn: DuckDB connection
        session: SQLAlchemy session
        llm_service: LLM service (optional - returns metrics only if None)

    Returns:
        Result containing:
        - financial_metrics: Computed numbers from financial.py
        - topological_metrics: Computed structure from topological.py
        - classified_cycles: LLM classifications (if LLM available)
        - interpretation: LLM interpretation (if LLM available)

    Example:
        result = await analyze_complete_financial_quality(
            table_id="table-123",
            duckdb_conn=conn,
            session=session,
            llm_service=llm_service
        )

        if result.success:
            data = result.value
            print(f"Quality Score: {data['interpretation']['overall_quality_score']}")
            print(f"Critical Issues: {data['interpretation']['critical_issues']}")
    """
    try:
        # ============================================================================
        # LAYER 1: COMPUTE ALL METRICS (Pure Python + SQL)
        # ============================================================================

        logger.info(f"Computing financial metrics for table {table_id}")

        # 1.1: Financial accounting metrics
        financial_result = await analyze_financial_quality(
            table_id=table_id, duckdb_conn=duckdb_conn, session=session
        )

        if not financial_result.success:
            logger.error(f"Financial analysis failed: {financial_result.error}")
            return Result.fail(f"Financial analysis failed: {financial_result.error}")

        financial_quality = financial_result.unwrap()

        # Build financial metrics dict for LLM
        financial_metrics = {
            "double_entry_balanced": financial_quality.double_entry_balanced,
            "net_difference": financial_quality.balance_difference or 0.0,
            "total_debits": (
                financial_quality.double_entry_details.total_debits
                if financial_quality.double_entry_details
                else 0.0
            ),
            "total_credits": (
                financial_quality.double_entry_details.total_credits
                if financial_quality.double_entry_details
                else 0.0
            ),
            "trial_balance_holds": financial_quality.accounting_equation_holds,
            "total_assets": financial_quality.assets_total or 0.0,
            "total_liabilities": financial_quality.liabilities_total or 0.0,
            "total_equity": financial_quality.equity_total or 0.0,
            "trial_balance_difference": (financial_quality.assets_total or 0.0)
            - (
                (financial_quality.liabilities_total or 0.0)
                + (financial_quality.equity_total or 0.0)
            ),
            "sign_compliance_rate": financial_quality.sign_convention_compliance or 0.0,
            "sign_violations": len(financial_quality.sign_violations)
            if financial_quality.sign_violations
            else 0,
            "period_start": (
                financial_quality.period_integrity_details[0].period_start.isoformat()
                if financial_quality.period_integrity_details
                else None
            ),
            "period_end": (
                financial_quality.period_integrity_details[0].period_end.isoformat()
                if financial_quality.period_integrity_details
                else None
            ),
            "missing_days": (
                financial_quality.period_integrity_details[0].missing_days
                if financial_quality.period_integrity_details
                else 0
            ),
            "has_cutoff_issues": not financial_quality.period_end_cutoff_clean,
        }

        # 1.2: Topological structure metrics
        logger.info(f"Computing topological metrics for table {table_id}")

        topological_result = await analyze_topological_quality(
            table_id=table_id, duckdb_conn=duckdb_conn, session=session
        )

        if not topological_result.success:
            logger.warning(f"Topological analysis failed: {topological_result.error}")
            # Continue with None - not critical for financial analysis
            topological_quality = None
            topological_metrics: dict[str, Any] = {
                "betti_0": None,
                "betti_1": None,
                "betti_2": None,
                "total_complexity": None,
                "persistent_entropy": None,
                "orphaned_components": None,
                "cycles": [],
            }
        else:
            topological_quality = topological_result.unwrap()
            topological_metrics = {
                "betti_0": topological_quality.betti_numbers.betti_0,
                "betti_1": topological_quality.betti_numbers.betti_1,
                "betti_2": topological_quality.betti_numbers.betti_2,
                "total_complexity": topological_quality.betti_numbers.total_complexity,
                "persistent_entropy": topological_quality.persistent_entropy,
                "orphaned_components": topological_quality.orphaned_components,
                "cycles": topological_quality.persistent_cycles,
            }

        # If no LLM, still run domain rules and return
        if llm_service is None:
            logger.info("No LLM service - running domain rules only")

            # Run Layer 1.5 without LLM
            stability = (
                getattr(topological_quality, "stability", None) if topological_quality else None
            )
            fiscal_stability = assess_fiscal_stability(stability, {})
            financial_anomalies = detect_financial_anomalies(topological_quality, [])
            domain_quality_score = compute_financial_quality_score(
                topological_quality, financial_anomalies, []
            )

            domain_analysis = {
                "fiscal_stability": fiscal_stability,
                "anomalies": [
                    {
                        "type": a.anomaly_type,
                        "severity": a.severity,
                        "description": a.description,
                        "evidence": a.evidence,
                    }
                    for a in financial_anomalies
                ],
                "quality_score": domain_quality_score,
            }

            return Result.ok(
                {
                    "financial_metrics": financial_metrics,
                    "topological_metrics": topological_metrics,
                    "classified_cycles": [],
                    "domain_analysis": domain_analysis,
                    "llm_interpretation": None,
                    "llm_available": False,
                }
            )

        # ============================================================================
        # LAYER 2: LLM CLASSIFICATION (Cycle interpretation with config context)
        # ============================================================================

        logger.info("Classifying cycles with LLM")

        # Get table and column info for context
        table = await session.get(Table, table_id)
        if not table:
            return Result.fail(f"Table {table_id} not found")

        # Get columns
        stmt = select(Column).where(Column.table_id == table_id)
        columns_result = await session.execute(stmt)
        columns = list(columns_result.scalars().all())

        # Get semantic annotations (join through columns)
        stmt_semantic = (
            select(SemanticAnnotation, Column.column_name)
            .join(Column, SemanticAnnotation.column_id == Column.column_id)
            .where(Column.table_id == table_id)
        )
        semantic_result = await session.execute(stmt_semantic)
        semantic_annotations = {
            col_name: ann.semantic_role
            for ann, col_name in semantic_result.all()
            if ann.semantic_role
        }

        # Classify each cycle with LLM
        classified_cycles = []

        cycles_list = topological_metrics.get("cycles", [])
        if cycles_list:
            for cycle in cycles_list:
                # Determine involved columns (simplified - would need graph analysis)
                # For now, use all columns as potential context
                column_names = [col.column_name for col in columns]

                classification_result = await classify_financial_cycle_with_llm(
                    cycle=cycle,
                    table_name=table.table_name,
                    column_names=column_names,
                    semantic_roles=semantic_annotations,
                    llm_service=llm_service,
                )

                if classification_result.success:
                    classification = classification_result.unwrap()
                    classified_cycles.append(classification)
                else:
                    logger.warning(f"Cycle classification failed: {classification_result.error}")
                    # Include unclassified cycle
                    classified_cycles.append(
                        {
                            "cycle_type": "UNKNOWN",
                            "confidence": 0.0,
                            "explanation": f"Classification failed: {classification_result.error}",
                            "business_value": "unknown",
                            "is_expected": False,
                            "recommendation": None,
                        }
                    )

        # ============================================================================
        # LAYER 1.5: DOMAIN RULES (Deterministic, Auditable)
        # ============================================================================

        logger.info("Applying domain rules for anomaly detection and scoring")

        # Assess fiscal stability (if stability info available)
        stability = getattr(topological_quality, "stability", None) if topological_quality else None
        temporal_context: dict[str, Any] = {}  # Could be passed as parameter in future
        fiscal_stability = assess_fiscal_stability(stability, temporal_context)

        # Detect financial-specific anomalies
        financial_anomalies = detect_financial_anomalies(topological_quality, classified_cycles)

        # Compute domain-weighted quality score
        domain_quality_score = compute_financial_quality_score(
            topological_quality, financial_anomalies, classified_cycles
        )

        # Build domain analysis summary
        domain_analysis = {
            "fiscal_stability": fiscal_stability,
            "anomalies": [
                {
                    "type": a.anomaly_type,
                    "severity": a.severity,
                    "description": a.description,
                    "evidence": a.evidence,
                }
                for a in financial_anomalies
            ],
            "quality_score": domain_quality_score,
        }

        # ============================================================================
        # LAYER 3: LLM INTERPRETATION (Holistic assessment)
        # ============================================================================

        logger.info("Generating LLM interpretation of all metrics")

        interpretation_result = await interpret_financial_quality_with_llm(
            financial_metrics=financial_metrics,
            topological_metrics=topological_metrics,
            classified_cycles=classified_cycles,
            llm_service=llm_service,
            domain_analysis=domain_analysis,  # Pass domain analysis as context
        )

        if not interpretation_result.success:
            logger.error(f"LLM interpretation failed: {interpretation_result.error}")
            # Return with domain analysis even if LLM fails
            return Result.ok(
                {
                    "financial_metrics": financial_metrics,
                    "topological_metrics": topological_metrics,
                    "classified_cycles": classified_cycles,
                    "domain_analysis": domain_analysis,
                    "llm_interpretation": None,
                    "interpretation_error": interpretation_result.error,
                    "llm_available": True,
                }
            )

        llm_interpretation = interpretation_result.unwrap()

        # ============================================================================
        # RETURN COMPLETE RESULT
        # ============================================================================

        return Result.ok(
            {
                "financial_metrics": financial_metrics,
                "topological_metrics": topological_metrics,
                "classified_cycles": classified_cycles,
                "domain_analysis": domain_analysis,
                "llm_interpretation": llm_interpretation,
                "llm_available": True,
            }
        )

    except Exception as e:
        logger.error(f"Complete financial quality analysis failed: {e}")
        return Result.fail(f"Analysis failed: {e}")
