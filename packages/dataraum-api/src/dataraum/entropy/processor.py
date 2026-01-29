"""Entropy processor for running detectors and aggregating results.

This module orchestrates entropy detection:
- Runs all applicable detectors for a given context
- Aggregates results into column/table summaries
- Detects compound risks
- Finds top resolution cascades

Usage:
    from dataraum.entropy.processor import EntropyProcessor

    processor = EntropyProcessor()
    summary = processor.process_column(
        table_name="orders",
        column_name="amount",
        analysis_results={
            "typing": type_candidate,
            "statistics": column_profile,
            "semantic": semantic_annotation,
        }
    )
"""

from dataclasses import dataclass
from typing import Any

from dataraum.core.logging import get_logger
from dataraum.entropy.analysis.aggregator import ColumnSummary, TableSummary
from dataraum.entropy.compound_risk import detect_compound_risks_for_column
from dataraum.entropy.config import get_entropy_config
from dataraum.entropy.detectors.base import (
    DetectorContext,
    DetectorRegistry,
    get_default_registry,
)
from dataraum.entropy.models import EntropyObject, ResolutionOption
from dataraum.entropy.resolution import find_top_resolutions

logger = get_logger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for entropy processor.

    Defaults are loaded from config/entropy/thresholds.yaml.
    Can be overridden by passing explicit values.
    """

    # Score weights for composite calculation
    layer_weights: dict[str, float] | None = None

    # Thresholds
    high_entropy_threshold: float | None = None
    critical_entropy_threshold: float | None = None

    # Resolution settings
    max_resolution_hints: int = 3

    # Parallel execution
    max_concurrent_detectors: int = 10

    def __post_init__(self) -> None:
        """Load defaults from config if not explicitly set."""
        config = get_entropy_config()

        if self.layer_weights is None:
            self.layer_weights = config.composite_weights

        if self.high_entropy_threshold is None:
            self.high_entropy_threshold = config.high_entropy_threshold

        if self.critical_entropy_threshold is None:
            self.critical_entropy_threshold = config.critical_entropy_threshold


class EntropyProcessor:
    """Orchestrates entropy detection across detectors.

    Runs detectors, aggregates results into summaries, and detects
    compound risks.
    """

    def __init__(
        self,
        registry: DetectorRegistry | None = None,
        config: ProcessorConfig | None = None,
    ):
        """Initialize processor.

        Args:
            registry: Detector registry to use. Defaults to global registry.
            config: Processor configuration.
        """
        self.registry = registry or get_default_registry()
        self.config = config or ProcessorConfig()

    def process_column(
        self,
        table_name: str,
        column_name: str,
        analysis_results: dict[str, Any],
        source_id: str | None = None,
        table_id: str | None = None,
        column_id: str | None = None,
    ) -> ColumnSummary:
        """Process a single column and return its entropy summary.

        Args:
            table_name: Name of the table
            column_name: Name of the column
            analysis_results: Dict of module name to analysis result
            source_id: Optional source ID
            table_id: Optional table ID
            column_id: Optional column ID

        Returns:
            ColumnSummary with aggregated entropy
        """
        # Create detector context
        context = DetectorContext(
            source_id=source_id,
            table_id=table_id,
            table_name=table_name,
            column_id=column_id,
            column_name=column_name,
            analysis_results=analysis_results,
        )

        # Run all applicable detectors
        entropy_objects = self._run_detectors(context)

        # Aggregate into summary
        summary = self._aggregate_to_summary(
            table_name=table_name,
            column_name=column_name,
            table_id=table_id or "",
            column_id=column_id or "",
            entropy_objects=entropy_objects,
        )

        # Detect compound risks
        summary.compound_risks = detect_compound_risks_for_column(summary, entropy_objects)

        # Find top resolution hints
        cascades = find_top_resolutions(entropy_objects, limit=self.config.max_resolution_hints)
        # Convert cascades to resolution options for the summary
        for cascade in cascades:
            if cascade.entropy_reductions:
                opt = ResolutionOption(
                    action=cascade.action,
                    parameters=cascade.parameters,
                    expected_entropy_reduction=cascade.total_reduction,
                    effort=cascade.effort,
                    description=cascade.description,
                    cascade_dimensions=list(cascade.entropy_reductions.keys()),
                )
                summary.top_resolution_hints.append(opt)

        return summary

    def process_table(
        self,
        table_name: str,
        columns: list[dict[str, Any]],
        source_id: str | None = None,
        table_id: str | None = None,
    ) -> TableSummary:
        """Process all columns in a table and return table entropy summary.

        Args:
            table_name: Name of the table
            columns: List of column specs with 'name' and 'analysis_results'
            source_id: Optional source ID
            table_id: Optional table ID

        Returns:
            TableSummary with aggregated entropy
        """
        # Process each column
        column_summaries: list[ColumnSummary] = []

        for col_spec in columns:
            column_name = col_spec.get("name", "")
            analysis_results = col_spec.get("analysis_results", {})
            column_id = col_spec.get("column_id")

            summary = self.process_column(
                table_name=table_name,
                column_name=column_name,
                analysis_results=analysis_results,
                source_id=source_id,
                table_id=table_id,
                column_id=column_id,
            )
            column_summaries.append(summary)

        # Create table summary using aggregator logic
        return self._create_table_summary(
            table_id=table_id or "",
            table_name=table_name,
            column_summaries=column_summaries,
        )

    def _run_detectors(self, context: DetectorContext) -> list[EntropyObject]:
        """Run all applicable detectors and collect results.

        Args:
            context: Detector context

        Returns:
            List of EntropyObject instances from all detectors
        """
        # Get detectors that can run with this context
        detectors = self.registry.get_runnable_detectors(context)

        if not detectors:
            logger.debug(f"No detectors can run for {context.target_ref}")
            return []

        logger.debug(f"Running {len(detectors)} detectors for {context.target_ref}")

        # Run detectors (could be parallel in future)
        all_objects: list[EntropyObject] = []

        for detector in detectors:
            try:
                objects = detector.detect(context)
                all_objects.extend(objects)
            except Exception as e:
                logger.error(f"Detector {detector.detector_id} failed: {e}")
                # Continue with other detectors

        return all_objects

    def _aggregate_to_summary(
        self,
        table_name: str,
        column_name: str,
        table_id: str,
        column_id: str,
        entropy_objects: list[EntropyObject],
    ) -> ColumnSummary:
        """Aggregate entropy objects into a column summary.

        Args:
            table_name: Table name
            column_name: Column name
            table_id: Table ID
            column_id: Column ID
            entropy_objects: List of entropy objects

        Returns:
            ColumnSummary with aggregated scores
        """
        summary = ColumnSummary(
            column_id=column_id,
            column_name=column_name,
            table_id=table_id,
            table_name=table_name,
            entropy_objects=entropy_objects,  # Store raw objects for persistence
        )

        if not entropy_objects:
            summary.readiness = "ready"
            return summary

        # Group by layer and calculate layer scores
        layer_scores_raw: dict[str, list[float]] = {
            "structural": [],
            "semantic": [],
            "value": [],
            "computational": [],
        }

        for obj in entropy_objects:
            if obj.layer in layer_scores_raw:
                layer_scores_raw[obj.layer].append(obj.score)

            # Also track dimension-level scores
            summary.dimension_scores[obj.dimension_path] = obj.score

        # Calculate layer averages (or 0 if no detectors ran)
        layer_scores: dict[str, float] = {}
        for layer in ["structural", "semantic", "value", "computational"]:
            if layer_scores_raw[layer]:
                layer_scores[layer] = sum(layer_scores_raw[layer]) / len(layer_scores_raw[layer])
            else:
                layer_scores[layer] = 0.0

        summary.layer_scores = layer_scores

        # Calculate composite score using config weights
        weights = self.config.layer_weights or get_entropy_config().composite_weights
        summary.composite_score = (
            layer_scores.get("structural", 0.0) * weights["structural"]
            + layer_scores.get("semantic", 0.0) * weights["semantic"]
            + layer_scores.get("value", 0.0) * weights["value"]
            + layer_scores.get("computational", 0.0) * weights["computational"]
        )

        # Identify high-entropy dimensions
        high_threshold = self.config.high_entropy_threshold or 0.5
        summary.high_entropy_dimensions = [
            dim for dim, score in summary.dimension_scores.items() if score >= high_threshold
        ]

        # Determine readiness
        summary.readiness = get_entropy_config().get_readiness(summary.composite_score)

        return summary

    def _create_table_summary(
        self,
        table_id: str,
        table_name: str,
        column_summaries: list[ColumnSummary],
    ) -> TableSummary:
        """Create table summary from column summaries.

        Args:
            table_id: Table ID
            table_name: Table name
            column_summaries: List of column summaries

        Returns:
            TableSummary with aggregated scores
        """
        summary = TableSummary(
            table_id=table_id,
            table_name=table_name,
            columns=column_summaries,
        )

        if not column_summaries:
            summary.readiness = "ready"
            return summary

        # Calculate aggregate scores
        composite_scores = [c.composite_score for c in column_summaries]
        summary.avg_composite_score = sum(composite_scores) / len(composite_scores)
        summary.max_composite_score = max(composite_scores)

        # Per-layer aggregates
        layers = ["structural", "semantic", "value", "computational"]
        avg_layer: dict[str, float] = {}
        max_layer: dict[str, float] = {}

        for layer in layers:
            layer_scores = [c.layer_scores.get(layer, 0.0) for c in column_summaries]
            avg_layer[layer] = sum(layer_scores) / len(layer_scores)
            max_layer[layer] = max(layer_scores)

        summary.avg_layer_scores = avg_layer
        summary.max_layer_scores = max_layer

        # Identify problem columns
        config = get_entropy_config()
        high_threshold = config.high_entropy_threshold
        critical_threshold = config.critical_entropy_threshold

        summary.high_entropy_columns = [
            c.column_name for c in column_summaries if c.composite_score >= high_threshold
        ]
        summary.blocked_columns = [
            c.column_name for c in column_summaries if c.composite_score >= critical_threshold
        ]

        # Collect all compound risks
        for col in column_summaries:
            summary.compound_risks.extend(col.compound_risks)

        # Determine table readiness
        if summary.blocked_columns:
            summary.readiness = "blocked"
        elif summary.high_entropy_columns:
            summary.readiness = "investigate"
        else:
            summary.readiness = "ready"

        return summary


def process_column_entropy(
    table_name: str,
    column_name: str,
    analysis_results: dict[str, Any],
    **kwargs: Any,
) -> ColumnSummary:
    """Convenience function to process a single column.

    Args:
        table_name: Table name
        column_name: Column name
        analysis_results: Analysis results from other modules
        **kwargs: Additional arguments for process_column

    Returns:
        ColumnSummary
    """
    processor = EntropyProcessor()
    return processor.process_column(
        table_name=table_name,
        column_name=column_name,
        analysis_results=analysis_results,
        **kwargs,
    )


def process_table_entropy(
    table_name: str,
    columns: list[dict[str, Any]],
    **kwargs: Any,
) -> TableSummary:
    """Convenience function to process a table.

    Args:
        table_name: Table name
        columns: List of column specs
        **kwargs: Additional arguments for process_table

    Returns:
        TableSummary
    """
    processor = EntropyProcessor()
    return processor.process_table(
        table_name=table_name,
        columns=columns,
        **kwargs,
    )
