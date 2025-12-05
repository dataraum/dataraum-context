"""Complete data processing pipeline.

Unified pipeline from CSV files through profiling and enrichment.
All metadata stored in database. Results contain only health information.
"""

from dataraum_context.dataflows.pipeline import (
    PipelineResult,
    TablePipelineHealth,
    run_pipeline,
)

__all__ = [
    "run_pipeline",
    "TablePipelineHealth",
    "PipelineResult",
]
