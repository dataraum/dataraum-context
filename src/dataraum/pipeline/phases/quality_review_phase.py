"""Quality review checkpoint phase.

A no-op phase that sits after semantic and before enriched_views.
Its sole purpose is to run foundation entropy detectors in post_verification,
creating the first quality gate checkpoint for the inline fix system.

In PAUSE gate mode, this is where the pipeline pauses to show all foundation
entropy violations and allows the user to apply fixes before proceeding.
"""

from dataraum.pipeline.base import PhaseContext, PhaseResult
from dataraum.pipeline.phases.base import BasePhase
from dataraum.pipeline.registry import analysis_phase


@analysis_phase
class QualityReviewPhase(BasePhase):
    """Quality checkpoint — runs foundation entropy detectors."""

    @property
    def name(self) -> str:
        return "quality_review"

    @property
    def description(self) -> str:
        return "Quality checkpoint after semantic — runs foundation entropy detectors"

    @property
    def dependencies(self) -> list[str]:
        return ["semantic", "statistical_quality"]

    @property
    def is_quality_gate(self) -> bool:
        """Quality gates assess ALL accumulated scores against contracts."""
        return True

    def _run(self, ctx: PhaseContext) -> PhaseResult:
        """No-op — the checkpoint's value is in post_verification."""
        return PhaseResult.success(summary="Quality review checkpoint")

    def should_skip(self, ctx: PhaseContext) -> str | None:
        """Never skip — the checkpoint must always run to evaluate entropy."""
        return None
