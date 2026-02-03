"""Resolution tracking and cascade management.

This module handles:
- Finding resolution cascades (single fixes that improve multiple dimensions)
- Prioritizing resolutions by impact-to-effort ratio
"""

from collections import defaultdict
from dataclasses import dataclass, field

from dataraum.core.logging import get_logger
from dataraum.entropy.models import (
    EntropyObject,
    ResolutionCascade,
    ResolutionOption,
)

logger = get_logger(__name__)


@dataclass
class ResolutionFinder:
    """Finds and prioritizes resolution options across entropy objects.

    Groups related resolutions, identifies cascades, and ranks
    by impact-to-effort ratio.
    """

    # Resolution options grouped by action type
    options_by_action: dict[str, list[ResolutionOption]] = field(default_factory=dict)

    # Cascade relationships discovered
    cascades: list[ResolutionCascade] = field(default_factory=list)

    def analyze_entropy_objects(
        self,
        entropy_objects: list[EntropyObject],
    ) -> list[ResolutionCascade]:
        """Analyze entropy objects to find resolution cascades.

        A cascade occurs when multiple entropy objects can be fixed
        by a single action.

        Args:
            entropy_objects: List of entropy objects with resolution options

        Returns:
            List of resolution cascades, sorted by priority
        """
        self.options_by_action = defaultdict(list)
        self.cascades = []

        # Group options by action type
        for obj in entropy_objects:
            for option in obj.resolution_options:
                key = self._action_key(option)
                self.options_by_action[key].append(option)

        # Find cascades (actions that appear multiple times)
        for action_key, options in self.options_by_action.items():
            if len(options) > 1:
                cascade = self._build_cascade(action_key, options)
                if cascade:
                    self.cascades.append(cascade)

        # Also look for resolutions with cascade_dimensions
        for obj in entropy_objects:
            for option in obj.resolution_options:
                if option.cascade_dimensions:
                    cascade = self._build_cascade_from_option(obj, option)
                    if cascade and not self._cascade_exists(cascade):
                        self.cascades.append(cascade)

        # Calculate priorities and sort
        for cascade in self.cascades:
            cascade.calculate_priority()

        self.cascades.sort(key=lambda c: c.priority_score, reverse=True)

        return self.cascades

    def _action_key(self, option: ResolutionOption) -> str:
        """Create a unique key for an action based on type and parameters."""
        # Create a stable key from action and key parameters
        params_str = ""
        if "column" in option.parameters:
            params_str = f":{option.parameters['column']}"
        elif "table" in option.parameters:
            params_str = f":{option.parameters['table']}"
        return f"{option.action}{params_str}"

    def _build_cascade(
        self,
        action_key: str,
        options: list[ResolutionOption],
    ) -> ResolutionCascade | None:
        """Build a cascade from multiple related options."""
        if not options:
            return None

        # Use first option as template
        template = options[0]

        # Collect all affected targets and entropy reductions
        affected_targets: list[str] = []
        entropy_reductions: dict[str, float] = {}

        # Get highest reduction per dimension
        for option in options:
            for dim in option.cascade_dimensions or [template.action]:
                if dim not in entropy_reductions:
                    entropy_reductions[dim] = option.expected_entropy_reduction
                else:
                    entropy_reductions[dim] = max(
                        entropy_reductions[dim], option.expected_entropy_reduction
                    )

        # If no cascade dimensions, use the action as dimension key
        if not entropy_reductions:
            entropy_reductions[action_key] = sum(
                o.expected_entropy_reduction for o in options
            ) / len(options)

        return ResolutionCascade(
            action=template.action,
            parameters=template.parameters,
            affected_targets=affected_targets,
            entropy_reductions=entropy_reductions,
            effort=template.effort,
            description=template.description or f"Apply {template.action}",
        )

    def _build_cascade_from_option(
        self,
        entropy_object: EntropyObject,
        option: ResolutionOption,
    ) -> ResolutionCascade:
        """Build a cascade from a single option with cascade_dimensions."""
        entropy_reductions = dict.fromkeys(
            option.cascade_dimensions, option.expected_entropy_reduction
        )

        return ResolutionCascade(
            action=option.action,
            parameters=option.parameters,
            affected_targets=[entropy_object.target],
            entropy_reductions=entropy_reductions,
            effort=option.effort,
            description=option.description or f"Apply {option.action}",
        )

    def _cascade_exists(self, new_cascade: ResolutionCascade) -> bool:
        """Check if an equivalent cascade already exists."""
        for existing in self.cascades:
            if (
                existing.action == new_cascade.action
                and existing.parameters == new_cascade.parameters
            ):
                return True
        return False


def find_top_resolutions(
    entropy_objects: list[EntropyObject],
    limit: int = 5,
) -> list[ResolutionCascade]:
    """Find the top resolution cascades by priority.

    Args:
        entropy_objects: List of entropy objects with resolution options
        limit: Maximum number of resolutions to return

    Returns:
        Top resolution cascades sorted by priority score
    """
    finder = ResolutionFinder()
    cascades = finder.analyze_entropy_objects(entropy_objects)
    return cascades[:limit]
