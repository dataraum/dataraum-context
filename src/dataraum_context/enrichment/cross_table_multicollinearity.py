"""Cross-table multicollinearity detection.

NOTE: This module is being repurposed.

The previous global matrix approach has been removed. Cross-table analysis will be
rebuilt for quality-focused analysis that runs AFTER relationships are
confirmed by the semantic agent.

For relationship evaluation BEFORE semantic agent, see:
    analysis/relationships/evaluator.py

Re-exports for backward compatibility:
    - EnrichedRelationship: from analysis/correlation/models
    - gather_relationships: from enrichment/relationships/gathering
"""

# Re-export from relationships package for backward compatibility
from dataraum_context.analysis.correlation.models import EnrichedRelationship
from dataraum_context.enrichment.relationships.gathering import (
    CONFIDENCE_THRESHOLDS,
    gather_relationships,
)

__all__ = [
    "EnrichedRelationship",
    "gather_relationships",
    "CONFIDENCE_THRESHOLDS",
]

# Placeholder - module is being repurposed
# See PLAN_CROSS_TABLE_CORRELATION.md for details
