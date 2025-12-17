"""Cross-table quality analysis (VDP-based).

NOTE: This module is being repurposed.

The previous global matrix approach has been removed. This module will be
rebuilt for quality-focused analysis that runs AFTER relationships are
confirmed by the semantic agent.

For relationship evaluation BEFORE semantic agent, see:
    analysis/relationships/evaluator.py

The pure algorithms remain available:
    - algorithms/numeric.py: compute_pairwise_correlations
    - algorithms/categorical.py: compute_cramers_v
    - algorithms/multicollinearity.py: compute_multicollinearity

TODO: Rebuild this module for post-confirmation quality analysis:
    - VDP-based redundancy detection on confirmed relationships
    - Cross-table correlation analysis for quality context
"""

# Placeholder - module is being repurposed
# See PLAN_CROSS_TABLE_CORRELATION.md for details
