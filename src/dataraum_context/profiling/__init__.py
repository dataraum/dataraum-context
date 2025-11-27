"""Profiling module for statistical analysis and type inference.

This module profiles staged data to:
- Generate statistical metadata (counts, nulls, distributions)
- Detect patterns in VARCHAR data (dates, emails, UUIDs, etc.)
- Infer type candidates with confidence scores
- Detect units using Pint (currencies, measurements)
"""

from dataraum_context.profiling.profiler import profile_table

__all__ = ["profile_table"]
