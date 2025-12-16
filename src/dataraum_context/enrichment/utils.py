"""Shared utility functions for enrichment operations.

NOTE: These functions have been moved to analysis/semantic/utils.py.
Re-exported here for backwards compatibility.
"""

# Re-export from new location for backwards compatibility
from dataraum_context.analysis.semantic.utils import (  # noqa: F401
    load_column_mappings,
    load_table_mappings,
)

__all__ = [
    "load_column_mappings",
    "load_table_mappings",
]
