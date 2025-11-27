"""Staging module for loading data from various sources into DuckDB.

This module implements a hybrid approach based on source type system strength:
- Untyped sources (CSV, JSON): VARCHAR-first with full type inference
- Weakly-typed sources (SQLite): VARCHAR-first with type hints
- Strongly-typed sources (Parquet, PostgreSQL): Direct load with semantic profiling only
"""

from dataraum_context.staging.base import LoaderBase, TypeSystemStrength

__all__ = [
    "LoaderBase",
    "TypeSystemStrength",
]
