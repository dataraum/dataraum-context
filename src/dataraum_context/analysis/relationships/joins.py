"""Join column detection using value overlap.

Uses parallel processing and caching for efficient detection.

Performance optimizations:
1. Pre-compute column stats (distinct count, total count) once per column
2. Filter column pairs by cardinality compatibility before expensive intersection
3. Use parallel processing for intersection queries
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


@dataclass
class ColumnStats:
    """Pre-computed statistics for a column."""

    column_name: str
    distinct_count: int
    total_count: int
    is_unique: bool  # distinct_count == total_count


def _precompute_column_stats(
    conn: duckdb.DuckDBPyConnection,
    table_path: str,
    columns: list[str],
    max_distinct: int,
) -> dict[str, ColumnStats]:
    """Pre-compute statistics for all columns in a table.

    This runs ONE query per table instead of per column pair.
    """
    stats: dict[str, ColumnStats] = {}

    for col in columns:
        try:
            result = conn.execute(f"""
                SELECT
                    COUNT(DISTINCT "{col}") AS distinct_count,
                    COUNT(*) FILTER (WHERE "{col}" IS NOT NULL) AS total_count
                FROM {table_path}
            """).fetchone()

            if result:
                distinct_count = min(result[0], max_distinct)
                total_count = result[1]
                stats[col] = ColumnStats(
                    column_name=col,
                    distinct_count=distinct_count,
                    total_count=total_count,
                    is_unique=(distinct_count == total_count and distinct_count > 0),
                )
        except Exception:
            # Skip columns that fail (e.g., unsupported types)
            pass

    return stats


def _should_compare_columns(
    stats1: ColumnStats,
    stats2: ColumnStats,
    cardinality_ratio_threshold: float = 100.0,
) -> bool:
    """Filter column pairs that are unlikely to be related.

    Skip pairs where:
    - Either column has 0 or 1 distinct values (constant/boolean)
    - Cardinality ratio is extreme (e.g., 1000:1)
    """
    # Skip constant or near-constant columns
    if stats1.distinct_count <= 1 or stats2.distinct_count <= 1:
        return False

    # Skip if cardinality ratio is extreme
    ratio = max(stats1.distinct_count, stats2.distinct_count) / max(
        min(stats1.distinct_count, stats2.distinct_count), 1
    )
    if ratio > cardinality_ratio_threshold:
        return False

    return True


def _compute_join_score_with_stats(
    conn: duckdb.DuckDBPyConnection,
    table1_path: str,
    table2_path: str,
    col1: str,
    col2: str,
    stats1: ColumnStats,
    stats2: ColumnStats,
    max_distinct: int,
) -> tuple[str, str, float, str]:
    """Compute join score using pre-computed stats.

    Only computes the intersection, not the counts (already known).
    """
    cursor = conn.cursor()
    try:
        # Only compute intersection - counts are already known
        result = cursor.execute(f"""
            WITH
            vals1 AS (SELECT DISTINCT "{col1}" AS v FROM {table1_path} WHERE "{col1}" IS NOT NULL LIMIT {max_distinct}),
            vals2 AS (SELECT DISTINCT "{col2}" AS v FROM {table2_path} WHERE "{col2}" IS NOT NULL LIMIT {max_distinct})
            SELECT COUNT(*) FROM vals1 WHERE v IN (SELECT v FROM vals2)
        """).fetchone()

        if result is None:
            return (col1, col2, 0.0, "unknown")

        intersection = result[0]
        count1 = stats1.distinct_count
        count2 = stats2.distinct_count

        if count1 == 0 or count2 == 0:
            return (col1, col2, 0.0, "unknown")

        union = count1 + count2 - intersection
        jaccard = intersection / union if union > 0 else 0.0

        # Check containment
        containment = 1.0 if (intersection == count1 or intersection == count2) else 0.0

        score = max(jaccard, containment)

        # Determine cardinality using pre-computed uniqueness
        if stats1.is_unique and stats2.is_unique:
            cardinality = "one-to-one"
        elif stats1.is_unique and not stats2.is_unique:
            cardinality = "one-to-many"
        elif not stats1.is_unique and stats2.is_unique:
            cardinality = "many-to-one"
        else:
            cardinality = "many-to-many"

        return (col1, col2, score, cardinality)

    except Exception:
        return (col1, col2, 0.0, "unknown")
    finally:
        cursor.close()


def find_join_columns(
    conn: duckdb.DuckDBPyConnection,
    table1_path: str,
    table2_path: str,
    columns1: list[str],
    columns2: list[str],
    min_score: float = 0.3,
    max_distinct: int = 100000,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Find join columns using DuckDB for accurate distinct value comparison.

    Uses a two-phase approach for efficiency:
    1. Pre-compute column statistics (distinct count, total count) once per column
    2. Filter column pairs by cardinality compatibility
    3. Only run expensive intersection queries for promising pairs

    Returns dicts with:
    - column1, column2: column names
    - join_confidence: value overlap score (Jaccard/containment)
    - cardinality: one-to-one, one-to-many, etc.
    """
    # Phase 1: Pre-compute column statistics
    stats1 = _precompute_column_stats(conn, table1_path, columns1, max_distinct)
    stats2 = _precompute_column_stats(conn, table2_path, columns2, max_distinct)

    # Phase 2: Filter column pairs
    pairs_to_check = []
    for col1 in columns1:
        if col1 not in stats1:
            continue
        for col2 in columns2:
            if col2 not in stats2:
                continue
            if _should_compare_columns(stats1[col1], stats2[col2]):
                pairs_to_check.append((col1, col2))

    logger.debug(
        f"Filtered {len(columns1) * len(columns2)} pairs to {len(pairs_to_check)} "
        f"for {table1_path} <-> {table2_path}"
    )

    if not pairs_to_check:
        return []

    # Phase 3: Compute intersection scores in parallel
    candidates = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _compute_join_score_with_stats,
                conn,
                table1_path,
                table2_path,
                col1,
                col2,
                stats1[col1],
                stats2[col2],
                max_distinct,
            )
            for col1, col2 in pairs_to_check
        ]

        for future in futures:
            col1, col2, score, cardinality = future.result()
            if score > min_score:
                candidates.append(
                    {
                        "column1": col1,
                        "column2": col2,
                        "join_confidence": score,
                        "cardinality": cardinality,
                    }
                )

    def _get_conf(x: dict[str, Any]) -> float:
        c = x["join_confidence"]
        return float(c) if isinstance(c, (int, float)) else 0.0

    candidates.sort(key=_get_conf, reverse=True)
    return candidates
