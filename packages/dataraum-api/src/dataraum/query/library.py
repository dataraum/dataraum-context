"""Query Library for storing and retrieving reusable queries.

The library enables RAG-based query reuse:
1. Search by semantic similarity (embeddings)
2. Retrieve SQL and metadata from SQLite
3. Track usage for continuous improvement

Usage:
    library = QueryLibrary(session, manager)

    # Find similar queries
    match = library.find_similar(question, source_id, min_similarity=0.8)
    if match:
        # Reuse existing query
        sql = match.entry.final_sql

    # Save a new query
    library.save(
        source_id=source_id,
        question=question,
        sql=sql,
        assumptions=assumptions,
        confidence_level=confidence_level,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import select, update

from dataraum.core.logging import get_logger
from dataraum.query.db_models import QueryExecutionRecord, QueryLibraryEntry
from dataraum.query.embeddings import (
    QueryEmbeddings,
    build_embedding_text_for_graph,
    build_embedding_text_for_question,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from dataraum.core.connections import ConnectionManager

logger = get_logger(__name__)


@dataclass
class LibraryMatch:
    """A match from the query library."""

    entry: QueryLibraryEntry
    similarity: float


class QueryLibrary:
    """Service for managing the query library.

    Combines SQLite (via SQLAlchemy) for metadata with DuckDB for vector search.
    """

    def __init__(self, session: Session, manager: ConnectionManager):
        """Initialize with database connections.

        Args:
            session: SQLAlchemy session for metadata
            manager: ConnectionManager for vectors database
        """
        self.session = session
        self.manager = manager
        self._embeddings = QueryEmbeddings(manager) if manager.vectors_enabled else None

    def find_similar(
        self,
        question: str,
        source_id: str,
        min_similarity: float = 0.7,
        limit: int = 1,
    ) -> LibraryMatch | None:
        """Find the most similar query in the library.

        Args:
            question: Natural language question
            source_id: Source ID to search within
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            limit: Maximum number of matches to consider

        Returns:
            LibraryMatch if found above threshold, None otherwise
        """
        if self._embeddings is None:
            logger.debug("Vectors not enabled, skipping similarity search")
            return None

        # Search embeddings
        similar = self._embeddings.find_similar(
            text=question,
            limit=limit * 2,  # Get extra to filter by source
            min_similarity=min_similarity,
        )

        if not similar:
            return None

        # Filter by source and get metadata
        for sim in similar:
            entry = self._get_entry(sim.query_id)
            if entry and entry.source_id == source_id:
                logger.info(
                    f"Found similar query: {sim.query_id} (similarity: {sim.similarity:.3f})"
                )
                return LibraryMatch(entry=entry, similarity=sim.similarity)

        return None

    def find_similar_all(
        self,
        question: str,
        source_id: str,
        min_similarity: float = 0.5,
        limit: int = 5,
    ) -> list[LibraryMatch]:
        """Find all similar queries above threshold.

        Args:
            question: Natural language question
            source_id: Source ID to search within
            min_similarity: Minimum similarity threshold
            limit: Maximum number of results

        Returns:
            List of LibraryMatch ordered by similarity
        """
        if self._embeddings is None:
            return []

        similar = self._embeddings.find_similar(
            text=question,
            limit=limit * 3,
            min_similarity=min_similarity,
        )

        results = []
        for sim in similar:
            entry = self._get_entry(sim.query_id)
            if entry and entry.source_id == source_id:
                results.append(LibraryMatch(entry=entry, similarity=sim.similarity))
                if len(results) >= limit:
                    break

        return results

    def save(
        self,
        source_id: str,
        question: str,
        sql: str,
        *,
        name: str | None = None,
        description: str | None = None,
        assumptions: list[dict[str, Any]] | None = None,
        column_mappings: dict[str, str] | None = None,
        contract_name: str | None = None,
        confidence_level: str = "GREEN",
        graph_id: str | None = None,
    ) -> QueryLibraryEntry:
        """Save a query to the library.

        Args:
            source_id: Source ID this query belongs to
            question: Original natural language question
            sql: Generated SQL
            name: Optional name for the query
            description: Optional description
            assumptions: List of assumption dicts
            column_mappings: Column name mappings
            contract_name: Contract used
            confidence_level: Confidence level string
            graph_id: Graph ID if seeded from graph definition

        Returns:
            Created QueryLibraryEntry
        """
        query_id = str(uuid4())

        # Build embedding text
        if graph_id:
            embedding_text = build_embedding_text_for_graph(
                name=name or graph_id,
                description=description,
            )
        else:
            embedding_text = build_embedding_text_for_question(question)

        entry = QueryLibraryEntry(
            query_id=query_id,
            source_id=source_id,
            original_question=question if not graph_id else None,
            graph_id=graph_id,
            name=name,
            description=description,
            final_sql=sql,
            column_mappings=column_mappings or {},
            assumptions=assumptions or [],
            contract_name=contract_name,
            confidence_level=confidence_level,
            embedding_text=embedding_text,
            created_at=datetime.now(UTC),
        )

        self.session.add(entry)
        self.session.flush()  # Get the ID without committing

        # Store embedding
        if self._embeddings:
            self._embeddings.add_query(query_id, embedding_text)

        logger.info(f"Saved query to library: {query_id}")
        return entry

    def record_execution(
        self,
        source_id: str,
        question: str,
        sql: str,
        *,
        library_entry_id: str | None = None,
        similarity_score: float | None = None,
        success: bool = True,
        row_count: int | None = None,
        error_message: str | None = None,
        confidence_level: str = "GREEN",
        contract_name: str | None = None,
    ) -> QueryExecutionRecord:
        """Record a query execution.

        Args:
            source_id: Source ID
            question: Question asked
            sql: SQL executed
            library_entry_id: Library entry if reused
            similarity_score: Similarity if reused
            success: Whether execution succeeded
            row_count: Number of rows returned
            error_message: Error if failed
            confidence_level: Confidence level
            contract_name: Contract used

        Returns:
            Created QueryExecutionRecord
        """
        record = QueryExecutionRecord(
            execution_id=str(uuid4()),
            library_entry_id=library_entry_id,
            source_id=source_id,
            question=question,
            sql_executed=sql,
            executed_at=datetime.now(UTC),
            success=success,
            row_count=row_count,
            error_message=error_message,
            confidence_level=confidence_level,
            contract_name=contract_name,
            similarity_score=similarity_score,
        )

        self.session.add(record)

        # Update usage count if reused
        if library_entry_id:
            self.session.execute(
                update(QueryLibraryEntry)
                .where(QueryLibraryEntry.query_id == library_entry_id)
                .values(
                    usage_count=QueryLibraryEntry.usage_count + 1,
                    last_used_at=datetime.now(UTC),
                )
            )

        return record

    def get_entry(self, query_id: str) -> QueryLibraryEntry | None:
        """Get a library entry by ID.

        Args:
            query_id: Query ID

        Returns:
            QueryLibraryEntry or None
        """
        return self._get_entry(query_id)

    def list_entries(
        self,
        source_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[QueryLibraryEntry]:
        """List library entries for a source.

        Args:
            source_id: Source ID to filter by
            limit: Maximum entries to return
            offset: Offset for pagination

        Returns:
            List of QueryLibraryEntry
        """
        stmt = (
            select(QueryLibraryEntry)
            .where(QueryLibraryEntry.source_id == source_id)
            .order_by(QueryLibraryEntry.usage_count.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def delete_entry(self, query_id: str) -> bool:
        """Delete a library entry.

        Args:
            query_id: Query ID to delete

        Returns:
            True if deleted, False if not found
        """
        entry = self._get_entry(query_id)
        if not entry:
            return False

        # Remove embedding
        if self._embeddings:
            self._embeddings.remove_query(query_id)

        self.session.delete(entry)
        return True

    def count(self, source_id: str | None = None) -> int:
        """Count library entries.

        Args:
            source_id: Optional source ID filter

        Returns:
            Count of entries
        """
        from sqlalchemy import func

        stmt = select(func.count(QueryLibraryEntry.query_id))
        if source_id:
            stmt = stmt.where(QueryLibraryEntry.source_id == source_id)
        result = self.session.execute(stmt).scalar()
        return result or 0

    def _get_entry(self, query_id: str) -> QueryLibraryEntry | None:
        """Get entry by ID (internal helper)."""
        return self.session.get(QueryLibraryEntry, query_id)

    def seed_from_graphs(self, source_id: str, graphs_dir: str | None = None) -> int:
        """Seed the library from graph definitions with generated SQL.

        Finds all metric graphs that have generated SQL (GeneratedCodeRecord)
        and adds them to the library for semantic search.

        Args:
            source_id: Source ID to associate entries with
            graphs_dir: Optional custom graphs directory

        Returns:
            Number of entries seeded
        """
        from pathlib import Path

        from dataraum.graphs.db_models import GeneratedCodeRecord
        from dataraum.graphs.loader import GraphLoader

        # Load graph definitions
        loader = GraphLoader(graphs_dir=Path(graphs_dir) if graphs_dir else None)
        loader.load_all()

        metric_graphs = loader.get_metric_graphs()
        if not metric_graphs:
            logger.info("No metric graphs found to seed")
            return 0

        seeded = 0

        for graph in metric_graphs:
            # Check if already seeded
            existing = (
                self.session.execute(
                    select(QueryLibraryEntry).where(
                        QueryLibraryEntry.source_id == source_id,
                        QueryLibraryEntry.graph_id == graph.graph_id,
                    )
                )
                .scalars()
                .first()
            )
            if existing:
                logger.debug(f"Graph {graph.graph_id} already seeded, skipping")
                continue

            # Find generated SQL for this graph
            code_record = (
                self.session.execute(
                    select(GeneratedCodeRecord).where(
                        GeneratedCodeRecord.graph_id == graph.graph_id,
                        GeneratedCodeRecord.is_validated == True,  # noqa: E712
                    )
                )
                .scalars()
                .first()
            )

            if not code_record:
                logger.debug(f"No validated SQL for graph {graph.graph_id}, skipping")
                continue

            # Build natural language question from graph metadata
            question = _build_question_from_graph(graph)

            # Save to library
            self.save(
                source_id=source_id,
                question=question,
                sql=code_record.final_sql,
                name=graph.metadata.name,
                description=graph.metadata.description,
                column_mappings=code_record.column_mappings,
                confidence_level="GREEN",
                graph_id=graph.graph_id,
            )
            seeded += 1
            logger.info(f"Seeded graph {graph.graph_id}: {graph.metadata.name}")

        return seeded

    def seed_from_graph(
        self,
        source_id: str,
        graph_id: str,
        sql: str,
        name: str,
        description: str | None = None,
        column_mappings: dict[str, str] | None = None,
    ) -> QueryLibraryEntry | None:
        """Seed a single graph entry to the library.

        Use this when you have the SQL already (e.g., after graph execution).

        Args:
            source_id: Source ID to associate with
            graph_id: Graph ID being seeded
            sql: Generated SQL
            name: Graph name
            description: Graph description
            column_mappings: Column mappings from generation

        Returns:
            Created entry, or None if already exists
        """
        # Check if already seeded
        existing = (
            self.session.execute(
                select(QueryLibraryEntry).where(
                    QueryLibraryEntry.source_id == source_id,
                    QueryLibraryEntry.graph_id == graph_id,
                )
            )
            .scalars()
            .first()
        )
        if existing:
            return None

        # Build question from name/description
        question = f"Calculate {name}"
        if description:
            question = f"{question}: {description}"

        return self.save(
            source_id=source_id,
            question=question,
            sql=sql,
            name=name,
            description=description,
            column_mappings=column_mappings,
            confidence_level="GREEN",
            graph_id=graph_id,
        )


def _build_question_from_graph(graph: Any) -> str:
    """Build a natural language question from a graph definition.

    Args:
        graph: TransformationGraph instance

    Returns:
        Natural language question
    """
    name = graph.metadata.name
    description = graph.metadata.description
    category = graph.metadata.category

    # Build question
    if description:
        question = f"Calculate {name}: {description}"
    else:
        question = f"Calculate {name}"

    # Add category context
    if category:
        question = f"{question} ({category} metric)"

    return question


__all__ = ["QueryLibrary", "LibraryMatch"]
