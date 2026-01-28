"""Integration tests for the query library and embeddings.

Tests verify:
- Embedding generation and similarity search
- Library save/search/reuse cycle
- Library persistence and retrieval
- Usage tracking
"""

from __future__ import annotations

import pytest

from dataraum.query.document import QueryDocument, SQLStep
from dataraum.query.embeddings import (
    QueryEmbeddings,
    build_embedding_text,
    generate_embedding,
)
from dataraum.query.library import QueryLibrary

from .conftest import PipelineTestHarness

pytestmark = pytest.mark.integration


class TestEmbeddingGeneration:
    """Test embedding generation from query documents."""

    def test_generate_embedding_dimension(self):
        """Embeddings should be 384-dimensional vectors."""
        embedding = generate_embedding("What is total revenue?")
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_build_embedding_text_with_document(self):
        """build_embedding_text should combine summary, steps, and assumptions."""
        text = build_embedding_text(
            summary="Calculates total revenue from completed orders.",
            step_descriptions=[
                "Filters orders by completed status",
                "Sums the amount column",
            ],
            assumption_texts=["Currency is USD"],
        )

        assert "revenue" in text
        assert "completed" in text
        assert "Sums" in text
        assert "USD" in text

    def test_similar_queries_have_higher_similarity(self):
        """Revenue-related queries should be more similar than unrelated ones."""
        emb_revenue = generate_embedding("What was total revenue?")
        emb_sales = generate_embedding("How much money did we make in sales?")
        emb_weather = generate_embedding("What is the weather forecast?")

        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            norm_a = sum(x**2 for x in a) ** 0.5
            norm_b = sum(x**2 for x in b) ** 0.5
            return dot / (norm_a * norm_b)

        sim_revenue_sales = cosine_sim(emb_revenue, emb_sales)
        sim_revenue_weather = cosine_sim(emb_revenue, emb_weather)

        assert sim_revenue_sales > sim_revenue_weather


class TestVectorSearch:
    """Test vector similarity search with real embeddings."""

    def test_add_and_find_similar(self, mock_connection_manager):
        """Save queries and find similar ones by text."""
        embeddings = QueryEmbeddings(mock_connection_manager)

        embeddings.add_query("q_revenue", "Calculate total revenue from orders")
        embeddings.add_query("q_count", "Count the number of customers")
        embeddings.add_query("q_avg", "Average order value per customer")

        results = embeddings.find_similar("How much money did we make?", limit=3)

        assert len(results) > 0
        # Revenue query should be most similar to "how much money"
        assert results[0].query_id == "q_revenue"

    def test_find_similar_with_threshold(self, mock_connection_manager):
        """High similarity threshold should filter out poor matches."""
        embeddings = QueryEmbeddings(mock_connection_manager)

        embeddings.add_query("q1", "Total revenue by region for Q3 2024")
        embeddings.add_query("q2", "The quick brown fox jumps over the lazy dog")

        results = embeddings.find_similar(
            "Revenue breakdown by geographic region",
            min_similarity=0.5,
        )

        query_ids = [r.query_id for r in results]
        assert "q1" in query_ids

    def test_batch_add_queries(self, mock_connection_manager):
        """Batch adding queries should work correctly."""
        embeddings = QueryEmbeddings(mock_connection_manager)

        queries = [
            ("q1", "Total revenue"),
            ("q2", "Customer count"),
            ("q3", "Average order value"),
        ]
        embeddings.add_queries_batch(queries)

        assert embeddings.count() == 3


class TestQueryLibrarySaveAndSearch:
    """Test the full save/search cycle using QueryLibrary."""

    def test_save_and_find_similar(
        self,
        analyzed_small_finance: PipelineTestHarness,
        mock_connection_manager,
    ):
        """Save a query and find it by similarity search."""
        with analyzed_small_finance.session_factory() as session:
            library = QueryLibrary(session, mock_connection_manager)
            source_id = analyzed_small_finance.source_id

            doc = QueryDocument(
                summary="Calculates total revenue from all transactions.",
                steps=[
                    SQLStep(
                        step_id="total",
                        sql='SELECT SUM("Amount") FROM typed_transactions',
                        description="Sum the Amount column",
                    ),
                ],
                final_sql='SELECT SUM("Amount") AS total FROM typed_transactions',
                column_mappings={"revenue": "Amount"},
            )

            entry = library.save(
                source_id=source_id,
                document=doc,
                original_question="What is total revenue?",
                name="Total Revenue",
            )
            session.commit()

            assert entry.query_id is not None
            assert entry.source_id == source_id
            assert entry.final_sql == doc.final_sql

            # Search for similar query
            match = library.find_similar(
                "How much money did we make?",
                source_id=source_id,
                min_similarity=0.3,
            )

            assert match is not None
            assert match.entry.query_id == entry.query_id
            assert match.similarity > 0.0

    def test_find_returns_none_for_unrelated_query(
        self,
        analyzed_small_finance: PipelineTestHarness,
        mock_connection_manager,
    ):
        """Search for unrelated topic should return no match above threshold."""
        with analyzed_small_finance.session_factory() as session:
            library = QueryLibrary(session, mock_connection_manager)
            source_id = analyzed_small_finance.source_id

            doc = QueryDocument(
                summary="Counts total customers.",
                steps=[],
                final_sql="SELECT COUNT(*) FROM typed_customers",
            )

            library.save(
                source_id=source_id,
                document=doc,
                original_question="How many customers?",
            )
            session.commit()

            # Search for something very different with high threshold
            match = library.find_similar(
                "What is the temperature on Mars?",
                source_id=source_id,
                min_similarity=0.9,
            )

            assert match is None

    def test_multiple_queries_ranked_by_similarity(
        self,
        analyzed_small_finance: PipelineTestHarness,
        mock_connection_manager,
    ):
        """Multiple saved queries should rank by similarity."""
        with analyzed_small_finance.session_factory() as session:
            library = QueryLibrary(session, mock_connection_manager)
            source_id = analyzed_small_finance.source_id

            # Save revenue query
            library.save(
                source_id=source_id,
                document=QueryDocument(
                    summary="Total revenue from all transactions.",
                    steps=[],
                    final_sql='SELECT SUM("Amount") FROM typed_transactions',
                ),
                original_question="Total revenue?",
                name="Revenue",
            )

            # Save count query
            library.save(
                source_id=source_id,
                document=QueryDocument(
                    summary="Count of all customers.",
                    steps=[],
                    final_sql="SELECT COUNT(*) FROM typed_customers",
                ),
                original_question="How many customers?",
                name="Customer Count",
            )

            # Save order query
            library.save(
                source_id=source_id,
                document=QueryDocument(
                    summary="Average transaction amount.",
                    steps=[],
                    final_sql='SELECT AVG("Amount") FROM typed_transactions',
                ),
                original_question="What is average order value?",
                name="Avg Order",
            )
            session.commit()

            # Find all similar to a revenue question
            matches = library.find_similar_all(
                "Show me total sales figures",
                source_id=source_id,
                min_similarity=0.2,
                limit=3,
            )

            assert len(matches) > 0
            # First match should be the revenue query
            assert (
                "revenue" in matches[0].entry.summary.lower()
                or "amount" in matches[0].entry.summary.lower()
            )


class TestLibraryPersistence:
    """Test library entries persist correctly."""

    def test_entry_fields_persisted(
        self,
        analyzed_small_finance: PipelineTestHarness,
        mock_connection_manager,
    ):
        """All fields of a saved query should be retrievable."""
        with analyzed_small_finance.session_factory() as session:
            library = QueryLibrary(session, mock_connection_manager)
            source_id = analyzed_small_finance.source_id

            doc = QueryDocument(
                summary="Test query for persistence.",
                steps=[
                    SQLStep(step_id="s1", sql="SELECT 1", description="Step one"),
                ],
                final_sql="SELECT 1 AS result",
                column_mappings={"output": "result"},
            )

            entry = library.save(
                source_id=source_id,
                document=doc,
                original_question="Test question?",
                name="Test Query",
                description="A test query for verifying persistence.",
                contract_name="exploratory_analysis",
                confidence_level="GREEN",
            )
            session.commit()

            # Retrieve entry
            retrieved = library.get_entry(entry.query_id)

            assert retrieved is not None
            assert retrieved.query_id == entry.query_id
            assert retrieved.source_id == source_id
            assert retrieved.original_question == "Test question?"
            assert retrieved.name == "Test Query"
            assert retrieved.description == "A test query for verifying persistence."
            assert retrieved.summary == "Test query for persistence."
            assert retrieved.final_sql == "SELECT 1 AS result"
            assert retrieved.contract_name == "exploratory_analysis"
            assert retrieved.confidence_level == "GREEN"
            assert retrieved.column_mappings == {"output": "result"}
            assert len(retrieved.steps_json) == 1

    def test_list_queries_for_source(
        self,
        analyzed_small_finance: PipelineTestHarness,
        mock_connection_manager,
    ):
        """list() should return all queries for a source."""
        with analyzed_small_finance.session_factory() as session:
            library = QueryLibrary(session, mock_connection_manager)
            source_id = analyzed_small_finance.source_id

            for i in range(3):
                library.save(
                    source_id=source_id,
                    document=QueryDocument(
                        summary=f"Query {i}.",
                        steps=[],
                        final_sql=f"SELECT {i}",
                    ),
                )
            session.commit()

            entries = library.list_entries(source_id)

            assert len(entries) == 3


class TestUsageTracking:
    """Test query usage tracking."""

    def test_initial_usage_count_is_zero(
        self,
        analyzed_small_finance: PipelineTestHarness,
        mock_connection_manager,
    ):
        """New queries should have zero usage count."""
        with analyzed_small_finance.session_factory() as session:
            library = QueryLibrary(session, mock_connection_manager)
            source_id = analyzed_small_finance.source_id

            entry = library.save(
                source_id=source_id,
                document=QueryDocument(
                    summary="Test.",
                    steps=[],
                    final_sql="SELECT 1",
                ),
            )
            session.commit()

            assert entry.usage_count == 0

    def test_record_execution_increments_usage(
        self,
        analyzed_small_finance: PipelineTestHarness,
        mock_connection_manager,
    ):
        """Recording an execution should increment usage count."""
        with analyzed_small_finance.session_factory() as session:
            library = QueryLibrary(session, mock_connection_manager)
            source_id = analyzed_small_finance.source_id

            entry = library.save(
                source_id=source_id,
                document=QueryDocument(
                    summary="Revenue query.",
                    steps=[],
                    final_sql='SELECT SUM("Amount") FROM typed_transactions',
                ),
                original_question="Total revenue?",
            )
            session.commit()

            # Record an execution
            library.record_execution(
                source_id=source_id,
                question="How much total revenue?",
                sql='SELECT SUM("Amount") FROM typed_transactions',
                library_entry_id=entry.query_id,
                similarity_score=0.92,
                success=True,
                row_count=1,
            )
            session.commit()

            # Refresh to get updated values
            session.expire(entry)
            updated = library.get_entry(entry.query_id)
            assert updated is not None
            assert updated.usage_count == 1
            assert updated.last_used_at is not None
