"""Query screen - natural language query interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Input, Static

from dataraum.cli.common import get_manager


class QueryScreen(Screen[None]):
    """Query screen with natural language input and results display."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("ctrl+l", "clear_history", "Clear"),
    ]

    def __init__(self, output_dir: Path, initial_query: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.initial_query = initial_query
        self._history: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        """Create the screen layout."""
        yield Container(
            Vertical(
                Static("Query", classes="screen-title"),
                Static("Ask questions about your data in natural language", id="subtitle"),
                # Input section
                Container(
                    Input(placeholder="Enter your question...", id="query-input"),
                    classes="input-section",
                ),
                # Results section
                Container(
                    Static("", id="answer-display"),
                    classes="answer-section",
                ),
                # Data table
                Container(
                    Static("Results", classes="section-title"),
                    DataTable(id="results-table"),
                    classes="results-section",
                ),
                # Query info
                Container(
                    Static("", id="query-info"),
                    classes="info-section",
                ),
                # History
                Container(
                    Static("Recent Queries", classes="section-title"),
                    DataTable(id="history-table"),
                    classes="history-section",
                ),
                classes="main-content",
            ),
            classes="screen-container",
        )

    def on_mount(self) -> None:
        """Initialize the screen."""
        # Set up history table
        history_table = self.query_one("#history-table", DataTable)
        history_table.add_column("Question", key="question")
        history_table.add_column("Confidence", key="confidence")

        # Set up results table
        results_table = self.query_one("#results-table", DataTable)
        results_table.cursor_type = "row"

        # Load history from database
        self._load_history()

        # Focus input
        self.query_one("#query-input", Input).focus()

        # If initial query provided, run it
        if self.initial_query:
            input_widget = self.query_one("#query-input", Input)
            input_widget.value = self.initial_query

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle query submission."""
        if event.input.id == "query-input" and event.value.strip():
            self._execute_query(event.value.strip())
            event.input.value = ""

    def action_clear_history(self) -> None:
        """Clear the query history display."""
        self._history.clear()
        history_table = self.query_one("#history-table", DataTable)
        history_table.clear()

    def _load_history(self) -> None:
        """Load recent queries from database."""
        from sqlalchemy import select

        from dataraum.query.db_models import QueryLibraryEntry
        from dataraum.storage import Source

        manager = get_manager(self.output_dir)

        try:
            with manager.session_scope() as session:
                sources_result = session.execute(select(Source))
                sources = sources_result.scalars().all()

                if not sources:
                    return

                source = sources[0]

                # Get recent queries
                queries_result = session.execute(
                    select(QueryLibraryEntry)
                    .where(QueryLibraryEntry.source_id == source.source_id)
                    .order_by(QueryLibraryEntry.created_at.desc())
                    .limit(10)
                )
                queries = queries_result.scalars().all()

                history_table = self.query_one("#history-table", DataTable)
                for q in queries:
                    confidence = q.confidence_level or "-"
                    question_text = q.original_question or q.name or "-"
                    question_display = (
                        question_text[:50] + "..." if len(question_text) > 50 else question_text
                    )
                    history_table.add_row(question_display, confidence)
        finally:
            manager.close()

    def _execute_query(self, question: str) -> None:
        """Execute a natural language query."""
        from sqlalchemy import select

        from dataraum.query import answer_question
        from dataraum.storage import Source

        manager = get_manager(self.output_dir)

        # Show loading state
        answer_display = self.query_one("#answer-display", Static)
        answer_display.update("[dim]Processing query...[/dim]")

        try:
            with manager.session_scope() as session:
                sources_result = session.execute(select(Source))
                sources = sources_result.scalars().all()

                if not sources:
                    answer_display.update("[red]No sources found[/red]")
                    return

                source = sources[0]

                with manager.duckdb_cursor() as cursor:
                    result = answer_question(
                        question=question,
                        session=session,
                        duckdb_conn=cursor,
                        source_id=source.source_id,
                        manager=manager,
                    )

                if not result.success or not result.value:
                    answer_display.update(f"[red]Error: {result.error}[/red]")
                    return

                query_result = result.value

                # Display answer
                confidence_colors = {
                    "green": "green",
                    "yellow": "yellow",
                    "orange": "yellow",
                    "red": "red",
                }
                color = confidence_colors.get(query_result.confidence_level.value, "white")
                emoji = query_result.confidence_level.emoji

                answer_display.update(
                    f"[{color}]{emoji} {query_result.confidence_level.label}[/{color}]\n\n"
                    f"{query_result.answer}"
                )

                # Display data table
                self._update_results_table(query_result.columns, query_result.data)

                # Display query info
                info_parts = []
                if query_result.sql:
                    sql_preview = (
                        query_result.sql[:100] + "..."
                        if len(query_result.sql) > 100
                        else query_result.sql
                    )
                    info_parts.append(f"[dim]SQL: {sql_preview}[/dim]")
                if query_result.assumptions:
                    info_parts.append(f"[dim]Assumptions: {len(query_result.assumptions)}[/dim]")

                query_info = self.query_one("#query-info", Static)
                query_info.update("\n".join(info_parts))

                # Add to history display
                history_table = self.query_one("#history-table", DataTable)
                q_display = question[:50] + "..." if len(question) > 50 else question
                history_table.add_row(q_display, query_result.confidence_level.label, key=None)

        finally:
            manager.close()

    def _update_results_table(
        self, columns: list[str] | None, data: list[dict[str, Any]] | None
    ) -> None:
        """Update the results data table."""
        table = self.query_one("#results-table", DataTable)
        table.clear(columns=True)

        if not columns or not data:
            return

        # Add columns
        for col in columns:
            table.add_column(col, key=col)

        # Add rows (limit to 20)
        for row in data[:20]:
            table.add_row(*[str(row.get(c, "")) for c in columns])

        if len(data) > 20:
            # Add indicator row
            table.add_row(
                *[f"... +{len(data) - 20} more" if i == 0 else "" for i in range(len(columns))]
            )
