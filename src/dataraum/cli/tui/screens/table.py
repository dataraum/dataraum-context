"""Table detail screen - column details and entropy breakdown."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Static

from dataraum.cli.common import get_manager


class TableScreen(Screen[None]):
    """Table detail screen showing columns, types, and entropy."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self, output_dir: Path, table_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.table_name = table_name
        self._data_loaded = False

    def compose(self) -> ComposeResult:
        """Create the screen layout."""
        yield Container(
            Vertical(
                Static(f"Table: {self.table_name}", classes="screen-title"),
                Static("Loading...", id="table-info"),
                # Columns section
                Container(
                    Static("Columns", classes="section-title"),
                    DataTable(id="columns-table"),
                    classes="columns-section",
                ),
                # Statistics section
                Container(
                    Static("Statistics", classes="section-title"),
                    Horizontal(
                        Static("", id="stats-left"),
                        Static("", id="stats-right"),
                        classes="stats-row",
                    ),
                    classes="stats-section",
                ),
                # Sample data section
                Container(
                    Static("Sample Data", classes="section-title"),
                    DataTable(id="sample-table"),
                    classes="sample-section",
                ),
                classes="main-content",
            ),
            classes="screen-container",
        )

    def on_mount(self) -> None:
        """Load data when screen mounts."""
        self._load_data()

    def action_refresh(self) -> None:
        """Refresh the data."""
        self._data_loaded = False
        self._load_data()

    def _load_data(self) -> None:
        """Load table data from database."""
        if self._data_loaded:
            return

        from sqlalchemy import select

        from dataraum.analysis.typing.db_models import TypeDecision
        from dataraum.entropy.db_models import EntropyInterpretationRecord
        from dataraum.storage import Column, Source, Table

        manager = get_manager(self.output_dir)

        try:
            with manager.session_scope() as session:
                # Get source
                sources_result = session.execute(select(Source))
                sources = sources_result.scalars().all()

                if not sources:
                    self._show_error("No sources found")
                    return

                source = sources[0]

                # Get table
                table_result = session.execute(
                    select(Table).where(
                        Table.source_id == source.source_id,
                        Table.table_name == self.table_name,
                    )
                )
                table = table_result.scalar_one_or_none()

                if not table:
                    self._show_error(f"Table not found: {self.table_name}")
                    return

                # Get columns
                columns_result = session.execute(
                    select(Column).where(Column.table_id == table.table_id)
                )
                columns = columns_result.scalars().all()

                # Get type decisions
                type_decisions = {}
                for col in columns:
                    td_result = session.execute(
                        select(TypeDecision).where(TypeDecision.column_id == col.column_id)
                    )
                    td = td_result.scalar_one_or_none()
                    if td:
                        type_decisions[col.column_id] = td

                # Get entropy interpretations
                entropy_interps = {}
                for col in columns:
                    ei_result = session.execute(
                        select(EntropyInterpretationRecord).where(
                            EntropyInterpretationRecord.column_id == col.column_id
                        )
                    )
                    ei = ei_result.scalar_one_or_none()
                    if ei:
                        entropy_interps[col.column_id] = ei

                # Update UI
                self._update_table_info(table)
                self._update_columns_table(columns, type_decisions, entropy_interps)
                self._update_stats(table, columns)

                # Load sample data
                with manager.duckdb_cursor() as cursor:
                    self._load_sample_data(cursor, table.table_name, columns)

                self._data_loaded = True
        finally:
            manager.close()

    def _show_error(self, message: str) -> None:
        """Display an error message."""
        info = self.query_one("#table-info", Static)
        info.update(f"[red]{message}[/red]")

    def _update_table_info(self, table: Any) -> None:
        """Update the table info section."""
        info = self.query_one("#table-info", Static)
        layer = table.layer or "unknown"
        rows = table.row_count or "?"
        info.update(f"Layer: {layer} | Rows: {rows}")

    def _update_columns_table(
        self,
        columns: Sequence[Any],
        type_decisions: dict[str, Any],
        entropy_interps: dict[str, Any],
    ) -> None:
        """Update the columns data table."""
        table = self.query_one("#columns-table", DataTable)
        table.clear(columns=True)

        table.add_column("Column", key="column")
        table.add_column("Type", key="type")
        table.add_column("Entropy", key="entropy")
        table.add_column("Status", key="status")

        readiness_icons = {
            "ready": "[green]Ready[/green]",
            "investigate": "[yellow]Investigate[/yellow]",
            "blocked": "[red]Blocked[/red]",
        }

        for col in columns:
            td = type_decisions.get(col.column_id)
            ei = entropy_interps.get(col.column_id)

            col_type = td.decided_type if td else col.inferred_type or "-"
            entropy = f"{ei.composite_score:.3f}" if ei else "-"
            status = readiness_icons.get(ei.readiness, "-") if ei else "-"

            table.add_row(col.column_name, col_type, entropy, status)

    def _update_stats(self, table: Any, columns: Sequence[Any]) -> None:
        """Update the statistics section."""
        stats_left = self.query_one("#stats-left", Static)
        stats_right = self.query_one("#stats-right", Static)

        stats_left.update(f"Total columns: {len(columns)}\nRow count: {table.row_count or '?'}")

        # Count by type
        type_counts: dict[str, int] = {}
        for col in columns:
            t = col.inferred_type or "unknown"
            type_counts[t] = type_counts.get(t, 0) + 1

        type_summary = "\n".join(f"{t}: {c}" for t, c in sorted(type_counts.items())[:5])
        stats_right.update(f"Types:\n{type_summary}")

    def _load_sample_data(self, cursor: Any, table_name: str, columns: Sequence[Any]) -> None:
        """Load sample data from DuckDB."""
        sample_table = self.query_one("#sample-table", DataTable)
        sample_table.clear(columns=True)

        try:
            # Query sample data
            col_names = [c.column_name for c in columns[:10]]  # Limit columns
            cols_sql = ", ".join(f'"{c}"' for c in col_names)
            cursor.execute(f'SELECT {cols_sql} FROM "{table_name}" LIMIT 10')
            rows = cursor.fetchall()

            # Add columns
            for col_name in col_names:
                display_name = col_name[:15] + "..." if len(col_name) > 15 else col_name
                sample_table.add_column(display_name, key=col_name)

            # Add rows
            for row in rows:
                sample_table.add_row(*[str(v)[:30] if v is not None else "" for v in row])

        except Exception as e:
            sample_table.add_column("Error")
            sample_table.add_row(f"Could not load sample: {e}")
