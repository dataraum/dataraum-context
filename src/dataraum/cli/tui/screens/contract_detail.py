"""Contract detail screen - deep dive into contract evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Static

from dataraum.cli.common import get_manager


class ContractDetailScreen(Screen[None]):
    """Contract detail screen showing dimensions, violations, and path to compliance."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(
        self,
        output_dir: Path,
        contract_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.contract_name = contract_name
        self._data_loaded = False
        self._violations: list[Any] = []

    def compose(self) -> ComposeResult:
        """Create the screen layout."""
        yield Container(
            ScrollableContainer(
                Vertical(
                    # Header
                    Static(f"Contract: {self.contract_name}", classes="screen-title"),
                    Static("Loading...", id="status-line"),
                    Static("", id="description-text"),
                    # Dimension scores section
                    Container(
                        Static("Dimension Scores", classes="section-title"),
                        DataTable(id="dimensions-table"),
                        classes="dimensions-section",
                    ),
                    # Violations section
                    Container(
                        Static("Violations", classes="section-title"),
                        DataTable(id="violations-table"),
                        classes="violations-section",
                    ),
                    # Warnings section
                    Container(
                        Static("Warnings", classes="section-title"),
                        Static("", id="warnings-text"),
                        classes="warnings-section",
                    ),
                    # Path to compliance section
                    Container(
                        Static("Path to Compliance", classes="section-title"),
                        Static("", id="compliance-text"),
                        classes="compliance-section",
                    ),
                    classes="main-content",
                ),
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
        """Load contract evaluation data from database."""
        if self._data_loaded:
            return

        from sqlalchemy import select

        from dataraum.entropy.analysis.aggregator import ColumnSummary, EntropyAggregator
        from dataraum.entropy.contracts import (
            evaluate_contract,
            get_contract,
        )
        from dataraum.entropy.core.storage import EntropyRepository
        from dataraum.storage import Source, Table

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

                # Get contract profile
                profile = get_contract(self.contract_name)
                if not profile:
                    self._show_error(f"Contract not found: {self.contract_name}")
                    return

                # Get tables
                tables_result = session.execute(
                    select(Table).where(Table.source_id == source.source_id)
                )
                tables = tables_result.scalars().all()

                if not tables:
                    self._show_error("No tables found")
                    return

                table_ids = [t.table_id for t in tables]

                # Build column summaries
                repo = EntropyRepository(session)
                aggregator = EntropyAggregator()

                typed_table_ids = repo.get_typed_table_ids(table_ids)
                column_summaries: dict[str, ColumnSummary] = {}
                compound_risks: list[Any] = []

                if typed_table_ids:
                    table_map, column_map = repo.get_table_column_mapping(typed_table_ids)
                    entropy_objects = repo.load_for_tables(typed_table_ids, enforce_typed=True)

                    if entropy_objects:
                        column_summaries, _ = aggregator.summarize_columns_by_table(
                            entropy_objects=entropy_objects,
                            table_map=table_map,
                            column_map=column_map,
                        )
                        for summary in column_summaries.values():
                            compound_risks.extend(summary.compound_risks)

                # Evaluate the contract
                evaluation = evaluate_contract(column_summaries, self.contract_name, compound_risks)

                # Update UI
                self._update_status(evaluation, profile)
                self._update_description(profile)
                self._update_dimensions(evaluation, profile)
                self._update_violations(evaluation)
                self._update_warnings(evaluation)
                self._update_compliance(evaluation)

                self._data_loaded = True
        finally:
            manager.close()

    def _show_error(self, message: str) -> None:
        """Display an error message."""
        status = self.query_one("#status-line", Static)
        status.update(f"[red]{message}[/red]")

    def _update_status(self, evaluation: Any, profile: Any) -> None:
        """Update the status line."""
        from dataraum.entropy.contracts import ConfidenceLevel

        status = self.query_one("#status-line", Static)
        title = self.query_one(".screen-title", Static)

        # Traffic light status
        if evaluation.confidence_level == ConfidenceLevel.GREEN:
            status_str = "[green]PASS[/green]"
        elif evaluation.confidence_level == ConfidenceLevel.YELLOW:
            status_str = "[yellow]WARN[/yellow]"
        elif evaluation.confidence_level == ConfidenceLevel.ORANGE:
            status_str = "[yellow]LOW CONFIDENCE[/yellow]"
        else:
            status_str = "[red]FAIL[/red]"

        title.update(f"Contract: {profile.display_name}")

        # Overall score
        overall_pass = evaluation.overall_score <= profile.overall_threshold
        overall_mark = "[green]Pass[/green]" if overall_pass else "[red]Fail[/red]"

        status.update(
            f"Status: {status_str} | "
            f"Overall Score: {evaluation.overall_score:.3f} "
            f"(threshold: {profile.overall_threshold}) {overall_mark}"
        )

    def _update_description(self, profile: Any) -> None:
        """Update the description text."""
        description = self.query_one("#description-text", Static)
        description.update(f"[dim]{profile.description}[/dim]")

    def _update_dimensions(self, evaluation: Any, profile: Any) -> None:
        """Update the dimensions table."""
        table = self.query_one("#dimensions-table", DataTable)
        table.clear(columns=True)

        table.add_column("Dimension", key="dimension")
        table.add_column("Score", key="score")
        table.add_column("Threshold", key="threshold")
        table.add_column("Status", key="status")

        for dim, threshold in profile.dimension_thresholds.items():
            score = evaluation.dimension_scores.get(dim, 0.0)

            if score <= threshold:
                status = "[green]Pass[/green]"
            elif score <= threshold * 1.2:
                status = "[yellow]Warn[/yellow]"
            else:
                status = "[red]Fail[/red]"

            table.add_row(
                dim,
                f"{score:.3f}",
                f"{threshold:.3f}",
                status,
            )

    def _update_violations(self, evaluation: Any) -> None:
        """Update the violations table."""
        table = self.query_one("#violations-table", DataTable)
        table.clear(columns=True)
        table.cursor_type = "row"

        table.add_column("Dimension", key="dimension")
        table.add_column("Score", key="score")
        table.add_column("Threshold", key="threshold")
        table.add_column("Affected", key="affected")

        self._violations = list(evaluation.violations) if evaluation.violations else []

        if not self._violations:
            table.add_row("-", "No violations", "-", "-")
            return

        for v in self._violations:
            if v.dimension:
                affected_count = len(v.affected_columns) if v.affected_columns else 0
                affected_str = f"{affected_count} columns"
                table.add_row(
                    v.dimension,
                    f"[red]{v.actual:.3f}[/red]",
                    f"{v.max_allowed:.3f}",
                    affected_str,
                )
            elif v.condition:
                table.add_row(
                    v.condition,
                    "[red]Violated[/red]",
                    "-",
                    v.details[:30] if v.details else "-",
                )
            else:
                table.add_row(
                    "-",
                    "[red]Violated[/red]",
                    "-",
                    v.details[:40] if v.details else "-",
                )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection to show affected columns."""
        if event.data_table.id != "violations-table":
            return

        if event.cursor_row >= len(self._violations):
            return

        violation = self._violations[event.cursor_row]

        if not violation.affected_columns:
            self.notify("No affected columns for this violation")
            return

        # Show affected columns in a notification
        cols = ", ".join(violation.affected_columns[:10])
        if len(violation.affected_columns) > 10:
            cols += f" (+{len(violation.affected_columns) - 10} more)"

        self.notify(
            f"Affected columns:\n{cols}",
            title=violation.dimension or "Violation",
            timeout=10,
        )

    def _update_warnings(self, evaluation: Any) -> None:
        """Update the warnings text."""
        warnings_widget = self.query_one("#warnings-text", Static)

        if not evaluation.warnings:
            warnings_widget.update("[dim]No warnings[/dim]")
            return

        warning_lines = []
        for w in evaluation.warnings:
            warning_lines.append(f"[yellow]Warning:[/yellow] {w.details}")

        warnings_widget.update("\n".join(warning_lines))

    def _update_compliance(self, evaluation: Any) -> None:
        """Update the path to compliance text."""
        compliance_widget = self.query_one("#compliance-text", Static)

        if evaluation.is_compliant:
            compliance_widget.update("[green]Contract is compliant![/green]")
            return

        parts = []

        if evaluation.worst_dimension:
            parts.append(
                f"[bold]Focus on:[/bold] [cyan]{evaluation.worst_dimension}[/cyan] "
                f"(score: {evaluation.worst_dimension_score:.3f})"
            )

        if evaluation.estimated_effort_to_comply:
            parts.append(f"[bold]Estimated effort:[/bold] {evaluation.estimated_effort_to_comply}")

        # Add specific recommendations based on violations
        if evaluation.violations:
            parts.append("")
            parts.append("[bold]Recommendations:[/bold]")
            for v in evaluation.violations[:3]:  # Top 3 violations
                if v.dimension:
                    parts.append(
                        f"  - Reduce {v.dimension} from {v.actual:.3f} to below {v.max_allowed:.3f}"
                    )

        if parts:
            compliance_widget.update("\n".join(parts))
        else:
            compliance_widget.update("[dim]No compliance path available[/dim]")
