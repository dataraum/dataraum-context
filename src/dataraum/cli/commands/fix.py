"""Fix command — re-run pipeline interactively to resolve data quality issues."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Annotated

import typer

from dataraum.cli.common import OutputDirArg, console, setup_logging


def fix(
    output_dir: OutputDirArg = Path("./pipeline_output"),
    contract: Annotated[
        str | None,
        typer.Option("--contract", help="Target contract name for gate evaluation"),
    ] = None,
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Increase logging verbosity"),
    ] = 0,
    log_format: Annotated[
        str,
        typer.Option("--log-format", help="Log output format (console or json)"),
    ] = "console",
) -> None:
    """Re-run pipeline interactively to fix data quality issues.

    Pauses at quality gates so you can review violations and apply fixes.
    The pipeline re-uses existing metadata — only affected phases re-run.

    Examples:

        dataraum fix ./pipeline_output

        dataraum fix ./pipeline_output --contract aggregation_safe
    """
    setup_logging(verbosity=verbose, log_format=log_format)

    # Validate output_dir has existing pipeline data
    metadata_db = output_dir / "metadata.db"
    if not metadata_db.exists():
        console.print(f"[red]Error: No pipeline data found at {output_dir}[/red]")
        console.print("[dim]Run 'dataraum run' first to create pipeline output.[/dim]")
        raise typer.Exit(1)

    # TTY check
    if not sys.stdin.isatty():
        console.print("[red]Error: fix command requires an interactive terminal[/red]")
        raise typer.Exit(1)

    # Interactive contract selection (same as run command)
    if contract is None:
        from dataraum.entropy.contracts import list_contracts

        available = list_contracts()
        if available:
            contract_names = [c["name"] for c in available]
            console.print()
            console.print("[bold]Available contracts:[/bold]")
            for i, c in enumerate(available, 1):
                console.print(f"  [{i}] {c['display_name']} — {c['description']}")
            console.print("  [0] None (skip contract evaluation)")
            console.print()
            try:
                from rich.prompt import Prompt

                choice = Prompt.ask("  Select contract", default="0", console=console)
                choice = choice.strip()
                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= len(contract_names):
                        contract = contract_names[idx - 1]
                        console.print(f"  Using contract: [bold]{contract}[/bold]")
            except (KeyboardInterrupt, EOFError):
                pass

    if contract is None:
        contract = "aggregation_safe"
        console.print(
            f"[dim]No contract specified, using {contract} (override with --contract)[/dim]"
        )

    # Setup pipeline using registered sources
    from dataraum.pipeline.setup import setup_pipeline

    setup = setup_pipeline(
        source_path=None,
        output_dir=output_dir,
        contract=contract,
    )

    gen = setup.scheduler.run()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

        from dataraum.cli.commands.run import _drive_pipeline, _print_summary

        result, stats = _drive_pipeline(
            gen=gen,
            console=console,
            interactive=True,
            contract_name=setup.contract_name,
            contract_thresholds=setup.contract_thresholds,
            session=setup.session,
            source_id=setup.source_id,
        )

    # Update PipelineRun status
    from dataraum.pipeline.db_models import PipelineRun

    run_record = setup.session.get(PipelineRun, setup.run_id)
    if run_record:
        run_record.status = "completed" if result.success else "failed"
        setup.session.commit()

    _print_summary(
        console,
        result,
        stats,
        contract,
        setup.contract_thresholds,
        output_dir=output_dir,
    )

    raise typer.Exit(0 if result.success else 1)
