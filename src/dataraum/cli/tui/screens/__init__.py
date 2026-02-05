"""TUI screens for DataRaum."""

from dataraum.cli.tui.screens.column_detail import ColumnDetailScreen
from dataraum.cli.tui.screens.contract_detail import ContractDetailScreen
from dataraum.cli.tui.screens.contracts import ContractsScreen
from dataraum.cli.tui.screens.entropy import EntropyScreen
from dataraum.cli.tui.screens.home import HomeScreen
from dataraum.cli.tui.screens.query import QueryScreen
from dataraum.cli.tui.screens.table import TableScreen

__all__ = [
    "ColumnDetailScreen",
    "ContractDetailScreen",
    "ContractsScreen",
    "EntropyScreen",
    "HomeScreen",
    "QueryScreen",
    "TableScreen",
]
