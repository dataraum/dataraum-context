"""Tests for _parse_target and apply_fix helpers."""

from __future__ import annotations

import pytest

from dataraum.mcp.server import _parse_target


class TestParseTarget:
    def test_column_prefixed(self) -> None:
        assert _parse_target("column:orders.amount") == ("orders", "amount")

    def test_table_prefixed(self) -> None:
        assert _parse_target("table:orders") == ("orders", None)

    def test_bare_table_dot_column(self) -> None:
        assert _parse_target("orders.amount") == ("orders", "amount")

    def test_bare_table_only(self) -> None:
        assert _parse_target("orders") == ("orders", None)

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _parse_target("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _parse_target("   ")

    def test_unknown_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown target prefix"):
            _parse_target("view:orders.amount")

    def test_empty_table_after_column_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty table name"):
            _parse_target("column:")

    def test_empty_table_after_table_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty table name"):
            _parse_target("table:")

    def test_trailing_dot_gives_none_column(self) -> None:
        # "column:orders." → table="orders", col="" → treated as None
        assert _parse_target("column:orders.") == ("orders", None)
