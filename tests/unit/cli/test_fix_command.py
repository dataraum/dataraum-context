"""Tests for the fix CLI command."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from dataraum.cli.main import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class TestFixCommandRegistered:
    def test_help(self) -> None:
        result = runner.invoke(app, ["fix", "--help"])
        assert result.exit_code == 0
        assert "Re-run pipeline interactively" in result.output

    def test_listed_in_main_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "fix" in result.output

    def test_contract_flag_in_help(self) -> None:
        """--contract is listed as a valid option."""
        result = runner.invoke(app, ["fix", "--help"])
        assert result.exit_code == 0
        assert "--contract" in _strip_ansi(result.output)


class TestFixRequiresMetadata:
    def test_missing_metadata_db(self, tmp_path) -> None:
        """Fix command fails when no metadata.db exists in output dir."""
        # Create the directory but no metadata.db
        result = runner.invoke(app, ["fix", str(tmp_path)])
        assert result.exit_code != 0
        assert "No pipeline data found" in result.output
