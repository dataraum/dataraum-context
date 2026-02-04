"""MCP Server for DataRaum.

Provides high-level tools for LLM integration:
- get_context: Full context document for data analysis
- get_entropy: Entropy summary with optional table filter
- evaluate_contract: Contract evaluation results
- query: Natural language query execution
"""

from dataraum.mcp.server import create_server, run_server

__all__ = ["create_server", "run_server"]
