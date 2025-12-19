#!/usr/bin/env python3
"""Phase 11: Business Cycle Detection with Expert LLM Agent.

This script runs an LLM-powered expert agent to detect business cycles
in the data using semantic metadata as context and tools for on-demand
data exploration.

Prerequisites:
    - Phase 5 must be completed (semantic analysis with annotations)
    - ANTHROPIC_API_KEY must be set in environment or .env file

Usage:
    uv run python scripts/run_phase11_business_cycles.py
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from infra import (
    cleanup_connections,
    get_duckdb_conn,
    get_test_session,
    get_typed_table_ids,
    print_database_summary,
)
from sqlalchemy import func, select

# Load environment variables from .env
load_dotenv()


async def main() -> int:
    """Run Phase 11: Business Cycle Detection."""
    print("=" * 70)
    print("Phase 11: Business Cycle Detection (Expert LLM Agent)")
    print("=" * 70)

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not found!")
        print("Please set ANTHROPIC_API_KEY in your environment or .env file.")
        return 1

    print(f"\nUsing Anthropic API key: {api_key[:8]}...{api_key[-4:]}")

    # Get connections
    session_factory = await get_test_session()
    duckdb_conn = get_duckdb_conn()

    async with session_factory() as session:
        from dataraum_context.analysis.semantic.db_models import SemanticAnnotation

        # Check prerequisites
        print("\n1. Checking prerequisites...")
        print("-" * 50)

        # Check for semantic annotations
        annotation_count = (
            await session.execute(select(func.count(SemanticAnnotation.annotation_id)))
        ).scalar() or 0

        if annotation_count == 0:
            print("   ERROR: No semantic annotations found!")
            print("   Please run run_phase5_semantic.py first.")
            await cleanup_connections()
            return 1

        print(f"   Found {annotation_count} semantic annotations")

        # Get typed table IDs
        table_ids = await get_typed_table_ids(session)
        print(f"   Found {len(table_ids)} typed tables")

        # Setup LLM provider
        print("\n2. Setting up LLM provider...")
        print("-" * 50)

        from dataraum_context.llm.config import load_llm_config
        from dataraum_context.llm.providers import create_provider

        try:
            llm_config = load_llm_config()
            provider_config = llm_config.providers["anthropic"]
            provider = create_provider(
                "anthropic",
                {
                    "api_key_env": provider_config.api_key_env,
                    "default_model": provider_config.default_model,
                    "models": provider_config.models,
                    "base_url_env": provider_config.base_url_env,
                },
            )
            print(f"   Using model: {provider_config.default_model}")
        except Exception as e:
            print(f"   ERROR creating LLM provider: {e}")
            await cleanup_connections()
            return 1

        # Create and run the business cycle agent
        print("\n3. Running business cycle detection...")
        print("-" * 50)

        from dataraum_context.analysis.cycles import BusinessCycleAgent

        agent = BusinessCycleAgent(
            provider=provider,
            model=provider_config.default_model,
        )

        print(f"   Analyzing {len(table_ids)} tables...")
        print("   This may take a minute as the agent explores the data...")

        result = await agent.analyze(
            session=session,
            duckdb_conn=duckdb_conn,
            table_ids=table_ids,
            max_tool_calls=20,  # Increased to allow more exploration
        )

        if not result.success:
            print(f"   ERROR: {result.error}")
            await cleanup_connections()
            return 1

        analysis = result.unwrap()

        # Print results
        print("\n" + "=" * 70)
        print("BUSINESS CYCLE ANALYSIS RESULTS")
        print("=" * 70)

        print(f"\nAnalysis ID: {analysis.analysis_id}")
        print(f"Duration: {analysis.analysis_duration_seconds:.1f} seconds")
        print(f"Tool calls made: {len(analysis.tool_calls_made)}")

        # Tool call summary
        if analysis.tool_calls_made:
            print("\nTool calls:")
            for tc in analysis.tool_calls_made:
                print(f"  - {tc['tool']}: {tc['input']}")

        # Detected cycles
        print(f"\n{'=' * 50}")
        print(f"DETECTED CYCLES: {analysis.total_cycles_detected}")
        print(f"High-value cycles: {analysis.high_value_cycles}")
        print(f"Overall cycle health: {analysis.overall_cycle_health:.1%}")
        print(f"{'=' * 50}")

        for i, cycle in enumerate(analysis.cycles, 1):
            print(f"\n--- Cycle {i}: {cycle.cycle_name} ---")
            print(f"Type: {cycle.cycle_type}")
            print(f"Business value: {cycle.business_value}")
            print(f"Confidence: {cycle.confidence:.0%}")
            print(f"Description: {cycle.description[:200]}...")

            if cycle.status_column:
                print(f"Status column: {cycle.status_table}.{cycle.status_column}")
                print(f"Completion value: {cycle.completion_value}")

            if cycle.completion_rate is not None:
                print(f"Completion rate: {cycle.completion_rate:.1%}")
                print(f"Total records: {cycle.total_records:,}")
                print(f"Completed: {cycle.completed_cycles:,}")

            if cycle.entity_flows:
                print("Entity flows:")
                for ef in cycle.entity_flows:
                    print(f"  - {ef.entity_type}: {ef.entity_table}.{ef.entity_column}")

            if cycle.stages:
                print("Stages:")
                for stage in cycle.stages:
                    print(f"  {stage.stage_order}. {stage.stage_name}")

            if cycle.evidence:
                print("Evidence:")
                for ev in cycle.evidence[:3]:
                    print(f"  - {ev}")

        # Business summary
        print(f"\n{'=' * 50}")
        print("BUSINESS SUMMARY")
        print(f"{'=' * 50}")
        print(analysis.business_summary or "(no summary provided)")

        if analysis.detected_processes:
            print("\nDetected business processes:")
            for proc in analysis.detected_processes:
                print(f"  - {proc}")

        if analysis.data_quality_observations:
            print("\nData quality observations:")
            for obs in analysis.data_quality_observations:
                print(f"  - {obs}")

        if analysis.recommendations:
            print("\nRecommendations:")
            for rec in analysis.recommendations:
                print(f"  - {rec}")

        # Save results to JSON
        output_file = "data/business_cycle_analysis.json"
        with open(output_file, "w") as f:
            json.dump(analysis.model_dump(mode="json"), f, indent=2, default=str)
        print(f"\nFull results saved to: {output_file}")

        # Database summary
        await print_database_summary(session, duckdb_conn)

    await cleanup_connections()

    print("\n" + "=" * 70)
    print("Phase 11 COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
