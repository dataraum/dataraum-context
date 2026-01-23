"""Contract evaluation endpoints."""

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from dataraum.api.deps import SessionDep
from dataraum.api.schemas import (
    AllContractsEvaluationResponse,
    ContractEvaluationResponse,
    ContractListResponse,
    ContractSummary,
    ViolationResponse,
)
from dataraum.entropy.context import build_entropy_context
from dataraum.entropy.contracts import (
    ContractEvaluation,
    evaluate_all_contracts,
    evaluate_contract,
    find_best_contract,
    get_contract,
    list_contracts,
)
from dataraum.storage import Source, Table

router = APIRouter()


@router.get("/contracts", response_model=ContractListResponse)
def get_contracts_list() -> ContractListResponse:
    """List all available contracts.

    Returns contract names, descriptions, and overall thresholds.
    """
    contracts = list_contracts()
    return ContractListResponse(
        contracts=[
            ContractSummary(
                name=c["name"],
                display_name=c["display_name"],
                description=c["description"],
                overall_threshold=c["overall_threshold"],
            )
            for c in contracts
        ]
    )


@router.get("/contracts/{name}", response_model=ContractSummary)
def get_contract_detail(name: str) -> ContractSummary:
    """Get details for a specific contract.

    Args:
        name: Contract name (e.g., "executive_dashboard")
    """
    contract = get_contract(name)
    if contract is None:
        raise HTTPException(status_code=404, detail=f"Contract not found: {name}")

    return ContractSummary(
        name=contract.name,
        display_name=contract.display_name,
        description=contract.description,
        overall_threshold=contract.overall_threshold,
    )


@router.get("/contracts/{name}/evaluate", response_model=ContractEvaluationResponse)
def evaluate_contract_for_source(
    name: str,
    source_id: str,
    session: SessionDep,
) -> ContractEvaluationResponse:
    """Evaluate a specific contract against a source's data.

    Args:
        name: Contract name (e.g., "executive_dashboard")
        source_id: Source ID to evaluate
    """
    # Verify contract exists
    contract = get_contract(name)
    if contract is None:
        raise HTTPException(status_code=404, detail=f"Contract not found: {name}")

    # Verify source exists
    stmt = select(Source).where(Source.source_id == source_id)
    result = session.execute(stmt)
    source = result.scalar_one_or_none()

    if source is None:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")

    # Get tables
    tables_stmt = select(Table).where(Table.source_id == source_id)
    tables_result = session.execute(tables_stmt)
    tables = list(tables_result.scalars().all())

    if not tables:
        raise HTTPException(status_code=400, detail="Source has no tables")

    table_ids = [t.table_id for t in tables]

    # Build entropy context
    entropy_context = build_entropy_context(session=session, table_ids=table_ids)

    # Evaluate contract
    evaluation = evaluate_contract(entropy_context, name)

    # Convert to response
    return _evaluation_to_response(evaluation)


@router.get("/sources/{source_id}/contracts", response_model=AllContractsEvaluationResponse)
def evaluate_all_contracts_for_source(
    source_id: str,
    session: SessionDep,
) -> AllContractsEvaluationResponse:
    """Evaluate all contracts against a source's data.

    Returns evaluations for all contracts and identifies the strictest passing contract.
    """
    # Verify source exists
    stmt = select(Source).where(Source.source_id == source_id)
    result = session.execute(stmt)
    source = result.scalar_one_or_none()

    if source is None:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")

    # Get tables
    tables_stmt = select(Table).where(Table.source_id == source_id)
    tables_result = session.execute(tables_stmt)
    tables = list(tables_result.scalars().all())

    if not tables:
        raise HTTPException(status_code=400, detail="Source has no tables")

    table_ids = [t.table_id for t in tables]

    # Build entropy context
    entropy_context = build_entropy_context(session=session, table_ids=table_ids)

    # Evaluate all contracts
    evaluations = evaluate_all_contracts(entropy_context)

    # Find strictest passing
    best_name, best_evaluation = find_best_contract(entropy_context)
    strictest_passing = best_name if best_evaluation and best_evaluation.is_compliant else None

    # Count passing
    passing = [e for e in evaluations.values() if e.is_compliant]

    return AllContractsEvaluationResponse(
        source_id=source_id,
        evaluations={name: _evaluation_to_response(e) for name, e in evaluations.items()},
        strictest_passing=strictest_passing,
        passing_count=len(passing),
        total_count=len(evaluations),
    )


def _evaluation_to_response(evaluation: ContractEvaluation) -> ContractEvaluationResponse:
    """Convert ContractEvaluation to API response."""
    violations = [
        ViolationResponse(
            type=v.violation_type,
            severity=v.severity,
            dimension=v.dimension,
            max_allowed=v.max_allowed,
            actual=v.actual,
            details=v.details,
            affected_columns=v.affected_columns,
        )
        for v in evaluation.violations
    ]

    warnings = [
        ViolationResponse(
            type=w.violation_type,
            severity=w.severity,
            dimension=w.dimension,
            max_allowed=w.max_allowed,
            actual=w.actual,
            details=w.details,
            affected_columns=getattr(w, "affected_columns", []),
        )
        for w in evaluation.warnings
    ]

    return ContractEvaluationResponse(
        contract_name=evaluation.contract_name,
        contract_display_name=evaluation.contract_display_name,
        is_compliant=evaluation.is_compliant,
        confidence_level=evaluation.confidence_level.value,
        confidence_emoji=evaluation.confidence_level.emoji,
        confidence_label=evaluation.confidence_level.label,
        overall_score=round(evaluation.overall_score, 3),
        dimension_scores={k: round(v, 3) for k, v in evaluation.dimension_scores.items()},
        violations=violations,
        warnings=warnings,
        compliance_percentage=round(evaluation.compliance_percentage * 100, 1),
        worst_dimension=evaluation.worst_dimension,
        worst_dimension_score=round(evaluation.worst_dimension_score, 3),
        estimated_effort_to_comply=evaluation.estimated_effort_to_comply,
        evaluated_at=evaluation.evaluated_at.isoformat(),
    )
