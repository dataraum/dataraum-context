# Entropy Contracts: Data Readiness Thresholds

This document defines data readiness contracts for different use cases. Each contract specifies acceptable entropy thresholds that data must meet before being used for that purpose.

**Related Documentation:**
- [ENTROPY_IMPLEMENTATION_PLAN.md](./ENTROPY_IMPLEMENTATION_PLAN.md) - Implementation roadmap
- [ENTROPY_MODELS.md](./ENTROPY_MODELS.md) - Data model specifications
- [ENTROPY_QUERY_BEHAVIOR.md](./ENTROPY_QUERY_BEHAVIOR.md) - Agent response policies

---

## Concept

Different use cases have different tolerance for uncertainty:

- **Regulatory reporting** requires near-zero entropy — every number must be defensible
- **Executive dashboards** need low entropy but can tolerate some assumptions
- **Ad-hoc exploration** can proceed with higher entropy if assumptions are disclosed
- **Data science/ML** needs value quality but can tolerate semantic ambiguity

A **data readiness contract** defines the maximum acceptable entropy score for each dimension, given a specific use case.

---

## Standard Contract Profiles

### 1. Regulatory Reporting

**Use case:** Financial statements, compliance reports, audit submissions

**Characteristics:**
- Every number must be traceable and defensible
- No assumptions allowed — all ambiguity must be resolved
- Requires sign-off before use

| Dimension | Max Score | Rationale |
|-----------|-----------|-----------|
| Structural.Types | 0.1 | Type mismatches cause calculation errors |
| Structural.Relations | 0.1 | Wrong joins = wrong numbers |
| Semantic.BusinessMeaning | 0.1 | Terms must have precise definitions |
| Semantic.Units | 0.1 | Currency/unit errors are material |
| Semantic.Temporal | 0.1 | Period alignment critical |
| Value.Nulls | 0.1 | Null handling must be defined |
| Value.Outliers | 0.2 | Outliers must be documented |
| Computational.DerivedValues | 0.1 | Formulas must be auditable |
| Computational.Aggregations | 0.1 | Rollup rules must be explicit |

**Overall threshold:** 0.1 (average across dimensions)

**Blocking conditions:**
- ANY dimension > 0.2
- ANY critical compound risk
- Missing business definition on financial columns

---

### 2. Executive Dashboard

**Use case:** C-level dashboards, board presentations, KPI tracking

**Characteristics:**
- Numbers should be reliable but not audit-grade
- Minor assumptions acceptable if disclosed
- Focus on trends rather than exact values

| Dimension | Max Score | Rationale |
|-----------|-----------|-----------|
| Structural.Types | 0.2 | Type issues should be minimal |
| Structural.Relations | 0.2 | Joins should be clear |
| Semantic.BusinessMeaning | 0.2 | Key metrics need definitions |
| Semantic.Units | 0.2 | Currency must be clear |
| Semantic.Temporal | 0.3 | Some temporal ambiguity OK |
| Value.Nulls | 0.3 | Null handling should be consistent |
| Value.Outliers | 0.3 | Outliers acceptable if noted |
| Computational.DerivedValues | 0.2 | Key calculations documented |
| Computational.Aggregations | 0.2 | Aggregation rules clear |

**Overall threshold:** 0.25 (average across dimensions)

**Blocking conditions:**
- ANY dimension > 0.5
- Critical compound risk on key metrics
- Missing unit on financial columns

---

### 3. Operational Analytics

**Use case:** Operational reports, team dashboards, process monitoring

**Characteristics:**
- Focused on actionable insights
- Can tolerate some ambiguity in non-critical dimensions
- Speed of insight more important than precision

| Dimension | Max Score | Rationale |
|-----------|-----------|-----------|
| Structural.Types | 0.2 | Type issues cause operational problems |
| Structural.Relations | 0.3 | Some join ambiguity OK |
| Semantic.BusinessMeaning | 0.3 | Operational terms need context |
| Semantic.Units | 0.3 | Units should be mostly clear |
| Semantic.Temporal | 0.4 | Temporal flexibility needed |
| Value.Nulls | 0.4 | Nulls common in operational data |
| Value.Outliers | 0.4 | Operational data has variance |
| Computational.DerivedValues | 0.3 | Derived values should be traceable |
| Computational.Aggregations | 0.3 | Aggregation approach clear |

**Overall threshold:** 0.35 (average across dimensions)

**Blocking conditions:**
- ANY dimension > 0.6
- Critical compound risk
- Type fidelity < 0.7 on key columns

---

### 4. Ad-Hoc Exploration

**Use case:** Data exploration, hypothesis testing, initial analysis

**Characteristics:**
- High tolerance for uncertainty
- Assumptions acceptable if clearly stated
- Goal is insight, not precision

| Dimension | Max Score | Rationale |
|-----------|-----------|-----------|
| Structural.Types | 0.3 | Type issues noted, not blocking |
| Structural.Relations | 0.5 | Multiple join paths OK |
| Semantic.BusinessMeaning | 0.5 | Exploration tolerates ambiguity |
| Semantic.Units | 0.4 | Unit assumptions OK if stated |
| Semantic.Temporal | 0.5 | Temporal exploration flexible |
| Value.Nulls | 0.5 | Nulls are data points |
| Value.Outliers | 0.5 | Outliers are interesting |
| Computational.DerivedValues | 0.5 | Derived values explored |
| Computational.Aggregations | 0.5 | Try different aggregations |

**Overall threshold:** 0.5 (average across dimensions)

**Blocking conditions:**
- ANY dimension > 0.8
- Data physically unreadable
- No relationship path exists

---

### 5. Data Science / ML

**Use case:** Feature engineering, model training, statistical analysis

**Characteristics:**
- Value quality critical (nulls, outliers matter for models)
- Semantic ambiguity less important (models don't need business definitions)
- Type consistency essential for feature pipelines

| Dimension | Max Score | Rationale |
|-----------|-----------|-----------|
| Structural.Types | 0.2 | Types must be consistent for pipelines |
| Structural.Relations | 0.3 | Join quality affects training data |
| Semantic.BusinessMeaning | 0.5 | ML doesn't need business terms |
| Semantic.Units | 0.3 | Units affect feature scaling |
| Semantic.Temporal | 0.4 | Temporal patterns are features |
| Value.Nulls | 0.3 | Nulls affect model quality |
| Value.Outliers | 0.3 | Outliers affect training |
| Computational.DerivedValues | 0.4 | Derived features are common |
| Computational.Aggregations | 0.4 | Aggregation is feature engineering |

**Overall threshold:** 0.35 (average across dimensions)

**Blocking conditions:**
- Type fidelity > 0.4 (pipelines will fail)
- Null ratio > 0.5 without imputation strategy
- Outlier ratio > 0.2 without handling strategy

---

## Contract Evaluation

### Algorithm

```python
def evaluate_contract(
    entropy_context: EntropyContext,
    contract: ContractProfile,
) -> ContractEvaluation:
    """Evaluate entropy against a contract.

    Returns:
        ContractEvaluation with compliance status, violations, and recommendations
    """
    violations = []
    warnings = []

    # Check each dimension threshold
    for dimension, max_score in contract.dimension_thresholds.items():
        actual_score = get_dimension_score(entropy_context, dimension)
        if actual_score > max_score:
            violations.append(DimensionViolation(
                dimension=dimension,
                max_allowed=max_score,
                actual=actual_score,
                severity="blocking" if actual_score > contract.blocking_threshold else "warning"
            ))

    # Check overall average
    overall = calculate_overall_entropy(entropy_context)
    if overall > contract.overall_threshold:
        violations.append(OverallViolation(
            max_allowed=contract.overall_threshold,
            actual=overall
        ))

    # Check blocking conditions
    for condition in contract.blocking_conditions:
        if condition.evaluate(entropy_context):
            violations.append(BlockingConditionViolation(
                condition=condition.description,
                details=condition.get_details(entropy_context)
            ))

    # Determine compliance
    is_compliant = len([v for v in violations if v.severity == "blocking"]) == 0

    return ContractEvaluation(
        contract_name=contract.name,
        is_compliant=is_compliant,
        violations=violations,
        warnings=warnings,
        recommendations=generate_recommendations(violations),
        evaluated_at=datetime.utcnow()
    )
```

### Result Structure

```python
@dataclass
class ContractEvaluation:
    """Result of evaluating entropy against a contract."""

    contract_name: str
    is_compliant: bool

    # Violations (blocking and non-blocking)
    violations: list[Violation]

    # Warnings (approaching threshold)
    warnings: list[Warning]

    # Prioritized recommendations to achieve compliance
    recommendations: list[ResolutionCascade]

    # Metadata
    evaluated_at: datetime

    # Summary for UI
    compliance_percentage: float  # % of dimensions within threshold
    worst_dimension: str  # Dimension furthest from compliance
    estimated_effort_to_comply: str  # low, medium, high

    def get_blocking_violations(self) -> list[Violation]:
        """Get only violations that block compliance."""
        return [v for v in self.violations if v.severity == "blocking"]

    def get_path_to_compliance(self) -> list[ResolutionCascade]:
        """Get ordered list of resolutions to achieve compliance."""
        return sorted(
            self.recommendations,
            key=lambda r: r.priority_score,
            reverse=True
        )
```

---

## Configuration File Format

Contracts are loaded from `config/entropy/contracts.yaml`:

```yaml
# config/entropy/contracts.yaml

contracts:
  regulatory_reporting:
    name: "Regulatory Reporting"
    description: "For financial statements, compliance reports, audit submissions"
    overall_threshold: 0.1

    dimension_thresholds:
      structural.types: 0.1
      structural.relations: 0.1
      semantic.business_meaning: 0.1
      semantic.units: 0.1
      semantic.temporal: 0.1
      value.nulls: 0.1
      value.outliers: 0.2
      computational.derived_values: 0.1
      computational.aggregations: 0.1

    blocking_conditions:
      - type: any_dimension_exceeds
        threshold: 0.2
      - type: has_critical_compound_risk
      - type: missing_definition
        applies_to:
          semantic_role: measure
          has_tag: financial

  executive_dashboard:
    name: "Executive Dashboard"
    description: "For C-level dashboards, board presentations, KPI tracking"
    overall_threshold: 0.25

    dimension_thresholds:
      structural.types: 0.2
      structural.relations: 0.2
      semantic.business_meaning: 0.2
      semantic.units: 0.2
      semantic.temporal: 0.3
      value.nulls: 0.3
      value.outliers: 0.3
      computational.derived_values: 0.2
      computational.aggregations: 0.2

    blocking_conditions:
      - type: any_dimension_exceeds
        threshold: 0.5
      - type: has_critical_compound_risk
        applies_to:
          has_tag: key_metric
      - type: missing_unit
        applies_to:
          semantic_role: measure

  # ... additional contracts

# Custom contracts can be defined per dataset
custom_contracts:
  my_project_strict:
    extends: regulatory_reporting
    overrides:
      dimension_thresholds:
        value.outliers: 0.1  # Stricter outlier threshold
```

---

## API Integration

### Endpoints

```
GET /entropy/contracts
  → List all available contract profiles

GET /entropy/contracts/{contract_name}
  → Get contract definition

GET /entropy/contracts/{contract_name}/evaluate?table_ids=...
  → Evaluate tables against contract

GET /entropy/contracts/{contract_name}/path-to-compliance?table_ids=...
  → Get prioritized resolutions to achieve compliance
```

### Response Examples

**Contract Evaluation:**
```json
{
  "contract_name": "executive_dashboard",
  "is_compliant": false,
  "compliance_percentage": 0.72,
  "worst_dimension": "semantic.units",
  "violations": [
    {
      "type": "dimension_violation",
      "dimension": "semantic.units",
      "max_allowed": 0.2,
      "actual": 0.45,
      "severity": "blocking",
      "affected_columns": ["revenue", "costs", "margin"]
    }
  ],
  "recommendations": [
    {
      "action": "declare_unit",
      "parameters": {"column": "revenue", "unit": "EUR"},
      "expected_entropy_reduction": 0.4,
      "effort": "low",
      "priority_score": 0.4
    }
  ],
  "estimated_effort_to_comply": "medium"
}
```

---

## UI Integration

### Compliance Dashboard

The UI should show:

1. **Contract Selector** - Choose which contract to evaluate against
2. **Compliance Status** - Traffic light (green/yellow/red)
3. **Dimension Breakdown** - Bar chart showing each dimension vs threshold
4. **Violation List** - Sorted by severity
5. **Path to Compliance** - Ordered resolution checklist

### Visual Elements

| Status | Color | Icon | Meaning |
|--------|-------|------|---------|
| Compliant | Green | Checkmark | All thresholds met |
| Warning | Yellow | Warning | Some dimensions approaching threshold |
| Non-compliant | Red | X | Blocking violations exist |
| Unknown | Gray | ? | Not evaluated |

---

## Usage in Graph Agent

The graph agent should check contract compliance before executing graphs:

```python
async def execute_with_contract_check(
    graph: TransformationGraph,
    context: ExecutionContext,
    contract_name: str = "ad_hoc_exploration",
) -> Result[GraphExecution]:
    """Execute graph with contract compliance check."""

    # Evaluate contract
    evaluation = await evaluate_contract(
        entropy_context=context.rich_context.entropy,
        contract_name=contract_name,
    )

    if not evaluation.is_compliant:
        blocking = evaluation.get_blocking_violations()
        if blocking:
            # For strict contracts, refuse to execute
            if contract_name in ["regulatory_reporting"]:
                return Result.fail(
                    f"Cannot execute: {len(blocking)} blocking violations for {contract_name} contract",
                    details={"violations": blocking}
                )

            # For lenient contracts, warn but proceed
            context.warnings.append(
                f"Data does not meet {contract_name} contract: {len(blocking)} issues"
            )

    # Proceed with execution
    return await self._execute(graph, context)
```

---

## Best Practices

### Choosing a Contract

1. **Start with ad-hoc exploration** for initial data understanding
2. **Upgrade to operational analytics** when building regular reports
3. **Use executive dashboard** for leadership-facing content
4. **Require regulatory reporting** only for actual compliance needs

### Custom Contracts

Create custom contracts when:
- Your use case doesn't fit standard profiles
- You need stricter thresholds for specific dimensions
- You want to gradually tighten requirements over time

### Contract Evolution

As data quality improves:
1. Start with lenient contract (ad-hoc exploration)
2. Identify and fix high-entropy areas
3. Upgrade to stricter contract
4. Repeat until target contract is met
