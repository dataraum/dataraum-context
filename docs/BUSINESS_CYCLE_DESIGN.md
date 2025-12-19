# Business Cycle Detection - Design Document

## Problem Statement

Current approach detects **table-level topology** but business cycles are **transaction flows**
that happen at the **column and row level**. We need to redesign to detect actual business
processes in the data.

## What Are Business Cycles?

Business cycles are transaction chains that represent complete business processes:

| Cycle Type | Transaction Flow | Key Columns |
|------------|------------------|-------------|
| Order-to-Cash | Order → Invoice → Payment | order_ref, invoice_ref, payment_ref |
| Procure-to-Pay | PO → Receipt → Invoice → Payment | po_ref, receipt_ref, vendor_invoice_ref |
| AR Cycle | Invoice → Due → Payment → Reconciliation | invoice_id, due_date, payment_date |
| AP Cycle | Vendor Invoice → Approval → Payment | vendor_invoice_id, approval_date, payment_date |

## Detection Strategy

### Level 1: Column Relationship Graph (Dataset-wide)

Build a graph where:
- **Nodes** = Columns across all tables
- **Edges** = Relationships (FK, semantic match, reference link, self-join)

```
customer_table.customer_id ←──FK──→ master_txn_table.customer_id
vendor_table.vendor_id ←──FK──→ master_txn_table.vendor_id
master_txn_table.invoice_ref ←──self-ref──→ master_txn_table.txn_id (where type='Invoice')
```

### Level 2: Transaction Type Detection

Identify the column(s) that classify transaction types:
- Look for: `type`, `txn_type`, `transaction_type`, `doc_type`
- Or infer from patterns in string columns

Extract distinct transaction types:
```
['Sale', 'Invoice', 'Payment', 'Purchase', 'Bill', 'Credit Memo', ...]
```

### Level 3: Reference Chain Analysis

Find columns that link transactions together:
- `invoice_number` / `invoice_ref` / `parent_id`
- Date sequences (transaction_date < due_date < payment_date)
- Amount relationships (invoice.amount = payment.amount)

### Level 4: Cycle Pattern Matching

Match detected chains to known business cycle patterns:

```python
KNOWN_CYCLES = {
    "order_to_cash": {
        "stages": ["Sale", "Invoice", "Payment"],
        "indicators": {
            "customer_reference": True,
            "amount_flow": "debit_then_credit",
            "date_sequence": True
        }
    },
    "procure_to_pay": {
        "stages": ["Purchase", "Bill", "Payment"],
        "indicators": {
            "vendor_reference": True,
            "amount_flow": "credit_then_debit",
            "date_sequence": True
        }
    }
}
```

## Output Model

### For Downstream Business Analysis Agents

```python
@dataclass
class DetectedBusinessCycle:
    """A complete business cycle detected in the dataset."""

    cycle_type: str  # "order_to_cash", "procure_to_pay", etc.
    confidence: float  # How confident are we this cycle exists

    # Structure
    transaction_stages: list[str]  # ["Sale", "Invoice", "Payment"]
    stage_columns: dict[str, list[str]]  # {"Sale": ["customer_id", "amount"], ...}
    linking_columns: list[LinkingColumn]  # How stages connect

    # Completeness
    expected_stages: list[str]
    present_stages: list[str]
    missing_stages: list[str]
    completeness_score: float  # 0.0 - 1.0

    # Volume metrics
    total_transactions_in_cycle: int
    complete_chains: int  # Transactions that complete full cycle
    incomplete_chains: int  # Transactions that stall mid-cycle

    # Health metrics (if dates available)
    avg_cycle_days: float | None
    median_cycle_days: float | None
    p90_cycle_days: float | None

    # Financial metrics (if amounts available)
    total_cycle_volume: float | None
    avg_transaction_amount: float | None


@dataclass
class LinkingColumn:
    """How two stages in a cycle are linked."""

    from_stage: str
    to_stage: str
    link_type: str  # "reference", "temporal", "amount_match", "fk"
    from_column: str
    to_column: str
    match_rate: float  # What % of from_stage records link to to_stage


@dataclass
class TransactionChainSample:
    """Example of a complete transaction chain."""

    cycle_type: str
    chain_id: str
    transactions: list[dict]  # Actual row data for each stage
    total_amount: float
    start_date: date
    end_date: date | None
    duration_days: int | None
    is_complete: bool


@dataclass
class BusinessCycleAnalysis:
    """Complete business cycle analysis for a dataset."""

    # Scope
    dataset_id: str
    tables_analyzed: list[str]
    total_rows_analyzed: int
    analysis_timestamp: datetime

    # Transaction classification
    transaction_type_column: str | None
    transaction_types_found: list[str]
    transaction_type_counts: dict[str, int]

    # Detected cycles
    detected_cycles: list[DetectedBusinessCycle]

    # Sample chains (for LLM context)
    sample_chains: list[TransactionChainSample]

    # Column relationship graph
    column_graph: ColumnRelationshipGraph

    # Overall assessment
    primary_business_domain: str  # "retail", "b2b", "services", etc.
    data_completeness: float
    cycle_health_score: float

    # LLM-generated insights
    business_summary: str
    detected_processes: list[str]
    data_quality_issues: list[str]
    recommendations: list[str]


@dataclass
class ColumnRelationshipGraph:
    """Graph of column relationships across the dataset."""

    nodes: list[ColumnNode]
    edges: list[ColumnEdge]

    # Computed properties
    hub_columns: list[str]  # Highly connected columns (likely keys)
    leaf_columns: list[str]  # Low connectivity (likely attributes)
    reference_chains: list[list[str]]  # Chains of reference columns


@dataclass
class ColumnNode:
    table_name: str
    column_name: str
    column_type: str  # "key", "reference", "amount", "date", "category", "attribute"
    semantic_role: str | None  # "customer_id", "transaction_date", etc.


@dataclass
class ColumnEdge:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str  # "fk", "semantic_match", "reference", "temporal_sequence"
    confidence: float
    sample_matches: int
```

## Implementation Phases

### Phase 1: Column Relationship Graph

1. Start with existing FK relationships from Phase 5
2. Add semantic matches (columns with same/similar names across tables)
3. Detect self-referential patterns (reference columns within same table)
4. Detect temporal sequences (date columns that form logical order)

### Phase 2: Transaction Type Detection

1. Find transaction type column (heuristic + LLM fallback)
2. Extract and classify transaction types
3. Map types to standard business categories

### Phase 3: Chain Detection

1. For each transaction type, find linking columns
2. Trace actual chains in the data (SQL queries)
3. Compute chain statistics (completion rate, timing, amounts)

### Phase 4: Pattern Matching

1. Match detected chains to known cycle patterns
2. Score completeness and health
3. Identify missing elements

### Phase 5: LLM Synthesis

1. Provide structured cycle data to LLM
2. Generate business interpretation
3. Identify data quality issues and recommendations

## Example Output (for booksql data)

```json
{
  "detected_cycles": [
    {
      "cycle_type": "order_to_cash",
      "confidence": 0.92,
      "transaction_stages": ["Sale", "Invoice", "Payment"],
      "completeness_score": 1.0,
      "complete_chains": 45000,
      "incomplete_chains": 5000,
      "avg_cycle_days": 32.5,
      "total_cycle_volume": 12500000.00
    },
    {
      "cycle_type": "procure_to_pay",
      "confidence": 0.88,
      "transaction_stages": ["Purchase", "Bill", "Payment"],
      "completeness_score": 1.0,
      "complete_chains": 38000,
      "incomplete_chains": 2000,
      "avg_cycle_days": 28.0,
      "total_cycle_volume": 9800000.00
    }
  ],
  "business_summary": "This is a complete double-entry bookkeeping system for a
    retail/wholesale business. The data shows healthy AR and AP cycles with
    typical DSO of 32 days and DPO of 28 days...",
  "data_quality_issues": [
    "5000 invoices have no matching payment (10% AR aging)",
    "Reference column 'parent_id' has 2% null values"
  ]
}
```

## Key Insight

The **relationship graph between tables** tells us the data model structure.
The **business cycles** tell us how the business actually operates.

We need both, but they're different analyses:
- Table relationships → Data Model Understanding
- Transaction chains → Business Process Understanding

Current implementation conflates these. We should separate them.

---

## Concrete Implementation for BookSQL Data

### Data Structure Analysis

The `master_txn_table` is a **transaction register** with:
- `Transaction type`: 21 distinct types (Invoice, Payment, Bill, etc.)
- `Transaction ID`: Groups line items within a journal entry
- `A/R paid` / `A/P paid`: Cycle completion status
- `Sale` / `Purchase`: Boolean flags for transaction classification
- `Customer name` / `Vendor name`: Entity references
- `Debit` / `Credit`: Double-entry amounts

### Detected Business Cycles in This Dataset

```
1. ORDER-TO-CASH CYCLE (Revenue)
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │  Estimate   │ ──► │   Invoice   │ ──► │   Payment   │
   └─────────────┘     └─────────────┘     └─────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
   Customer name        A/R created          A/R paid=Paid

2. PROCURE-TO-PAY CYCLE (Expenses)
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │Purchase Order│ ──► │    Bill     │ ──► │Bill Payment │
   └─────────────┘     └─────────────┘     └─────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
   Vendor name          A/P created          A/P paid=Paid

3. DIRECT SALES (No Invoice)
   ┌─────────────┐     ┌─────────────┐
   │Sales Receipt│ ──► │   Deposit   │
   └─────────────┘     └─────────────┘
        │
        ▼
   Customer name, Payment method

4. EXPENSE PROCESSING
   ┌─────────────┐     ┌─────────────┐
   │   Expense   │ ──► │Check/CC Exp │
   └─────────────┘     └─────────────┘
```

### Detection Queries

```sql
-- Order-to-Cash cycle completeness
SELECT
    "Customer name",
    COUNT(CASE WHEN "Transaction type" = 'Estimate' THEN 1 END) as estimates,
    COUNT(CASE WHEN "Transaction type" = 'Invoice' THEN 1 END) as invoices,
    COUNT(CASE WHEN "Transaction type" = 'Payment' THEN 1 END) as payments,
    COUNT(CASE WHEN "A/R paid" = 'Paid' THEN 1 END) as ar_cleared
FROM typed_master_txn_table
WHERE Sale = TRUE
GROUP BY "Customer name"
HAVING COUNT(*) > 1;

-- Cycle time analysis (Invoice to Payment)
SELECT
    "Customer name",
    MIN("Transaction date") as first_invoice,
    MAX("Transaction date") as last_payment,
    DATEDIFF('day', MIN("Transaction date"), MAX("Transaction date")) as cycle_days
FROM typed_master_txn_table
WHERE "Transaction type" IN ('Invoice', 'Payment')
AND Sale = TRUE
GROUP BY "Customer name", "Transaction ID";
```

### Implementation Approach

1. **Identify Transaction Flow Patterns**
   - Group by `Transaction type`
   - Map to standard cycle stages

2. **Detect Cycle Linkages**
   - Same `Customer name` / `Vendor name` = potential cycle
   - `A/R paid` = 'Paid' = completed AR cycle
   - `A/P paid` = 'Paid' = completed AP cycle

3. **Compute Cycle Metrics**
   - Completion rate: % with Paid status
   - Cycle time: Date difference between stages
   - Volume: Sum of amounts per cycle type

4. **Build Structured Output**
   - DetectedBusinessCycle objects with all metrics
   - Sample transaction chains
   - LLM-ready context

### Key Columns for Cycle Detection

| Purpose | Column(s) |
|---------|-----------|
| Transaction classification | `Transaction type`, `Sale`, `Purchase` |
| Cycle completion status | `A/R paid`, `A/P paid` |
| Entity grouping | `Customer name`, `Vendor name` |
| Line item grouping | `Transaction ID` |
| Timing | `Transaction date`, `Due date` |
| Amounts | `Debit`, `Credit`, `Amount`, `Open balance` |
| Account classification | `Account` |
