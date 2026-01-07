# Entropy Query Behavior: Agent Response Policies

This document defines how the graph agent should behave when encountering different entropy levels. It specifies when to answer confidently, when to add caveats, when to ask clarifying questions, and when to refuse.

**Related Documentation:**
- [ENTROPY_IMPLEMENTATION_PLAN.md](./ENTROPY_IMPLEMENTATION_PLAN.md) - Implementation roadmap
- [ENTROPY_MODELS.md](./ENTROPY_MODELS.md) - Data model specifications
- [ENTROPY_CONTRACTS.md](./ENTROPY_CONTRACTS.md) - Data readiness thresholds

---

## Core Principle

**Entropy determines confidence, not capability.**

The agent can always *attempt* to answer, but entropy determines:
- How confident the answer should be
- What assumptions must be stated
- Whether to ask for clarification
- Whether to refuse entirely

---

## Entropy Level Classification

| Level | Score Range | Deterministic? | Meaning |
|-------|-------------|----------------|---------|
| **Low** | 0.0 - 0.3 | Yes | High confidence, minimal uncertainty |
| **Medium** | 0.3 - 0.6 | Partial | Some uncertainty, assumptions needed |
| **High** | 0.6 - 0.8 | No | Significant uncertainty, clarification recommended |
| **Critical** | 0.8 - 1.0 | No | Extreme uncertainty, consider refusing |

---

## Behavior Decision Matrix

### Standard Behavior (Default)

| Entropy Level | Agent Behavior | Response Style |
|---------------|----------------|----------------|
| Low (<0.3) | Answer confidently | Direct answer, no caveats |
| Medium (0.3-0.6) | Answer with assumptions | Include assumptions in response |
| High (0.6-0.8) | Ask OR caveat strongly | Prefer clarification; if answering, strong caveat |
| Critical (>0.8) | Refuse or ask | Explain what needs resolution first |

### By Dimension (When One Dimension is High)

| High Dimension | Behavior | Example |
|----------------|----------|---------|
| Structural.Types | Warn about data type issues | "Note: Some dates may be incorrectly parsed" |
| Structural.Relations | Ask about intended join | "Which path should I use: purchased products or recommended products?" |
| Semantic.Units | Ask about currency/unit | "Should I assume EUR, or is there a mix of currencies?" |
| Semantic.Temporal | Ask about time period | "For what time period? YTD, last 12 months, or all time?" |
| Value.Nulls | Explain null handling | "Null values excluded from average (15% of records)" |
| Value.Outliers | Report with/without | "Average: $847 (with outliers) or $312 (excluding enterprise orders)" |
| Computational.Aggregations | Ask about aggregation | "Should I sum, average, or show breakdown?" |

---

## Response Templates

### Low Entropy (Confident Answer)

```
Revenue by region for Q3 2024:

| Region | Revenue |
|--------|---------|
| EMEA   | €4.2M   |
| APAC   | €2.8M   |
| Americas | €5.1M |

Total: €12.1M
```

### Medium Entropy (Answer with Assumptions)

```
Revenue by region for Q3 2024:

| Region | Revenue |
|--------|---------|
| EMEA   | €4.2M   |
| APAC   | €2.8M   |
| Americas | €5.1M |

Total: €12.1M

**Assumptions made:**
- Revenue is in EUR (based on system default; some entities may report in local currency)
- Q3 defined as Jul-Sep (calendar quarter, not fiscal)
- Excluding cancelled orders
```

### High Entropy (Ask Clarification)

```
I can calculate revenue by region, but I need clarification:

1. **Currency**: Revenue data includes multiple currencies (EUR, USD, GBP). Should I:
   - Convert all to EUR using current rates?
   - Show each currency separately?
   - Use a specific conversion date?

2. **Time period**: "Q3" could mean:
   - Calendar Q3 (Jul-Sep)
   - Fiscal Q3 (varies by entity)

Please specify, or I can proceed with defaults (EUR, calendar quarter).
```

### Critical Entropy (Refuse with Explanation)

```
I cannot reliably calculate revenue by region because:

**Blocking issues:**
1. **Missing currency information** (entropy: 0.92)
   - 40% of revenue records have no currency indicator
   - Cannot determine if values are EUR, USD, or local currency

2. **Ambiguous join path** (entropy: 0.85)
   - Multiple paths exist between orders and regions
   - Results differ by 23% depending on path chosen

**To proceed, please resolve:**
- [ ] Add currency field to revenue table, OR specify default currency
- [ ] Designate canonical path from orders to regions

Would you like me to show what needs to be fixed?
```

---

## Assumption Handling

### Assumption Types

| Type | Example | Disclosure Required? |
|------|---------|---------------------|
| **Explicit default** | "Using EUR as default currency" | Yes |
| **Inferred from context** | "Assuming calendar year based on date patterns" | Yes |
| **System convention** | "Excluding cancelled orders per standard practice" | If entropy > 0.3 |
| **User-confirmed** | "Using gross margin as you specified" | No (already confirmed) |

### Assumption Disclosure Format

```
**Assumptions made:**
- [Assumption 1] (confidence: X%)
- [Assumption 2] (confidence: Y%)

To verify or change these assumptions, [action].
```

### Assumption Tracking

Assumptions should be:
1. **Logged** with the query execution
2. **Returnable** via `get_assumptions(execution_id)`
3. **Promotable** to permanent rules if confirmed

```python
@dataclass
class QueryAssumption:
    """An assumption made during query execution."""

    assumption_id: str
    execution_id: str

    # What was assumed
    dimension: str  # e.g., "semantic.units"
    target: str  # e.g., "column:orders.amount"
    assumption: str  # e.g., "Currency is EUR"

    # Basis for assumption
    basis: str  # "system_default", "inferred", "user_specified"
    confidence: float

    # For promotion to permanent rule
    can_promote: bool = True
    promoted_at: datetime | None = None
    promoted_by: str | None = None
```

---

## Compound Risk Behavior

When compound risks exist, behavior should be more cautious:

### Critical Compound Risk

**Example:** High Semantic.Units + High Computational.Aggregations

```
⚠️ **Critical data quality concern**

I cannot safely calculate this metric because:

- **Unknown currencies** are being **summed** without conversion
- This combination could produce meaningless results

The revenue figure could be off by 20-40% depending on currency mix.

**Options:**
1. I can show the breakdown by currency (no summing)
2. I can proceed with explicit currency assumption (you specify)
3. Please resolve the currency entropy first

Which would you prefer?
```

### High Compound Risk

**Example:** High Value.Nulls + High Computational.Aggregations

```
**Warning:** This calculation has data quality concerns.

Average order value: $312

**Data quality note:**
- 30% of orders have null values
- I excluded nulls from the average
- If nulls represent zero-value orders, the true average would be $218

Would you like me to show both figures, or specify how to handle nulls?
```

---

## Configurable Behavior Modes

Different users may want different behavior. Support these modes:

### Mode: Strict

- Ask clarification for ANY entropy > 0.3
- Never make assumptions automatically
- Always show entropy scores

```yaml
behavior_mode: strict
clarification_threshold: 0.3
auto_assume: false
show_entropy_scores: true
```

### Mode: Balanced (Default)

- Ask for high entropy (> 0.6)
- Make reasonable assumptions for medium entropy
- Show assumptions when made

```yaml
behavior_mode: balanced
clarification_threshold: 0.6
auto_assume: true
show_entropy_scores: false
assumption_disclosure: true
```

### Mode: Lenient

- Only refuse for critical entropy (> 0.8)
- Make assumptions freely
- Minimal disclosure

```yaml
behavior_mode: lenient
clarification_threshold: 0.8
auto_assume: true
show_entropy_scores: false
assumption_disclosure: minimal
```

---

## SQL Generation with Entropy

When generating SQL, include entropy context in comments:

### Low Entropy SQL

```sql
-- Generated with high confidence (entropy: 0.15)
SELECT
    region,
    SUM(amount) as revenue
FROM orders
JOIN customers ON orders.customer_id = customers.id
WHERE order_date BETWEEN '2024-07-01' AND '2024-09-30'
  AND status != 'cancelled'
GROUP BY region
ORDER BY revenue DESC
```

### Medium Entropy SQL

```sql
-- Generated with assumptions (entropy: 0.45)
-- ASSUMPTION: Currency is EUR (no explicit currency field)
-- ASSUMPTION: Using calendar Q3 (Jul-Sep)
-- ASSUMPTION: Excluding cancelled orders
SELECT
    region,
    SUM(amount) as revenue_eur  -- Assumed EUR
FROM orders
JOIN customers ON orders.customer_id = customers.id
WHERE order_date BETWEEN '2024-07-01' AND '2024-09-30'
  AND status != 'cancelled'
GROUP BY region
ORDER BY revenue_eur DESC
```

### High Entropy SQL (with warning)

```sql
-- ⚠️ HIGH ENTROPY WARNING (entropy: 0.72)
-- Multiple join paths exist - using orders->customers (purchased)
-- Alternative path orders->recommendations->customers would give different results
--
-- ASSUMPTION: Using direct customer relationship
-- ASSUMPTION: EUR currency
-- VERIFY: Results before using in reports

SELECT
    c.region,
    SUM(o.amount) as revenue
FROM orders o
JOIN customers c ON o.customer_id = c.id  -- Direct path (not via recommendations)
WHERE o.order_date BETWEEN '2024-07-01' AND '2024-09-30'
GROUP BY c.region
```

---

## Implementation in Graph Agent

### Context Injection

```python
def format_entropy_warnings_for_prompt(
    entropy_context: EntropyContext,
    columns_used: list[str],
) -> str:
    """Generate entropy warnings section for LLM prompt."""

    warnings = []

    for col in columns_used:
        profile = entropy_context.get_column_entropy(*col.split("."))
        if profile and profile.composite_score > 0.3:
            warnings.append(
                f"- {col}: entropy {profile.composite_score:.2f} "
                f"({profile.high_entropy_dimensions})"
            )

    if entropy_context.compound_risks:
        warnings.append("\n**Compound risks:**")
        for risk in entropy_context.compound_risks:
            warnings.append(f"- {risk.risk_level}: {risk.impact}")

    if not warnings:
        return ""

    return """
## ENTROPY WARNINGS

The following columns have elevated uncertainty:

{}

When generating SQL:
- Include comments noting assumptions
- Consider asking for clarification if entropy > 0.6
- Refuse if critical compound risks exist
""".format("\n".join(warnings))
```

### Response Generation

```python
def generate_response_with_entropy(
    result: GraphExecution,
    entropy_context: EntropyContext,
    behavior_mode: str = "balanced",
) -> str:
    """Generate response with appropriate entropy handling."""

    max_entropy = get_max_entropy_for_execution(result, entropy_context)
    assumptions = get_assumptions_for_execution(result)

    # Determine response style
    if max_entropy < 0.3:
        # Confident response
        return format_confident_response(result)

    elif max_entropy < 0.6:
        # Response with assumptions
        return format_response_with_assumptions(result, assumptions)

    elif max_entropy < 0.8:
        if behavior_mode == "strict":
            return format_clarification_request(result, entropy_context)
        else:
            return format_response_with_strong_caveat(result, assumptions, entropy_context)

    else:
        # Critical - refuse or explain
        return format_refusal_with_explanation(result, entropy_context)
```

---

## User Feedback Loop

### Feedback Signals

| Signal | Meaning | Action |
|--------|---------|--------|
| User accepts answer | Assumption was correct | Increase assumption confidence |
| User provides correction | Assumption was wrong | Log for review, decrease confidence |
| User confirms assumption | Make permanent | Promote to rule |
| User asks follow-up | May need clarification | Track for pattern analysis |

### Feedback Collection

```python
@dataclass
class QueryFeedback:
    """User feedback on a query response."""

    execution_id: str
    feedback_type: str  # "accept", "reject", "correct", "clarify"

    # For corrections
    correction: str | None = None
    corrected_assumption: str | None = None

    # For promotions
    promote_assumption: str | None = None

    # Metadata
    user_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

### Learning from Feedback

Over time, the system should:
1. **Track assumption success rate** per dimension
2. **Identify common clarification patterns** → suggest defaults
3. **Detect recurring corrections** → flag for data fix
4. **Promote validated assumptions** → reduce future entropy

---

## Summary: Decision Flow

```
┌─────────────────────────────────────┐
│ User Query                          │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ Calculate entropy for relevant      │
│ columns and relationships           │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ Check for compound risks            │
└─────────────────┬───────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│ Critical risk │   │ No critical   │
│ exists?       │   │ risk          │
└───────┬───────┘   └───────┬───────┘
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────────────────┐
│ REFUSE        │   │ Check max entropy level   │
│ Explain why   │   └───────────────┬───────────┘
│ Show fixes    │                   │
└───────────────┘     ┌─────────────┼─────────────┐
                      │             │             │
                      ▼             ▼             ▼
                 ┌─────────┐  ┌──────────┐  ┌──────────┐
                 │ < 0.3   │  │ 0.3-0.6  │  │ > 0.6    │
                 │ LOW     │  │ MEDIUM   │  │ HIGH     │
                 └────┬────┘  └────┬─────┘  └────┬─────┘
                      │            │             │
                      ▼            ▼             ▼
                 ┌─────────┐  ┌──────────┐  ┌──────────┐
                 │ ANSWER  │  │ ANSWER   │  │ ASK or   │
                 │ directly│  │ with     │  │ CAVEAT   │
                 │         │  │ assump-  │  │ strongly │
                 └─────────┘  │ tions    │  └──────────┘
                              └──────────┘
```

---

## Configuration File

```yaml
# config/entropy/query_behavior.yaml

default_mode: balanced

modes:
  strict:
    clarification_threshold: 0.3
    refusal_threshold: 0.6
    auto_assume: false
    show_entropy_scores: true
    assumption_disclosure: always

  balanced:
    clarification_threshold: 0.6
    refusal_threshold: 0.8
    auto_assume: true
    show_entropy_scores: false
    assumption_disclosure: when_made

  lenient:
    clarification_threshold: 0.8
    refusal_threshold: 0.95
    auto_assume: true
    show_entropy_scores: false
    assumption_disclosure: minimal

# Dimension-specific overrides
dimension_behavior:
  semantic.units:
    # Always ask about currency, even at lower thresholds
    clarification_threshold: 0.4

  structural.relations:
    # Always ask about join paths
    clarification_threshold: 0.5

# Compound risk responses
compound_risk_behavior:
  critical:
    action: refuse
    explain: true

  high:
    action: warn_strongly
    require_confirmation: true

  medium:
    action: note_in_response
```
