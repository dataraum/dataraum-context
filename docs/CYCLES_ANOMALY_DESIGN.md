# Cycle Anomaly Detection - Design Analysis

## Overview

Post-processing step to detect anomalies in business cycle analysis results.
Runs **after** the LLM agent completes detection, using deterministic rules.

## Why Post-Processing (Not LLM)?

1. **Deterministic**: Comparing detected vs expected cycles is rule-based
2. **Consistent**: Same input always produces same anomalies
3. **Fast**: No additional LLM calls needed
4. **Configurable**: Rules can be tuned via config without prompt changes

## Anomaly Types

### 1. Missing Expected Cycles
**Trigger**: Domain config specifies expected cycles, but they weren't detected.

```yaml
# From config/cycles/cycle_vocabulary.yaml
domains:
  financial:
    expected_cycles:
      - order_to_cash
      - procure_to_pay
      - accounts_receivable
      - accounts_payable
```

**Detection Logic**:
```python
expected = get_domain_config(domain).get("expected_cycles", [])
detected = {c.canonical_type for c in analysis.cycles if c.canonical_type}
missing = set(expected) - detected
```

**Severity**: Medium (data might not contain these processes)

---

### 2. Low Completion Rates
**Trigger**: Cycle has completion_rate < threshold (e.g., 50%)

**Detection Logic**:
```python
for cycle in analysis.cycles:
    if cycle.completion_rate and cycle.completion_rate < 0.5:
        # Flag as anomaly
```

**Severity**: High for high-value cycles, Medium otherwise

---

### 3. Unclassified Cycles
**Trigger**: Detected cycles that don't match vocabulary (is_known_type=False)

**Detection Logic**:
```python
unclassified = [c for c in analysis.cycles if not c.is_known_type]
```

**Severity**: Low (might be valid domain-specific cycles)

---

### 4. Excessive Cycles
**Trigger**: Too many cycles detected (suggests fragmented/noisy data)

**Detection Logic**:
```python
if len(analysis.cycles) > threshold:  # e.g., 15
    # Flag as anomaly
```

**Severity**: Medium

---

### 5. Orphan Entities (Future)
**Trigger**: Entities that start cycles but never complete them

**Detection Logic**: Would require additional data queries (not just analysis results)

**Severity**: High

---

## Data Model

### CycleAnomaly (Pydantic)

```python
class CycleAnomaly(BaseModel):
    """A detected anomaly in cycle analysis."""

    anomaly_type: Literal[
        "missing_expected",
        "low_completion",
        "unclassified",
        "excessive_cycles",
    ]
    severity: Literal["low", "medium", "high", "critical"]
    title: str  # Short title for display
    description: str  # Detailed explanation
    affected_cycles: list[str] = []  # cycle_ids
    affected_tables: list[str] = []  # table names
    recommendation: str  # Actionable suggestion
    metadata: dict[str, Any] = {}  # Additional context
```

### Extended BusinessCycleAnalysis

```python
class BusinessCycleAnalysis(BaseModel):
    # ... existing fields ...

    # NEW: Anomaly detection results
    anomalies: list[CycleAnomaly] = Field(default_factory=list)
    anomaly_count: int = 0
    has_critical_anomalies: bool = False
```

---

## Function Design

### Option A: Standalone Function

```python
# In analysis/cycles/anomalies.py

def detect_cycle_anomalies(
    analysis: BusinessCycleAnalysis,
    domain: str | None = None,
    *,
    completion_threshold: float = 0.5,
    excessive_threshold: int = 15,
) -> list[CycleAnomaly]:
    """Detect anomalies in a business cycle analysis.

    Args:
        analysis: Completed cycle analysis
        domain: Domain for expected cycle checks
        completion_threshold: Min completion rate (default 50%)
        excessive_threshold: Max cycles before flagging (default 15)

    Returns:
        List of detected anomalies
    """
    anomalies = []

    # 1. Missing expected cycles
    if domain:
        anomalies.extend(_check_missing_expected(analysis, domain))

    # 2. Low completion rates
    anomalies.extend(_check_low_completion(analysis, completion_threshold))

    # 3. Unclassified cycles
    anomalies.extend(_check_unclassified(analysis))

    # 4. Excessive cycles
    anomalies.extend(_check_excessive(analysis, excessive_threshold))

    return anomalies
```

### Option B: Method on BusinessCycleAnalysis

```python
class BusinessCycleAnalysis(BaseModel):
    # ... existing fields ...

    def detect_anomalies(
        self,
        domain: str | None = None,
        **kwargs
    ) -> list[CycleAnomaly]:
        """Detect anomalies and populate self.anomalies."""
        from dataraum_context.analysis.cycles.anomalies import detect_cycle_anomalies
        self.anomalies = detect_cycle_anomalies(self, domain, **kwargs)
        self.anomaly_count = len(self.anomalies)
        self.has_critical_anomalies = any(a.severity == "critical" for a in self.anomalies)
        return self.anomalies
```

### Option C: Integrated into Agent

```python
# In agent.py _parse_response() or analyze()

# After creating analysis:
analysis = BusinessCycleAnalysis(...)

# Detect anomalies
anomalies = detect_cycle_anomalies(analysis, domain=domain)
analysis.anomalies = anomalies
analysis.anomaly_count = len(anomalies)
```

---

## Recommendation

**Use Option A (standalone function) + Option C (integrate in agent)**:

1. Create `analysis/cycles/anomalies.py` with detection logic
2. Add `anomalies` field to `BusinessCycleAnalysis`
3. Call `detect_cycle_anomalies()` in agent's `analyze()` after parsing
4. Persist anomalies to DB (new table or JSON field)

**Benefits**:
- Clean separation: Detection logic isolated in its own module
- Testable: Easy to unit test anomaly detection
- Reusable: Can run anomaly detection on historical analyses
- Integrated: Anomalies included in standard analysis output

---

## Configuration

Add anomaly thresholds to `config/cycles/cycle_vocabulary.yaml`:

```yaml
# Add to cycle_vocabulary.yaml
anomaly_detection:
  # Completion rate threshold
  low_completion_threshold: 0.5  # 50%

  # Excessive cycles threshold
  excessive_cycles_threshold: 15

  # Severity mappings
  severity_rules:
    missing_expected:
      default: medium
      high_value_cycle: high  # If missing cycle was high-value
    low_completion:
      high_value_cycle: high
      other: medium
    unclassified:
      default: low
    excessive_cycles:
      default: medium
```

---

## DB Persistence

### Option 1: JSON Field (Simpler)

Add to `BusinessCycleAnalysisRun`:
```python
anomalies: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
anomaly_count: Mapped[int] = mapped_column(Integer, default=0)
```

### Option 2: Separate Table (More Queryable)

```python
class CycleAnomalyRecord(Base):
    __tablename__ = "cycle_anomalies"

    anomaly_id: Mapped[str] = mapped_column(String, primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey(...))
    anomaly_type: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    # ...
```

**Recommendation**: Start with Option 1 (JSON field) for simplicity.

---

## Implementation Steps

1. Add `CycleAnomaly` model to `models.py`
2. Add `anomalies` field to `BusinessCycleAnalysis`
3. Create `anomalies.py` with detection functions
4. Add anomaly thresholds to config
5. Integrate in agent's `analyze()` method
6. Add DB persistence (JSON field initially)
7. Update status doc
8. Write tests

**Estimated Effort**: 1-2 hours
