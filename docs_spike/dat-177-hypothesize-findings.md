# DAT-177: Can the BBN predict fix outcomes without LLM?

**Status:** Spike complete
**Date:** 2026-03-23
**Author:** Research spike

## Executive Summary

**Yes.** The BBN can predict fix outcomes purely through computation. The existing `what_if_analysis` function already implements do-calculus interventions via pgmpy's `CausalInference`. The `compute_network_priorities` function already computes `impact_delta` for every non-low node. A `hypothesize` tool can be built as pure BBN propagation with no LLM involvement for the common case. LLM reasoning is only needed for edge cases where the fix action does not map cleanly to a single BBN node state change.

## 1. How the BBN Works Today

### Network Structure

The BBN (`config/entropy/network.yaml`) is an 18-node DAG with 3 discrete states (`low`, `medium`, `high`):

- **12 observable root nodes** (have detectors): `type_fidelity`, `null_ratio`, `outlier_rate`, `naming_clarity`, `unit_declaration`, `time_role`, `temporal_drift`, `benford_compliance`, `slice_stability`, `dimension_coverage`, `cross_table_consistency`, `business_cycle_health`
- **2 observable children** (have detectors + causal parents): `join_path_determinism`, `relationship_quality`
- **2 inferred children** (causal composites, no detectors): `formula_match`, `aggregation_safety`
- **3 intent leaves** (readiness scores): `query_intent`, `aggregation_intent`, `reporting_intent`

### Per-Column Execution

The network runs independently per column target (`network_context.py:_build_column_result`):

1. **Subgraph pruning** (`model.py:subgraph`): Only nodes with evidence (from detectors that fired for this column) and their inferrable descendants are kept. Unobserved root nodes are pruned to prevent prior leakage.
2. **Forward propagation** (`inference.py:forward_propagate`): pgmpy `VariableElimination` computes posterior P(intent=high) given observed evidence.
3. **Priority computation** (`priority.py:compute_network_priorities`): For each non-low node, runs `what_if_analysis` with `intervention={node: "low"}` to compute how much P(intent=high) drops. This is `impact_delta`.

### Key Insight: `impact_delta` Is Already a Fix Prediction

`impact_delta` answers exactly the question `hypothesize` needs: "If this detector score drops to low, how much does readiness improve?" The computation is:

```
impact_delta = baseline_P(intent=high) - P(intent=high | do(node=low))
```

This uses pgmpy's `CausalInference.query()` with `do=` (do-calculus), which severs the node from its parents and forces it to a specific state. This is the correct causal intervention semantics.

## 2. Mapping Fixes to BBN Nodes

Each fix in `config/entropy/fixes.yaml` has a `dimension_path` that maps directly to a BBN node via the bridge (`bridge.py:build_dimension_path_to_node_map`). The mapping:

| Fix detector_id | dimension_path | BBN node | Fix actions |
|---|---|---|---|
| type_fidelity | structural.types.type_fidelity | `type_fidelity` | document_type_pattern, document_type_override, accept |
| null_ratio | value.nulls.null_ratio | `null_ratio` | accept |
| outlier_rate | value.outliers.outlier_rate | `outlier_rate` | accept |
| benford | value.distribution.benford_compliance | `benford_compliance` | accept |
| business_meaning | semantic.business_meaning.naming_clarity | `naming_clarity` | document_business_name |
| unit_entropy | semantic.units.unit_declaration | `unit_declaration` | document_unit, document_unit_source |
| temporal_entropy | semantic.temporal.time_role | `time_role` | document_timestamp_role, document_type_pattern |
| temporal_drift | value.temporal.temporal_drift | `temporal_drift` | accept |
| relationship_entropy | structural.relations.relationship_quality | `relationship_quality` | document_relationship, accept |
| derived_value | computational.derived_values.formula_match | `formula_match` | accept |
| cross_table_consistency | computational.reconciliation.cross_table_consistency | `cross_table_consistency` | accept |
| business_cycle_health | semantic.cycles.business_cycle_health | `business_cycle_health` | accept |
| dimensional_entropy | semantic.dimensional.dimensional_patterns | *unmapped (direct signal)* | confirm_expected_pattern |
| dimension_coverage | semantic.coverage.dimension_coverage | `dimension_coverage` | *no fix schemas yet* |

Every fix detector except `dimensional_entropy` maps to a BBN node. `dimensional_entropy` is a direct signal (table-scoped, not in the column-level network).

## 3. Comparing BBN Predictions vs. Actual Fix Calibration

### Calibration data (from dataraum-eval)

The fix calibration tested 10 fixes across 3 phases. Here is the analysis of whether BBN `impact_delta` would have predicted the outcome:

#### Phase 1: accept_finding (score unchanged, contract overruled)

| Detector | Target | Pre | Post | BBN prediction |
|---|---|---|---|---|
| outlier_rate | journal_lines.credit | 1.000 | ~1.000 | **N/A** -- accept does not change the score, it changes the contract evaluation. BBN would correctly show impact_delta > 0 for `outlier_rate`, indicating what *would* happen if the score went to low. But accept_finding doesn't change the score. |
| benford | bank_transactions.amount | 0.803 | ~0.803 | Same as above |
| null_ratio | journal_lines.cost_center | 0.711 | ~0.711 | Same as above |
| relationship_entropy | payments.invoice_id | 0.447 | ~0.447 | Same as above |

**Finding:** For `accept_finding`, the BBN prediction is "score stays the same, readiness impact is zero for the detector score itself." This is correct -- the BBN should report that readiness *would* improve if the node went to low, but `accept_finding` does not change the node state. `hypothesize` needs to distinguish between "score drops to low" (metadata/config fix) and "score stays, but it's excluded from contract evaluation" (accept).

#### Phase 2: metadata fixes (score drops to 0)

| Detector | Target | Pre | Post | BBN node | Predicted behavior |
|---|---|---|---|---|---|
| business_meaning | invoices.rrflp_11_zp00 | 0.375 | 0.000 | `naming_clarity` | `what_if(naming_clarity=low)` would correctly predict the readiness improvement. Score goes from medium (0.375 > 0.3) to low (0.0). |
| business_meaning | invoices.xq_v7kl | 0.350 | 0.000 | `naming_clarity` | Same -- medium to low transition. |

**Finding:** BBN prediction is accurate. The metadata fix (`document_business_name`) directly resolves the detector finding, score drops to 0, node state goes from medium to low. `what_if_analysis(intervention={naming_clarity: low})` gives the correct readiness delta.

#### Phase 3: config fixes requiring re-run (score drops significantly)

| Detector | Target | Pre | Post | BBN node | Predicted behavior |
|---|---|---|---|---|---|
| temporal_entropy | payments.date | 0.800 | ~0 | `time_role` | `what_if(time_role=low)` correctly predicts the readiness delta. Score goes from high to low. |
| type_fidelity | journal_lines.debit | 0.585 | 0.100 | `type_fidelity` | `what_if(type_fidelity=low)` correctly predicts the *direction*. But the actual post-fix score (0.100) is still in the "low" band (<=0.3), so the node state does go to low. Prediction is correct. |

**Finding:** BBN prediction is accurate for config fixes too. The key is that the fix resolves the root cause, and the detector score drops into the "low" band.

#### xfail cases (fix doesn't address root cause)

| Detector | Target | Fix action | Why it fails |
|---|---|---|---|
| temporal_entropy | payments.date | set_timestamp_role | Column already has timestamp role; real issue is type mismatch |
| relationship_entropy | payments.invoice_id | confirm_relationship | ri_entropy dominates (0.447 from orphans); confirm_relationship only reduces semantic component |

**Finding:** These are cases where the fix action targets the wrong underlying cause. The BBN *cannot* predict this because it models node-level state, not the sub-components within a detector. `relationship_entropy` maps to `relationship_quality` in the BBN, but within that detector, the score is a max-aggregation of 3 sub-components (ri_entropy, cardinality_entropy, semantic_entropy). `confirm_relationship` only reduces `semantic_entropy`, but `ri_entropy` is the dominant contributor.

## 4. Can `hypothesize` Be Pure Computation?

### Yes, for the common case

For fixes that map to a BBN node and are expected to resolve the finding (score drops to low), `hypothesize` is pure computation:

```python
def predict_fix(
    network: EntropyNetwork,
    evidence: dict[str, str],  # current column evidence
    target_node: str,           # BBN node the fix targets
    new_state: str = "low",     # expected post-fix state
) -> dict[str, dict[str, float]]:
    """Predict readiness change from a fix.

    Returns per-intent P(high) delta.
    """
    # Current baseline
    baseline = forward_propagate(network, evidence, query_nodes=network.get_intent_nodes())

    # Intervention: set target node to new_state
    remaining = {k: v for k, v in evidence.items() if k != target_node}
    intervened = what_if_analysis(network, remaining, {target_node: new_state})

    # Compute deltas
    deltas = {}
    for intent in network.get_intent_nodes():
        before = baseline.get(intent, {}).get("high", 0.0)
        after = intervened.get(intent, {}).get("high", 0.0)
        deltas[intent] = {"before": before, "after": after, "delta": before - after}

    return deltas
```

This is essentially what `compute_network_priorities` already does, but scoped to a single chosen node rather than iterating over all non-low nodes.

### Proposed API for `hypothesize`

```python
@dataclass
class HypothesisResult:
    target: str                  # column target (e.g., "column:journal_lines.debit")
    fix_action: str              # e.g., "document_type_pattern"
    bbn_node: str                # e.g., "type_fidelity"
    current_state: str           # e.g., "high"
    predicted_state: str         # e.g., "low"
    intent_deltas: dict[str, float]  # per-intent P(high) reduction
    readiness_before: str        # e.g., "blocked"
    readiness_after: str         # e.g., "ready"
    confidence: str              # "high" | "medium" | "low"
    caveats: list[str]           # e.g., ["fix may only resolve sub-component"]

def hypothesize(
    session: Session,
    table_ids: list[str],
    column_target: str,
    fix_action: str,
    *,
    assume_resolved: bool = True,  # assume fix fully resolves finding
) -> HypothesisResult:
    ...
```

The `confidence` field reflects whether the BBN prediction is reliable:
- **high**: Fix action maps to a root BBN node and is expected to fully resolve the finding (score -> low)
- **medium**: Fix action maps to a BBN node but may only partially resolve (e.g., `accept_finding` doesn't change score)
- **low**: Fix action targets something the BBN doesn't model (e.g., `dimensional_entropy` direct signal)

### Cases Where LLM Reasoning Is Needed

1. **Sub-component ambiguity**: When a detector aggregates multiple sub-signals (e.g., `relationship_entropy` = max(ri, cardinality, semantic)), and the fix only addresses one sub-component. The BBN treats `relationship_quality` as a single node. To predict whether `confirm_relationship` will reduce the score, you need to know which sub-component dominates -- this requires reading the evidence breakdown, which the BBN doesn't model.

   **Mitigation without LLM**: Include the detector's sub-component breakdown in the evidence. If the fix action maps to a specific sub-component (e.g., `confirm_relationship` -> semantic_entropy), check if that sub-component is the dominant contributor. This is deterministic logic, not LLM reasoning.

2. **Config fixes with uncertain outcome**: `document_type_pattern` (adding a date parsing pattern) may or may not resolve `type_fidelity`. If the pattern is correct, score goes to ~0. If it's wrong, score stays. The BBN can predict the *readiness change if the fix works*, but cannot predict *whether* the fix will work.

   **Mitigation**: Always present the prediction as conditional: "If this fix resolves the finding, readiness improves by X." The `hypothesize` tool doesn't need to predict fix correctness, just fix impact.

3. **Cross-column effects**: A fix to one column (e.g., typing a date column) can affect other columns' detectors downstream (e.g., `relationship_entropy` improves because joins now work). The per-column BBN doesn't model cross-column dependencies.

   **Mitigation**: Run `hypothesize` per-column. Cross-column effects are a second-order concern that can be noted as a caveat.

4. **Direct signals** (unmapped detectors like `dimensional_entropy`): The BBN has no node for these, so it cannot predict impact. However, `dimensional_entropy`'s fix (`confirm_expected_pattern`) is an `accept`-style fix that labels the pattern as expected. The score itself doesn't change; the contract evaluation changes.

   **Mitigation**: For direct signals, `hypothesize` returns a fixed response: "This detector is not modeled in the network; fix impact cannot be predicted via BBN."

## 5. Implementation Complexity

### What Already Exists

| Component | Status | Location |
|---|---|---|
| BBN model + CPTs | Done | `entropy/network/model.py`, `cpts.py` |
| Forward propagation | Done | `entropy/network/inference.py:forward_propagate` |
| do-calculus intervention | Done | `entropy/network/inference.py:what_if_analysis` |
| Priority ranking (per-node impact_delta) | Done | `entropy/network/priority.py:compute_network_priorities` |
| Per-column subgraph + evidence | Done | `entropy/views/network_context.py:_build_column_result` |
| Fix-to-node mapping | Done | `config/entropy/fixes.yaml` dimension_path -> bridge -> node |
| Sub-component breakdown | Done | Detector evidence contains sub-signals (e.g., ri_entropy) |

### What Needs Building

| Component | Effort | Notes |
|---|---|---|
| `hypothesize` MCP tool | S | Thin wrapper: load evidence, look up BBN node, call `what_if_analysis`, format result |
| `predict_fix` function | S | ~30 lines; extract from `compute_network_priorities` logic |
| Accept-finding special case | S | Detect `accept_*` actions, report "score unchanged, contract overruled" |
| Sub-component dominance check | M | Read detector evidence breakdown, check which sub-component a fix targets |
| Direct signal fallback | S | Return "not modeled" for unmapped detectors |

**Total estimate: S-M task.** No new inference machinery needed. The BBN already has everything.

## 6. Recommendations

1. **Build `hypothesize` as pure computation.** No LLM call needed. The BBN's `what_if_analysis` is the prediction engine.

2. **Three prediction modes:**
   - `resolve` (default): Assumes fix drops node to "low". Returns readiness delta.
   - `accept`: Score unchanged, contract evaluation changes. Returns "overruled" status.
   - `partial`: Fix targets a sub-component. Uses detector evidence to estimate whether dominant component is addressed.

3. **Include `impact_delta` from existing priority computation.** The MCP `get_quality` response already includes `impact_delta` per node per at-risk column. `hypothesize` could simply re-surface this with the fix action context.

4. **Defer LLM integration.** If we discover cases where pure computation is insufficient (e.g., user asks "what if I clean up the orphan records manually?"), we can add an LLM reasoning layer later. Start with BBN-only.

5. **Test with calibration data.** The fix calibration results (8/10 pass) provide ground truth for `hypothesize` predictions. The 2 xfail cases (wrong fix for root cause) are exactly the cases where sub-component dominance checking would add value.
