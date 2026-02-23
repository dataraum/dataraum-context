# Bayesian Entropy Network: Probabilistic Inference for Data Quality

## Abstract

This document specifies a **Bayesian Network** that models the probabilistic dependencies between dataraum's 18 entropy sub-dimensions. The network serves three purposes:

1. **Inference**: Propagate evidence from measured sub-dimensions to estimate unmeasured ones
2. **Impact Analysis**: Quantify how resolving one entropy source affects the entire system
3. **Explanation**: Trace the causal path from root causes to observable query failures

The network unifies the knowledge graph (what depends on what) with weighted probabilistic edges (how strongly), producing a single visual and computational artifact that both humans and LLM agents can reason over.

---

## 1. Conceptual Foundation

### 1.1 Why Bayesian Networks?

Traditional entropy scoring aggregates sub-dimension scores using weighted sums. This assumes independence — that fixing Schema entropy has no bearing on Business Meaning entropy. In practice, these are deeply coupled through metadata lineage:

- A CSV with no header row (Physical entropy) makes Schema detection uncertain, which makes Type inference unreliable, which means Unit detection fails, which means Query Intent resolution requires guesswork.

A Bayesian Network encodes these dependencies as a **Directed Acyclic Graph (DAG)** where:
- Each **node** is an entropy sub-dimension (18 nodes)
- Each **directed edge** represents a conditional dependency (parent influences child)
- Each node has a **Conditional Probability Distribution (CPD)** quantifying how parent states affect child state probabilities

This gives us three capabilities that weighted sums cannot:

| Capability | Weighted Sum | Bayesian Network |
|---|---|---|
| Forward propagation (cause → effect) | ❌ | ✅ Predict downstream impact |
| Backward propagation (effect → cause) | ❌ | ✅ Diagnose root causes |
| Partial observability | ❌ Manual imputation | ✅ Automatic marginalization |
| What-if analysis | ❌ | ✅ Interventional queries |
| Explain why a score is high | ❌ | ✅ Most probable explanation |

### 1.2 The Knowledge Graph IS the Bayesian Network

A key insight: the graph you want to show users in the UI is the *same* structure used for probabilistic inference. There is no separate "knowledge graph" and "inference model" — they are one artifact. The visual representation shows:

- **Nodes** colored by entropy level (green/yellow/red)
- **Edges** with thickness proportional to conditional dependency strength
- **Clusters** corresponding to the 6 entropy dimensions
- **Flow** generally left-to-right from Source → Query, showing the data lineage path

When a user clicks a node and asks "why is this high?", the answer comes from running **Most Probable Explanation (MPE)** inference on the network — finding the most likely parent states that explain the observed child state.

---

## 2. Network Topology

### 2.1 Full DAG Specification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BAYESIAN ENTROPY NETWORK                            │
│                                                                             │
│  SOURCE                STRUCTURAL          SEMANTIC                         │
│  ┌──────────┐         ┌──────────┐        ┌──────────────┐                  │
│  │ Physical │────────▶│  Schema  │───────▶│ Biz Meaning  │                  │
│  └──────────┘         └──────────┘        └──────────────┘                  │
│       │                    │                    │    │                       │
│       │                    ▼                    │    │                       │
│       │               ┌──────────┐              │    │                       │
│       │               │  Types   │──────────────┼────┼───┐                  │
│       │               └──────────┘              │    │   │                  │
│       │                    │                    ▼    │   │                  │
│  ┌──────────┐              │              ┌─────────┐│   │                  │
│  │Provenance│──────────────┼─────────────▶│  Units  ││   │                  │
│  └──────────┘              │              └─────────┘│   │                  │
│       │                    ▼                    │    │   │                  │
│       │               ┌──────────┐              │    ▼   │                  │
│       │               │Relations │──────────────┼▶┌─────────────┐           │
│       │               └──────────┘              │ │  Temporal    │           │
│       │                    │                    │ │  Resolution  │           │
│  ┌──────────┐              │                    │ └─────────────┘           │
│  │ Access   │              │                    │        │                  │
│  │ Rights   │              │                    ▼        │                  │
│  └──────────┘              │              ┌─────────────┐│                  │
│       │                    │              │ Categorical  ││                  │
│  ┌──────────┐              │              │ Dimensions   ││                  │
│  │Lifecycle │              │              └─────────────┘│                  │
│  └──────────┘              │                    │        │                  │
│       │                    │                    │        │                  │
│       ▼                    ▼                    ▼        ▼                  │
│  VALUE                 COMPUTATIONAL       QUERY                            │
│  ┌──────────┐         ┌──────────────┐    ┌─────────────┐                  │
│  │Categories│         │Derived Values│───▶│ Linguistics │                  │
│  └──────────┘         └──────────────┘    └─────────────┘                  │
│  ┌──────────┐         ┌──────────────┐         │                           │
│  │  Nulls   │         │Business Rules│─────────┤                           │
│  └──────────┘         └──────────────┘         │                           │
│  ┌──────────┐         ┌──────────────┐         ▼                           │
│  │ Outliers │         │ Aggregations │───▶┌──────────┐                     │
│  └──────────┘         └──────────────┘    │  Intent  │                     │
│  ┌──────────┐                             └──────────┘                     │
│  │ Patterns │                                                              │
│  └──────────┘                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Edge Definitions

Each edge encodes a specific causal hypothesis about why entropy in one sub-dimension creates entropy in another. These are not arbitrary — they follow data lineage logic.

#### Source → Structural Edges

| Edge | Rationale | Strength |
|---|---|---|
| Physical → Schema | File format constrains schema detection. A well-typed Parquet file has perfect schema; a headerless CSV has none. | Strong |
| Physical → Types | Binary formats embed types; text formats require inference. | Strong |
| Provenance → Schema | Known systems have known schemas (ERP exports are predictable). Unknown origins mean unknown structure. | Moderate |

#### Structural → Semantic Edges

| Edge | Rationale | Strength |
|---|---|---|
| Schema → Business Meaning | Clear column naming (`total_revenue_eur`) carries semantic information. Opaque names (`col_7`, `FIELD_A`) do not. | Strong |
| Types → Units | Numeric types enable unit detection; VARCHAR columns can't have unit semantics reliably inferred. | Strong |
| Types → Categories | Categorical type detection depends on knowing whether a column is truly categorical vs. a numeric code. | Moderate |
| Relations → Temporal Resolution | Foreign keys to date/time dimension tables reveal temporal semantics. Standalone date columns are ambiguous. | Moderate |
| Relations → Categorical Dimensions | Relationships to lookup tables indicate dimensional hierarchies. | Strong |
| Provenance → Business Meaning | Known data source → known business context. An SAP export column `BUKRS` has known meaning; a generic CSV column does not. | Strong |

#### Semantic → Value Edges

| Edge | Rationale | Strength |
|---|---|---|
| Business Meaning → Categories (Value) | If you know a column represents "payment status", you know what categories to expect, making anomalous values detectable. | Strong |
| Business Meaning → Nulls | Business meaning determines null semantics: a null in `discount_pct` means 0%, but a null in `shipping_date` means "not yet shipped". Without meaning, nulls are opaque. | Strong |
| Units → Outliers | Knowing the unit constrains valid ranges. 1,000,000 CHF revenue is plausible; 1,000,000 km/h speed is not. | Moderate |
| Categorical Dimensions → Categories (Value) | Defined hierarchies constrain valid category values. If you know "Region > Country > City", you can validate each level. | Strong |
| Temporal Resolution → Patterns | Known temporal granularity defines expected patterns (daily data should have daily records, monthly should aggregate). | Moderate |

#### Structural → Computational Edges

| Edge | Rationale | Strength |
|---|---|---|
| Relations → Derived Values | Join paths enable cross-table calculations. Without known relations, derived values can't be computed reliably. | Strong |
| Relations → Aggregations | Cardinality and join types determine valid aggregation paths. Many-to-many joins produce incorrect sums without awareness. | Strong |
| Types → Business Rules | Type information constrains what operations are valid (you can't SUM a VARCHAR, even if it looks numeric). | Moderate |

#### Semantic → Computational Edges

| Edge | Rationale | Strength |
|---|---|---|
| Business Meaning → Business Rules | Domain rules require business context. "Revenue = Quantity × Price" only applies if you know which columns represent what. | Strong |
| Business Meaning → Derived Values | Knowing what a column means enables correct derivation formulas. | Strong |
| Temporal Resolution → Aggregations | Time grain determines valid rollups. Aggregating monthly averages by summing is wrong; aggregating monthly totals by summing is right. | Strong |
| Units → Derived Values | Unit consistency is required for valid calculations. Adding CHF to EUR produces garbage without conversion. | Strong |

#### → Query Edges (Leaf Nodes)

| Edge | Rationale | Strength |
|---|---|---|
| Business Meaning → Linguistics | Semantic definitions populate the synonym/abbreviation mappings that resolve linguistic ambiguity. | Strong |
| Categorical Dimensions → Linguistics | Known dimension values enable fuzzy matching ("Q1" → "Quarter 1", "EMEA" → specific countries). | Moderate |
| Business Rules → Linguistics | Business logic terminology must map to natural language ("margin" might mean gross or net depending on rules). | Moderate |
| Derived Values → Intent | Available derived calculations constrain what questions can be answered and what "show me revenue" actually computes. | Strong |
| Business Rules → Intent | Domain rules define defaults ("revenue" means "net revenue after returns" per company policy). | Strong |
| Aggregations → Intent | Available aggregation paths determine whether "total revenue by region" is computable and how. | Strong |
| Temporal Resolution → Intent | Time semantics determine what "last quarter" means and what granularity is available. | Moderate |
| Access Rights → Intent | RLS/PII constraints limit what data can be shown, affecting query resolution. | Moderate |
| Lifecycle → Intent | Data freshness affects whether "current" queries can be answered. | Weak |

### 2.3 Formal Edge List

For implementation, the complete edge list:

```yaml
# bayesian_network_edges.yaml
# Format: [parent, child, strength, category]

edges:
  # Source → Structural
  - [physical, schema, 0.85, "source_structural"]
  - [physical, types, 0.80, "source_structural"]
  - [provenance, schema, 0.60, "source_structural"]

  # Source → Semantic (cross-dimension)
  - [provenance, business_meaning, 0.75, "source_semantic"]

  # Structural → Semantic
  - [schema, business_meaning, 0.80, "structural_semantic"]
  - [types, units, 0.85, "structural_semantic"]
  - [types, categories_value, 0.55, "structural_semantic"]
  - [relations, temporal_resolution, 0.60, "structural_semantic"]
  - [relations, categorical_dimensions, 0.75, "structural_semantic"]

  # Semantic → Value
  - [business_meaning, categories_value, 0.80, "semantic_value"]
  - [business_meaning, nulls, 0.75, "semantic_value"]
  - [units, outliers, 0.65, "semantic_value"]
  - [categorical_dimensions, categories_value, 0.80, "semantic_value"]
  - [temporal_resolution, patterns, 0.60, "semantic_value"]

  # Structural → Computational
  - [relations, derived_values, 0.80, "structural_computational"]
  - [relations, aggregations, 0.85, "structural_computational"]
  - [types, business_rules, 0.55, "structural_computational"]

  # Semantic → Computational
  - [business_meaning, business_rules, 0.85, "semantic_computational"]
  - [business_meaning, derived_values, 0.75, "semantic_computational"]
  - [temporal_resolution, aggregations, 0.80, "semantic_computational"]
  - [units, derived_values, 0.70, "semantic_computational"]

  # → Query (leaf edges)
  - [business_meaning, linguistics, 0.80, "to_query"]
  - [categorical_dimensions, linguistics, 0.55, "to_query"]
  - [business_rules, linguistics, 0.60, "to_query"]
  - [derived_values, intent, 0.75, "to_query"]
  - [business_rules, intent, 0.80, "to_query"]
  - [aggregations, intent, 0.80, "to_query"]
  - [temporal_resolution, intent, 0.65, "to_query"]
  - [access_rights, intent, 0.50, "to_query"]
  - [lifecycle, intent, 0.35, "to_query"]
```

---

## 3. Probabilistic Model

### 3.1 State Space

Each node (sub-dimension) has a discrete state representing its entropy level:

```python
ENTROPY_STATES = ["low", "medium", "high"]

# Mapping from continuous entropy score [0.0, 1.0] to discrete states:
# low:    [0.0, 0.33)  — well-defined, minimal ambiguity
# medium: [0.33, 0.66) — partially defined, some ambiguity
# high:   [0.66, 1.0]  — undefined or highly ambiguous
```

Three states balance expressiveness with tractability. Each CPT has at most 3^k entries where k is the number of parents (max 4 parents in this network = 81 entries).

### 3.2 Conditional Probability Tables (CPTs)

#### Root Nodes (No Parents)

Root nodes use **prior distributions** reflecting baseline expectations for enterprise data:

```yaml
# Prior probabilities for root nodes
# These represent "before we measure anything, how likely is each state?"

priors:
  physical:
    # Most enterprise data arrives in semi-structured formats
    low: 0.30    # Well-typed binary formats (Parquet, DB exports)
    medium: 0.45 # Common formats with some ambiguity (CSV with headers)
    high: 0.25   # Opaque formats (headerless CSV, PDFs, Excel with merged cells)

  provenance:
    # Many enterprise datasets lack proper lineage metadata
    low: 0.20    # Full lineage chain documented
    medium: 0.35 # Partial provenance (system known, pipeline unknown)
    high: 0.45   # Unknown origin or undocumented

  access_rights:
    # Access control metadata is frequently missing in analytics contexts
    low: 0.25    # Full RBAC/RLS metadata present
    medium: 0.40 # Partial (some PII marking, incomplete roles)
    high: 0.35   # No access metadata

  lifecycle:
    # Most data has some freshness indicators
    low: 0.35    # Clear versioning and freshness metadata
    medium: 0.40 # Timestamps present but semantics unclear
    high: 0.25   # No lifecycle metadata
```

#### Parameterized CPTs from Edge Strength

Rather than manually specifying all CPTs, we derive them from the edge strengths defined in Section 2.3 using a parameterized approach.

For a **single-parent** node:

```python
def make_cpt_single_parent(strength: float) -> np.ndarray:
    """
    Generate CPT from edge strength.
    
    strength = 1.0: child perfectly mirrors parent
    strength = 0.0: child is independent of parent
    
    Returns 3x3 matrix: CPT[parent_state][child_state]
    """
    # Base rate (uniform when no influence)
    base = np.array([1/3, 1/3, 1/3])
    
    # Influence matrix: parent state pushes child toward same state
    influence = np.array([
        [0.80, 0.15, 0.05],  # parent=low → child likely low
        [0.15, 0.70, 0.15],  # parent=medium → child likely medium
        [0.05, 0.15, 0.80],  # parent=high → child likely high
    ])
    
    # Blend based on strength
    cpt = (1 - strength) * base + strength * influence
    
    # Normalize rows
    cpt = cpt / cpt.sum(axis=1, keepdims=True)
    return cpt
```

For **multi-parent** nodes, we use a **Noisy-OR** inspired aggregation:

```python
def make_cpt_multi_parent(parent_strengths: dict[str, float]) -> np.ndarray:
    """
    Generate CPT for node with multiple parents using weighted combination.
    
    parent_strengths: {parent_name: strength}
    
    The key insight: entropy is pessimistic — high entropy from ANY strong 
    parent propagates. This is not a simple average; it models the reality 
    that one broken link in the lineage chain can cascade.
    """
    parents = list(parent_strengths.keys())
    strengths = list(parent_strengths.values())
    n_parents = len(parents)
    states = ["low", "medium", "high"]
    n_states = 3
    
    # Generate all parent state combinations
    import itertools
    combos = list(itertools.product(range(n_states), repeat=n_parents))
    
    cpt = np.zeros((len(combos), n_states))
    
    for i, combo in enumerate(combos):
        # Weighted aggregation: higher-strength parents have more influence
        total_weight = sum(strengths)
        weighted_state = sum(
            s * w for s, w in zip(combo, strengths)
        ) / total_weight
        
        # Pessimistic shift: the max parent entropy pulls the distribution
        max_parent = max(combo)
        pessimism_weight = 0.3  # How much the worst parent dominates
        effective_state = (
            (1 - pessimism_weight) * weighted_state + 
            pessimism_weight * max_parent
        )
        
        # Convert effective state to probability distribution
        # using softmax-like mapping
        for j in range(n_states):
            distance = abs(j - effective_state)
            cpt[i, j] = np.exp(-2 * distance)
        
        # Normalize
        cpt[i] /= cpt[i].sum()
    
    return cpt
```

**The pessimistic shift is conceptually important**: in entropy terms, ambiguity is not averaged away by good neighbors. If your Schema is well-defined but your Provenance is unknown, Business Meaning still suffers — you know the column names but not what context they were defined in. The `pessimism_weight` parameter controls this cascading effect.

### 3.3 Learning CPTs from Data

Initial CPTs come from the parameterized functions above (expert priors). As dataraum profiles more datasets, the CPTs are updated using **Bayesian parameter learning**:

```python
def update_cpts_from_observations(
    network: BayesianNetwork,
    observations: pd.DataFrame
) -> BayesianNetwork:
    """
    Update CPTs using Maximum A Posteriori (MAP) estimation
    with Bayesian Dirichlet equivalent priors (BDeu).
    
    observations: DataFrame where each row is a profiled dataset,
                  columns are sub-dimension names,
                  values are discretized states (0=low, 1=med, 2=high)
    """
    from pgmpy.estimators import BayesianEstimator
    
    estimator = BayesianEstimator(network, observations)
    
    for node in network.nodes():
        # equivalent_sample_size controls how much prior knowledge 
        # we retain vs. learning from data.
        # Higher = more conservative (trust priors more)
        # Lower = learn faster from observations
        cpd = estimator.estimate_cpd(
            node, 
            prior_type="BDeu",
            equivalent_sample_size=10  # moderate prior strength
        )
        network.add_cpds(cpd)
    
    return network
```

The **equivalent_sample_size** of 10 means the prior CPTs are worth approximately 10 observations. After profiling ~50 datasets, the learned distributions will dominate. This is the **learning flywheel**: every dataset profiled makes the network's predictions more accurate.

---

## 4. Inference Operations

### 4.1 Core Operations

The network supports four inference operations that map directly to user-facing features:

#### Operation 1: Forward Propagation (Predictive)

**Question**: "I just profiled Source and Structural dimensions. What should I expect for Semantic and downstream?"

```python
from pgmpy.inference import VariableElimination

infer = VariableElimination(network)

# Set observed evidence
evidence = {
    "physical": "low",      # Parquet file, well-typed
    "provenance": "high",   # Unknown origin
    "schema": "low",        # Good column names
    "types": "low",         # Correct types from Parquet
    "relations": "medium",  # Some FKs detected, not all
}

# Predict unobserved dimensions
for node in ["business_meaning", "units", "intent"]:
    posterior = infer.query([node], evidence=evidence)
    print(f"{node}: {posterior}")

# Example output:
# business_meaning: [low=0.25, medium=0.50, high=0.25]
#   → Schema is good (low) but provenance is unknown (high)
#     so business meaning is uncertain despite clear column names
#
# units: [low=0.70, medium=0.25, high=0.05]
#   → Types are well-defined, so unit detection will likely succeed
#
# intent: [low=0.15, medium=0.45, high=0.40]
#   → Query resolution is still risky due to unknown provenance
#     cascading through business rules and aggregation paths
```

**UI Feature**: After initial profiling (fast Source/Structural scan), show predicted entropy levels for deeper dimensions with confidence intervals. Users see "we predict your semantic layer has medium entropy — click to verify" before the expensive semantic profiling runs.

#### Operation 2: Backward Propagation (Diagnostic)

**Question**: "Query Intent entropy is high. Why?"

```python
# Observe the symptom
evidence = {"intent": "high"}

# Diagnose most likely causes
for root in ["physical", "provenance", "access_rights", "lifecycle"]:
    posterior = infer.query([root], evidence=evidence)
    print(f"P({root} | intent=high): {posterior}")

# Also check intermediate nodes
for mid in ["business_meaning", "aggregations", "business_rules"]:
    posterior = infer.query([mid], evidence=evidence)
    print(f"P({mid} | intent=high): {posterior}")
```

**UI Feature**: When a query fails or produces ambiguous results, trace backward through the graph to highlight which upstream entropy sources are most likely responsible. Show the path as a highlighted subgraph: "Query intent is ambiguous because → Business Rules are undefined because → Business Meaning is uncertain because → Provenance metadata is missing."

#### Operation 3: What-If Analysis (Interventional)

**Question**: "If I define all business meanings (reduce Business Meaning entropy to low), how much does overall query entropy improve?"

```python
from pgmpy.inference import CausalInference

causal = CausalInference(network)

# Current state (observed evidence)
current_evidence = {
    "physical": "low",
    "provenance": "high",
    "schema": "low",
    "types": "low",
    "relations": "medium",
    "business_meaning": "high",  # Currently unmapped
}

# Predict intent with current state
current_intent = infer.query(["intent"], evidence=current_evidence)

# Now intervene: force business_meaning to low
intervention_evidence = current_evidence.copy()
intervention_evidence["business_meaning"] = "low"  # After resolution

new_intent = infer.query(["intent"], evidence=intervention_evidence)

# Compare
improvement = (
    current_intent.values[2] -  # P(intent=high) before
    new_intent.values[2]        # P(intent=high) after
)
print(f"Reducing business_meaning entropy decreases P(intent=high) by {improvement:.1%}")
```

**UI Feature**: This is the "what-if" slider. Users drag a node's entropy from high to low and see the entire downstream graph update in real time. This directly answers "where should I invest my configuration effort?" — the node whose intervention produces the largest decrease in Query Entropy is the highest priority.

#### Operation 4: Most Probable Explanation (MPE)

**Question**: "Given what we observe, what's the single most likely state of all unmeasured sub-dimensions?"

```python
from pgmpy.inference import VariableElimination

infer = VariableElimination(network)

# Partial observations
evidence = {
    "physical": "low",
    "schema": "low",
    "types": "low",
    "intent": "high",  # But queries are still failing!
}

# Find most probable explanation for all unobserved nodes
mpe = infer.map_query(
    variables=[
        "provenance", "access_rights", "lifecycle",
        "relations", "business_meaning", "units",
        "temporal_resolution", "categorical_dimensions",
        "categories_value", "nulls", "outliers", "patterns",
        "derived_values", "business_rules", "aggregations",
        "linguistics"
    ],
    evidence=evidence
)
print(mpe)
# Output: most probable state assignment for all unmeasured nodes
# Likely shows: provenance=high, business_meaning=high, business_rules=high
# → "Your data format is fine but business context is completely missing"
```

**UI Feature**: One-click "Explain" button that produces a narrative: "Your query entropy is high despite good structural quality. The most likely explanation is that business context is missing across provenance, business meaning, and business rules — the data is clean but undocumented."

---

## 5. Integration with Entropy Objects

### 5.1 Entropy Object → Network Evidence

Each entropy object produced by a detector maps to network evidence:

```yaml
# Existing entropy object (from current framework)
entropy_object:
  dimension: semantic
  sub_dimension: units
  scope:
    table: financial_transactions
    column: amount
  score: 0.78  # High entropy
  evidence:
    detected_values: [100, 250.50, 1000000]
    possible_units: ["CHF", "EUR", "USD"]
    confidence: 0.22
  hints:
    - action: "add_unit_annotation"
      target: "amount"
      priority: "high"

# Maps to network evidence:
network_evidence:
  units: "high"  # score 0.78 → discretized to "high"
```

The mapping function:

```python
def entropy_objects_to_evidence(
    objects: list[EntropyObject]
) -> dict[str, str]:
    """
    Convert a collection of entropy objects into network evidence.
    
    When multiple objects exist for the same sub-dimension 
    (e.g., units entropy for multiple columns), aggregate using
    the maximum (pessimistic) score.
    """
    scores_by_subdim: dict[str, list[float]] = defaultdict(list)
    
    for obj in objects:
        key = obj.sub_dimension  # e.g., "units", "schema", "nulls"
        scores_by_subdim[key].append(obj.score)
    
    evidence = {}
    for subdim, scores in scores_by_subdim.items():
        # Pessimistic aggregation: worst column determines table-level state
        max_score = max(scores)
        evidence[subdim] = discretize(max_score)
    
    return evidence

def discretize(score: float) -> str:
    if score < 0.33:
        return "low"
    elif score < 0.66:
        return "medium"
    else:
        return "high"
```

### 5.2 Network Posteriors → Enhanced Entropy Objects

After inference, the network enriches entropy objects with causal context:

```yaml
# Enhanced entropy object (after Bayesian inference)
entropy_object:
  dimension: semantic
  sub_dimension: units
  scope:
    table: financial_transactions
    column: amount
  score: 0.78
  
  # NEW: Bayesian context
  bayesian_context:
    # What's driving this score?
    primary_causes:
      - sub_dimension: types
        state: medium
        contribution: 0.35  # 35% of the entropy explained by type ambiguity
      - sub_dimension: provenance
        state: high
        contribution: 0.45  # 45% explained by unknown data origin
    
    # What does fixing this improve?
    downstream_impact:
      - sub_dimension: outliers
        current_p_high: 0.72
        after_fix_p_high: 0.31  # 57% reduction
      - sub_dimension: derived_values
        current_p_high: 0.65
        after_fix_p_high: 0.28  # 57% reduction
      - sub_dimension: intent
        current_p_high: 0.81
        after_fix_p_high: 0.54  # 33% reduction
    
    # Updated priority based on network-wide impact
    network_priority: 0.92  # Very high — fixing this cascades widely
    
  evidence:
    detected_values: [100, 250.50, 1000000]
    possible_units: ["CHF", "EUR", "USD"]
    confidence: 0.22
    
  hints:
    - action: "add_unit_annotation"
      target: "amount"
      priority: "critical"  # Upgraded from "high" due to network impact
      rationale: >
        Annotating units on 'amount' reduces entropy across 3 downstream
        dimensions. The primary cause of unit ambiguity is unknown data
        provenance (45% contribution) — if you can also document the data
        source, the combined effect reduces query intent entropy by ~40%.
```

### 5.3 Network Priority vs. Local Priority

A critical distinction: the **local** entropy score tells you how bad a single sub-dimension is. The **network priority** tells you how much fixing it matters for the system overall.

```python
def compute_network_priority(
    network: BayesianNetwork,
    infer: VariableElimination,
    evidence: dict[str, str],
    target_node: str = "intent"
) -> dict[str, float]:
    """
    For each observed high-entropy node, compute how much fixing it
    would reduce entropy at the target (usually 'intent').
    
    Returns: {sub_dimension: priority_score}
    """
    # Current target entropy
    current = infer.query([target_node], evidence=evidence)
    current_p_high = current.values[2]  # P(target=high)
    
    priorities = {}
    
    for node, state in evidence.items():
        if state in ["medium", "high"]:
            # What if we fixed this node?
            modified = evidence.copy()
            modified[node] = "low"
            
            new = infer.query([target_node], evidence=modified)
            new_p_high = new.values[2]
            
            # Priority = reduction in target entropy
            priorities[node] = current_p_high - new_p_high
    
    # Normalize to [0, 1]
    max_priority = max(priorities.values()) if priorities else 1
    return {k: v / max_priority for k, v in priorities.items()}
```

This produces the **prioritized resolution order** that your hints system needs. Instead of arbitrary priority weights, the network computes impact empirically.

---

## 6. YAML Configuration Integration

### 6.1 Network Configuration in YAML

The Bayesian network structure and parameters are configurable, allowing customization per deployment:

```yaml
# dataraum_config.yaml (extend existing config)

entropy_network:
  # Network structure can be customized per domain
  # Default edges from Section 2.3 are used unless overridden
  
  custom_edges:
    # Add domain-specific dependencies
    # Example: In healthcare, Access Rights strongly affects everything
    - parent: access_rights
      child: business_meaning
      strength: 0.80
      rationale: "HIPAA compliance requires semantic understanding of PHI fields"
    
    # Example: In finance, Lifecycle is critical for compliance
    - parent: lifecycle
      child: business_rules
      strength: 0.70
      rationale: "Financial regulations require data freshness for reporting"
  
  removed_edges:
    # Remove edges that don't apply in this domain
    - parent: lifecycle
      child: intent
      rationale: "Not relevant for static reference data"
  
  # Override prior distributions
  priors:
    physical:
      low: 0.60   # Most of our data is Parquet/DB exports
      medium: 0.30
      high: 0.10
  
  # Discretization thresholds (override defaults)
  discretization:
    low_threshold: 0.25    # More conservative for this deployment
    high_threshold: 0.60
  
  # Inference settings
  inference:
    pessimism_weight: 0.3  # How much worst-parent dominates (0-1)
    equivalent_sample_size: 10  # Prior strength for learning
    
  # Aggregation scope
  scope:
    # Run network at which level?
    # Options: table, schema, dataset, source
    level: table
    # How to aggregate column-level scores to table level?
    aggregation: max  # Options: max, mean, p90
```

### 6.2 Per-Table Network Overrides

Different tables may have different dependency structures:

```yaml
# In table-specific configuration
tables:
  financial_transactions:
    entropy_network:
      # For this table, temporal resolution is critical
      custom_edges:
        - parent: temporal_resolution
          child: intent
          strength: 0.85  # Override default 0.65
      
      # Pin known states (these won't be inferred)
      known_states:
        physical: low      # We know this is a Parquet export
        provenance: low    # From SAP, fully documented
        access_rights: low # RBAC configured
```

---

## 7. Query Agent Integration

### 7.1 Network Summary for LLM Context

When the query agent processes a natural language question, it receives a network summary as context:

```python
def generate_query_context(
    network: BayesianNetwork,
    evidence: dict[str, str],
    query: str
) -> str:
    """
    Generate a context block for the LLM query agent that includes
    the Bayesian network's assessment of data reliability.
    """
    infer = VariableElimination(network)
    
    # Get posteriors for query-relevant nodes
    intent_posterior = infer.query(["intent"], evidence=evidence)
    linguistics_posterior = infer.query(["linguistics"], evidence=evidence)
    
    # Identify highest-risk paths
    priorities = compute_network_priority(network, infer, evidence)
    top_risks = sorted(priorities.items(), key=lambda x: -x[1])[:3]
    
    context = f"""
## Data Reliability Assessment (Bayesian Network)

**Query Intent Confidence**: {1 - intent_posterior.values[2]:.0%}
**Linguistic Resolution Confidence**: {1 - linguistics_posterior.values[2]:.0%}

### Active Entropy Sources (by downstream impact):
"""
    for node, priority in top_risks:
        state = evidence.get(node, "unknown")
        context += f"- **{node}** (state: {state}, impact: {priority:.0%})\n"
    
    context += f"""
### Recommendation:
"""
    if intent_posterior.values[2] > 0.5:
        context += "⚠️ High query entropy — express uncertainty in the response. "
        context += "Flag specific ambiguities to the user.\n"
    elif intent_posterior.values[2] > 0.3:
        context += "⚡ Moderate query entropy — proceed but note assumptions.\n"
    else:
        context += "✅ Low query entropy — high confidence in query resolution.\n"
    
    return context
```

### 7.2 Resolution Workflow Trigger

The network also determines when to trigger interactive resolution:

```python
def should_trigger_resolution(
    network: BayesianNetwork,
    evidence: dict[str, str],
    query: str,
    threshold: float = 0.5
) -> tuple[bool, list[str]]:
    """
    Determine if the query agent should ask the user for clarification
    before executing, based on network state.
    
    Returns: (should_ask, list_of_ambiguous_dimensions)
    """
    infer = VariableElimination(network)
    
    ambiguous = []
    
    # Check query-layer nodes
    for node in ["intent", "linguistics"]:
        posterior = infer.query([node], evidence=evidence)
        if posterior.values[2] > threshold:  # P(high) > threshold
            ambiguous.append(node)
    
    # Check semantic nodes that affect query resolution
    for node in ["business_meaning", "units", "temporal_resolution"]:
        posterior = infer.query([node], evidence=evidence)
        if posterior.values[2] > threshold:
            ambiguous.append(node)
    
    return len(ambiguous) > 0, ambiguous
```

---

## 8. Visualization Specification

### 8.1 Graph Layout

The primary visualization is the DAG itself, rendered as an interactive graph:

```yaml
visualization:
  layout: hierarchical_left_to_right
  
  # Cluster by dimension
  clusters:
    source:
      nodes: [physical, provenance, access_rights, lifecycle]
      color: "#E8F4FD"
      label: "Source Entropy"
      position: left
    
    structural:
      nodes: [schema, types, relations]
      color: "#FDF2E8"
      label: "Structural Entropy"
      position: center_left
    
    semantic:
      nodes: [business_meaning, units, temporal_resolution, categorical_dimensions]
      color: "#F0E8FD"
      label: "Semantic Entropy"
      position: center
    
    value:
      nodes: [categories_value, nulls, outliers, patterns]
      color: "#E8FDE8"
      label: "Value Entropy"
      position: center_right
    
    computational:
      nodes: [derived_values, business_rules, aggregations]
      color: "#FDE8E8"
      label: "Computational Entropy"
      position: center_right
    
    query:
      nodes: [linguistics, intent]
      color: "#FDFDE8"
      label: "Query Entropy"
      position: right
  
  # Node appearance
  node_style:
    shape: rounded_rectangle
    size_by: network_priority  # Higher priority = larger node
    color_by: entropy_state    # low=green, medium=amber, high=red
    border_by: observation     # Solid=measured, dashed=inferred
    label: sub_dimension_name
    tooltip: |
      Score: {score}
      State: {state} ({observed|inferred})
      Network Priority: {priority}
      Top cause: {top_parent} ({contribution}%)
  
  # Edge appearance
  edge_style:
    width_by: strength         # Stronger dependency = thicker edge
    color_by: active_path      # Gray=inactive, colored=on explanation path
    style: directed_arrow
    opacity_by: relevance      # Dim edges not relevant to current selection
  
  # Interaction
  interactions:
    click_node:
      - Show posterior distribution
      - Show parent contributions
      - Show downstream impact
    
    drag_node_state:
      - "What-if" slider: drag entropy from high → low
      - Real-time update of all downstream posteriors
      - Show delta annotations on affected nodes
    
    hover_edge:
      - Show conditional probability summary
      - Highlight full causal path
    
    right_click_node:
      - "Explain this score" → MPE inference
      - "What fixes this?" → resolution hints
      - "Show entropy objects" → link to raw evidence
```

### 8.2 Dashboard Integration

The graph appears alongside existing entropy dashboards:

```
┌──────────────────────────────────────────────────────────────┐
│  ENTROPY MANAGEMENT DASHBOARD                                 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─ Summary ───────────────────┐  ┌─ Bayesian Network ─────┐ │
│  │ Overall Score: 0.62 (Med)   │  │                         │ │
│  │ Tables Profiled: 12/18      │  │  [Interactive DAG]      │ │
│  │ Query Confidence: 38%       │  │                         │ │
│  │                             │  │  Source → Struct →      │ │
│  │ Top Priority:               │  │  Semantic → Query       │ │
│  │ 1. business_meaning (0.92)  │  │                         │ │
│  │ 2. aggregations (0.78)      │  │  Click any node for     │ │
│  │ 3. provenance (0.71)        │  │  "why" and "what-if"    │ │
│  └─────────────────────────────┘  └─────────────────────────┘ │
│                                                               │
│  ┌─ Resolution Queue ──────────────────────────────────────┐  │
│  │ 1. Define business meaning for 'amount' column          │  │
│  │    Impact: Reduces intent entropy by 33%                │  │
│  │    Cascade: → business_rules → aggregations → intent    │  │
│  │    [Configure] [Dismiss]                                │  │
│  │                                                         │  │
│  │ 2. Document data provenance for financial_transactions  │  │
│  │    Impact: Reduces 5 downstream dimensions              │  │
│  │    Cascade: → business_meaning → units → intent         │  │
│  │    [Configure] [Dismiss]                                │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Implementation Roadmap

### Phase 1: Static Network (Week 1-2)

Build the DAG with expert-defined CPTs. No learning, no dynamic updates.

**Deliverables:**
- `entropy_network/model.py` — DAG definition using pgmpy
- `entropy_network/cpts.py` — Parameterized CPT generation from edge strengths
- `entropy_network/inference.py` — Forward/backward propagation, what-if, MPE
- `entropy_network/config.py` — Load network configuration from YAML
- Unit tests: verify CPTs sum to 1, verify known scenarios produce expected results

**Dependencies:**
- `pgmpy` (Bayesian network library)
- `networkx` (graph operations, already likely in stack)
- `numpy` (matrix operations)

**Validation:**
Create 5-10 synthetic dataset profiles with known entropy states. Verify the network produces sensible inferences (e.g., high provenance entropy → elevated business meaning entropy).

### Phase 2: Integration with Entropy Objects (Week 3-4)

Connect the network to existing entropy detectors.

**Deliverables:**
- `entropy_network/bridge.py` — Convert entropy objects ↔ network evidence
- `entropy_network/priority.py` — Compute network priorities, enhance hints
- `entropy_network/context.py` — Generate query agent context blocks
- Integration tests: end-to-end from detection → network → enhanced objects

### Phase 3: Visualization (Week 4-6)

Build the interactive graph UI.

**Technology considerations:**
- D3.js for custom DAG rendering with force-directed layout
- Or: Cytoscape.js for graph-specific interactions (better for DAGs)
- Or: React Flow for node-based UI (good for the what-if drag interaction)
- Server-side: expose inference API endpoints for real-time what-if queries

**Deliverables:**
- Interactive DAG component with click, drag, hover interactions
- Real-time posterior updates via WebSocket or SSE
- Integration with existing dashboard layout

### Phase 4: Learning (Week 6-8)

Enable CPT learning from profiled datasets.

**Deliverables:**
- `entropy_network/learning.py` — BDeu parameter learning
- Observation storage: append discretized profiles to learning dataset
- Scheduled re-estimation: update CPTs after N new observations
- A/B comparison: show "expert prior" vs "learned" network side by side

### Phase 5: Structure Learning (Future)

Let the network discover new dependencies from data.

**Deliverables:**
- Structure learning using PC or GES algorithms
- Expert-in-the-loop: propose new edges for human approval
- Edge discovery reports: "We observed a strong dependency between X and Y that isn't in the current model"

---

## 10. Technical Notes

### 10.1 Scalability

The 18-node network is small enough for exact inference (Variable Elimination is polynomial for bounded treewidth). No approximation needed. Inference runs in <10ms for any query.

For per-table networks (same structure, different evidence), inference is embarrassingly parallel.

### 10.2 Handling Continuous Scores

If discrete states (low/medium/high) prove too coarse, two upgrade paths:

1. **Finer discretization**: 5 states (very_low, low, medium, high, very_high). CPTs grow but remain tractable.
2. **Gaussian Bayesian Network**: Each node has a continuous Gaussian distribution conditioned linearly on parents. pgmpy supports this via `LinearGaussianCPD`.

Start with 3 states. Upgrade only if users need finer granularity.

### 10.3 Multi-Table Aggregation

The network runs at **table level** by default. For dataset-level or schema-level views, aggregate table-level posteriors:

```python
def aggregate_table_networks(
    table_posteriors: dict[str, dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    """
    Aggregate per-table posteriors to dataset level.
    Uses pessimistic aggregation (max of P(high) across tables).
    """
    dataset_posteriors = {}
    all_nodes = list(next(iter(table_posteriors.values())).keys())
    
    for node in all_nodes:
        p_highs = [
            posteriors[node][2]  # P(high) for this node
            for posteriors in table_posteriors.values()
        ]
        # Dataset entropy is driven by worst table
        dataset_posteriors[node] = max(p_highs)
    
    return dataset_posteriors
```

### 10.4 Relationship to Information-Theoretic Entropy

The "entropy" in entropy sub-dimension scores is conceptually aligned with Shannon entropy but is not computed using H(X) = -Σ p(x) log p(x) directly. The Bayesian Network adds a separate probabilistic layer on top:

- **Sub-dimension scores** (0-1): Domain-specific measurement of ambiguity
- **Network CPDs**: Probabilistic model of how these measurements relate
- **Network posteriors**: Bayesian-updated beliefs incorporating both local evidence and causal structure

This layered approach means you're doing information-theoretic measurement locally and probabilistic inference globally — which is the right decomposition.
