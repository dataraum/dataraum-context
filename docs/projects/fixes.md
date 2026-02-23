# Project: Data Fixes

*Reproducible, auditable data corrections that survive re-imports.*

---

## Problem

The plugin can detect data quality issues and suggest actions, but there's no way to actually fix them. When a user resolves an action today, nothing changes in the data. The fix lives only in conversation history and is lost on the next session.

Worse: even if we could apply fixes, they'd need to be re-applied every time the data is re-imported. A fix that isn't reproducible isn't a fix — it's a one-time patch.

## Scope

This project owns:
1. **Persistent state** — DB models for tracking resolutions and quality over time
2. **Resolve workflow** — guided interview that walks users through actions
3. **Fixes ledger** — recorded, replayable operations in three categories
4. **Re-application** — applying fixes automatically during import

---

## 1. Persistent State Layer

Every conversation currently starts from scratch. This is the foundation everything else builds on.

**Implementation:** Use the existing SQLAlchemy database in `pipeline_output/`. Adding tables to the existing DB is consistent with the architecture and gives queryability, transactions, and schema enforcement.

**New DB models:**

```
ResolutionRecord
  resolution_id    UUID PK
  source_id        FK → Source
  action_type      str          # document_unit, investigate_outliers, etc.
  action_key       str          # unique key for the specific action
  status           str          # pending | resolved | skipped | escalated
  answer           JSON         # user's response
  output           JSON         # generated artifact (metadata entry, script, etc.)
  resolved_by      str
  resolved_at      datetime
  session_id       str

QualitySnapshot
  snapshot_id      UUID PK
  source_id        FK → Source
  run_id           FK → PipelineRun
  created_at       datetime
  overall_score    float
  dimension_scores JSON         # {structural: 0.3, semantic: 0.5, ...}
  action_counts    JSON         # {high: 5, medium: 12, low: 8}
  resolved_count   int
```

**New MCP tools:**
- `get_progress` — returns resolution status (resolved/pending/skipped counts), quality score from latest snapshot, last session timestamp
- `save_resolution(action_key, answer, output)` — writes to `ResolutionRecord`

These tools are consumed by the plugin's `progress` and `resolve` skills.

---

## 2. Progress Dashboard Entry Point

Every returning session should open with a status summary, not a blank slate:

> *"Last session (Feb 19): 3 actions resolved, quality score 62% → 71%. 12 actions remaining — 2 high priority. Continue?"*

**Implementation:** New skill `progress/SKILL.md` (`alwaysApply: true`). Uses `get_progress` tool to read resolution status and quality snapshots. If data exists, leads with continuity. If not, guides to `analyze`.

This is the primary retention hook — users come back because they can see momentum.

**New skill:** `progress/SKILL.md`

---

## 3. Guided Fix Workflow (`resolve` skill)

A new skill that walks through the actions list as a structured interview. This is the primary way users create fixes.

```
For each action (sorted: high priority + low effort first):
  → Show: what's wrong, why it matters
  → Optionally: show example rows (plugin's show_examples tool)
  → Ask: the specific question for that action type
  → On answer: produce concrete output
  → Save to ResolutionRecord via save_resolution
  → Move to next
```

**Question library** in SKILL.md maps action types to prompts and output templates:

| Action type | Question pattern | Output |
|---|---|---|
| `document_unit` | "What unit is `{column}` stored in?" | metadata YAML entry → SemanticFix |
| `document_null_semantics` | "When `{column}` is null — not applicable, zero, or missing?" | metadata YAML entry → SemanticFix |
| `document_entity_type` | "What do the numeric codes in `{column}` mean?" | code lookup table → SemanticFix |
| `investigate_*` | "Check rows where `{condition}` — is this expected?" | close action or escalate to transform |
| `transform_*` | "Confirm: cap `{column}` at 99th percentile?" | SQL script → FixScript |
| `create_validation_rule` | "Confirm: validate SUM(Soll) = SUM(Haben) per `{column}`?" | SQL assertion → ClassifiedFix |

Session continuity: picks up from the last unresolved action on next visit (reads `ResolutionRecord.status`).

Each resolution produces a fix record in the ledger — semantic fixes become `SemanticFix` records, transforms become `FixScript` records.

**New skill:** `resolve/SKILL.md`

---

## 4. Fixes Ledger

### Core Principle

Every fix is a **recorded, replayable operation**. The ledger stores:
- What was wrong (the action that triggered the fix)
- What was done (the fix operation, as executable code)
- Why (user's rationale / confirmation)
- When and by whom

### Fix Categories

#### 4a. Reproducible Database Scripts

Deterministic transformations expressed as SQL or DuckDB operations. Same input always produces same output.

**Examples:**
- Date format standardization: `STRFTIME(TRY_CAST(buchungsdatum AS DATE), '%Y-%m-%d')`
- Outlier capping: `LEAST(betrag, PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY betrag))`
- Null replacement: `COALESCE(kost1kostenstelle, 'UNASSIGNED')`
- Type coercion: `CAST(belegnummer AS INTEGER)`
- Deduplication: `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...) = 1`

**Storage:**
```
FixScript
  fix_id           UUID PK
  source_id        FK → Source
  action_key       str          # links back to the triggering action
  category         str          # 'transform_script'
  dialect          str          # 'duckdb_sql' | 'python'
  script           text         # the executable fix
  parameters       JSON         # configurable values (percentile, replacement, etc.)
  description      str          # human-readable summary
  created_by       str
  created_at       datetime
  verified         bool         # has been reviewed and confirmed
```

#### 4b. Semantic Ambiguity Fixes

Metadata corrections that resolve meaning, not data values. These don't change rows but change how the data is interpreted.

**Examples:**
- Column unit declaration: "betrag is in EUR (cents)"
- Null semantics: "NULL in kost1kostenstelle means 'not applicable', not 'missing'"
- Entity type clarification: "buchungsart codes map to: 1=Debit, 2=Credit, 3=Adjustment"
- Column role correction: "belegnummer is an identifier, not a measure"

**Storage:**
```
SemanticFix
  fix_id           UUID PK
  source_id        FK → Source
  action_key       str
  category         str          # 'document_unit', 'document_null_semantics', etc.
  target_column    str          # table.column
  metadata_key     str          # what's being declared
  metadata_value   JSON         # the declaration
  description      str
  created_by       str
  created_at       datetime
```

Semantic fixes feed back into the pipeline: on re-analysis, declared metadata overrides LLM inference. This is how you **pin the semantic layer** and eliminate the LLM reproducibility variance (see Performance Notes below).

#### 4c. Other Classified Fixes

Fixes that don't fit neatly into SQL transforms or metadata:

- **Validation rules**: "SUM(Soll) must equal SUM(Haben) per Buchungskreis" → stored as SQL assertions
- **Filter rules**: "Exclude rows where status = 'DELETED'" → stored as WHERE clauses
- **Derived columns**: "net_amount = brutto - mwst" → stored as column expressions
- **Business rules**: "Fiscal year starts April 1" → stored as configuration

**Storage:**
```
ClassifiedFix
  fix_id           UUID PK
  source_id        FK → Source
  action_key       str
  category         str          # 'validation_rule', 'filter_rule', 'derived_column', 'business_rule'
  definition       JSON         # category-specific structure
  description      str
  created_by       str
  created_at       datetime
```

### Unified Ledger View

All three fix types are unified through the ledger — a single view of everything that's been fixed:

```
FixesLedger (view across all fix tables)
  fix_id
  source_id
  category          # transform_script | document_* | validation_rule | ...
  action_key        # original action that prompted this
  description
  status            # pending | applied | failed | superseded
  applied_at        # when last applied
  applied_run_id    # which pipeline run applied it
  created_by
  created_at
```

---

## 5. Re-application During Import

When new data arrives or the pipeline re-runs:
1. Load the fixes ledger for the source
2. Apply transform scripts in order (SQL scripts against DuckDB)
3. Inject semantic fixes as metadata overrides (bypass LLM inference for declared columns)
4. Evaluate validation rules against the result
5. Report: which fixes applied cleanly, which failed (schema change?), which are new issues

This creates a **reproducible transformation pipeline**: raw data → fixes ledger → clean data.

---

## New MCP Tools

| Tool | Purpose |
|---|---|
| `get_progress` | Read resolution status + quality snapshots |
| `save_resolution` | Write resolution answer + output to DB |
| `save_fix` | Record a fix (any category) to the ledger |
| `list_fixes` | Return all fixes for a source, with status |
| `apply_fixes` | Re-apply all fixes to current data, report results |
| `remove_fix` | Mark a fix as superseded or remove it |

## New Skills

| Skill | Purpose |
|---|---|
| `progress` | Session warm start, shows fix continuity |
| `resolve` | Guided fix interview with question library |

---

## Performance Notes: LLM Reproducibility

From pipeline benchmarks (Feb 2026, 4 runs on same financial dataset):

| Metric | Run 2 | Run 3 | Run 4 |
|---|---|---|---|
| Tables | 42 | 42 | 41 |
| Relationships | 17 | 17 | 17 |
| Critical issues | 33 | 33 | 37 |
| Warnings | 18 | 18 | 25 |
| journal_entries type | FACT | FACT | DIMENSION |
| fx_rates type | DIMENSION | DIMENSION | FACT |

The structural core is rock-solid and fully reproducible. The variance is in three LLM-decided areas:
1. **Entity type classification** (FACT vs DIMENSION) — flips between runs
2. **Slice dimension selection** — swapped `account_type` for `payment_method` between runs
3. **Quality grading** — warning counts vary

**Semantic fixes directly address this**: once a user confirms "journal_entries is a FACT table", that declaration overrides LLM inference on all future runs. Pinning the semantic layer after human review eliminates this variance entirely.

---

## Dependencies

- Pipeline DB infrastructure (already exists: `PipelineRun`, `PhaseCheckpoint`, `Source`, etc.)
- Plugin's `show_examples` tool (for the resolve workflow to surface example rows)
- Pipeline integration for re-application (new: fixes applied during import phase)

## Open Questions

- Should fixes be exportable (YAML/JSON) for sharing between environments?
- How do we handle fix conflicts when schema changes?
- Should there be a "dry run" mode that shows what fixes would do without applying them?
- Fix ordering: some transforms depend on others (e.g., type cast before outlier cap). Do we need explicit ordering or is category-based ordering sufficient?
