# Project: Data Fixes

*Reproducible, auditable data corrections that survive re-imports.*

---

## Problem

The plugin can detect data quality issues and suggest actions, but there's no way to actually fix them. When a user resolves an action today, nothing changes in the data. The fix lives only in conversation history and is lost on the next session.

Worse: even if we could apply fixes, they'd need to be re-applied every time the data is re-imported. A fix that isn't reproducible isn't a fix — it's a one-time patch.

## Core Principle: Fixes Ledger

Every fix is a **recorded, replayable operation**. The ledger stores:
- What was wrong (the action that triggered the fix)
- What was done (the fix operation, as executable code)
- Why (user's rationale / confirmation)
- When and by whom

Fixes can be re-applied automatically during import, creating an auditable transformation pipeline from raw data to clean data.

## Fix Categories

### 1. Reproducible Database Scripts

Deterministic transformations expressed as SQL or DuckDB operations. These are the cleanest fixes — same input always produces same output.

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

### 2. Semantic Ambiguity Fixes

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

### 3. Other Classified Fixes

Catch-all for fixes that don't fit neatly into SQL transforms or metadata:

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

## The Fixes Ledger

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

### Re-application During Import

When new data arrives or the pipeline re-runs:
1. Load the fixes ledger for the source
2. Apply transform scripts in order (SQL scripts against DuckDB)
3. Inject semantic fixes as metadata overrides (bypass LLM inference for declared columns)
4. Evaluate validation rules against the result
5. Report: which fixes applied cleanly, which failed (schema change?), which are new issues

This creates a **reproducible transformation pipeline**: raw data → fixes ledger → clean data.

## New MCP Tools

| Tool | Purpose |
|---|---|
| `save_fix` | Record a fix (any category) to the ledger |
| `list_fixes` | Return all fixes for a source, with status |
| `apply_fixes` | Re-apply all fixes to current data, report results |
| `remove_fix` | Mark a fix as superseded or remove it |

## Integration with Plugin

The `resolve` skill (from the plugin roadmap) is the primary entry point for creating fixes:
- User walks through actions via the guided interview
- Each resolution produces a fix record in the ledger
- Transform fixes generate SQL scripts; semantic fixes generate metadata entries

The `compare` skill (from the plugin roadmap) surfaces fix effectiveness:
- "3 fixes applied successfully, quality score improved 62% → 71%"
- "1 fix failed: column 'betrag' no longer exists in new data"

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

## Dependencies

- Persistent state layer (DB infrastructure)
- `resolve` skill (primary way users create fixes)
- Pipeline integration (applying fixes during re-analysis)

## Open Questions

- Should fixes be exportable (YAML/JSON) for sharing between environments?
- How do we handle fix conflicts when schema changes?
- Should there be a "dry run" mode that shows what fixes would do without applying them?
- Fix ordering: some transforms depend on others (e.g., type cast before outlier cap). Do we need explicit ordering or is category-based ordering sufficient?
