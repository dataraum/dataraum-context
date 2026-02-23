# DataRaum Plugin — Product Roadmap

*Based on product review session, Feb 2026. Revised Feb 23, 2026.*

---

## Overview

This roadmap covers the **MCP plugin** — the Claude Desktop integration. It is one of several projects. Items that belong elsewhere are scoped into their own project definitions.

**Current state:** 6 MCP tools (`analyze`, `get_context`, `get_entropy`, `evaluate_contract`, `query`, `get_actions`) and 6 skills. The `analyze` tool returns immediately with background execution; `get_context` polls for progress.

### Related Projects

| Project | Scope | Definition |
|---|---|---|
| [Onboarding](projects/onboarding.md) | Role identification, source configuration, multi-source management | Separate project |
| [Data Fixes](projects/fixes.md) | Reproducible fix scripts, semantic corrections, fixes ledger, re-application during import | Separate project |
| [Quality UI](projects/quality-ui.md) | Web dashboard, report export, trend charts, fix review — lives in `dataraum-ui` | Separate project |
| [Incremental Imports](projects/incremental-imports.md) | Change detection, partial re-analysis, "what changed?" reporting | Separate project |

---

## Pipeline Performance Baseline

From benchmarks (Feb 2026, financial dataset, 41 tables):

| Phase group | Time | % of total |
|---|---|---|
| Structural (import → correlations) | 4.7s | 0.7% |
| semantic | 84s | 12% |
| slicing + slice_analysis | 17s | 2.5% |
| quality_summary | 51s | 7.4% |
| **entropy_interpretation** | **539s** | **78%** |
| **Total** | **693s (~11.5 min)** | **100%** |

The structural core is fully reproducible across runs. LLM-decided areas (entity type classification, slice dimension selection, quality grading) vary between runs. Pinning the semantic layer after human review via the [fixes ledger](projects/fixes.md) eliminates this variance.

Note: LLM call counters show 0 in phase metadata — counters may not be wired to actual API calls yet.

---

## Phase 1 — Pre-Release Fixes

*Must-haves before any external user touches this*

### 1.1 Fix the analyze UX (timeout problem) — DONE

**Status:** Implemented.

- `analyze` returns immediately. Pipeline runs in background via `asyncio.create_task(asyncio.to_thread(...))`.
- `get_context` checks `PipelineRun.status`. If `status == "running"`, returns progress (completed/total phases, currently running phase names) instead of partial context.
- MCP Task API path preserved (`taskSupport="optional"`) for future client support.
- Tool descriptions and `analyze/SKILL.md` document the polling pattern.

### 1.2 Auto-detect the file path

Path discovery currently requires the user to provide their full system path.

**Implementation:** Skill-level change only. Update `analyze/SKILL.md` to instruct Claude to scan the workspace folder for `.csv`/`.parquet` files before asking for a path. Claude already has file system access — the skill just needs to direct it to look first, ask second.

**Effort:** Low

### 1.3 Persistent state layer

Every conversation starts from scratch. The plugin needs session continuity.

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

**New MCP tool:** `get_progress` — returns resolution status, quality score, last session timestamp.

**Effort:** Medium

### 1.4 Improve output formatting

The current Markdown output is functional but dense. Since Claude Desktop renders Markdown (not HTML), the improvement is better-structured Markdown.

**Implementation:** Update `mcp/formatters.py`:
- **Actions**: Group by type prefix (`document_` / `investigate_` / `transform_`), priority markers, effort inline
- **Context**: Compact tables, quality inline markers
- **Entropy**: Letter grades (A–D) alongside scores, dimension one-liners
- **Contracts**: Lead with pass/fail verdict and score, then details

HTML output belongs in the [Quality UI](projects/quality-ui.md) project.

**Effort:** Low-Medium

---

## Phase 2 — Core Workflow

*Make the plugin useful for non-developers*

### 2.1 Show example bad rows

Actions describe problems statistically. Users need to see actual data.

**Implementation:** New MCP tool `show_examples(action_key, table_name, limit=5)`. Executes DuckDB query to return rows matching the quality issue condition. The action's `parameters` dict already contains column names and thresholds — these translate to SQL WHERE clauses.

**Effort:** Low-Medium

### 2.2 Guided fix workflow (`resolve` skill)

A new skill that walks through the actions list as a structured interview:

```
For each action (sorted: high priority + low effort first):
  → Show: what's wrong, why it matters
  → Optionally: show example rows (2.1)
  → Ask: the specific question for that action type
  → On answer: produce concrete output
  → Save to ResolutionRecord
  → Move to next
```

**Question library** in SKILL.md maps action types to prompts and output templates:

| Action type | Question pattern | Output |
|---|---|---|
| `document_unit` | "What unit is `{column}` stored in?" | metadata YAML entry |
| `document_null_semantics` | "When `{column}` is null — not applicable, zero, or missing?" | metadata YAML entry |
| `document_entity_type` | "What do the numeric codes in `{column}` mean?" | code lookup table |
| `investigate_*` | "Check rows where `{condition}` — is this expected?" | close action or escalate |
| `transform_*` | "Confirm: cap `{column}` at 99th percentile?" | transform script |
| `create_validation_rule` | "Confirm: validate SUM(Soll) = SUM(Haben) per `{column}`?" | SQL validation |

Session continuity: picks up from the last unresolved action on next visit.

Resolutions feed into the [fixes ledger](projects/fixes.md) — semantic fixes become `SemanticFix` records, transforms become `FixScript` records.

**New MCP tool:** `save_resolution(action_key, answer, output)`
**New skill:** `resolve/SKILL.md`
**Depends on:** 1.3, 2.1
**Effort:** Medium-High

---

## Phase 3 — Continuity

*Give users a reason to come back*

### 3.1 Progress dashboard entry point

Every returning session opens with a status summary:

> *"Last session (Feb 19): 3 actions resolved, quality score 62% → 71%. 12 actions remaining — 2 high priority. Continue?"*

**Implementation:** New skill `progress/SKILL.md` (`alwaysApply: true`). Uses `get_progress` tool to read resolution status and quality snapshots.

**Depends on:** 1.3
**Effort:** Low

### 3.2 Compare runs (`compare` skill)

When the pipeline re-runs, diff against the previous run:
- Quality scores: improved / regressed / unchanged
- Actions: resolved in source data / new issues
- Row count / schema changes

**Implementation:** New MCP tool `compare_runs(run_id_a?, run_id_b?)`. `PipelineRun`, `PhaseCheckpoint`, and `QualitySnapshot` already store the needed data.

**New skill:** `compare/SKILL.md`
**Depends on:** 1.3
**Effort:** Medium

---

## Implementation Summary

### New MCP Tools (this project)

| Tool | Purpose | Phase | Effort |
|---|---|---|---|
| `get_progress` | Resolution status + quality snapshots | 1.3 | Medium |
| `show_examples` | N real rows matching a quality issue | 2.1 | Low |
| `save_resolution` | Write resolution to DB | 2.2 | Low |
| `compare_runs` | Diff two pipeline runs | 3.2 | Medium |

### New / Updated Skills

| Skill | Change | Phase | Effort |
|---|---|---|---|
| `analyze` | Path auto-detect (DONE: timeout fix) | 1.2 | Low |
| `actions` | Better Markdown grouping | 1.4 | Low |
| `context` | Compact format, inline markers | 1.4 | Low |
| `entropy` | Letter grades, dimension summaries | 1.4 | Low |
| `contracts` | Lead with verdict, compact layout | 1.4 | Low |
| `resolve` *(new)* | Guided fix interview | 2.2 | Medium-High |
| `progress` *(new)* | Session warm start | 3.1 | Low |
| `compare` *(new)* | Run delta analysis | 3.2 | Medium |

### Recommended Build Order

| Step | Item | What it unlocks | Effort |
|---|---|---|---|
| 1 | 1.2 Auto-detect path | Remove friction for new users | Low |
| 2 | 1.4 Formatter improvements | Cleaner output for all tools | Low-Medium |
| 3 | 2.1 Show example rows | Users see real data behind issues | Low-Medium |
| 4 | 1.3 Persistent state (DB) | Foundation for session continuity | Medium |
| 5 | 3.1 Progress dashboard | Warm start on every session | Low |
| 6 | 2.2 Resolve workflow | Users make progress on fixes | Medium-High |
| 7 | 3.2 Compare runs | Users see improvement over time | Medium |

Steps 1–3 are independent. Steps 4–5 form a pair. Step 6 is the highest-value feature. Step 7 follows naturally.

Items that were previously in this roadmap but belong in other projects:
- Role/profession onboarding → [Onboarding](projects/onboarding.md)
- Apply transforms / save cleaned dataset → [Data Fixes](projects/fixes.md)
- HTML output / quality dashboard → [Quality UI](projects/quality-ui.md)
- Quality score trend charts → [Quality UI](projects/quality-ui.md)
- "What changed in my data?" → [Incremental Imports](projects/incremental-imports.md)
