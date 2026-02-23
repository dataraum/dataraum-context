# Project: Onboarding

*First-run experience: identify the user, configure data sources, and set the stage for everything else.*

---

## Problem

Today, the first interaction with the plugin is either "give me a file path" or nothing. There's no guided setup, no role detection, and no way to manage data sources after initial analysis. Users who aren't the developer have no idea what to do.

## Scope

This project covers the first-run and returning-user setup flow:

1. **Role identification** — who is using this and what do they care about?
2. **Source configuration** — what data are we working with?
3. **Source management** — adding, removing, re-analyzing sources over time

## 1. Role Identification

Ask once at the start of the first session, persist the answer.

> *"What's your role? (a) Accountant / Controller, (b) Data Engineer, (c) Business Analyst, (d) Manager / Executive"*

### Why it matters

The role determines how the entire plugin communicates:

| Role | Leads with | Language style |
|---|---|---|
| **Accountant / Controller** | Double-entry balance checks, VAT key gaps, period closing readiness | Accounting terminology, regulatory framing |
| **Data Engineer** | Null handling, join paths, derived column formulas, transform scripts | Technical, schema-focused |
| **Business Analyst / Controller** | Cost center completeness, aggregation readiness, KPI reliability | Business metrics, reporting confidence |
| **Manager / Executive** | Overall quality grade, contract compliance, what's safe to report on | High-level, risk-focused |

### Storage

Store in the plugin's persistent state (SQLite DB in `pipeline_output/`):

```
UserPreference
  preference_id    UUID PK
  source_id        FK → Source (nullable, for source-specific prefs)
  key              str          # "user_role", "language", etc.
  value            JSON
  created_at       datetime
  updated_at       datetime
```

### Integration points

- `get_actions` tool: role-aware sorting via weight multiplier in `merge_actions()`
- `actions/SKILL.md`: adapt language and emphasis based on role
- `entropy/SKILL.md`: surface dimensions most relevant to the role
- `contracts/SKILL.md`: recommend contracts relevant to the role (e.g., `regulatory_reporting` for accountants)

## 2. Source Configuration

### First-run: auto-detect and confirm

When no data has been analyzed:
1. Scan the workspace folder for `.csv` and `.parquet` files
2. Present found files and ask which to analyze
3. Allow naming the source (default: filename)
4. Fall back to manual path only if nothing found

This is partly a skill-level change (instruct Claude to look before asking) and partly a `list_sources` tool that returns configured sources.

### Multi-source support

The pipeline already supports multiple sources via `Source` records. What's missing:

- **`list_sources` MCP tool** — return all configured sources with status (analyzed, stale, running)
- **`add_source` MCP tool** — register a new source path without immediately running the pipeline
- **Source selection** — skills need to know which source the user is asking about when multiple exist
- **Skill prompt updates** — every skill should handle the multi-source case gracefully

### Source lifecycle

| State | Meaning |
|---|---|
| `configured` | Path registered, not yet analyzed |
| `analyzed` | Pipeline completed at least once |
| `stale` | Source file modified since last analysis |
| `error` | Last pipeline run failed |

Detecting staleness: compare file mtime against `PipelineRun.started_at`.

## 3. Adding Sources Later

Users should be able to add new sources at any time, not just during onboarding:

- "Analyze this new file too" → `add_source` + `analyze`
- "What sources do I have?" → `list_sources`
- "Remove the test data" → `remove_source` (mark as archived, don't delete DB records)

## New MCP Tools

| Tool | Purpose |
|---|---|
| `list_sources` | Return configured sources with status |
| `add_source` | Register a new source path |
| `remove_source` | Archive a source |

## New / Updated Skills

| Skill | Change |
|---|---|
| `analyze` | Auto-detect files in workspace, handle multi-source |
| All skills | Check for role preference and adapt |

## Dependencies

- Persistent state layer (DB models for `UserPreference`)
- This project should be implemented before role-aware action ordering

## Open Questions

- Should role be per-workspace or per-source? (Probably per-workspace)
- Should we support changing role mid-session? (Yes, but don't prompt — just let the user say "I'm actually an engineer")
- How do we handle team scenarios where multiple people use the same workspace? (Defer — single user for now)
