---
description: "Use when the user asks about data quality issues, how to fix data problems, improvement recommendations, or resolution actions. Trigger phrases: 'fix data quality', 'improve data', 'resolution actions', 'what's wrong with the data', 'data issues', 'how do I fix', 'what should I fix first'."
tools:
  - WebFetch
alwaysApply: false
---

# Resolution Actions

Get prioritized actions to improve data quality from the DataRaum API.

## How to Use

Call the DataRaum API to get resolution actions:

```
GET ~~dataraum_api~~/api/v1/actions/{source_id}
```

Optional query parameters:
- `priority`: Filter by "high", "medium", or "low"
- `table_name`: Filter to actions affecting a specific table

The response includes a `prompt_text` field optimized for LLM consumption.

## API Response Structure

```json
{
  "source_id": "...",
  "actions": [...],
  "summary": {"high": N, "medium": N, "low": N},
  "prompt_text": "..."  // Use this for your response
}
```

## What You Get

Resolution actions are concrete steps to fix data issues:

- **Priority**: high, medium, or low based on impact
- **Effort**: low, medium, or high (time/complexity to implement)
- **Affected Columns**: Which data is impacted
- **Expected Impact**: What will improve after fixing
- **Parameters**: Actionable details (thresholds, strategies, targets)

## Action Types

| Prefix | Meaning |
|--------|---------|
| `document_` | Add documentation or metadata |
| `investigate_` | Requires human review |
| `transform_` | Data transformation needed |
| `create_` | Create new artifact |

## Priority Levels

| Level | When to Address |
|-------|-----------------|
| **HIGH** | Fix immediately - blocking reliable analysis |
| **MEDIUM** | Address soon - affects confidence in results |
| **LOW** | Nice to have - minor improvements |

## Quick Wins

Actions marked as **high priority + low effort** are quick wins. Start with these for immediate impact with minimal work.

## If No Data Exists Yet

If the API returns 404 or "No tables found", the user needs to:
1. Upload their CSV file first via `POST ~~dataraum_api~~/api/v1/upload`
2. The pipeline will run automatically
3. Then call this endpoint with the returned source_id

## Response Pattern

1. Call the actions API endpoint
2. Extract the `prompt_text` from the response
3. Summarize the counts by priority
4. Present the top high-priority actions
5. Highlight any quick wins (high priority + low effort)
6. Suggest a remediation order
