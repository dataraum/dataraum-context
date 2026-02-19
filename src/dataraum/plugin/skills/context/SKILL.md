---
description: "Use when the user asks about data schema, table structure, column types, relationships between tables, or needs to understand what data is available before analysis. Trigger phrases: 'what tables do I have', 'show me the schema', 'data context', 'what columns are in', 'describe the data', 'what data is available'."
tools:
  - WebFetch
alwaysApply: false
---

# Data Context

Retrieve comprehensive context about a dataset from the DataRaum API.

## How to Use

Call the DataRaum API to get the data context:

```
GET ~~dataraum_api~~/api/v1/context/{source_id}
```

The response includes a `prompt_text` field optimized for LLM consumption.

## API Response Structure

```json
{
  "source_id": "...",
  "tables": [...],
  "relationships": [...],
  "entropy_summary": {...},
  "prompt_text": "..."  // Use this for your response
}
```

## What You Get

The context document includes:

- **Schema Information**: Tables, columns, and their data types
- **Semantic Annotations**: What each column represents (identifiers, measures, dimensions, timestamps)
- **Entity Types**: Business concepts like customer, order, product, transaction
- **Relationships**: Foreign keys and join candidates between tables with confidence scores
- **Entropy Summary**: Overall data readiness status (ready, investigate, or blocked)
- **Quality Indicators**: Per-column and per-table quality assessments

## Understanding the Output

### Readiness Levels

| Status | Meaning |
|--------|---------|
| **ready** | Safe for analysis with high confidence |
| **investigate** | Review assumptions before using |
| **blocked** | Needs remediation before reliable analysis |

### Semantic Roles

- **identifier**: Primary keys, unique identifiers
- **foreign_key**: References to other tables
- **measure**: Numeric values for aggregation
- **dimension**: Categorical values for grouping
- **temporal**: Dates and timestamps
- **attribute**: Descriptive fields

## If No Data Exists Yet

If the API returns 404 or "No tables found", the user needs to:
1. Upload their CSV file first via `POST ~~dataraum_api~~/api/v1/upload`
2. The pipeline will run automatically
3. Then call this endpoint with the returned source_id

## Response Pattern

1. Call the context API endpoint
2. Extract the `prompt_text` from the response
3. Summarize the tables and their row counts
4. Highlight key relationships
5. Note any quality concerns (investigate/blocked status)
6. Suggest what questions the data can answer
