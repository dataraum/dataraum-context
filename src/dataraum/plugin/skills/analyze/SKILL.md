---
description: "Use when the user wants to analyze new data, process a CSV or Parquet file, load a dataset, or run the pipeline on their data. Trigger phrases: 'analyze this data', 'process my CSV', 'load this file', 'analyze the data at', 'run the pipeline', 'import this dataset'."
tools:
  - dataraum:analyze
alwaysApply: false
---

# Analyze Data

Run the DataRaum analysis pipeline on a CSV or Parquet file (or directory of files) using the MCP tool.

## How to Use

This takes **several minutes** and always returns immediately. The pipeline runs in the background.

- **With task support**: progress updates are delivered automatically via the tasks API
- **Without task support**: call `get_context` periodically (~2 min intervals) to check progress — it reports the current phase while running and returns the full context document when done

Call the `analyze` MCP tool:

```
analyze(path="/path/to/data.csv")
analyze(path="/path/to/data.parquet")
analyze(path="/path/to/data/directory")
analyze(path="/path/to/data.csv", name="my_dataset")
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | **Required**: path to CSV/Parquet file or directory of files |
| `name` | string | Optional: name for the data source (defaults to filename) |

### Supported Formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Loaded as VARCHAR, types inferred by pipeline |
| Parquet | `.parquet`, `.pq` | Types preserved from file schema |

## What You Get

After analysis completes, you receive:

- **Tables found**: Names and row counts
- **Phases completed**: How many of the 18 analysis phases succeeded
- **Duration**: How long the analysis took
- **Next steps**: Suggested tools to call next

## What Happens During Analysis

The pipeline runs 18 phases:

1. **Import**: Load raw data into the analysis engine
2. **Typing**: Infer column types (or preserve Parquet types)
3. **Statistics**: Profile column distributions
4. **Correlations**: Detect numeric/categorical correlations
5. **Relationships**: Find join candidates and foreign keys
6. **Semantic Analysis**: Identify business meaning of columns (uses LLM)
7. **Temporal Analysis**: Detect time series patterns
8. **Quality Rules**: Generate validation rules (uses LLM)
9. **Entropy Detection**: Measure data uncertainty across dimensions
10. **Entropy Interpretation**: Generate human-readable quality assessments (uses LLM)
11. **Context Assembly**: Build the full data context document

## Response Pattern

1. Call the `analyze` tool with the user's data path — it returns immediately
2. The pipeline takes several minutes. Call `get_context` periodically (~2 min intervals) to check progress
3. When the pipeline completes, `get_context` returns the full context document
4. Note any phase failures or warnings
5. Suggest next steps:
   - `get_context` to explore the schema
   - `get_entropy` to check data quality
   - `query` to ask questions about the data
