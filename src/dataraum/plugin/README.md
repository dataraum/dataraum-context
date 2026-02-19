# DataRaum Plugin

Data context and quality analysis for knowledge work. Understand your data structure, identify quality issues, and get prioritized actions to fix them.

## How It Works

This plugin connects to a DataRaum API server to analyze your data. It provides two main skills:

1. **Context**: Get schema, relationships, and quality indicators
2. **Actions**: Get prioritized steps to improve data quality

## Quick Start

### 1. Start the DataRaum API

```bash
# Install
pip install dataraum

# Start API server
dataraum-api --output-dir ./pipeline_output --port 8000
```

### 2. Configure the Connector

In your Cowork settings, set the connector:
- `~~dataraum_api~~` → `http://localhost:8000` (or your deployed URL)

### 3. Upload and Analyze

Ask questions like:
- "What tables do I have in my data?"
- "What data quality issues should I fix?"

The plugin will call the API to get answers.

## Skills

### Context
**Trigger phrases:** "what tables do I have", "show me the schema", "describe the data"

Returns:
- Table and column schema
- Semantic annotations
- Relationships between tables
- Quality indicators (ready, investigate, blocked)

### Actions
**Trigger phrases:** "what should I fix", "data quality issues", "improve the data"

Returns:
- Prioritized resolution actions (high/medium/low)
- Effort estimates
- Expected impact
- Quick wins (high priority + low effort)

## For Cowork Cloud

To use this plugin in Cowork cloud (not local Claude Code):

1. **Expose your API** using ngrok or deploy to cloud:
   ```bash
   ngrok http 8000
   ```

2. **Configure connector** with the public URL

3. **Upload your CSV** - the API handles the rest

See [CONNECTORS.md](CONNECTORS.md) for detailed setup instructions.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/upload` | POST | Upload CSV, auto-runs pipeline |
| `/api/v1/context/{source_id}` | GET | Get data context |
| `/api/v1/actions/{source_id}` | GET | Get resolution actions |
| `/api/v1/sources` | GET | List all sources |
| `/health` | GET | Health check |

## Requirements

- DataRaum API server running and accessible
- Connector configured with API URL
