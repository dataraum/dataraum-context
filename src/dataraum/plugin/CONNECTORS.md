# Connectors

## How tool references work

Plugin files use `~~category~~` as a placeholder for whatever tool the user connects in that category. Plugins are tool-agnostic — they describe workflows in terms of categories rather than specific products.

## Connectors for this plugin

| Category | Placeholder | Description |
|----------|-------------|-------------|
| DataRaum API | `~~dataraum_api~~` | The base URL of the DataRaum API server |

## Setup

This plugin requires a running DataRaum API server that the skills can call via WebFetch.

### Option 1: Self-Hosted API

Run the DataRaum API server locally or on your infrastructure:

```bash
# Install DataRaum
pip install dataraum

# Start the API server
dataraum-api --output-dir ./pipeline_output --port 8000

# The API will be available at http://localhost:8000
```

Then configure the connector:
- `~~dataraum_api~~` → `http://localhost:8000`

### Option 2: Expose via ngrok (for Cowork cloud access)

If using Cowork cloud, expose your local API:

```bash
# Start the API
dataraum-api --output-dir ./pipeline_output --port 8000

# In another terminal, expose via ngrok
ngrok http 8000

# Use the ngrok URL as your connector
# ~~dataraum_api~~ → https://xxxx.ngrok.io
```

### Option 3: Deploy to Cloud

Deploy the DataRaum API to a cloud provider:
- Fly.io
- Railway
- AWS/GCP/Azure
- Docker on any VPS

## API Endpoints Used

The skills call these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/upload` | POST | Upload CSV and run pipeline |
| `/api/v1/context/{source_id}` | GET | Get data context |
| `/api/v1/actions/{source_id}` | GET | Get resolution actions |
| `/api/v1/sources` | GET | List sources |
| `/health` | GET | Health check |

## File Upload Flow

When a user provides a CSV file:

1. Upload to `/api/v1/upload` (multipart form)
2. Pipeline runs automatically
3. Returns `source_id` for subsequent queries
4. Use `source_id` with `/context` and `/actions` endpoints
