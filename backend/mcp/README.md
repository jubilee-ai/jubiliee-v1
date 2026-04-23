# Jubilee MCP server

Single-tool MCP (`assign_task`) that delegates to the same code path as `POST /api/chat`
([`backend/chat/service.py`](../chat/service.py)), so tool inventory and routing stay aligned
with the in-app Jubilee agent.

## Run locally (stdio)

From the repository root, with `.env` configured like the backend API:

```bash
pip install -r requirements.txt
python -m backend.mcp.server
```

Add to Cursor MCP settings (example):

```json
{
  "mcpServers": {
    "jubilee": {
      "command": "python",
      "args": ["-m", "backend.mcp.server"],
      "cwd": "/absolute/path/to/jubiliee-v1",
      "env": {
        "OPENAI_API_KEY": "...",
        "DATABASE_URL": "..."
      }
    }
  }
}
```

## Tool: `assign_task`

- **`message`**: user task (same as chat `message`).
- **`options_json`**: JSON object with optional fields matching
  [`ChatRequest`](../chat/schemas.py): `experiment_id`, `linked_datasets`,
  `model_preference`, `mode`, `conversation`, `chat_thread_id`, `org_id`,
  `background_intake`, `resume_training`, `resume`, `force_orchestrator`.

The tool returns a JSON string with `thread_id`, `summary` (assistant text,
tool trace, errors, training log), and `ok`.

## Railway

Use the same environment variables as the main backend (database, OpenAI,
optional Clerk org if your `chat` paths require it). Start command:

```bash
python -m backend.mcp.server
```

See [`deploy/mcp/Dockerfile`](../../deploy/mcp/Dockerfile) for an image that runs
the MCP process instead of uvicorn.

## Tests

Set `JUBILEE_MCP_SKIP_DB=1` to skip `init_db()` when importing the server in
lightweight test environments.
