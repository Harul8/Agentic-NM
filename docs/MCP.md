# Nyaymalaw MCP Server

This project exposes **Model Context Protocol (MCP)** tools so Cursor, Claude Desktop, or other MCP clients can use Nyaymalaw’s legal research and opinion features.

## Tools

| Tool | Description |
|------|-------------|
| **legal_research** | Search Indian legal materials (Bare Acts + Case Laws) for a given issue. Returns relevant sections from the vector store. |
| **get_legal_opinion** | Generate a legal opinion from a facts summary. Optionally pass `bare_act_sections` and `case_laws` from `legal_research` for a grounded opinion. |

## Run the server

Use the same Python environment as the rest of the app (e.g. `ai-gpu` venv with dependencies installed).

**Stdio (recommended for Cursor / Claude Desktop):**

```bash
python mcp_server.py
```

**SSE (HTTP) on port 8010:**

```bash
python mcp_server.py --transport sse --port 8010
```

## Use in Cursor

1. Open **Cursor Settings** → **MCP** (or **Features** → **MCP**).
2. Add a new MCP server.

**Option A – Stdio (local process):**

Add to your Cursor MCP config (e.g. `%APPDATA%\Cursor\User\globalStorage\cursor.mcp\mcp.json` on Windows, or via Settings UI):

```json
{
  "mcpServers": {
    "nyaymalaw": {
      "command": "python",
      "args": ["c:\\Users\\rahul\\Nyaymalaw 3.0\\mcp_server.py"],
      "cwd": "c:\\Users\\rahul\\Nyaymalaw 3.0",
      "env": {}
    }
  }
}
```

Use the path to your project and ensure `python` is the venv interpreter (e.g. `c:\Users\rahul\Nyaymalaw 3.0\ai-gpu\Scripts\python.exe`) if you use a virtualenv.

**Option B – SSE (if you run the server with `--transport sse`):**

```json
{
  "mcpServers": {
    "nyaymalaw": {
      "url": "http://127.0.0.1:8010/sse"
    }
  }
}
```

Restart Cursor or reload MCP after changing config. You can then ask the AI to use “legal_research” or “get_legal_opinion” in the chat.

## Dependencies

- `mcp>=1.23.0` and `anyio` are in `requirements.txt`. Install with:  
  `pip install -r requirements.txt`  
  (or use your existing `ai-gpu` venv.)

- The MCP server uses the same data (vector store, Bare Acts) and LLM (Ollama) as the main app. Ensure:
  - `data/vector_store` is populated (indexed bare acts and case laws).
  - Ollama is running if you use `get_legal_opinion` (response generator uses the LLM).
