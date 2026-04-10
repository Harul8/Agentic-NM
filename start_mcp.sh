#!/bin/bash
# Nyaymalaw MCP Server — WSL launcher for Claude Desktop
# Claude Desktop (Windows) invokes this via: wsl.exe bash /path/to/start_mcp.sh

set -e

PROJECT_DIR="/mnt/c/Users/rahul/Agentic NM"
cd "$PROJECT_DIR"
source .venv/bin/activate

# ── Required ──────────────────────────────────────────────────────────────────
export OPENAI_API_KEY="sk-YOUR_KEY_HERE"

# ── Optional: LangSmith tracing (leave commented out to keep inert) ───────────
# export LANGCHAIN_TRACING_V2="true"
# export LANGCHAIN_API_KEY="ls-YOUR_LANGSMITH_KEY"
# export LANGCHAIN_PROJECT="nyaymalaw"

# ── Start the MCP server on stdio transport ───────────────────────────────────
exec python mcp_server.py --transport stdio
