#!/bin/bash
# code-review-graph MCP Server — WSL launcher for Windows-side clients
# Used by: Claude Desktop, VS Code (Windows), any host-side tool
#
# Invoked via:  wsl.exe bash /mnt/c/Users/rahul/Agentic\ NM/start_crg.sh
#
# Requirements: uv / uvx must be installed in WSL
#   curl -LsSf https://astral.sh/uv/install.sh | sh

set -e

PROJECT_DIR="/mnt/c/Users/rahul/Agentic NM"
cd "$PROJECT_DIR"

# Ensure uvx is on PATH (uv installs to ~/.local/bin by default)
export PATH="$HOME/.local/bin:$PATH"

exec uvx code-review-graph serve --repo "$PROJECT_DIR"
