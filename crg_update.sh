#!/bin/bash
# code-review-graph update wrapper for host-side clients via WSL

set -e

PROJECT_DIR="/mnt/c/Users/rahul/Agentic NM"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$PATH"

exec uvx code-review-graph update --repo "$PROJECT_DIR" --skip-flows
