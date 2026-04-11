#!/bin/bash
# code-review-graph detect-changes wrapper for host-side clients via WSL

set -e

PROJECT_DIR="/mnt/c/Users/rahul/Agentic NM"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$PATH"

exec uvx code-review-graph detect-changes --repo "$PROJECT_DIR" --brief
