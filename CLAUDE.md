<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes_tool` | Reviewing code changes - gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review - token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |
| `list_flows_tool` | Listing execution flows in the codebase |
| `get_minimal_context_tool` | Ultra-cheap context fetch for quick lookups |
| `find_large_functions_tool` | Spotting oversized functions for refactoring |
| `generate_wiki_tool` | Auto-generate codebase wiki pages |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.

### Graph Stats (as of last build)

- **1 130 nodes** - **14 932 edges** - **201 flows** - version 2.2.2

## Default Chat Behavior

Apply these preferences by default in this repository unless the user explicitly asks otherwise.

### Context Discipline

- Treat the current request as the main source of truth.
- Use the last two user-assistant exchanges as secondary context when relevant.
- Ignore older conversation history unless it is explicitly referenced again or is required to avoid a concrete mistake.
- Do not re-summarize old plans, prior attempts, or unrelated history unless asked.

### Execution Style

- Prefer doing the work over describing the work.
- Avoid verbose intermediate narration, long progress updates, and detailed step-by-step explanations unless the user asks for them.
- For code changes, make the change first, then report completion briefly.

### Response Style

- Keep responses short and practical.
- After completing work, reply with `Done.` followed by a 2-3 line summary of what changed.
- Do not include long implementation details, patch walkthroughs, or file-by-file change logs unless requested.
- Ask clarifying questions only when the risk of guessing is meaningful.
