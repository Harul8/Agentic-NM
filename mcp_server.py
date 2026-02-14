"""
Nyaymalaw MCP Server – Model Context Protocol server exposing legal research tools.

Run with stdio (for Cursor / Claude Desktop):
    python mcp_server.py

Or with SSE (for HTTP clients):
    python mcp_server.py --transport sse --port 8010

Then add to Cursor: Settings → MCP → Add server (stdio or SSE URL).
"""

import json
import os
import sys

# Ensure project root is on path when run as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def _run_legal_research(issue: str) -> dict:
    """Run bare act + case law fusion for a legal issue. Uses existing agents."""
    from agents.Legal_Research.act_case_fusion_agent import fuse_bare_act_and_case_law

    return fuse_bare_act_and_case_law.run(issue=issue)


def _run_legal_opinion(facts_summary: str, bare_act_sections: list = None, case_laws: list = None) -> dict:
    """Generate a legal opinion from facts and optional retrieved materials."""
    from services.response_generator import generate_response

    confirmed = {}
    if bare_act_sections:
        confirmed["bare_acts"] = bare_act_sections
    if case_laws:
        confirmed["case_laws"] = case_laws
    return generate_response(facts_summary, confirmed_materials=confirmed or None)


def main() -> None:
    import argparse
    from mcp.server.fastmcp import FastMCP

    parser = argparse.ArgumentParser(description="Nyaymalaw MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport: stdio for Cursor/Claude Desktop, sse for HTTP",
    )
    parser.add_argument("--port", type=int, default=8010, help="Port for SSE transport")
    args = parser.parse_args()

    app = FastMCP(
        name="Nyaymalaw",
        instructions="Legal research and opinion tools for Indian law (bare acts and case laws).",
        port=args.port,
    )

    @app.tool(
        description="Search Indian legal materials for a given issue. Returns relevant Bare Act sections and Case Laws from the vector store. Use this for legal research on any issue (e.g. contract breach, bail, divorce, IPC sections).",
    )
    def legal_research(issue: str) -> str:
        """Run legal research: retrieve relevant bare act sections and case laws for an issue."""
        try:
            result = _run_legal_research(issue)
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "issue": issue})

    @app.tool(
        description="Generate a legal opinion from a summary of facts. Optionally pass pre-retrieved bare_act_sections and case_laws (lists of dicts) from legal_research to get an opinion grounded in those materials.",
    )
    def get_legal_opinion(
        facts_summary: str,
        bare_act_sections: list = None,
        case_laws: list = None,
    ) -> str:
        """Generate a legal opinion from facts and optional materials."""
        try:
            result = _run_legal_opinion(
                facts_summary,
                bare_act_sections=bare_act_sections or [],
                case_laws=case_laws or [],
            )
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "facts_summary": facts_summary[:200]})

    if args.transport == "stdio":
        app.run(transport="stdio")
    else:
        app.run(transport="sse")


if __name__ == "__main__":
    main()
