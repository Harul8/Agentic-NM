"""
Response Generator — BACKWARD-COMPAT WRAPPER.

All logic now lives in:
  - services.response_generator_v2  (pipeline orchestrator)
  - retrieval.hybrid_retriever      (FAISS + BM25 + cross-encoder)
  - retrieval.tiered_search          (4-tier internet search)
  - retrieval.auto_enricher          (PDF save + incremental indexing)
  - retrieval.sufficiency_analyzer   (LLM gap detection)

This file only re-exports `generate_response()` with the old signature
so that api_server.py and interactive_chat.py keep working until they
are migrated to import directly from v2.
"""

from services.response_generator_v2 import generate_response_v2


def generate_response(
    facts_summary: str,
    confirmed_materials: dict = None,
    top_k: int = 5,
    intent: str = "legal_opinion",
) -> dict:
    """
    Backward-compatible wrapper around generate_response_v2.

    Old callers pass (facts_summary, confirmed_materials, top_k, intent).
    v2 expects   (facts_summary, jurisdiction_state, intent, confirmed_materials).
    """
    return generate_response_v2(
        facts_summary=facts_summary,
        jurisdiction_state="",
        intent=intent,
        confirmed_materials=confirmed_materials,
    )


# Re-export expand_legal_query for any old imports
def expand_legal_query(facts: str) -> str:
    """Delegate to v2's query expansion."""
    from llm.ollama_client import ask_llm
    from prompts.advocate_prompts import EXPAND_LEGAL_QUERY_SYSTEM

    prompt = f"""{EXPAND_LEGAL_QUERY_SYSTEM}

FACTS:
{facts[:1500]}

Query:"""
    try:
        return ask_llm(prompt).strip()[:500] or facts[:300]
    except Exception:
        return facts[:300]
