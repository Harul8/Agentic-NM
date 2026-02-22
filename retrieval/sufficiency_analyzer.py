"""
Sufficiency Analyzer — LLM-driven gap detection for legal research results.

Instead of hard numeric thresholds (e.g., "≥3 bare acts = sufficient"), this module
asks the LLM to analyze the dispute and determine:
1. What are ALL the legal aspects of this dispute?
2. Which bare act sections cover each aspect?
3. Are there gaps — aspects with no coverage?
4. For each section, are there supporting case laws?

The output is a structured assessment that drives targeted internet search
(only for the gaps, not broadly).

Also provides get_broad_discovery_queries() for web_only flows: when the user
asks for "all acts" / broad discovery with no local results, generates 4–8
broad search queries (e.g. Family laws India Code, Labour laws Telangana).
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts for sufficiency analysis
# ---------------------------------------------------------------------------

SUFFICIENCY_ANALYSIS_PROMPT = """You are a senior Indian advocate analyzing whether retrieved legal materials are sufficient for a client's case.

CASE FACTS:
{facts}

RETRIEVED BARE ACT SECTIONS:
{bare_acts_summary}

RETRIEVED CASE LAWS:
{case_laws_summary}

TASK: Analyze the dispute and determine if the retrieved materials are sufficient.

Step 1: List ALL legal aspects/issues in this dispute (e.g., "tenant's right to deposit refund", "limitation period for recovery", "forum for complaint").

Step 2: For each aspect, check if there is at least one relevant bare act section in the retrieved materials.

Step 3: For each bare act section (or group of related sections), check if there are 2-3 supporting case laws in the retrieved materials.

Step 4: Identify GAPS — aspects not covered, sections without case law support.

OUTPUT FORMAT (valid JSON only):
{{
  "aspects": [
    {{
      "aspect": "description of legal aspect",
      "covered_by_bare_acts": true/false,
      "bare_act_sections": ["Section X of Act Y", ...],
      "has_case_law_support": true/false,
      "case_laws_found": ["Case Name 1", "Case Name 2"]
    }}
  ],
  "gaps": [
    {{
      "aspect": "description of uncovered aspect",
      "missing": "bare_act" or "case_law" or "both",
      "search_query": "specific query to find the missing material"
    }}
  ],
  "overall_sufficient": true/false,
  "confidence": "high" or "medium" or "low"
}}

RULES:
- Be thorough. Don't miss any aspect of the dispute.
- A section is "relevant" only if its ingredients match the facts.
- Don't flag a gap if a section is covered even loosely — only flag genuine absences.
- search_query should be SPECIFIC (e.g., "Section 22 Karnataka Rent Control Act security deposit"), not broad.
- Output ONLY valid JSON. No preamble, no explanation."""


def _summarize_bare_acts(bare_acts: list) -> str:
    """Create a concise summary of retrieved bare act sections for the LLM."""
    if not bare_acts:
        return "NONE FOUND"
    lines = []
    for i, ba in enumerate(bare_acts[:30]):  # cap to avoid prompt overflow
        act = ba.get("act_name", "")
        sec = ba.get("section_number", "")
        title = ba.get("section_title", "")
        text = (ba.get("full_text") or ba.get("text", ""))[:300]
        entry = f"{i+1}. {act}"
        if sec:
            entry += f" Section {sec}"
        if title:
            entry += f" — {title}"
        entry += f"\n   {text}"
        lines.append(entry)
    return "\n".join(lines)


def _summarize_case_laws(case_laws: list) -> str:
    """Create a concise summary of retrieved case laws for the LLM."""
    if not case_laws:
        return "NONE FOUND"
    lines = []
    for i, cl in enumerate(case_laws[:20]):
        name = cl.get("case_name", cl.get("source", "Unknown"))
        court = cl.get("court", "")
        year = cl.get("year", "")
        text = (cl.get("full_text") or cl.get("text", ""))[:300]
        entry = f"{i+1}. {name}"
        if court:
            entry += f" ({court}"
            if year:
                entry += f", {year}"
            entry += ")"
        entry += f"\n   {text}"
        lines.append(entry)
    return "\n".join(lines)


def analyze_sufficiency(
    facts: str,
    bare_acts: list,
    case_laws: list,
    llm_fn=None,
) -> dict:
    """
    Analyze whether retrieved materials are sufficient for the dispute.

    Args:
        facts: Plain language description of the legal dispute
        bare_acts: List of retrieved bare act chunks
        case_laws: List of retrieved case law chunks
        llm_fn: Function to call LLM (default: llm.ollama_client.ask_llm)

    Returns:
        Dict with aspects, gaps, overall_sufficient, confidence
    """
    if llm_fn is None:
        from llm.ollama_client import ask_llm
        llm_fn = ask_llm

    bare_summary = _summarize_bare_acts(bare_acts)
    case_summary = _summarize_case_laws(case_laws)

    prompt = SUFFICIENCY_ANALYSIS_PROMPT.format(
        facts=facts[:2000],
        bare_acts_summary=bare_summary[:3000],
        case_laws_summary=case_summary[:3000],
    )

    try:
        response = llm_fn(prompt)
        # Try to extract JSON from response
        result = _parse_json_response(response)
        if result:
            logger.info(
                f"Sufficiency analysis: overall_sufficient={result.get('overall_sufficient')}, "
                f"gaps={len(result.get('gaps', []))}"
            )
            return result
    except Exception as e:
        logger.error(f"Sufficiency analysis failed: {e}")

    # Fallback: simple heuristic analysis
    return _heuristic_sufficiency(bare_acts, case_laws)


def _parse_json_response(response: str) -> Optional[dict]:
    """Extract and parse JSON from LLM response."""
    if not response:
        return None

    # Try direct parse
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in response
    import re
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _heuristic_sufficiency(bare_acts: list, case_laws: list) -> dict:
    """
    Heuristic fallback when LLM analysis fails.

    Be SKEPTICAL: having chunks in the vector store doesn't mean they're relevant.
    Check that results look like real legal content (have act names, case names, etc.)
    rather than just trusting raw count.
    """
    # Filter to results that look like real legal content
    real_bare_acts = [
        ba for ba in bare_acts
        if ba.get("act_name") and ba.get("section_number")
        and len((ba.get("full_text") or ba.get("text", "")).strip()) > 50
    ]
    real_case_laws = [
        cl for cl in case_laws
        if cl.get("case_name") and cl.get("case_name", "").lower() != "unknown"
        and len((cl.get("full_text") or cl.get("text", "")).strip()) > 50
    ]

    has_bare_acts = len(real_bare_acts) >= 2
    has_case_laws = len(real_case_laws) >= 1

    gaps = []
    if not has_bare_acts:
        gaps.append({
            "aspect": "Insufficient relevant bare act sections",
            "missing": "bare_act",
            "search_query": "relevant Indian bare act sections",
        })
    if not has_case_laws:
        gaps.append({
            "aspect": "Insufficient relevant case laws",
            "missing": "case_law",
            "search_query": "relevant Indian court judgments",
        })

    return {
        "aspects": [],
        "gaps": gaps,
        "overall_sufficient": has_bare_acts and has_case_laws,
        "confidence": "low",  # heuristic = low confidence
    }


def get_targeted_search_queries(sufficiency_result: dict) -> list:
    """
    Extract specific search queries from sufficiency analysis gaps.

    Returns list of dicts: [{"query": "...", "type": "bare_act"|"case_law"|"both"}]
    """
    queries = []
    for gap in sufficiency_result.get("gaps", []):
        query = gap.get("search_query", "").strip()
        missing = gap.get("missing", "both")
        if query:
            queries.append({"query": query, "type": missing})
    return queries


def get_broad_discovery_queries(
    user_request: str,
    legal_query: str,
    document_types: str = "both",
    llm_fn=None,
    intent: dict = None,
) -> list:
    """
    Generate 4–8 broad web search queries for web_only / "all acts" style requests.
    Uses extracted intent (states, domains, topics) when provided—no hardcoded states/domains.

    Returns list of dicts: [{"query": "...", "type": "bare_act"|"case_law"}]
    """
    if llm_fn is None:
        from llm.ollama_client import ask_llm
        llm_fn = ask_llm

    # Build intent block from extracted intent (dynamic)
    intent_block = ""
    if intent and isinstance(intent, dict):
        intent_block = "\nEXTRACTED INTENT (use only these; do not add others):\n" + json.dumps(intent)

    try:
        from prompts.advocate_prompts import BROAD_DISCOVERY_QUERIES_PROMPT
        prompt = BROAD_DISCOVERY_QUERIES_PROMPT.format(
            user_request=(user_request or "")[:1500],
            legal_query=(legal_query or "")[:800],
            intent_block=intent_block,
        )
        response = llm_fn(prompt)
        out = _parse_json_response(response)
        if not out or "queries" not in out:
            raise ValueError("No queries in LLM response")
        raw = out["queries"]
        if not isinstance(raw, list):
            raw = [raw]
        queries = []
        for q in raw[:10]:
            if isinstance(q, dict):
                query = (q.get("query") or "").strip()
                qtype = (q.get("type") or "bare_act").strip().lower()
                if qtype not in ("bare_act", "case_law", "both"):
                    qtype = "bare_act"
                if query:
                    queries.append({"query": query, "type": qtype})
            elif isinstance(q, str) and q.strip():
                queries.append({"query": q.strip(), "type": "bare_act"})
        if not queries:
            raise ValueError("Empty queries list")
        # Respect document_types: filter to bare_act or case_law only if requested
        if document_types == "acts_only":
            queries = [g for g in queries if g.get("type") == "bare_act"]
            if not queries:
                queries = [{"query": (user_request or legal_query)[:200], "type": "bare_act"}]
        elif document_types == "case_laws_only":
            queries = [g for g in queries if g.get("type") == "case_law"]
            if not queries:
                queries = [{"query": (user_request or legal_query)[:200], "type": "case_law"}]
        logger.info("Broad discovery: %d queries for web_only", len(queries))
        return queries[:8]
    except Exception as e:
        logger.warning("Broad discovery LLM failed (%s), retrying with simpler prompt", e)
        # Retry with simpler LLM prompt (model-driven before static fallback)
        try:
            simple_prompt = (
                f"User wants to find Indian acts/laws via web search. Request: {(user_request or legal_query or '')[:400]}\n\n"
                "Output valid JSON only: {\"queries\": [{\"query\": \"search phrase 1\", \"type\": \"bare_act\"}, ...]}\n"
                "Generate 2-4 short web search phrases. Use type \"bare_act\" for acts/laws, \"case_law\" for judgments."
            )
            response = llm_fn(simple_prompt)
            out = _parse_json_response(response)
            if out and out.get("queries"):
                raw = out["queries"] if isinstance(out["queries"], list) else [out["queries"]]
                queries = []
                for q in raw[:8]:
                    if isinstance(q, dict) and (q.get("query") or "").strip():
                        qtype = (q.get("type") or "bare_act").strip().lower()
                        if qtype not in ("bare_act", "case_law"):
                            qtype = "bare_act"
                        queries.append({"query": (q.get("query") or "").strip(), "type": qtype})
                    elif isinstance(q, str) and q.strip():
                        queries.append({"query": q.strip(), "type": "bare_act"})
                if queries:
                    if document_types == "acts_only":
                        queries = [g for g in queries if g.get("type") == "bare_act"] or [{"query": (user_request or legal_query)[:200], "type": "bare_act"}]
                    elif document_types == "case_laws_only":
                        queries = [g for g in queries if g.get("type") == "case_law"] or [{"query": (user_request or legal_query)[:200], "type": "case_law"}]
                    logger.info("Broad discovery retry: %d queries", len(queries))
                    return queries[:8]
        except Exception as e2:
            logger.warning("Broad discovery retry failed (%s), using template fallback", e2)
        # Static fallback when both LLM attempts fail
        combined = (user_request or legal_query or "").strip()[:200]
        fallback = [{"query": f"India Code bare acts {combined}".strip(), "type": "bare_act"}]
        if "state" in (combined or "").lower():
            fallback.append({"query": f"state acts {combined[:120]}".strip(), "type": "bare_act"})
        if document_types == "case_laws_only":
            fallback = [{"query": combined or "Indian court judgments", "type": "case_law"}]
        return fallback[:8]
