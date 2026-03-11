"""
Few-Shot Example Retriever for Nyaymalaw
=========================================
Serves two purposes:

1. FEW-SHOT LEARNING (immediate) — retrieves the most relevant training
   example for a given client query and injects it into the LLM prompt
   at inference time. The model sees what a perfect response looks like
   for a similar case type without needing to carry all examples in every
   prompt.

2. FINE-TUNING DATA (future) — same JSONL files are the training dataset
   for Unsloth/LoRA fine-tuning. Use convert_to_finetune.py to export.

Retrieval strategy: keyword overlap scoring with case-type boosting.
Simple and fast — no embeddings needed for O(tens of examples).
Upgrade to FAISS/ChromaDB when the example store grows beyond ~200.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

EXAMPLES_DIR = Path(__file__).parent / "examples"

# ── Cache: loaded once per process ──────────────────────────────────────────
_intake_cache: list[dict] | None = None
_opinion_cache: list[dict] | None = None


# ── Loader ───────────────────────────────────────────────────────────────────

def _load(filename: str) -> list[dict]:
    path = EXAMPLES_DIR / filename
    if not path.exists():
        logger.warning("Few-shot examples file not found: %s", path)
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed example at line %d in %s: %s", i, filename, e)
    return out


def _intake_examples() -> list[dict]:
    global _intake_cache
    if _intake_cache is None:
        _intake_cache = _load("intake_conversations.jsonl")
    return _intake_cache


def _opinion_examples() -> list[dict]:
    global _opinion_cache
    if _opinion_cache is None:
        _opinion_cache = _load("final_opinions.jsonl")
    return _opinion_cache


# ── Scoring ──────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    """Simple word tokeniser — lowercase, strip punctuation."""
    return set(re.findall(r"[a-z0-9\u0900-\u097f\u0c00-\u0c7f]+", text.lower()))


def _score(example: dict, query_tokens: set[str]) -> float:
    """
    Score an example against the query.
    - +1.0 per keyword match
    - +2.0 if case_type words all appear in query
    - +0.5 per description word match
    """
    score = 0.0

    # Keyword matches
    for kw in example.get("keywords", []):
        kw_tokens = _tokenise(kw)
        if kw_tokens and kw_tokens.issubset(query_tokens):
            score += 1.0

    # Case-type boost
    case_type_tokens = _tokenise(example.get("case_type", "").replace("_", " "))
    if case_type_tokens and case_type_tokens.issubset(query_tokens):
        score += 2.0

    # Description partial match
    desc_tokens = _tokenise(example.get("description", ""))
    overlap = len(desc_tokens & query_tokens)
    score += overlap * 0.5

    return score


def _best_match(examples: list[dict], query: str) -> dict | None:
    if not examples:
        return None
    query_tokens = _tokenise(query)
    scored = [(ex, _score(ex, query_tokens)) for ex in examples]
    scored.sort(key=lambda x: x[1], reverse=True)
    best, best_score = scored[0]
    if best_score == 0.0:
        return None  # No match — don't inject an irrelevant example
    return best


# ── Formatters ───────────────────────────────────────────────────────────────

def _fmt_sep(label: str) -> str:
    return f"\n{'═' * 60}\n{label}\n{'═' * 60}"


def _format_intake(ex: dict) -> str:
    """
    Format an intake conversation example for injection into ROUTING_GATE2
    or FACT_COLLECTION_SYSTEM.  Shows the model exactly how the advocate
    should speak: empathy first, one question at a time, prayer elicitation,
    amount probing.
    """
    description = ex.get("description", "Example case")
    conversation = ex.get("conversation", [])
    if not conversation:
        return ""

    lines = [
        _fmt_sep(f"REFERENCE EXAMPLE — {description}"),
        "(Mirror this conversation style: empathy first, one question at a time,",
        " prayer elicitation, amount probing. Adapt to the actual client's facts.)",
        "",
    ]
    for turn in conversation:
        role = "Client" if turn.get("role") == "user" else "Advocate"
        lines.append(f"{role}: {turn.get('content', '').strip()}")
        lines.append("")

    lines.append("═" * 60)
    return "\n".join(lines)


def _format_opinion(ex: dict) -> str:
    """
    Format a final opinion example for injection into
    STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT.
    """
    description = ex.get("description", "Example opinion")
    opinion_text = ex.get("opinion_text", "").strip()
    if not opinion_text:
        return ""

    lines = [
        _fmt_sep(f"REFERENCE EXAMPLE — {description}"),
        "(Follow this structure, tone, and format exactly.",
        " Replace all section numbers, case names, and facts with what is in",
        " the RETRIEVED LEGAL MATERIALS above — never copy citations from here.)",
        "",
        opinion_text,
        "",
        "═" * 60,
    ]
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def get_intake_example(query: str) -> str | None:
    """
    Returns a formatted few-shot intake conversation example relevant to
    the query, or None if no match found.

    Inject this at the END of the ROUTING_GATE2_SYSTEM or
    FACT_COLLECTION_SYSTEM prompt, separated by a blank line.

    Args:
        query: The combined user messages so far (used for keyword matching).
    """
    ex = _best_match(_intake_examples(), query)
    if not ex:
        return None
    try:
        return _format_intake(ex)
    except Exception as e:
        logger.warning("Failed to format intake example: %s", e)
        return None


def get_opinion_example(query: str) -> str | None:
    """
    Returns a formatted few-shot final opinion example relevant to the
    query, or None if no match found.

    Inject this at the END of the STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT,
    before the RETRIEVED LEGAL MATERIALS section.

    Args:
        query: The facts_summary or combined conversation text.
    """
    ex = _best_match(_opinion_examples(), query)
    if not ex:
        return None
    try:
        return _format_opinion(ex)
    except Exception as e:
        logger.warning("Failed to format opinion example: %s", e)
        return None


def reload_examples() -> None:
    """Force-reload examples from disk (useful after adding new examples)."""
    global _intake_cache, _opinion_cache
    _intake_cache = None
    _opinion_cache = None
    logger.info("Few-shot example cache cleared — will reload on next request")


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    queries = [
        "My husband left me and not giving maintenance I have two kids",
        "Neighbour built wall inside my property I have sale deed",
        "Terminated from job without notice, gratuity not paid",
        "Cheque bounced my friend gave me for loan repayment",
        "Husband beating me and threw me out of house with my daughter",
    ]

    print("\n=== FEW-SHOT RETRIEVER — SMOKE TEST ===\n")
    for q in queries:
        print(f"QUERY : {q}")
        result = get_intake_example(q)
        if result:
            first_line = result.split("\n")[1] if "\n" in result else result[:80]
            print(f"MATCH : {first_line.strip()}")
        else:
            print("MATCH : (none)")
        print()

    print("Opinion examples:")
    for q in queries[:2]:
        print(f"QUERY : {q}")
        result = get_opinion_example(q)
        if result:
            first_line = result.split("\n")[1] if "\n" in result else result[:80]
            print(f"MATCH : {first_line.strip()}")
        else:
            print("MATCH : (none)")
        print()
