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

Retrieval strategy
------------------
Keyword overlap + legal-domain detection + bare-act section number matching.
Simple and fast — no embeddings needed for O(~100) examples.
Upgrade to FAISS/ChromaDB if the store grows beyond ~500 examples.

Schema support
--------------
Handles TWO record formats automatically:

  OLD format  (intake_conversations.jsonl / final_opinions.jsonl)
    keys: id, case_type, keywords, description, conversation / opinion_text

  NEW format  (rich_cases/rich_training_records.jsonl)
    keys: id, case_type, keywords, description, legal_domain, dispute_type,
          relevant_bare_acts, client_scenario, intake_conversation,
          advocate_reasoning, final_opinion

Both are normalised into the same internal shape by _normalise().
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

EXAMPLES_DIR = Path(__file__).parent / "examples"
RICH_FILE    = EXAMPLES_DIR / "rich_cases" / "rich_training_records.jsonl"

# ── Cache ────────────────────────────────────────────────────────────────────
_rich_cache:   list[dict] | None = None   # normalised rich records
_legacy_cache: list[dict] | None = None   # normalised legacy records


# ─────────────────────────────────────────────────────────────────────────────
# Schema normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(raw: dict) -> dict:
    """
    Convert any raw record (old or new schema) into a common internal dict:

      id            str
      case_type     str
      keywords      list[str]
      description   str
      legal_domain  str   ("" for legacy records)
      dispute_type  str   ("" for legacy records)
      bare_act_sections  list[str]   e.g. ["Section 138", "Section 125"]
      conversation  list[dict]   role/content turns
      opinion_text  str
      client_scenario dict  (empty dict for legacy records)
      advocate_reasoning dict (empty dict for legacy records)
    """
    # ── Conversation turns ──────────────────────────────────────────────────
    # New schema nests them at intake_conversation.conversation
    conversation = (
        raw.get("intake_conversation", {}).get("conversation")
        or raw.get("conversation")
        or []
    )

    # ── Opinion text ────────────────────────────────────────────────────────
    opinion_text = (
        raw.get("final_opinion", {}).get("opinion_text")
        or raw.get("opinion_text")
        or ""
    )

    # ── Bare act section numbers ─────────────────────────────────────────────
    # Collect from relevant_bare_acts[].sections for domain-signal boosting
    bare_act_sections: list[str] = []
    for ba in raw.get("relevant_bare_acts", []):
        for sec in ba.get("sections", []):
            bare_act_sections.append(sec.lower())

    # ── Keywords: merge top-level keywords + intake_conversation.keywords ────
    kws = list(raw.get("keywords", []))
    ic_kws = raw.get("intake_conversation", {}).get("keywords", [])
    for k in ic_kws:
        if k not in kws:
            kws.append(k)

    return {
        "id":               raw.get("id", ""),
        "case_type":        raw.get("case_type", ""),
        "keywords":         kws,
        "description":      raw.get("description", ""),
        "legal_domain":     raw.get("legal_domain", ""),
        "dispute_type":     raw.get("dispute_type", ""),
        "bare_act_sections": bare_act_sections,
        "conversation":     conversation,
        "opinion_text":     opinion_text,
        "client_scenario":  raw.get("client_scenario", {}),
        "advocate_reasoning": raw.get("advocate_reasoning", {}),
        "case_analysis":    raw.get("case_analysis", {}),
    }


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_file(path: Path) -> list[dict]:
    if not path.exists():
        logger.warning("Few-shot file not found: %s", path)
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(_normalise(json.loads(line)))
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Skipping line %d in %s: %s", i, path.name, e)
    return out


def _rich_examples() -> list[dict]:
    global _rich_cache
    if _rich_cache is None:
        _rich_cache = _load_file(RICH_FILE)
        logger.info("Loaded %d rich training examples", len(_rich_cache))
    return _rich_cache


def _legacy_examples() -> list[dict]:
    """Old 5+2 example files — used as additional pool if score > 0."""
    global _legacy_cache
    if _legacy_cache is None:
        pool: list[dict] = []
        for fname in ("intake_conversations.jsonl", "final_opinions.jsonl"):
            pool.extend(_load_file(EXAMPLES_DIR / fname))
        # Deduplicate by id
        seen: set[str] = set()
        uniq = []
        for ex in pool:
            if ex["id"] not in seen:
                seen.add(ex["id"])
                uniq.append(ex)
        _legacy_cache = uniq
        logger.info("Loaded %d legacy training examples", len(_legacy_cache))
    return _legacy_cache


def _all_examples() -> list[dict]:
    return _rich_examples() + _legacy_examples()


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

# Legal-domain keyword clusters used for domain detection in the query
_DOMAIN_SIGNALS: dict[str, list[str]] = {
    "fir_quashing":     ["fir", "quash", "false case", "section 482", "chargesheet", "complaint registered"],
    "criminal_bail":    ["bail", "anticipatory bail", "section 438", "section 439", "custody", "arrested"],
    "criminal_arrest":  ["arrested", "police custody", "illegal arrest", "section 41", "d.k. basu", "handcuff"],
    "criminal_general": ["conviction", "acquittal", "murder", "accused", "circumstantial", "sentence"],
    "motor_accident":   ["accident", "mact", "motor", "compensation", "insurance", "death claim", "multiplier"],
    "matrimonial":      ["maintenance", "divorce", "custody", "498a", "dowry", "domestic violence", "dv act", "cruelty", "wife", "husband"],
    "cheque_bounce":    ["cheque", "dishonour", "section 138", "negotiable", "drawer", "bounce"],
    "landlord_tenant":  ["tenant", "landlord", "eviction", "rent", "premises", "lease", "possession notice"],
    "income_tax":       ["income tax", "it notice", "section 148", "assessment", "tds", "tax demand", "itr"],
    "indirect_tax":     ["gst", "customs duty", "excise", "service tax", "import duty"],
    "service_law":      ["government employee", "promotion", "seniority", "pension", "termination", "dismissal", "disciplinary", "compulsory retirement"],
    "labour":           ["gratuity", "retrenchment", "pf", "epf", "esic", "workman", "factory", "labour", "regularisation"],
    "property":         ["property", "sale deed", "encroachment", "title", "partition", "injunction", "possession", "7/12"],
    "land_acquisition": ["land acquisition", "collector", "compensation award", "compulsory acquisition", "urgency clause",
                          "acquired my land", "acquired my field", "acquired my plot", "acquired my farm",
                          "government acquired", "government took my land", "compensation for land",
                          "market value land", "enhanced compensation"],
    "constitutional":   ["fundamental right", "article 14", "article 19", "article 21", "article 226", "article 32", "writ", "rti", "right to information"],
    "contract":         ["contract", "dealership", "agreement", "breach", "specific performance", "arbitration"],
    "medical_negligence": ["medical negligence", "doctor", "hospital", "surgery", "treatment", "malpractice", "patient"],
    "banking_recovery": ["bank", "loan", "npa", "sarfaesi", "recovery", "mortgage", "debt", "default"],
    "consumer":         ["consumer", "deficiency", "ncdrc", "district forum", "unfair trade"],
}


def _tokenise(text: str) -> set[str]:
    """Lowercase word tokeniser including Devanagari/Kannada ranges."""
    return set(re.findall(r"[a-z0-9\u0900-\u097f\u0c00-\u0c7f]+", text.lower()))


def _detect_domain(query_lower: str) -> str:
    """Best-guess legal domain from query text. Returns "" if unclear."""
    best_domain, best_hits = "", 0
    for domain, signals in _DOMAIN_SIGNALS.items():
        hits = sum(1 for sig in signals if sig in query_lower)
        if hits > best_hits:
            best_hits, best_domain = hits, domain
    return best_domain if best_hits >= 1 else ""


def _score(example: dict, query_tokens: set[str], query_lower: str,
           detected_domain: str) -> float:
    """
    Score a normalised example against the query.

    Weights:
      +3.0   legal_domain exact match with detected domain
      +2.0   all case_type words appear in query
      +1.5   each bare act section number found in query (e.g. "138", "498")
      +1.0   each keyword phrase fully matched in query
      +0.5   each description word matched
    """
    score = 0.0

    # 1. Domain match — strongest signal
    if detected_domain and example.get("legal_domain") == detected_domain:
        score += 3.0

    # 2. Case-type full-word match
    ct_tokens = _tokenise(example.get("case_type", "").replace("_", " "))
    if ct_tokens and ct_tokens.issubset(query_tokens):
        score += 2.0

    # 3. Bare-act section numbers in query (e.g. "138", "125", "482")
    for sec in example.get("bare_act_sections", []):
        # Extract digits: "section 138" → "138", "section 8(1)(j)" → "8"
        # Use word-boundary match so "8" does NOT match inside "138"
        nums = re.findall(r"\d+[a-z]?", sec)
        for n in nums:
            if re.search(r"\b" + re.escape(n) + r"\b", query_lower):
                score += 1.5

    # 4. Keyword phrase matches
    for kw in example.get("keywords", []):
        kw_tokens = _tokenise(kw)
        if kw_tokens and kw_tokens.issubset(query_tokens):
            score += 1.0

    # 5. Description word overlap
    desc_tokens = _tokenise(example.get("description", ""))
    score += len(desc_tokens & query_tokens) * 0.5

    return score


def _best_match(examples: list[dict], query: str) -> dict | None:
    if not examples:
        return None
    query_lower = query.lower()
    query_tokens = _tokenise(query)
    detected_domain = _detect_domain(query_lower)

    scored = [
        (ex, _score(ex, query_tokens, query_lower, detected_domain))
        for ex in examples
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    best, best_score = scored[0]
    return best if best_score > 0.0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────

def _sep(label: str) -> str:
    return f"\n{'═' * 64}\n{label}\n{'═' * 64}"


def _format_intake(ex: dict) -> str:
    """
    Format a rich intake example for injection into ROUTING_SINGLE_GATE_SYSTEM
    or FACT_COLLECTION_SYSTEM.

    Shows the model:
      • client profile and emotional context
      • the exact conversation arc (empathy → questions → prayer elicitation)
      • the advocate's internal reasoning (brief)

    The model should mirror the STYLE and ARC — not copy the facts.
    """
    description  = ex.get("description", "")
    domain       = ex.get("legal_domain", "")
    dispute_type = ex.get("dispute_type", "").replace("_", " ")
    conversation = ex.get("conversation", [])

    if not conversation:
        return ""

    # Build header
    heading = description or f"{domain} / {dispute_type}"
    lines = [
        _sep(f"REFERENCE EXAMPLE — {heading}"),
        "",
        "INSTRUCTION: Mirror the STYLE and ARC of this conversation for the",
        "current client. Do NOT copy the facts, names, or specific legal details.",
        "Key behaviours to replicate:",
        "  • Start with empathy — acknowledge what the client is feeling",
        "  • Ask ONE focused question at a time",
        "  • Elicit prayer/what the client wants if not yet stated",
        "  • If a monetary amount is mentioned — probe WHY that amount and the",
        "    other party's income before accepting it",
        "  • Close with action:complete once you have enough for a legal opinion",
        "",
    ]

    # Client context (rich records only)
    scenario = ex.get("client_scenario", {})
    if scenario:
        profile = scenario.get("client_profile", {})
        problem = scenario.get("presenting_problem", "")
        state   = scenario.get("client_emotional_state", "")
        prayer  = scenario.get("prayer_stated", "")
        if profile:
            lines.append(
                f"CLIENT: {profile.get('name','?')}, {profile.get('age','?')}, "
                f"{profile.get('occupation','?')}, {profile.get('location','?')}"
            )
        if state:
            lines.append(f"EMOTIONAL STATE: {state}")
        if problem:
            lines.append(f"PRESENTING PROBLEM: {problem[:200]}")
        if prayer:
            lines.append(f"PRAYER: {prayer[:150]}")
        lines.append("")

    # Conversation turns
    lines.append("─── CONVERSATION ───")
    lines.append("")
    for turn in conversation:
        role    = "Client" if turn.get("role") == "user" else "Advocate"
        content = turn.get("content", "").strip()

        # For assistant turns, try to extract just reply_to_client for readability
        if role == "Advocate":
            try:
                inner = json.loads(content)
                action = inner.get("action", "")
                if action == "ask":
                    content = f'[asks] {inner.get("reply_to_client", content)}'
                elif action == "complete":
                    facts_s = inner.get("facts_summary", "")
                    reply   = inner.get("reply_to_client", "")
                    content = (
                        f'[completes intake] {reply}\n'
                        f'  FACTS CAPTURED: {facts_s[:300]}'
                    )
            except (json.JSONDecodeError, TypeError):
                pass  # Show raw content if not parseable JSON

        lines.append(f"{role}: {content}")
        lines.append("")

    # Advocate reasoning summary (rich records only)
    reasoning = ex.get("advocate_reasoning", {})
    if reasoning:
        assessment = reasoning.get("initial_assessment", "")
        if assessment:
            lines.append("─── ADVOCATE'S REASONING ───")
            lines.append(f"{assessment[:300]}")
            lines.append("")

    lines.append("═" * 64)
    return "\n".join(lines)


def _format_opinion(ex: dict) -> str:
    """
    Format a rich opinion example for injection into
    STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT.

    Shows the model:
      • the exact opinion structure (Facts / Disputes / Legal Position /
        Reliefs Assessment / Next Steps)
      • how to refer to bare act sections and case laws inline
      • the tone and level of detail expected

    IMPORTANT: The model must use the bare acts and case laws from
    RETRIEVED LEGAL MATERIALS, NOT from this example. This is for
    structural and tonal guidance only.
    """
    description  = ex.get("description", "")
    domain       = ex.get("legal_domain", "")
    dispute_type = ex.get("dispute_type", "").replace("_", " ")
    opinion_text = ex.get("opinion_text", "").strip()

    if not opinion_text:
        return ""

    heading = description or f"{domain} / {dispute_type}"
    lines = [
        _sep(f"REFERENCE EXAMPLE — {heading}"),
        "",
        "INSTRUCTION: Follow this STRUCTURE, TONE, and FORMAT exactly.",
        "Replace ALL section numbers, case names, and facts with what is in",
        "the RETRIEVED LEGAL MATERIALS above — never copy citations from here.",
        "The sections and cases below are illustrative of the FORMAT only.",
        "",
    ]

    # Show bare acts structure (structural hint, not content)
    final_opinion = ex.get("final_opinion", {})
    bare_acts = final_opinion.get("bare_acts_cited", [])
    if bare_acts:
        lines.append("─── STRUCTURE: BARE ACTS USED ───")
        for ba in bare_acts[:3]:
            act_s  = ba.get("section", "")
            rel    = ba.get("relevance", "")
            lines.append(f"  • {act_s}: {rel}")
        lines.append("")

    case_laws = final_opinion.get("case_laws_cited", [])
    if case_laws:
        lines.append("─── STRUCTURE: CASE LAWS CITED ───")
        for cl in case_laws[:2]:
            name      = cl.get("case_name", "")
            principle = cl.get("principle", "")[:120]
            applied   = cl.get("how_applied", "")[:100]
            lines.append(f"  • {name}: {principle}")
            lines.append(f"    Applied: {applied}")
        lines.append("")

    # The full opinion text
    lines.append("─── OPINION FORMAT TO FOLLOW ───")
    lines.append("")
    # Truncate to ~600 words to avoid bloating the prompt
    words = opinion_text.split()
    if len(words) > 600:
        opinion_text = " ".join(words[:600]) + "\n[…opinion continues…]"
    lines.append(opinion_text)
    lines.append("")
    lines.append("═" * 64)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_intake_example(query: str) -> str | None:
    """
    Return a formatted few-shot intake conversation relevant to the query,
    or None if nothing relevant is found.

    Inject the returned string at the END of ROUTING_SINGLE_GATE_SYSTEM or
    FACT_COLLECTION_SYSTEM, preceded by two blank lines.

    Args:
        query: All user messages so far concatenated (used for matching).
    """
    ex = _best_match(_all_examples(), query)
    if not ex or not ex.get("conversation"):
        return None
    try:
        return _format_intake(ex)
    except Exception as e:
        logger.warning("Failed to format intake example: %s", e)
        return None


def get_opinion_example(query: str) -> str | None:
    """
    Return a formatted few-shot final opinion relevant to the query,
    or None if nothing relevant is found.

    Inject the returned string AFTER STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT
    and BEFORE the _allowlist_suffix.

    Args:
        query: The facts_summary or combined conversation text.
    """
    # Prefer rich records for opinion examples (they have full opinion_text)
    ex = _best_match(_rich_examples() or _all_examples(), query)
    if not ex or not ex.get("opinion_text"):
        return None
    try:
        return _format_opinion(ex)
    except Exception as e:
        logger.warning("Failed to format opinion example: %s", e)
        return None


def reload_examples() -> None:
    """Force-reload all examples from disk (useful after adding new records)."""
    global _rich_cache, _legacy_cache
    _rich_cache   = None
    _legacy_cache = None
    logger.info("Few-shot example cache cleared — will reload on next request")


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    queries = [
        ("matrimonial",      "My husband left me with two kids and is not paying maintenance. I want Rs 15000 per month."),
        ("property",         "Neighbour built a wall inside my property boundary. I have registered sale deed and patta."),
        ("cheque_bounce",    "My business partner gave me a cheque of Rs 3 lakh which bounced. Section 138 notice sent."),
        ("criminal_bail",    "My son was arrested under 498A IPC. Police have given Section 41A notice. Need anticipatory bail."),
        ("fir_quashing",     "False FIR registered against me for cheating. I have evidence proving I am innocent."),
        ("motor_accident",   "My husband died in a truck accident. He was 40 years old earning Rs 50000 per month. I want compensation."),
        ("income_tax",       "Income tax department sent me a notice under section 148 to reopen my assessment for three years ago."),
        ("labour",           "I worked for 7 years and was terminated without notice. Company is not paying gratuity."),
        ("service_law",      "I am a government employee and was denied promotion despite having more seniority than my junior."),
        ("land_acquisition", "Government acquired my 5 acres of agricultural land and gave only Rs 10 lakh. Market value is Rs 50 lakh."),
    ]

    print("\n" + "═" * 72)
    print("  FEW-SHOT RETRIEVER — SMOKE TEST")
    print(f"  Rich examples: {len(_rich_examples())}  |  Legacy examples: {len(_legacy_examples())}")
    print("═" * 72)

    all_ok = True
    for expected_domain, q in queries:
        # Test intake retrieval
        result = get_intake_example(q)
        if result:
            # Extract the heading line from the formatted block
            heading_line = ""
            for ln in result.split("\n"):
                if "REFERENCE EXAMPLE" in ln:
                    heading_line = ln.strip()
                    break
            # Verify domain detection
            detected = _detect_domain(q.lower())
            status = "PASS" if detected == expected_domain else f"WARN (detected={detected!r})"
            print(f"\n[{status}] QUERY: {q[:70]}")
            print(f"  DOMAIN DETECTED : {detected}")
            print(f"  INTAKE MATCH    : {heading_line[:70]}")
        else:
            print(f"\n[FAIL] QUERY: {q[:70]}")
            print(f"  INTAKE MATCH    : (none — check keywords/domain)")
            all_ok = False

        # Test opinion retrieval
        op_result = get_opinion_example(q)
        if op_result:
            op_line = ""
            for ln in op_result.split("\n"):
                if "REFERENCE EXAMPLE" in ln:
                    op_line = ln.strip()
                    break
            print(f"  OPINION MATCH   : {op_line[:70]}")
        else:
            print(f"  OPINION MATCH   : (none)")

    print("\n" + "═" * 72)
    print("ALL PASS" if all_ok else "SOME WARNINGS — review output above")
    print("═" * 72 + "\n")
