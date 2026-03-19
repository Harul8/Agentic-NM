"""
Few-Shot Example Retriever for Nyaymalaw  (v1.1)
=================================================
Serves two purposes:

1. FEW-SHOT LEARNING (immediate) — retrieves the most relevant training
   example for a given client query and injects it into the LLM prompt
   at inference time. The model sees what a perfect intake looks like for a
   similar case type, including:
     • contrastive turn annotations (ideal q / bad q / why bad loses)
     • decision-state snapshot (schema in action)
     • intake-layer action plan
   without needing to carry all 97 examples in every prompt.

2. FINE-TUNING DATA (future) — same JSONL files are the training dataset
   for Unsloth/LoRA fine-tuning. Use convert_to_finetune.py to export.

Retrieval strategy
------------------
Keyword overlap + legal-domain detection + bare-act section number matching.
Simple and fast — no embeddings needed for O(~100) examples.
Upgrade to FAISS/ChromaDB if the store grows beyond ~500 examples.

Source hierarchy (v1.1)
-----------------------
PRIMARY   nyaymalaw_training_examples.md  — v1.1 enriched blocks
           • Enriched Intake Conversation with contrastive annotations
           • Decision-State Snapshot (core runtime fields, compact JSON)
           • Stop Policy Check (why intake stopped + counter-test)
           • Intake-Layer Action Plan (what to do / not do today)
           • Grounded Advice Layer reference (for opinion examples)

FALLBACK   rich_training_records.jsonl + legacy JSONL files
           • Used when MD block unavailable or MD not yet generated
           • Formatted by the original _format_intake / _format_opinion logic

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

# v1.1: enriched MD source — lives one level above the training package
TRAINING_MD  = Path(__file__).parent.parent / "nyaymalaw_training_examples.md"

# ── Cache ────────────────────────────────────────────────────────────────────
_rich_cache:   list[dict] | None = None   # normalised rich records
_legacy_cache: list[dict] | None = None   # normalised legacy records
_md_cache:     dict[str, dict] | None = None  # keyed by record id e.g. "rich_001"
_curated_cache: list[dict] | None = None  # curated_* few-shot blocks from MD


# ═════════════════════════════════════════════════════════════════════════════
# MD BLOCK PARSER  (v1.1 primary source)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_table_field(block_text: str, field_name: str) -> str:
    """Return the value for a field in a markdown table row: | **Field** | value |"""
    pattern = rf"\|\s*\*\*{re.escape(field_name)}\*\*\s*\|\s*([^|\n]+?)\s*\|"
    m = re.search(pattern, block_text)
    return m.group(1).strip() if m else ""


def _extract_md_section(block_text: str, section_header: str,
                         next_section_header: str | None) -> str:
    """
    Extract the content between two ### headers inside a case block.
    Returns the raw markdown text (without the header line itself).
    """
    # Build start pattern — match the specific section header (level 3 ##)
    start_pat = rf"###\s+{re.escape(section_header)}[^\n]*\n"
    m_start = re.search(start_pat, block_text)
    if not m_start:
        return ""

    content_start = m_start.end()

    if next_section_header:
        end_pat = rf"\n###\s+{re.escape(next_section_header)}"
        m_end = re.search(end_pat, block_text[content_start:])
        if m_end:
            return block_text[content_start: content_start + m_end.start()].strip()

    # No next header — read to the end of this case block (next ## or end of string)
    m_end2 = re.search(r"\n## rich_", block_text[content_start:])
    if m_end2:
        return block_text[content_start: content_start + m_end2.start()].strip()

    return block_text[content_start:].strip()


def _extract_loader_index() -> dict[str, dict]:
    """
    Parse the curated loader index from the training markdown.
    Returns metadata keyed by curated id, e.g. "curated_01".
    """
    if not TRAINING_MD.exists():
        return {}

    content = TRAINING_MD.read_text(encoding="utf-8")
    m = re.search(
        r"### Curated Loader Index\s*\n(?P<body>.*?)(?:\n\*\*Loader hint:\*\*|\n### curated_01:)",
        content,
        flags=re.S,
    )
    if not m:
        return {}

    meta: dict[str, dict] = {}
    row_pat = re.compile(
        r"^\|\s*`(?P<id>curated_\d+)`\s*\|\s*(?P<client>[^|]+?)\s*\|\s*(?P<outcome>[^|]+?)\s*\|\s*(?P<shape>[^|]+?)\s*\|\s*(?P<move>[^|]+?)\s*\|$",
        flags=re.M,
    )
    for row in row_pat.finditer(m.group("body")):
        meta[row.group("id")] = {
            "client_type_tag": row.group("client").strip(),
            "outcome_pattern": row.group("outcome").strip(),
            "matter_shape": row.group("shape").strip(),
            "primary_teaching_move": row.group("move").strip(),
        }
    return meta


def _canonical_domain_from_text(text: str) -> str:
    """
    Best-effort mapping of a free-form domain label into the existing internal
    canonical domain names used by the scorer.
    """
    norm = text.lower().replace("/", " ").replace("+", " ").replace("-", " ")
    if "mixed" in norm or "multi" in norm:
        return "mixed"

    best_domain, best_hits = "", 0
    for domain, signals in _DOMAIN_SIGNALS.items():
        hits = sum(1 for sig in signals if sig in norm)
        if hits > best_hits:
            best_domain, best_hits = domain, hits

    # Light alias support for labels that may not hit the signal list directly
    aliases = {
        "criminal fir challenge": "fir_quashing",
        "criminal procedure fir registration": "criminal_general",
        "criminal bail": "criminal_bail",
        "administrative travel": "constitutional",
        "notice review": "contract",
        "rti information access": "constitutional",
        "property injunction": "property",
        "property partition": "property",
        "labour termination": "labour",
        "family protection": "matrimonial",
        "banking recovery": "banking_recovery",
    }
    alias_key = " ".join(norm.split())
    return aliases.get(alias_key, best_domain)


def _parse_curated_blocks() -> list[dict]:
    """
    Parse curated_* compact examples from the training markdown so they can be
    used directly at inference time without requiring JSONL backing records.
    """
    global _curated_cache
    if _curated_cache is not None:
        return _curated_cache

    if not TRAINING_MD.exists():
        _curated_cache = []
        return _curated_cache

    content = TRAINING_MD.read_text(encoding="utf-8")
    section_match = re.search(
        r"## Curated Few-Shot Starter Pack \(Preferred\)\s*(?P<body>.*?)(?:\n## rich_|\Z)",
        content,
        flags=re.S,
    )
    if not section_match:
        _curated_cache = []
        return _curated_cache

    curated_section = section_match.group("body")
    parts = re.split(r"\n### (curated_\d+): ([^\n]+)\n", curated_section)
    index_meta = _extract_loader_index()

    blocks: list[dict] = []
    i = 1
    while i + 2 < len(parts):
        rec_id = parts[i].strip()
        rec_name = parts[i + 1].strip()
        body = parts[i + 2].strip()

        domain_raw = _extract_table_field(body, "Domain")
        client_type = _extract_table_field(body, "Client Type")
        urgency = _extract_table_field(body, "Urgency Level")
        use_case = _extract_table_field(body, "Use Case")
        meta = index_meta.get(rec_id, {})

        blocks.append({
            "id": rec_id,
            "name": rec_name,
            "description": use_case or rec_name,
            "case_type": domain_raw,
            "domain": domain_raw.lower().replace(" ", "_"),
            "legal_domain": _canonical_domain_from_text(domain_raw),
            "dispute_type": "",
            "keywords": [rec_name, domain_raw, use_case, meta.get("primary_teaching_move", "")],
            "bare_act_sections": [],
            "client_type": client_type,
            "urgency": urgency,
            "intake_conv": body,
            "source": "curated",
            "outcome_pattern": meta.get("outcome_pattern", ""),
            "matter_shape": meta.get("matter_shape", ""),
            "primary_teaching_move": meta.get("primary_teaching_move", ""),
        })
        i += 3

    _curated_cache = blocks
    logger.info("Parsed %d curated few-shot blocks from %s", len(blocks), TRAINING_MD.name)
    return _curated_cache


def _parse_md_blocks() -> dict[str, dict]:
    """
    Parse nyaymalaw_training_examples.md into per-case blocks.
    Returns a dict keyed by record id (e.g. "rich_001").

    Each value contains:
      id           str
      name         str    Case name from the ## heading
      domain       str    snake_case domain from metadata table
      client_type  str    Lay Client / Junior Advocate
      urgency      str    IMMEDIATE / HIGH / MEDIUM / LOW
      intake_conv  str    ### Enriched Intake Conversation section
      decision_snap str   ### Decision-State Snapshot section
      stop_policy  str    ### Stop Policy Check section
      weakness     str    ### Weakness Stress-Test section
      commercial   str    ### Commercial Reality section
      action_plan  str    ### Intake-Layer Action Plan section
      grounded     str    ### Grounded Advice Layer section
    """
    global _md_cache
    if _md_cache is not None:
        return _md_cache

    if not TRAINING_MD.exists():
        logger.warning("Training MD not found: %s — falling back to JSONL-only formatting", TRAINING_MD)
        _md_cache = {}
        return _md_cache

    content = TRAINING_MD.read_text(encoding="utf-8")

    # Split on ## rich_NNN: headers; re.split includes capturing groups as elements
    parts = re.split(r"\n## (rich_\d+): ([^\n]+)\n", content)
    # parts = [preamble, id1, name1, body1, id2, name2, body2, ...]

    blocks: dict[str, dict] = {}
    i = 1
    while i + 2 < len(parts):
        rec_id   = parts[i].strip()
        rec_name = parts[i + 1].strip()
        body     = parts[i + 2]

        domain_raw  = _extract_table_field(body, "Domain")
        client_type = _extract_table_field(body, "Client Type")
        urgency     = _extract_table_field(body, "Urgency Level")

        blocks[rec_id] = {
            "id":           rec_id,
            "name":         rec_name,
            "domain":       domain_raw.lower().replace(" ", "_"),
            "client_type":  client_type,
            "urgency":      urgency,
            "intake_conv":  _extract_md_section(body, "Enriched Intake Conversation", "Decision-State Snapshot at Completion"),
            "decision_snap": _extract_md_section(body, "Decision-State Snapshot at Completion", "Stop Policy Check"),
            "stop_policy":  _extract_md_section(body, "Stop Policy Check", "Weakness Stress-Test"),
            "weakness":     _extract_md_section(body, "Weakness Stress-Test", "Commercial Reality"),
            "commercial":   _extract_md_section(body, "Commercial Reality", "Intake-Layer Action Plan"),
            "action_plan":  _extract_md_section(body, "Intake-Layer Action Plan", "Grounded Advice Layer"),
            "grounded":     _extract_md_section(body, "Grounded Advice Layer", None),
        }
        i += 3

    _md_cache = blocks
    logger.info("Parsed %d enriched MD blocks from %s", len(blocks), TRAINING_MD.name)
    return blocks


def _get_md_block(rec_id: str) -> dict | None:
    """Return the parsed MD block for a record id, or None."""
    blocks = _parse_md_blocks()
    return blocks.get(rec_id)


# ═════════════════════════════════════════════════════════════════════════════
# JSONL SCHEMA NORMALISATION  (scoring metadata source)
# ═════════════════════════════════════════════════════════════════════════════

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
    return _parse_curated_blocks() + _rich_examples() + _legacy_examples()


# ═════════════════════════════════════════════════════════════════════════════
# SCORING
# ═════════════════════════════════════════════════════════════════════════════

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
    return set(re.findall(r"[a-z0-9\u0900-\u097f\u0c00-\u0c7f]+", text.lower()))


def _detect_domain(query_lower: str) -> str:
    best_domain, best_hits = "", 0
    for domain, signals in _DOMAIN_SIGNALS.items():
        hits = sum(1 for sig in signals if sig in query_lower)
        if hits > best_hits:
            best_hits, best_domain = hits, domain
    return best_domain if best_hits >= 1 else ""


def _score(example: dict, query_tokens: set[str], query_lower: str,
           detected_domain: str) -> float:
    score = 0.0

    if detected_domain and example.get("legal_domain") == detected_domain:
        score += 3.0
    elif detected_domain and example.get("legal_domain") == "mixed":
        # Mixed examples should only win when there is enough signal they are relevant.
        if sum(1 for sigs in _DOMAIN_SIGNALS.values() for sig in sigs if sig in query_lower) >= 2:
            score += 1.5

    ct_tokens = _tokenise(example.get("case_type", "").replace("_", " "))
    if ct_tokens and ct_tokens.issubset(query_tokens):
        score += 2.0

    for sec in example.get("bare_act_sections", []):
        nums = re.findall(r"\d+[a-z]?", sec)
        for n in nums:
            if re.search(r"\b" + re.escape(n) + r"\b", query_lower):
                score += 1.5

    for kw in example.get("keywords", []):
        kw_tokens = _tokenise(kw)
        if kw_tokens and kw_tokens.issubset(query_tokens):
            score += 1.0

    desc_tokens = _tokenise(example.get("description", ""))
    score += len(desc_tokens & query_tokens) * 0.5

    # Curated examples are the preferred intake source when they are relevant.
    if example.get("source") == "curated":
        score += 1.0
        meta_tokens = _tokenise(
            " ".join(
                [
                    example.get("matter_shape", ""),
                    example.get("outcome_pattern", ""),
                    example.get("primary_teaching_move", ""),
                    example.get("intake_conv", "")[:1200],
                ]
            )
        )
        score += len(meta_tokens & query_tokens) * 0.15

    # v1.1 bonus: prefer records that have an enriched MD block
    if _get_md_block(example.get("id", "")):
        score += 0.5

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


def _best_intake_match(query: str) -> dict | None:
    """
    Prefer curated compact examples for intake behavior. Fall back to the rich
    / legacy pools only if no curated example scores positively.
    """
    curated_best = _best_match(_parse_curated_blocks(), query)
    if curated_best:
        return curated_best
    return _best_match(_rich_examples() + _legacy_examples(), query)


# ═════════════════════════════════════════════════════════════════════════════
# FORMATTERS  (v1.1 — MD-primary, JSONL-fallback)
# ═════════════════════════════════════════════════════════════════════════════

def _sep(label: str) -> str:
    return f"\n{'═' * 64}\n{label}\n{'═' * 64}"


# ── Truncation helper ─────────────────────────────────────────────────────────

def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "\n[…truncated…]"


# ── MD-backed intake formatter ────────────────────────────────────────────────

def _format_intake_from_md(ex: dict, block: dict) -> str:
    """
    Format a v1.1 enriched intake example from the parsed MD block.

    Injects into the prompt:
      • Domain / urgency / client-type header
      • Client scenario (brief)
      • Enriched intake conversation with contrastive turn annotations
        — teaches: ideal question, plausible bad alternative, why bad loses
      • Decision-state snapshot (compact JSON)
      • Intake-layer action plan

    Layer boundary respected: no section numbers in intake annotations.
    """
    rec_id      = ex.get("id", "")
    rec_name    = block.get("name", "")
    domain      = (block.get("domain") or ex.get("legal_domain", "")).replace("_", " ").title()
    client_type = block.get("client_type") or ""
    urgency     = block.get("urgency") or ""
    description = ex.get("description", "")

    heading = description or rec_name or f"{domain}"
    lines   = [
        _sep(f"REFERENCE EXAMPLE ({rec_id}) — {heading}"),
        "",
        "INSTRUCTION: Mirror the DECISION PATTERN of this intake — not the facts.",
        "  • Replicate the question discipline: one focused question per turn",
        "  • Note why each turn's question wins over the bad alternative",
        "  • Follow the decision-state schema shown in the snapshot",
        "  • Stop only when the three exit conditions are met (see Stop Policy)",
        "",
    ]

    # ── Header metadata ──────────────────────────────────────────────────────
    if domain or client_type or urgency:
        lines.append(f"DOMAIN: {domain}  |  CLIENT TYPE: {client_type}  |  URGENCY: {urgency}")
        lines.append("")

    # ── Client scenario ───────────────────────────────────────────────────────
    scenario = ex.get("client_scenario", {})
    if scenario:
        profile = scenario.get("client_profile", {})
        state   = scenario.get("client_emotional_state", "")
        problem = scenario.get("presenting_problem", "")
        prayer  = scenario.get("prayer_stated", "")
        if profile:
            lines.append(
                f"CLIENT: {profile.get('name', '?')}, {profile.get('age', '?')}, "
                f"{profile.get('occupation', '?')}"
            )
        if state:
            lines.append(f"EMOTIONAL STATE: {state}")
        if problem:
            lines.append(f"PRESENTING PROBLEM: {problem[:200]}")
        if prayer:
            lines.append(f"PRAYER: {prayer[:150]}")
        lines.append("")

    # ── Enriched intake conversation (core teaching signal) ───────────────────
    intake_conv = block.get("intake_conv", "").strip()
    if intake_conv:
        lines.append("─── ENRICHED INTAKE CONVERSATION (with contrastive annotations) ───")
        lines.append("")
        # Cap at ~700 words to keep token use reasonable while keeping all turns
        lines.append(_truncate_words(intake_conv, 700))
        lines.append("")

    # ── Decision-state snapshot ───────────────────────────────────────────────
    decision_snap = block.get("decision_snap", "").strip()
    if decision_snap:
        lines.append("─── DECISION-STATE SNAPSHOT AT COMPLETION ───")
        lines.append("")
        lines.append(_truncate_words(decision_snap, 250))
        lines.append("")

    # ── Intake-layer action plan ──────────────────────────────────────────────
    action_plan = block.get("action_plan", "").strip()
    if action_plan:
        lines.append("─── INTAKE-LAYER ACTION PLAN ───")
        lines.append("")
        lines.append(_truncate_words(action_plan, 200))
        lines.append("")

    lines.append("═" * 64)
    return "\n".join(lines)


def _format_intake_from_curated(block: dict) -> str:
    """
    Format a curated compact few-shot example. These examples are deliberately
    shorter and more behaviorally varied than the legacy long-form blocks.
    """
    heading = block.get("description") or block.get("name") or block.get("id", "")
    lines = [
        _sep(f"CURATED REFERENCE EXAMPLE ({block.get('id', '')}) — {heading}"),
        "",
        "INSTRUCTION: Mirror the diagnostic move, not the facts.",
        "  • Prefer the same question-selection discipline shown here",
        "  • Respect the Stop / Continue signal",
        "  • Preserve the intake-vs-grounded-advice boundary",
        "",
        f"DOMAIN: {(block.get('case_type') or '').strip()}  |  CLIENT TYPE: {block.get('client_type', '')}  |  URGENCY: {block.get('urgency', '')}",
    ]

    if block.get("outcome_pattern") or block.get("matter_shape"):
        lines.append(
            f"META: OUTCOME={block.get('outcome_pattern', '')}  |  MATTER SHAPE={block.get('matter_shape', '')}"
        )
    if block.get("primary_teaching_move"):
        lines.append(f"PRIMARY MOVE: {block.get('primary_teaching_move')}")

    lines.extend([
        "",
        "——— CURATED INTAKE EXAMPLE ——",
        "",
        _truncate_words(block.get("intake_conv", ""), 500),
        "",
        "═" * 64,
    ])
    return "\n".join(lines)


# ── MD-backed opinion formatter ───────────────────────────────────────────────

def _format_opinion_from_md(ex: dict, block: dict) -> str:
    """
    Format a v1.1 enriched opinion reference from the parsed MD block.
    Shows the Grounded Advice Layer for structural and tonal guidance.
    """
    rec_id      = ex.get("id", "")
    rec_name    = block.get("name", "")
    domain      = (block.get("domain") or ex.get("legal_domain", "")).replace("_", " ").title()
    description = ex.get("description", "")
    grounded    = block.get("grounded", "").strip()

    if not grounded:
        return ""

    heading = description or rec_name or domain
    lines = [
        _sep(f"REFERENCE EXAMPLE ({rec_id}) — {heading}"),
        "",
        "INSTRUCTION: Follow this STRUCTURE, TONE, and FORMAT exactly.",
        "Replace ALL section numbers, case names, and facts with what is in",
        "the RETRIEVED LEGAL MATERIALS above — never copy citations from here.",
        "The sections and cases below are illustrative of FORMAT only.",
        "",
        "─── GROUNDED ADVICE LAYER ───",
        "",
        _truncate_words(grounded, 600),
        "",
        "═" * 64,
    ]
    return "\n".join(lines)


# ── Legacy JSONL formatters (unchanged — used when MD block unavailable) ──────

def _format_intake(ex: dict) -> str:
    """
    Format an intake example.
    Uses enriched MD block when available; falls back to JSONL-based format.
    """
    # v1.1: prefer MD block
    if ex.get("source") == "curated":
        return _format_intake_from_curated(ex)

    block = _get_md_block(ex.get("id", ""))
    if block and (block.get("intake_conv") or block.get("action_plan")):
        return _format_intake_from_md(ex, block)

    # ── Legacy fallback ───────────────────────────────────────────────────────
    description  = ex.get("description", "")
    domain       = ex.get("legal_domain", "")
    dispute_type = ex.get("dispute_type", "").replace("_", " ")
    conversation = ex.get("conversation", [])

    if not conversation:
        return ""

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

    lines.append("─── CONVERSATION ───")
    lines.append("")
    for turn in conversation:
        role    = "Client" if turn.get("role") == "user" else "Advocate"
        content = turn.get("content", "").strip()

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
                pass

        lines.append(f"{role}: {content}")
        lines.append("")

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
    Format a final opinion example.
    Uses enriched MD block when available; falls back to JSONL-based format.
    """
    # v1.1: prefer MD block
    block = _get_md_block(ex.get("id", ""))
    if block and block.get("grounded"):
        return _format_opinion_from_md(ex, block)

    # ── Legacy fallback ───────────────────────────────────────────────────────
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

    final_opinion = ex.get("final_opinion", {})
    bare_acts = final_opinion.get("bare_acts_cited", [])
    if bare_acts:
        lines.append("─── STRUCTURE: BARE ACTS USED ───")
        for ba in bare_acts[:3]:
            act_s = ba.get("section", "")
            rel   = ba.get("relevance", "")
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

    lines.append("─── OPINION FORMAT TO FOLLOW ───")
    lines.append("")
    words = opinion_text.split()
    if len(words) > 600:
        opinion_text = " ".join(words[:600]) + "\n[…opinion continues…]"
    lines.append(opinion_text)
    lines.append("")
    lines.append("═" * 64)
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def get_intake_example(query: str) -> str | None:
    """
    Return a formatted few-shot intake conversation relevant to the query,
    or None if nothing relevant is found.

    v1.1: Prefers the enriched MD block (contrastive annotations, decision-
    state snapshot, intake-layer action plan). Falls back to JSONL formatting.

    Inject the returned string at the END of the active intake system prompt,
    preceded by two blank lines.

    Args:
        query: All user messages so far concatenated (used for matching).
    """
    ex = _best_intake_match(query)
    if not ex or (not ex.get("conversation") and not _get_md_block(ex.get("id", ""))):
        if not ex or ex.get("source") != "curated":
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

    v1.1: Prefers the enriched MD Grounded Advice Layer. Falls back to JSONL.

    Inject the returned string AFTER STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT
    and BEFORE the _allowlist_suffix.

    Args:
        query: The facts_summary or combined conversation text.
    """
    ex = _best_match(_rich_examples() or _all_examples(), query)
    if not ex:
        return None
    try:
        return _format_opinion(ex)
    except Exception as e:
        logger.warning("Failed to format opinion example: %s", e)
        return None


def preload_examples() -> None:
    """Load and cache curated, MD-backed, rich, and legacy few-shot sources."""
    _parse_curated_blocks()
    _parse_md_blocks()
    _rich_examples()
    _legacy_examples()
    logger.info(
        "Few-shot examples preloaded (curated=%d, md=%d, rich=%d, legacy=%d)",
        len(_curated_cache or []),
        len(_md_cache or {}),
        len(_rich_cache or []),
        len(_legacy_cache or []),
    )


def reload_examples() -> None:
    """Force-reload all examples from disk (useful after adding new records)."""
    global _rich_cache, _legacy_cache, _md_cache, _curated_cache
    _rich_cache   = None
    _legacy_cache = None
    _md_cache     = None
    _curated_cache = None
    logger.info("Few-shot example cache cleared — will reload on next request")


# ═════════════════════════════════════════════════════════════════════════════
# CLI SMOKE TEST
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    queries = [
        ("matrimonial",      "My husband left me with two kids and is not paying maintenance. I want Rs 15000 per month."),
        ("property",         "Neighbour built a wall inside my property boundary. I have registered sale deed and patta."),
        ("cheque_bounce",    "My business partner gave me a cheque of Rs 3 lakh which bounced. Section 138 notice sent."),
        ("criminal_arrest",  "My son was arrested under 498A IPC. Police have given Section 41A notice. Need anticipatory bail."),
        ("fir_quashing",     "False FIR registered against me for cheating. I have evidence proving I am innocent."),
        ("motor_accident",   "My husband died in a truck accident. He was 40 years old earning Rs 50000 per month. I want compensation."),
        ("income_tax",       "Income tax department sent me a notice under section 148 to reopen my assessment for three years ago."),
        ("labour",           "I worked for 7 years and was terminated without notice. Company is not paying gratuity."),
        ("service_law",      "I am a government employee and was denied promotion despite having more seniority than my junior."),
        ("land_acquisition", "Government acquired my 5 acres of agricultural land and gave only Rs 10 lakh. Market value is Rs 50 lakh."),
    ]

    # Pre-load caches
    md_blocks = _parse_md_blocks()
    rich      = _rich_examples()
    legacy    = _legacy_examples()

    print("\n" + "═" * 72)
    print("  FEW-SHOT RETRIEVER v1.1 — SMOKE TEST")
    print(f"  Rich examples : {len(rich):3d}  |  Legacy: {len(legacy):3d}  |  MD blocks: {len(md_blocks):3d}")
    print("═" * 72)

    all_ok = True
    for expected_domain, q in queries:
        examples = _all_examples()
        q_lower  = q.lower()
        q_tokens = _tokenise(q)
        detected = _detect_domain(q_lower)

        scored = [(ex, _score(ex, q_tokens, q_lower, detected)) for ex in examples]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_ex, best_score = scored[0] if scored else (None, 0)

        status = "PASS" if detected == expected_domain else f"WARN (detected={detected!r})"
        md_hit = "MD✓" if (best_ex and _get_md_block(best_ex.get("id", ""))) else "MD✗"

        print(f"\n[{status}] {md_hit}  QUERY: {q[:65]}")
        print(f"  DOMAIN DETECTED : {detected}")
        if best_ex:
            print(f"  BEST MATCH      : {best_ex['id']} — {best_ex['description'][:55]}  (score={best_score:.1f})")

        result = get_intake_example(q)
        if result:
            # Show first meaningful line to confirm MD vs legacy format
            for ln in result.split("\n"):
                if "REFERENCE EXAMPLE" in ln or "─── ENRICHED" in ln or "─── CONVERSATION" in ln:
                    print(f"  FORMAT          : {ln.strip()[:70]}")
                    break
        else:
            print(f"  FORMAT          : (no example returned)")
            all_ok = False

    print("\n" + "═" * 72)
    print("ALL PASS" if all_ok else "SOME WARNINGS — review output above")
    print("═" * 72 + "\n")
