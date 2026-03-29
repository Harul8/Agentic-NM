"""
Lightweight few-shot retriever for Nyaymalaw runtime prompting.

The runtime uses the same ideas as the training data, but keeps prompts small:
- intake-state examples for compact state extraction
- intake-reply examples for the next intake move
- final-opinion examples for grounded opinion style

Primary sources are the repo-local training files. You can override them with
NYAYMALAW_FEWSHOT_FILES using os.pathsep or comma-separated paths.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_FEWSHOT_FILES = [
    PROJECT_ROOT / "training" / "runtime_fewshot" / "runtime_examples.jsonl",
]

_RUNTIME_EXAMPLES: list[dict] | None = None


def _split_env_paths(raw: str) -> list[Path]:
    parts: list[str] = []
    for chunk in re.split(r"[\n,]+", raw or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([p.strip() for p in chunk.split(os.pathsep) if p.strip()])
    return [Path(p).expanduser() for p in parts]


def _candidate_files() -> list[Path]:
    env_paths = _split_env_paths(os.environ.get("NYAYMALAW_FEWSHOT_FILES", ""))
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in env_paths + _DEFAULT_FEWSHOT_FILES:
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = str(path)
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def _extract_json_block(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    if "```json" in text:
        candidate = text.split("```json", 1)[1].split("```", 1)[0].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _strip_json_block(text: str) -> str:
    text = (text or "").strip()
    if "```json" in text:
        head = text.split("```json", 1)[0].strip()
        tail = text.split("```", 2)[-1].strip() if text.count("```") >= 2 else ""
        return "\n\n".join(part for part in [head, tail] if part).strip()
    return text


def _clean_reply_text(text: str) -> str:
    text = " ".join((text or "").split())
    boilerplate_prefixes = [
        "That helps narrow the record. The next details will help me test whether the matter is ready to move forward on a stronger factual footing.",
    ]
    for prefix in boilerplate_prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.replace(
        "The picture is becoming clearer, and the next details will help me understand how strong the record is. The factual record now seems complete enough for grounded legal analysis.",
        "The factual record now seems complete enough for grounded legal analysis.",
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "for", "from", "had", "has", "have",
    "he", "her", "hers", "him", "his", "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "ours", "she", "that", "the", "their", "them", "they", "this", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your", "yours", "want", "help", "legal", "issue", "case"
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9_]+", (text or "").lower())
    return {tok for tok in tokens if len(tok) > 2 and tok not in _STOPWORDS}


def _trim_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return (text or "").strip()
    return " ".join(words[:max_words]).strip() + " ..."


def _sentences(text: str, limit: int = 4) -> list[str]:
    out: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip()):
        clean = part.strip(" -")
        if len(clean) < 16:
            continue
        out.append(clean[:220])
        if len(out) >= limit:
            break
    return out


def _extract_client_objective(text: str) -> str:
    text = (text or "").strip()
    patterns = [
        r"Prayer / Relief Sought:\s*(.+?)(?:\n|$)",
        r"Relief Sought:\s*(.+?)(?:\n|$)",
        r"Prayer:\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).strip()[:180]
    return ""


def _flatten_opinion_text(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", text or "", flags=re.M)
    text = re.sub(r"^[-*]{3,}$", "", text, flags=re.M)
    text = re.sub(r"^\*\*(.+?)\*\*$", r"\1", text, flags=re.M)
    text = re.sub(r"[\u2500-\u257f]+", " ", text)
    for label in ["Facts of the Case", "Disputes Identified", "Legal Protection", "Reliefs Sought & Assessment", "Next Steps & How to Strengthen Your Case", "Relevant precedents", "Judicial Precedents"]:
        text = text.replace(label, "")
    text = re.sub(r"\n{2,}", "\n\n", text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return " ".join(paragraphs)


def _infer_prior_actions(text: str) -> list[str]:
    low = (text or "").lower()
    signals = [
        (r"\bfir\b|police complaint|complaint filed", "police complaint or FIR mentioned"),
        (r"legal notice|notice sent|show cause|demand notice", "notice or formal communication mentioned"),
        (r"filed|petition|application|appeal|writ", "filing or court step mentioned"),
        (r"hospital|clinic|medical treatment|medical examination", "medical visit or treatment mentioned"),
        (r"representation|reply sent|responded", "reply or representation mentioned"),
        (r"order passed|dismissed|terminated|assessment order", "adverse order or official action mentioned"),
    ]
    actions: list[str] = []
    for pattern, label in signals:
        if re.search(pattern, low):
            actions.append(label)
    return actions[:4]


def _detect_audience(text: str, meta_audience: str = "") -> str:
    if meta_audience in {"lay_user", "legal_professional"}:
        return meta_audience
    low = (text or "").lower()
    if any(tok in low for tok in ["our client", "my client", "writ", "petitioner", "respondent", "article 14", "section "]):
        return "legal_professional"
    return "lay_user"


# ---------------------------------------------------------------------------
# Client-side / orientation detection — victim vs. accused/defence
# Used to prevent a defence-side DV/matrimonial example being shown for a
# victim-side DV query (and vice versa).
# ---------------------------------------------------------------------------

_VICTIM_SIDE_SIGNALS = (
    # First-person victim phrasing: "my husband assaulted me", "I was beaten", etc.
    r"\bmy (husband|wife|partner|spouse|boyfriend|girlfriend)\b.{0,60}(assault|beat|hit|kick|hurt|threaten|abuse|harass|slap|push|throw)",
    r"\b(i was|i have been|i am being) (beaten|assaulted|attacked|threatened|abused|harassed|hurt|injured)\b",
    r"\bi need (protection|a protection order|immediate safety)\b",
    r"\b(he|she) (beat|hit|slap|assault|hurt|threaten|abuse|harass)s? me\b",
    r"\bi am (scared|afraid|in danger|unsafe|living in fear)\b",
    r"\bbruises?\b.{0,40}\bphotos?\b",  # "I have bruises, photos"
    r"\bhe threw me out\b",
)

_ACCUSED_SIDE_SIGNALS = (
    # Third-person defence / accused phrasing
    r"\b(our clients?|my clients?)\b.{0,80}(named|accused|facing|implicated|charged)",
    r"\b(parents?-in-law|in-laws?|relatives?)\b.{0,60}(named|accused|facing|implicated|FIR|complaint)",
    r"\bfalse (FIR|case|complaint|allegation)\b",
    r"\banticipatory bail\b",
    r"\bprotective strateg",  # typical defence counsel language
    r"\blimited interaction with the complainant\b",
    r"\bresidence proof showing.{0,40}lived separately\b",
)


def _detect_client_side(text: str) -> str:
    """Return 'victim', 'accused', or '' (unknown)."""
    low = (text or "").lower()
    victim_score = sum(1 for pat in _VICTIM_SIDE_SIGNALS if re.search(pat, low))
    accused_score = sum(1 for pat in _ACCUSED_SIDE_SIGNALS if re.search(pat, low))
    if victim_score > accused_score:
        return "victim"
    if accused_score > victim_score:
        return "accused"
    return ""


def _snapshot_to_compact_state(snapshot: dict | None, user_turns: list[str]) -> dict | None:
    if not snapshot:
        return None
    known_facts = [str(x).strip() for x in (snapshot.get("known_facts") or []) if str(x).strip()][:4]
    open_points = [str(x).strip() for x in (snapshot.get("unknowns") or []) if str(x).strip()][:4]
    relief = (
        snapshot.get("relief_sought")
        or snapshot.get("current_best_decision")
        or snapshot.get("next_information_needed")
        or ""
    )
    presenting = snapshot.get("presenting_problem") or (user_turns[0] if user_turns else "")
    facts_summary = presenting.strip()
    if relief:
        facts_summary = f"{facts_summary} Relief focus: {relief}.".strip()
    return {
        "route": "legal_opinion",
        "client_objective": str(relief).strip(),
        "urgency_level": str(snapshot.get("urgency_level") or "unknown").strip() or "unknown",
        "known_facts": known_facts,
        "prior_actions_taken": _infer_prior_actions(" ".join(user_turns)),
        "open_points": open_points,
        "enough_to_proceed": bool(snapshot.get("enough_to_advise")),
        "facts_summary": facts_summary[:320],
    }


def _normalise_runtime_record(raw: dict, origin: str) -> dict | None:
    messages = raw.get("messages") or []
    meta = raw.get("_meta") or {}
    if not messages:
        return None

    layer = meta.get("training_layer") or ""
    user_turns = [
        (m.get("content") or "").strip()
        for m in messages
        if m.get("role") == "user" and (m.get("content") or "").strip()
    ]
    assistant_turns = [
        (m.get("content") or "").strip()
        for m in messages
        if m.get("role") == "assistant" and (m.get("content") or "").strip()
    ]
    if not user_turns or not assistant_turns:
        return None

    final_assistant = assistant_turns[-1]
    snapshot = _extract_json_block(final_assistant)
    compact_state = _snapshot_to_compact_state(snapshot, user_turns)
    reply_text = _clean_reply_text(_strip_json_block(final_assistant))

    input_messages = messages[:-1] if messages[-1].get("role") == "assistant" else messages
    input_tail = [
        {
            "role": m.get("role", ""),
            "content": (m.get("content") or "").strip(),
        }
        for m in input_messages
        if (m.get("content") or "").strip()
    ][-4:]

    query_text = " ".join(user_turns)
    return {
        "id": meta.get("source_id") or raw.get("id") or origin,
        "layer": layer,
        "audience": _detect_audience(query_text, meta.get("audience_type") or ""),
        "domain": (meta.get("domain") or "").strip(),
        "client_side": _detect_client_side(query_text + " " + reply_text),
        "query_text": query_text,
        "input_tail": input_tail,
        "compact_state": compact_state,
        "reply_text": reply_text,
        "opinion_text": final_assistant if layer == "final_opinion" else "",
        "origin": origin,
    }


def _normalise_legacy_record(raw: dict, origin: str) -> dict | None:
    conversation = raw.get("conversation") or []
    opinion_text = (raw.get("opinion_text") or "").strip()
    if conversation:
        user_turns = [
            (m.get("content") or "").strip()
            for m in conversation
            if m.get("role") == "user" and (m.get("content") or "").strip()
        ]
        assistant_turns = [
            (m.get("content") or "").strip()
            for m in conversation
            if m.get("role") == "assistant" and (m.get("content") or "").strip()
        ]
        if not user_turns or not assistant_turns:
            return None
        layer = "complete_intake" if len(user_turns) >= 3 else ("clarification_follow_up" if len(user_turns) >= 2 else "intake")
        facts_summary = (raw.get("facts_summary") or "").strip() or _trim_words(" ".join(user_turns), 60)
        client_objective = _extract_client_objective(facts_summary)
        compact_state = {
            "route": "legal_opinion",
            "client_objective": client_objective,
            "urgency_level": "high" if any(tok in facts_summary.lower() for tok in ["urgent", "scared", "arrest", "detained", "violence"]) else "medium",
            "known_facts": _sentences(facts_summary, 4),
            "prior_actions_taken": _infer_prior_actions(" ".join(user_turns)),
            "open_points": [],
            "enough_to_proceed": layer == "complete_intake",
            "facts_summary": facts_summary[:320],
        }
        query_text_leg = " ".join(user_turns)
        return {
            "id": raw.get("id") or origin,
            "layer": layer,
            "audience": _detect_audience(query_text_leg),
            "domain": raw.get("case_type") or raw.get("description") or "",
            "client_side": _detect_client_side(query_text_leg + " " + (assistant_turns[-1] if assistant_turns else "")),
            "query_text": query_text_leg,
            "input_tail": [{"role": m.get("role", ""), "content": (m.get("content") or "").strip()} for m in conversation[:-1]][-4:],
            "compact_state": compact_state,
            "reply_text": assistant_turns[-1],
            "opinion_text": "",
            "origin": origin,
        }
    if opinion_text:
        facts = raw.get("description") or raw.get("case_type") or ""
        return {
            "id": raw.get("id") or origin,
            "layer": "final_opinion",
            "audience": _detect_audience(facts),
            "domain": raw.get("case_type") or "",
            "query_text": facts,
            "input_tail": [{"role": "user", "content": facts}],
            "compact_state": None,
            "reply_text": "",
            "opinion_text": opinion_text,
            "origin": origin,
        }
    return None


def _load_examples() -> list[dict]:
    global _RUNTIME_EXAMPLES
    if _RUNTIME_EXAMPLES is not None:
        return _RUNTIME_EXAMPLES

    examples: list[dict] = []
    for path in _candidate_files():
        loaded = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                raw_line = line.strip()
                if not raw_line:
                    continue
                try:
                    raw = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON in %s line %d", path.name, line_no)
                    continue
                origin = f"{path.name}:{line_no}"
                if "messages" in raw:
                    example = _normalise_runtime_record(raw, origin)
                else:
                    example = _normalise_legacy_record(raw, origin)
                if example:
                    examples.append(example)
                    loaded += 1
        logger.info("Loaded %d few-shot examples from %s", loaded, path.name)

    _RUNTIME_EXAMPLES = examples
    return _RUNTIME_EXAMPLES


def _score(example: dict, query_tokens: set[str], query_lower: str, desired_layers: set[str], desired_audience: str, query_client_side: str = "") -> float:
    layer = example.get("layer") or ""
    if desired_layers and layer not in desired_layers:
        return -1.0

    # ── Orientation guard: victim-side query must never match accused/defence example ──
    # This prevents a defence-side DV/matrimonial example (e.g. 498A accused family)
    # from being retrieved for a victim-side DV query, and vice versa.
    if query_client_side:
        example_side = example.get("client_side") or ""
        if example_side and example_side != query_client_side:
            return -1.0  # hard block — opposite orientation

    example_text = " ".join([
        str(example.get("query_text") or ""),
        str(example.get("domain") or ""),
        str(((example.get("compact_state") or {}).get("client_objective") or "")),
        str(((example.get("compact_state") or {}).get("facts_summary") or "")),
    ]).strip()
    example_tokens = _tokenize(example_text)
    domain_lower = (example.get("domain") or "").lower()

    lexical_overlap = example_tokens & query_tokens
    lexical = len(lexical_overlap) * 0.55

    topic_rules = [
        ("criminal", {"bail", "fir", "arrest", "detention", "custody", "habeas", "remand", "chargesheet", "crime", "complaint"}, {"criminal", "detention", "bail", "police"}),
        ("domestic_family", {"husband", "wife", "marriage", "matrimonial", "domestic", "violence", "cruelty", "dowry", "maintenance", "498a", "assaulted", "bruises"}, {"family", "matrimonial", "domestic", "women", "maintenance"}),
        ("tax", {"tax", "assessment", "deduction", "tds", "royalty", "withholding", "penalty"}, {"tax"}),
        ("property", {"property", "land", "title", "mutation", "deed", "forgery", "encroachment", "possession"}, {"property"}),
        ("accident", {"accident", "injury", "insurer", "disability", "compensation", "hospital", "medical"}, {"accident", "compensation", "insurance"}),
        ("labour", {"salary", "wages", "termination", "overtime", "labour", "employee", "dismissal"}, {"labour", "employment", "service"}),
        ("housing_consumer", {"housing", "flat", "allotment", "seepage", "builder", "authority", "defect", "defects", "demolition", "municipal"}, {"housing", "consumer", "municipal", "public"}),
        ("insolvency", {"insolvency", "creditor", "creditors", "valuation", "auction", "sale", "revival"}, {"insolvency", "company"}),
        ("commercial", {"cheque", "security", "notice", "lender", "loan", "bank", "recovery"}, {"cheque", "commercial", "banking", "finance"}),
        ("recruitment", {"recruitment", "selection", "viva", "interview", "bias", "candidate", "reservation"}, {"recruitment", "service", "education"}),
    ]

    active_topics = {name for name, q_words, _ in topic_rules if query_tokens & q_words}
    topical = 0.0
    mismatch_penalty = 0.0
    for name, _q_words, d_words in topic_rules:
        domain_match = any(word in domain_lower for word in d_words)
        if name in active_topics and domain_match:
            topical += 1.6
        elif name in active_topics and not domain_match and any(word in example_tokens for word in d_words):
            mismatch_penalty += 0.8

    if "domestic_family" in active_topics:
        if any(word in domain_lower for word in {"family", "matrimonial", "domestic", "women", "maintenance"}):
            topical += 1.2
        else:
            mismatch_penalty += 1.1
    if "housing_consumer" in active_topics and not any(word in domain_lower for word in {"housing", "consumer", "municipal", "public"}):
        mismatch_penalty += 0.7

    if active_topics and topical <= 0 and lexical < 0.6:
        return -1.0
    if lexical <= 0 and topical <= 0:
        return -1.0

    score = lexical + topical - mismatch_penalty

    if desired_audience and example.get("audience") == desired_audience:
        score += 0.6
    if layer == "complete_intake":
        score += 0.1
    if layer == "final_opinion":
        score += 0.2

    return score if score > 0 else -1.0


def _rank_examples(query: str, desired_layers: set[str]) -> list[dict]:
    examples = _load_examples()
    query_lower = (query or "").lower()
    query_tokens = _tokenize(query or "")
    desired_audience = _detect_audience(query or "")
    query_client_side = _detect_client_side(query or "")
    ranked: list[tuple[float, dict]] = []
    for example in examples:
        score = _score(example, query_tokens, query_lower, desired_layers, desired_audience, query_client_side)
        if score > 0:
            ranked.append((score, example))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [example for _, example in ranked]


def _format_input_tail(input_tail: list[dict]) -> str:
    lines: list[str] = []
    for turn in input_tail:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = _trim_words(turn.get("content") or "", 80)
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_state_example(example: dict) -> str:
    state = example.get("compact_state") or {}
    if not state:
        return ""
    return "\n".join([
        "REFERENCE STATE EXAMPLE - mirror the structure, not the facts.",
        "Conversation:",
        _format_input_tail(example.get("input_tail") or []),
        "Target state:",
        json.dumps(state, ensure_ascii=False),
    ]).strip()


def _format_reply_example(example: dict) -> str:
    state = example.get("compact_state") or {}
    if not state:
        return ""
    if example.get("layer") == "complete_intake":
        target = {
            "action": "complete",
            "intent": "legal_opinion",
            "facts_summary": state.get("facts_summary", ""),
            "reply_to_client": _trim_words(example.get("reply_text") or "", 120),
        }
    else:
        target = {
            "action": "ask",
            "reply_to_client": _trim_words(example.get("reply_text") or "", 120),
        }
    return "\n".join([
        "REFERENCE NEXT-MOVE EXAMPLE - mirror the behavior, not the facts.",
        "Compact case state:",
        json.dumps(state, ensure_ascii=False),
        "Target output:",
        json.dumps(target, ensure_ascii=False),
    ]).strip()


def _format_opinion_example(example: dict) -> str:
    opinion = _trim_words(_flatten_opinion_text(example.get("opinion_text") or ""), 220)
    if not opinion:
        return ""
    facts = _trim_words(example.get("query_text") or "", 90)
    return "\n".join([
        "REFERENCE FINAL-OPINION EXAMPLE - mirror the structure and tone, not the facts or citations.",
        f"Client record: {facts}",
        "Opinion:",
        opinion,
    ]).strip()


def _pick_unique(ranked: list[dict], max_examples: int) -> list[dict]:
    chosen: list[dict] = []
    seen_ids: set[str] = set()
    for example in ranked:
        ex_id = example.get("id") or example.get("origin")
        if ex_id in seen_ids:
            continue
        seen_ids.add(ex_id)
        chosen.append(example)
        if len(chosen) >= max_examples:
            break
    return chosen


def get_intake_state_example_pack(query: str, max_examples: int = 1) -> str | None:
    ranked = _rank_examples(query, {"intake", "clarification_follow_up", "complete_intake"})
    chosen = _pick_unique(ranked, max_examples)
    rendered = [_format_state_example(example) for example in chosen]
    rendered = [item for item in rendered if item]
    return "\n\n".join(rendered) if rendered else None


def get_intake_reply_example_pack(query: str, max_examples: int = 2) -> str | None:
    ranked = _rank_examples(query, {"intake", "clarification_follow_up", "complete_intake"})
    chosen = _pick_unique(ranked, max_examples)
    rendered = [_format_reply_example(example) for example in chosen]
    rendered = [item for item in rendered if item]
    return "\n\n".join(rendered) if rendered else None


def get_intake_example(query: str) -> str | None:
    return get_intake_reply_example_pack(query, max_examples=1)


def get_intake_example_pack(query: str, max_examples: int = 2) -> str | None:
    return get_intake_reply_example_pack(query, max_examples=max_examples)


def get_opinion_example(query: str) -> str | None:
    ranked = _rank_examples(query, {"final_opinion"})
    chosen = _pick_unique(ranked, 1)
    if not chosen:
        return None
    return _format_opinion_example(chosen[0]) or None


def preload_examples() -> None:
    _load_examples()


def reload_examples() -> None:
    global _RUNTIME_EXAMPLES
    _RUNTIME_EXAMPLES = None
    logger.info("Few-shot example cache cleared")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(get_intake_state_example_pack("My husband threw me out and I need protection and maintenance.") or "<none>")
    print("-" * 80)
    print(get_intake_reply_example_pack("False FIR and need anticipatory bail") or "<none>")
    print("-" * 80)
    print(get_opinion_example("Road accident compensation dispute") or "<none>")
