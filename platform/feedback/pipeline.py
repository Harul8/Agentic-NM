"""
feedback/pipeline.py — Feedback → training artifact distillation.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from config import LEGAL_DB_CHAT_HISTORY
from platform.feedback.store import feedback_store_path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TRAINING_DIR = _PROJECT_ROOT / "training" / "finetune_ready"
_EVAL_DIR = _PROJECT_ROOT / "eval" / "data"
_CHAT_HISTORY_DIR = Path(LEGAL_DB_CHAT_HISTORY)

_STATE_PATH = _CHAT_HISTORY_DIR / "feedback_distillation_state.json"
_SUMMARY_PATH = _CHAT_HISTORY_DIR / "feedback_distillation_summary.json"
_PATTERNS_PATH = _TRAINING_DIR / "feedback_generalized_patterns.json"
_TRAINING_CANDIDATES_PATH = _TRAINING_DIR / "feedback_training_candidates.jsonl"
_EVAL_CANDIDATES_PATH = _EVAL_DIR / "feedback_eval_candidates.json"

_INTAKE_TRAINING_SYSTEM = (
    "You are a senior advocate at Nyaymalaw. Respond with empathy, legal judgment, "
    "and one high-value next step or question. Do not proceed too early. Do not ask "
    "low-value or repetitive questions. Prioritize urgency, prior actions, objective, "
    "and practical protection."
)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "with", "is", "are",
    "was", "were", "be", "been", "being", "my", "our", "his", "her", "their", "that",
    "this", "it", "at", "by", "from", "as", "into", "about", "after", "before", "me",
    "us", "you", "your", "he", "she", "they", "them", "we", "i",
}

_LESSON_MAP: dict[str, str] = {
    "wrong_followup": "Ask the single next question that most changes legal advice, not a low-value detail question.",
    "repeated_question": "Do not repeat a question once the fact has already been given or reasonably established.",
    "premature_proceed": "Do not move to legal analysis until urgency, client objective, and prior-action position are sufficiently clear.",
    "missed_urgency": "When ongoing harm or deadline risk appears, acknowledge it and assess immediate safety or timing before secondary details.",
    "missed_prior_actions": "Ask what has already been reported, filed, admitted, paid, or informally settled before deciding the path ahead.",
    "missed_client_objective": "Clarify what the client wants right now: protection, recovery, complaint, defence, settlement, or another concrete outcome.",
    "poor_empathy": "Acknowledge distress in plain human language before moving into legal questioning.",
    "poor_clarity": "Use simpler, more direct language and ask only one focused question at a time.",
    "unsupported_legal_reference": "Do not cite legal provisions or authorities unless grounded in approved retrieved materials.",
    "poor_grounding": "Keep legal statements tied to retrieved local materials and avoid unsupported claims.",
    "hallucinated_query_expansion": "Do not invent sections, Act names, or legal issues that the client did not raise and the materials do not support.",
    "bad_stop_continue_judgment": "Stop only when the next practical step is clear and no blocking fact remains unresolved.",
    "too_verbose": "Be concise and decision-oriented rather than expansive.",
    "too_slow": "Prefer the smallest useful decision step and avoid unnecessary processing for intake responses.",
    "strong_reasoning": "Preserve the same decision-oriented prioritization and issue framing.",
    "strong_empathy": "Preserve the same humane, reassuring tone.",
    "strong_grounding": "Preserve the same disciplined grounding in retrieved materials.",
}


def _ensure_dirs() -> None:
    _CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    _TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)


def _today_iso() -> str:
    return datetime.now().date().isoformat()


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_feedback_records() -> list[dict[str, Any]]:
    path = feedback_store_path()
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _extract_keywords(text: str, limit: int = 6) -> list[str]:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    counts = Counter(w for w in words if w not in _STOPWORDS)
    return [word for word, _ in counts.most_common(limit)]


def _classify_scenario(user_message: str) -> str:
    text = user_message.lower()
    if any(k in text for k in ("husband", "wife", "marriage", "maintenance", "alimony")):
        if any(k in text for k in ("beat", "assault", "harass", "terror", "violent", "drunk")):
            return "matrimonial_violence"
        return "matrimonial"
    if any(k in text for k in ("fired", "terminated", "salary", "gratuity", "job")):
        return "employment"
    if any(k in text for k in ("cheque", "loan", "payment", "money back")):
        return "recovery_financial"
    if any(k in text for k in ("property", "land", "encroach", "wall", "possession")):
        return "property"
    if any(k in text for k in ("police", "fir", "arrest", "complaint")):
        return "criminal_process"
    return "general_legal_intake"


def _generalize_user_issue(user_message: str, scenario_type: str, tags: list[str]) -> str:
    text = _clean_text(user_message)
    if scenario_type == "matrimonial_violence":
        if "missed_urgency" in tags:
            return "Client reports ongoing domestic violence affecting both the client and minor children, with immediate safety concerns."
        return "Client reports ongoing matrimonial abuse and child-safety concerns in a shared home."
    if scenario_type == "matrimonial":
        return "Client presents with a matrimonial dispute requiring objective, forum, and relief clarification."
    if scenario_type == "employment":
        return "Client presents with an employment dispute where facts, documents, and desired relief must be clarified."
    if scenario_type == "recovery_financial":
        return "Client presents with a recovery or payment dispute where timing, documentation, and remedy selection matter."
    if scenario_type == "property":
        return "Client presents with a property dispute where possession, documents, and immediate protective relief matter."
    if scenario_type == "criminal_process":
        return "Client presents with a criminal-process problem where urgency, prior actions, and procedural posture matter."
    keywords = _extract_keywords(text, limit=4)
    if keywords:
        return f"Client presents a legal intake involving: {', '.join(keywords)}."
    return "Client presents a legal problem that requires structured intake before analysis."


def _extract_preferred_response(free_text: str) -> str:
    text = _clean_text(free_text)
    if not text:
        return ""
    patterns = [
        r"^\s*a better response would have been[,:\-]?\s*",
        r"^\s*better response would have been[,:\-]?\s*",
        r"^\s*the better response would have been[,:\-]?\s*",
        r"^\s*better question would have been[,:\-]?\s*",
        r"^\s*the better question would have been[,:\-]?\s*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip(" \"'")


def _build_preferred_behavior(tags: list[str]) -> list[str]:
    behaviors = []
    for tag in tags:
        lesson = _LESSON_MAP.get(tag)
        if lesson and lesson not in behaviors:
            behaviors.append(lesson)
    return behaviors


def _build_training_candidate(record: dict[str, Any]) -> dict[str, Any] | None:
    rating = _clean_text(record.get("rating")).lower()
    tags = [str(t) for t in (record.get("reason_tags") or []) if str(t).strip()]
    free_text = _clean_text(record.get("free_text"))
    assistant_text = _clean_text(record.get("assistant_text"))
    user_message = _clean_text(record.get("user_message"))
    preferred_response = _extract_preferred_response(free_text)

    if rating in {"bad", "okay"} and not preferred_response:
        return None
    if rating == "good" and not assistant_text:
        return None

    target_response = preferred_response or assistant_text
    scenario_type = _classify_scenario(user_message)
    generalized_issue = _generalize_user_issue(user_message, scenario_type, tags)
    feedback_tags = tags or (["strong_reasoning"] if rating == "good" else [])
    preferred_behavior = _build_preferred_behavior(feedback_tags)

    return {
        "conversations": [
            {"role": "system", "content": _INTAKE_TRAINING_SYSTEM},
            {"role": "user", "content": user_message or generalized_issue},
            {"role": "assistant", "content": target_response},
        ],
        "_meta": {
            "source": "response_feedback",
            "message_id": record.get("message_id", ""),
            "chat_id": record.get("chat_id", ""),
            "rating": rating,
            "feedback_tags": feedback_tags,
            "stage": record.get("stage", ""),
            "response_type": record.get("response_type", ""),
            "model_used": record.get("model_used", ""),
            "generalized_issue": generalized_issue,
            "scenario_type": scenario_type,
            "preferred_behavior": preferred_behavior,
            "assistant_text_original": assistant_text,
            "free_text_feedback": free_text,
            "logged_at": record.get("logged_at", ""),
        },
    }


def _build_eval_candidate(record: dict[str, Any]) -> dict[str, Any] | None:
    tags = [str(t) for t in (record.get("reason_tags") or []) if str(t).strip()]
    user_message = _clean_text(record.get("user_message"))
    assistant_text = _clean_text(record.get("assistant_text"))
    free_text = _clean_text(record.get("free_text"))
    preferred_response = _extract_preferred_response(free_text)
    if not user_message or not assistant_text:
        return None
    scenario_type = _classify_scenario(user_message)
    return {
        "id": f"feedback_{record.get('message_id', '') or len(user_message)}",
        "stage": record.get("stage", ""),
        "rating": record.get("rating", ""),
        "scenario_type": scenario_type,
        "user_message": user_message,
        "assistant_text": assistant_text,
        "reason_tags": tags,
        "free_text_feedback": free_text,
        "preferred_response": preferred_response,
        "generalized_issue": _generalize_user_issue(user_message, scenario_type, tags),
        "preferred_behavior": _build_preferred_behavior(tags),
        "latency_ms": record.get("latency_ms"),
        "model_used": record.get("model_used", ""),
        "logged_at": record.get("logged_at", ""),
    }


def _build_patterns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        tags = sorted(str(t) for t in (record.get("reason_tags") or []) if str(t).strip())
        scenario = _classify_scenario(_clean_text(record.get("user_message")))
        grouped[(scenario, "|".join(tags))].append(record)

    patterns: list[dict[str, Any]] = []
    for (scenario, tags_key), items in grouped.items():
        tags = [t for t in tags_key.split("|") if t]
        sample = items[0]
        patterns.append(
            {
                "scenario_type": scenario,
                "reason_tags": tags,
                "count": len(items),
                "generalized_issue": _generalize_user_issue(
                    _clean_text(sample.get("user_message")),
                    scenario,
                    tags,
                ),
                "preferred_behavior": _build_preferred_behavior(tags),
                "sample_user_message": _clean_text(sample.get("user_message")),
                "sample_bad_response": _clean_text(sample.get("assistant_text")),
                "sample_preferred_response": _extract_preferred_response(_clean_text(sample.get("free_text"))),
                "message_ids": [str(item.get("message_id", "")) for item in items[:10]],
            }
        )
    patterns.sort(key=lambda p: (-p["count"], p["scenario_type"], "|".join(p["reason_tags"])))
    return patterns


def _build_summary(
    records: list[dict[str, Any]],
    training_candidates: list[dict[str, Any]],
    eval_candidates: list[dict[str, Any]],
    patterns: list[dict[str, Any]],
) -> dict[str, Any]:
    rating_counts = Counter(_clean_text(r.get("rating")).lower() for r in records if _clean_text(r.get("rating")))
    stage_counts = Counter(_clean_text(r.get("stage")).lower() for r in records if _clean_text(r.get("stage")))
    tag_counts = Counter()
    for record in records:
        tag_counts.update(str(tag) for tag in (record.get("reason_tags") or []) if str(tag).strip())

    top_failure_tags = [
        {"tag": tag, "count": count, "lesson": _LESSON_MAP.get(tag, "")}
        for tag, count in tag_counts.most_common(10)
    ]

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "feedback_records_total": len(records),
        "training_candidates_total": len(training_candidates),
        "eval_candidates_total": len(eval_candidates),
        "rating_counts": dict(rating_counts),
        "stage_counts": dict(stage_counts),
        "top_feedback_tags": top_failure_tags,
        "pattern_count": len(patterns),
        "top_patterns": patterns[:10],
        "output_files": {
            "summary": str(_SUMMARY_PATH),
            "patterns": str(_PATTERNS_PATH),
            "training_candidates": str(_TRAINING_CANDIDATES_PATH),
            "eval_candidates": str(_EVAL_CANDIDATES_PATH),
        },
    }


def run_daily_feedback_distillation(force: bool = False) -> dict[str, Any]:
    """
    Distill live response feedback into reusable artifacts once per day.
    """
    _ensure_dirs()
    today = _today_iso()
    state = _load_json(_STATE_PATH, {})
    if not force and state.get("last_processed_date") == today:
        return {
            "status": "skipped",
            "reason": "already_processed_today",
            "last_processed_at": state.get("last_processed_at", ""),
            "date": today,
        }

    records = _load_feedback_records()
    training_candidates = [c for c in (_build_training_candidate(r) for r in records) if c]
    eval_candidates = [c for c in (_build_eval_candidate(r) for r in records) if c]
    patterns = _build_patterns(records)
    summary = _build_summary(records, training_candidates, eval_candidates, patterns)

    _write_json(_SUMMARY_PATH, summary)
    _write_json(_PATTERNS_PATH, {
        "generated_at": summary["generated_at"],
        "patterns": patterns,
    })
    _write_jsonl(_TRAINING_CANDIDATES_PATH, training_candidates)
    _write_json(_EVAL_CANDIDATES_PATH, {
        "generated_at": summary["generated_at"],
        "records": eval_candidates,
    })

    state_payload = {
        "last_processed_date": today,
        "last_processed_at": summary["generated_at"],
        "feedback_records_total": len(records),
        "training_candidates_total": len(training_candidates),
        "eval_candidates_total": len(eval_candidates),
    }
    _write_json(_STATE_PATH, state_payload)
    logger.info(
        "Feedback distillation complete | feedback=%s training_candidates=%s eval_candidates=%s",
        len(records),
        len(training_candidates),
        len(eval_candidates),
    )
    return {
        "status": "processed",
        "date": today,
        **state_payload,
        "summary_path": str(_SUMMARY_PATH),
        "patterns_path": str(_PATTERNS_PATH),
        "training_candidates_path": str(_TRAINING_CANDIDATES_PATH),
        "eval_candidates_path": str(_EVAL_CANDIDATES_PATH),
    }
