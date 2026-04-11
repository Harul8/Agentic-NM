"""
retrieval/guard.py — Content safety, sanitization, prompt injection detection.
"""
import json
import logging
import re
import unicodedata

logger = logging.getLogger("nyaymalaw.guard")


# ---------------------------------------------------------------------------
# 1. Harmful-content classifier
#
# Architecture: two-stage to avoid burning LLM tokens on every legal query.
#
#   Stage 1 — O(1) regex keyword pre-screen.
#              The vast majority of legitimate Indian legal queries (Section X,
#              IPC, CrPC, contracts, property, divorce, etc.) contain none of
#              these keywords and exit here with return False — zero API cost.
#
#   Stage 2 — LLM call, ONLY when Stage 1 fires.
#              The LLM disambiguates edge cases like "how do I poison-pill a
#              merger?" (legal M&A term, not harmful) vs. genuinely harmful
#              requests.  Estimated trigger rate: < 1 % of normal legal traffic.
# ---------------------------------------------------------------------------
_HARMFUL_QUICK_KEYWORDS = re.compile(
    r"\b("
    r"kill\s+(?:someone|a\s+person|him|her|them|people)|"
    r"murder(?:ing)?\s+(?:someone|a\s+person)|"
    r"bomb(?:ing)?\s+(?:a|the)\s+\w+|"
    r"make\s+(?:a\s+)?(?:bomb|explosive|poison|meth|heroin|fentanyl)|"
    r"synthesize\s+\w+(?:\s+drug|s)?|"
    r"how\s+to\s+(?:make|build|create)\s+(?:a\s+)?(?:bomb|explosive|weapon|virus|malware)|"
    r"child\s+(?:porn|pornography|exploit|grooming)|"
    r"csam|"
    r"sexual\s+(?:content|material)\s+(?:involving\s+)?(?:minor|child)|"
    r"hack(?:ing)?\s+(?:into\s+)?(?:bank|government|hospital)|"
    r"ransomware|"
    r"ddos\s+attack|"
    r"commit\s+(?:fraud|terrorism|genocide)|"
    r"terrorist\s+attack"
    r")\b",
    re.IGNORECASE,
)


def _normalize_for_screening(text: str) -> str:
    """NFKC-normalize + remove zero-width chars to defeat Unicode evasion."""
    normalized = unicodedata.normalize("NFKC", text)
    # Strip zero-width / invisible characters
    normalized = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", normalized)
    return normalized


def _llm_is_harmful(text: str) -> bool:
    """
    Use the LLM to classify whether a query has genuinely harmful intent.

    ONLY called after the fast keyword pre-screen fires — not on every query.
    Falls back to False (allow) on any error so a classifier failure never
    silently blocks legitimate legal queries.
    """
    try:
        from platform_pkg.llm import ask_llm
        prompt = (
            "You are a content safety classifier for an Indian legal research assistant.\n"
            "Classify the user message below as harmful or not.\n"
            "Return ONLY JSON: {\"harmful\": true/false, \"reason\": \"<5 words max>\"}\n\n"
            "HARMFUL means the user is explicitly seeking help to commit violence, fraud, drug manufacturing, "
            "terrorism, child exploitation, or serious cybercrime — NOT legal questions about these topics.\n"
            "A victim describing abuse, a lawyer researching a crime, or a student asking about criminal law "
            "is NOT harmful.\n\n"
            f"User message: {text[:600]}\n"
        )
        raw = (ask_llm(prompt, task_hint="fast") or "").strip()
        if "{" in raw and "}" in raw:
            raw = raw[raw.find("{"): raw.rfind("}") + 1]
        data = json.loads(raw)
        return bool(data.get("harmful", False))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 2. Prompt injection patterns — attempts to override system prompts
#
# M3 fix: patterns now run against NFKC-normalised text so Unicode homoglyphs
# and invisible separators can't bypass them.
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    # Classic override phrases
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?|constraints?)",
    r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(everything|all|your)\s*(you|instructions?|rules?|training)?",
    # Role-switching
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"act\s+as\s+(a|an|the)?\s*(?!lawyer|counsel|judge|advocate)",  # allow legal roles
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"roleplay\s+as\s+",
    # System-prompt injection markers
    r"system\s*:\s*(?!\w+\s+(?:prompt|message|error))",  # allow "system error"
    r"<\s*system\s*>",
    r"\[\s*INST\s*\]",
    r"\[\s*/?SYS\s*\]",
    r"<\s*\|?\s*im_start\s*\|?\s*>",
    # Override / jailbreak vocabulary
    r"new\s+instruction\s*:",
    r"override\s+(safety|content|moderation|filter)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"god\s+mode",
    # Prompt leakage fishing
    r"(print|repeat|output|reveal|show|tell\s+me)\s+(your|the)\s+(system\s+)?prompt",
    r"what\s+(are|were)\s+your\s+(instructions?|rules?|guidelines?)",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# ---------------------------------------------------------------------------
# 3. Sensitive data patterns — warn if user shares PII
# ---------------------------------------------------------------------------
_PII_PATTERNS = [
    (r"\b\d{4}\s?\d{4}\s?\d{4}\b", "Aadhaar number"),
    (r"\b[A-Z]{5}\d{4}[A-Z]\b", "PAN card number"),
    (r"\b\d{9,18}\b(?=.*(?:account|bank|ifsc))", "bank account number"),
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b", "credit card number"),
]

_PII_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in _PII_PATTERNS]


def check_query_safety(text: str) -> dict:
    """
    Check if a user query is safe to process.

    Two-stage harmful-content check:
      1. Fast regex keyword pre-screen  — O(1), no API cost.
      2. LLM classifier                — only when Stage 1 fires (~1 % of traffic).

    Returns:
        {
            "safe": bool,
            "risk_level": "none" | "low" | "high" | "blocked",
            "reason": str (empty if safe),
            "pii_warning": str | None (if PII detected),
        }
    """
    if not text or not text.strip():
        return {"safe": True, "risk_level": "none", "reason": "", "pii_warning": None}

    result = {"safe": True, "risk_level": "none", "reason": "", "pii_warning": None}
    cleaned = text.strip()

    # Normalise once — used for both injection and harmful checks (M3 fix)
    normalized = _normalize_for_screening(cleaned)

    # ── Stage A: Prompt injection check (runs on normalized text) ────────────
    if is_prompt_injection(normalized):
        logger.warning("Prompt injection detected: %s", cleaned[:100])
        return {
            "safe": False,
            "risk_level": "blocked",
            "reason": (
                "Your message appears to contain instructions that aren't related to legal "
                "queries. Please rephrase your legal question."
            ),
            "pii_warning": None,
        }

    # ── Stage B: Harmful-content check — two-stage, API call is gated ────────
    # Do not block user input based on substantive content alone.
    # Safety handling should shape the model's response, not reject the query.
    if _HARMFUL_QUICK_KEYWORDS.search(normalized):
        # Only now do we pay for an LLM call to disambiguate edge cases
        # (e.g. "poison-pill merger" is M&A terminology, not harmful intent)
        if _llm_is_harmful(cleaned):
            logger.warning("High-risk query confirmed by LLM classifier: %s", cleaned[:100])
            result["risk_level"] = "high"
        else:
            logger.info("Keyword pre-screen fired but LLM cleared query: %s", cleaned[:80])
    # else: query contains no suspicious keywords → skip LLM call entirely

    # ── Stage C: PII detection (warn but don't block) ────────────────────────
    pii_warnings = []
    for regex, label in _PII_RE:
        if regex.search(cleaned):
            pii_warnings.append(label)

    if pii_warnings:
        result["pii_warning"] = (
            f"It looks like your message may contain sensitive information "
            f"({', '.join(pii_warnings)}). "
            "For your security, please avoid sharing personal identification numbers in chat. "
            "Your query will still be processed, but this data is not needed for legal research."
        )
        if result["risk_level"] == "none":
            result["risk_level"] = "low"
        logger.info("PII detected in query: %s", ", ".join(pii_warnings))

    return result


def sanitize_input(text: str) -> str:
    """
    Clean user input — remove control characters and excessive whitespace.
    Does NOT alter legal content or technical terms.
    """
    if not text:
        return ""

    # Remove null bytes and control chars (keep newlines, tabs)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse excessive whitespace (preserve single newlines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


def is_prompt_injection(text: str) -> bool:
    """
    Detect common prompt injection attempts.
    Input is NFKC-normalised before matching so Unicode homoglyphs can't bypass
    pattern detection (M3 fix — caller may pre-normalise; normalising twice is safe).
    """
    if not text:
        return False

    normalized = _normalize_for_screening(text)
    for regex in _INJECTION_RE:
        if regex.search(normalized):
            return True

    return False


# ---------------------------------------------------------------------------
# Response-level safety: ensure LLM output doesn't contain harmful content
# ---------------------------------------------------------------------------
_OUTPUT_REDFLAGS = [
    r"(?i)\b(here'?s?\s+how\s+to\s+)?(kill|murder|harm)\s+someone\b",
    r"(?i)\bstep[\s-]*by[\s-]*step\s+(guide|instructions?)\s+(to|for)\s+(kill|harm|forge|hack)\b",
]

_OUTPUT_RE = [re.compile(p) for p in _OUTPUT_REDFLAGS]


def check_response_safety(text: str) -> dict:
    """
    Light check on LLM output for red-flag content.

    Returns:
        {"safe": bool, "reason": str}
    """
    if not text:
        return {"safe": True, "reason": ""}

    for regex in _OUTPUT_RE:
        if regex.search(text):
            logger.warning("Unsafe LLM output detected: %s", text[:100])
            return {
                "safe": False,
                "reason": "The generated response was flagged for review. Please rephrase your query.",
            }

    return {"safe": True, "reason": ""}
