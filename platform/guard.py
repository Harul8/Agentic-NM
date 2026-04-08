"""
platform/guard.py — Content safety, sanitization, prompt injection detection.
"""
import logging
import re

logger = logging.getLogger("nyaymalaw.guard")

# ---------------------------------------------------------------------------
# 1. Harmful intent patterns — queries seeking help with illegal acts
# ---------------------------------------------------------------------------
_HARMFUL_PATTERNS = [
    # Violence / threats
    (r"\b(how\s+to\s+)?(kill|murder|poison|assassinate|attack)\s+(someone|a\s+person|my|wife|husband|boss)", "violence"),
    (r"\bhow\s+to\s+(harm|hurt|injure|torture)\b", "violence"),
    (r"\b(bomb|explosive|weapon)\s+(make|build|create|assemble)\b", "weapons"),
    (r"\b(make|build|create)\s+(a\s+)?(bomb|explosive|weapon)\b", "weapons"),
    # Sexual exploitation / minors
    (r"\b(child|minor|underage)\s+(marriage|sex|exploit|abuse|porn)\b", "child_safety"),
    (r"\b(rape|molest|trafficking)\s+(guide|how|tips|method)\b", "exploitation"),
    # Fraud / forgery
    (r"\bhow\s+to\s+(forge|fake|counterfeit|fabricate)\s+(document|evidence|signature|currency|will|affidavit)\b", "fraud"),
    (r"\b(create|make)\s+(fake|forged|false)\s+(id|passport|license|aadhaar|pan\s*card)\b", "identity_fraud"),
    # Drug manufacturing
    (r"\bhow\s+to\s+(make|manufacture|produce|synthesize)\s+(drugs|meth|cocaine|heroin|fentanyl)\b", "drugs"),
    # Hacking / cybercrime
    (r"\bhow\s+to\s+(hack|breach|crack|phish|ddos)\b", "cybercrime"),
    # Terrorism
    (r"\b(plan|execute|carry\s+out)\s+(attack|bombing|terror)\b", "terrorism"),
    (r"\bjihad\s+(guide|how|manual)\b", "terrorism"),
]

# Compiled for performance
_HARMFUL_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in _HARMFUL_PATTERNS]

# ---------------------------------------------------------------------------
# 2. Prompt injection patterns — attempts to override system prompts
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)",
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"\[\s*INST\s*\]",
    r"forget\s+(everything|all|your)\s+(you|instructions|rules)",
    r"new\s+instruction\s*:",
    r"override\s+(safety|content|moderation)",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"DAN\s+mode",
]

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

    # Check prompt injection
    if is_prompt_injection(cleaned):
        logger.warning("Prompt injection detected: %s", cleaned[:100])
        return {
            "safe": False,
            "risk_level": "blocked",
            "reason": "Your message appears to contain instructions that aren't related to legal queries. Please rephrase your legal question.",
            "pii_warning": None,
        }

    # Check harmful intent
    for regex, label in _HARMFUL_RE:
        if regex.search(cleaned):
            logger.warning("Harmful query detected (%s): %s", label, cleaned[:100])
            return {
                "safe": False,
                "risk_level": "blocked",
                "reason": (
                    "I'm designed to help with legitimate legal research and queries. "
                    "I cannot assist with requests that may involve harmful or illegal activities. "
                    "If you have a genuine legal concern, please rephrase your question."
                ),
                "pii_warning": None,
            }

    # Check PII (warn but don't block)
    pii_warnings = []
    for regex, label in _PII_RE:
        if regex.search(cleaned):
            pii_warnings.append(label)

    if pii_warnings:
        result["pii_warning"] = (
            f"It looks like your message may contain sensitive information ({', '.join(pii_warnings)}). "
            "For your security, please avoid sharing personal identification numbers in chat. "
            "Your query will still be processed, but this data is not needed for legal research."
        )
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
    """
    if not text:
        return False

    for regex in _INJECTION_RE:
        if regex.search(text):
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
