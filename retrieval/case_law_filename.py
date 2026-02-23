"""
Suggest a safe filename for a case law PDF from the first page.

Structure (first page):
- Court: "In the Supreme Court of India" → SC_; "In the High Court of [State]" → HC_
- Appellant: name(s) on the left; on the right "... Appellants?". Use first name only.
- Respondent: name(s) on the left; on the right "... Respondent(s)?". Use first name only.
- Format: SC_Appellant_name versus Respondent_name  or  HC_Appellant_name versus Respondent_name

Fallback when court not found: look for "PetitionType No. XXX of YYYY" (e.g. Criminal Appeal No. 192 of 2011,
Civil Appeal No. ..., Writ Petition No. ...). Then: PetitionType_No_XXX_of_YYYY.
"""

import re

# First page only for structure
_FIRST_PAGE_CHARS = 3500

_UNSAFE_FS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Court detection (case-insensitive)
_SUPREME_COURT = re.compile(r"\b(?:in\s+the\s+)?supreme\s+court\s+of\s+india\b", re.IGNORECASE)
_HIGH_COURT = re.compile(r"\b(?:in\s+the\s+)?high\s+court\s+of\s+(?:the\s+state\s+of\s+)?\w+\b", re.IGNORECASE)

# Appellant: on same line, text to the left of "... Appellants?" (no DOTALL = same line)
_APPELLANT_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9\s\.\,\'\-\@\&]+?)\s*\.{2,4}\s*Appellants?\b",
    re.IGNORECASE,
)
# Respondent: same idea
_RESPONDENT_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9\s\.\,\'\-\@\&]+?)\s*\.{2,4}\s*Respondents?\b",
    re.IGNORECASE,
)

# Fallback when court not detected: "Criminal Appeal No. 192 of 2011", "Writ Petition No. 123 of 2020", etc.
_APPEAL_NO_OF_YEAR = re.compile(
    r"\b(\w+(?:\s+\w+)+)\s+No\.?\s*([A-Za-z0-9\-]+)\s+of\s+(19|20)(\d{2})\b",
    re.IGNORECASE,
)


def _safe_filename(name: str) -> str:
    """Replace unsafe filesystem chars with space; collapse spaces; strip."""
    s = _UNSAFE_FS.sub(" ", name)
    return " ".join(s.split()).strip() or "document"


def _first_name_only(party_text: str) -> str:
    """If multiple parties (e.g. 'X & ANR.' or 'X @ Y'), return first name only."""
    if not party_text or not party_text.strip():
        return ""
    s = party_text.strip()
    # Split by " & " or " &" or "& " (multiple appellants/respondents)
    parts = re.split(r"\s+&\s+", s, maxsplit=1)
    s = parts[0].strip()
    # Split by " @ " (alternate names) and take first
    parts = re.split(r"\s+@\s+", s, maxsplit=1)
    s = parts[0].strip()
    return " ".join(s.split()).strip()


def _detect_court(text: str) -> str:
    """Return 'SC' for Supreme Court, 'HC' for High Court, '' if not found."""
    if _SUPREME_COURT.search(text):
        return "SC"
    if _HIGH_COURT.search(text):
        return "HC"
    return ""


def _extract_appellant(text: str) -> str:
    """Extract appellant name (first name only) from first page text."""
    m = _APPELLANT_PATTERN.search(text)
    if not m:
        return ""
    raw = m.group(1).strip()
    # Take last line if multiline (name is often the last line before ...Appellant)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        raw = lines[-1]
    return _first_name_only(raw)


def _extract_respondent(text: str) -> str:
    """Extract respondent name (first name only) from first page text."""
    m = _RESPONDENT_PATTERN.search(text)
    if not m:
        return ""
    raw = m.group(1).strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        raw = lines[-1]
    return _first_name_only(raw)


def suggest_case_law_basename(text: str, fallback_title: str = "document") -> str:
    """
    From case law first-page text, suggest a basename for the PDF (no extension).

    Format: SC_Appellant_name versus Respondent_name  or  HC_Appellant_name versus Respondent_name
    Uses first appellant and first respondent only when multiple are listed.
    """
    if not text or not text.strip():
        return _safe_filename(fallback_title)

    sample = text[: _FIRST_PAGE_CHARS]
    # Normalize newlines but keep structure for line-based extraction
    sample = re.sub(r"\r\n", "\n", sample)

    court = _detect_court(sample)
    appellant = _extract_appellant(sample)
    respondent = _extract_respondent(sample)

    if not appellant:
        appellant = "Appellant"
    if not respondent:
        respondent = "Respondent"

    prefix = f"{court}_" if court else ""
    middle = " versus "
    base = f"{prefix}{_safe_filename(appellant)}{middle}{_safe_filename(respondent)}"

    if not court and base == f"Appellant{middle}Respondent":
        # Try fallback: appeal/petition number (e.g. Criminal Appeal No. 192 of 2011)
        appeal_basename = _fallback_appeal_number(sample)
        if appeal_basename:
            return appeal_basename
        return _safe_filename(fallback_title)

    return base


def _fallback_appeal_number(text: str) -> str:
    """
    When court is not detected, look for "PetitionType No. XXX of YYYY"
    (e.g. Criminal Appeal No. 192 of 2011, Civil Appeal No. 5 of 2020, Writ Petition No. 10 of 2019).
    Return PetitionType_No_XXX_of_YYYY or empty string if not found.
    """
    m = _APPEAL_NO_OF_YEAR.search(text)
    if not m:
        return ""
    petition_type = m.group(1).strip()
    case_no = m.group(2).strip()
    year = m.group(3) + m.group(4)
    # Sanitize petition type for filename (spaces -> underscore, remove unsafe chars)
    type_safe = _safe_filename(petition_type).replace(" ", "_")
    case_safe = _UNSAFE_FS.sub("", case_no) or case_no
    return f"{type_safe}_No_{case_safe}_of_{year}"
