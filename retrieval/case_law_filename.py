"""
Suggest a safe filename for a case law PDF from the first page.

Structure (first page):
- Court: "In the Supreme Court of India" → SC_; "In the High Court of [State]" → HC_
- First party: name(s) to the left of "Appellant(s)" or "Petitioner(s)" (dots before the label optional).
- Second party: name(s) to the left of "Respondent(s)" (dots optional).
- Format: SC_FirstParty_name versus SecondParty_name  or  HC_... (same).

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

# First party (appellant/petitioner): text to the left of "Appellant(s)" or "Petitioner(s)"; dots optional
_FIRST_PARTY_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9\s\.\,\'\-\@\&]+?)\s*(?:\.{2,5})?\s*(?:Appellant|Petitioner)(?:s|\([sS]\))?\b",
    re.IGNORECASE,
)
# Second party (respondent): same idea; match "Respondent(s)" or "RESPONDENT(S)"
_RESPONDENT_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9\s\.\,\'\-\@\&]+?)\s*(?:\.{2,5})?\s*Respondent(?:s|\([sS]\))?\b",
    re.IGNORECASE,
)

# Fallback when court not detected: "Criminal Appeal No. 192 of 2011", "Writ Petition No. 123 of 2020", etc.
_APPEAL_NO_OF_YEAR = re.compile(
    r"\b(\w+(?:\s+\w+)+)\s+No\.?\s*([A-Za-z0-9\-]+)\s+of\s+(19|20)(\d{2})\b",
    re.IGNORECASE,
)

# Max length for a party name caption (avoids treating body text as name)
_MAX_PARTY_NAME_LEN = 80

# Sentence-like patterns: if capture contains these, treat as body text not caption
_BODY_TEXT_MARKERS = re.compile(
    r"^\d+\.\s|\.\s+[A-Z]|\bbetween\s+the\b|\bmarriage\b|\bpetitioner\s+and\s+the\b",
    re.IGNORECASE,
)


def _safe_filename(name: str) -> str:
    """Replace unsafe and special chars (&, @, etc.) with space; collapse spaces; strip."""
    s = _UNSAFE_FS.sub(" ", name)
    # Remove & and @ so filenames are safe and never contain these
    s = s.replace("&", " ").replace("@", " ")
    return " ".join(s.split()).strip() or "document"


def _is_likely_caption_name(raw: str) -> bool:
    """Return False if the capture looks like body text (e.g. '3. The marriage between...'), not a party name."""
    if not raw or len(raw) > _MAX_PARTY_NAME_LEN:
        return False
    if _BODY_TEXT_MARKERS.search(raw):
        return False
    return True


def _first_name_only(party_text: str) -> str:
    """
    Use only the main party name: if the text contains & or @, take the part before it.
    When the line starts with @ (e.g. '@ SK SALAUDDIN & ANR.'), take the part after @ then before &.
    """
    if not party_text or not party_text.strip():
        return ""
    s = party_text.strip()
    # If line starts with @, the real name is after it (alias line)
    if s.startswith("@"):
        s = s[1:].strip()
    # Take name before & (e.g. 'SUDHIRKUMAR SHRIVASTAVA & ORS.' -> 'SUDHIRKUMAR SHRIVASTAVA')
    if "&" in s:
        s = s.split("&", 1)[0].strip()
    # Take name before @ (e.g. 'X @ Alias' -> 'X')
    if "@" in s:
        s = s.split("@", 1)[0].strip()
    return " ".join(s.split()).strip()


def _detect_court(text: str) -> str:
    """Return 'SC' for Supreme Court, 'HC' for High Court, '' if not found."""
    if _SUPREME_COURT.search(text):
        return "SC"
    if _HIGH_COURT.search(text):
        return "HC"
    return ""


def _extract_first_party(text: str) -> str:
    """Extract first party name (Appellant or Petitioner) from first page text."""
    for m in _FIRST_PARTY_PATTERN.finditer(text):
        raw = m.group(1).strip()
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if lines:
            raw = lines[-1]
        if not _is_likely_caption_name(raw):
            continue
        name = _first_name_only(raw)
        if name:
            return name
    return ""


def _extract_respondent(text: str) -> str:
    """Extract respondent name (first name only) from first page text. Skip captures that look like body text."""
    for m in _RESPONDENT_PATTERN.finditer(text):
        raw = m.group(1).strip()
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if lines:
            raw = lines[-1]
        if not _is_likely_caption_name(raw):
            continue
        name = _first_name_only(raw)
        if name:
            return name
    return ""


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
    first_party = _extract_first_party(sample)
    respondent = _extract_respondent(sample)

    if not first_party:
        first_party = "Appellant"
    if not respondent:
        respondent = "Respondent"

    prefix = f"{court}_" if court else ""
    middle = " versus "
    base = f"{prefix}{_safe_filename(first_party)}{middle}{_safe_filename(respondent)}"

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
