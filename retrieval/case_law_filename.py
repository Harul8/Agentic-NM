"""
Case law filename generator — LLM-primary, regex fallback.

Target format:
  SC_YYYY_Appellant Name versus Respondent Name   (Supreme Court of India)
  HC_YYYY_Appellant Name versus Respondent Name   (any High Court)

The LLM reads the first ~3 000 chars (roughly the cover page) and extracts:
  court   — "SC" | "HC" | ""
  year    — 4-digit judgment year (from case number or date line), or ""
  first   — first/appellant/petitioner party name (main name only, no "& Ors.")
  second  — respondent/opposite party name (main name only)

A regex fast-path handles the most common well-structured formats without an LLM
call.  If it returns a complete result the LLM is skipped entirely.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIRST_PAGE_CHARS = 3000          # chars from start fed to the extractor
_MAX_PARTY_LEN    = 80            # ignore regex captures longer than this
_UNSAFE_FS        = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe(name: str) -> str:
    """Strip filesystem-unsafe chars, collapse whitespace.
    '/' is replaced with empty string so 'M/S' → 'MS' (not 'M S').
    """
    s = (name or "").replace("/", "")   # remove slash before other substitutions
    s = _UNSAFE_FS.sub(" ", s)
    s = s.replace("&", " ").replace("@", " ")
    return " ".join(s.split()).strip()


_RE_WO_SO = re.compile(r"\s+[WwSsDdCc][/\\][Oo]\b.*", re.S)  # W/O, S/O, D/O, C/O
_RE_NUMBERED_PREFIX = re.compile(r"^\d+[\.\)]\s*")            # "1. " or "1) "
_RE_BAD_PARTY = re.compile(                                    # fragments that are not party names
    r"^(?:"
    r"rep(?:resented)?\.?\s+(?:by|through|per)\b"  # REP. BY / REP. THROUGH / REP. PER
    r"|through\b"                                   # THROUGH THE SECRETARY …
    r"|via\b"
    r"|per\b"
    r"|the\s+state\s+of\b.*public\b"               # THE STATE OF … PUBLIC PROSECUTOR
    r"|public\s+prosecutor"
    r")",
    re.I,
)


def _clean_party(raw: str) -> str:
    """
    Extract the *main* party name from a raw regex capture:

    Multi-line captures: for numbered parties (1. Name\n2. Name) take the first
    non-empty line; for all other multi-line captures take the last line
    (which is closest to the role label and usually the correct name).

    Then clean:
    – strip numbered prefix "1. " / "1) "
    – strip leading '@' alias marker
    – drop W/O, S/O, D/O, C/O relationship suffixes and anything after
    – drop everything from '&' onward (strips ANR, ORS, co-parties)
    – drop alias suffix '@ ...'
    – remove accidentally-captured role words at the end
    """
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Numbered party list → take first name; otherwise take last line (closest to label)
    if _RE_NUMBERED_PREFIX.match(lines[0]):
        s = lines[0]
    else:
        s = lines[-1]

    # Strip numbered prefix
    s = _RE_NUMBERED_PREFIX.sub("", s).strip()
    # Leading alias marker '@'
    if s.startswith("@"):
        s = s[1:].strip()
    # W/O, S/O, D/O, C/O — "LALITHA DEVI W/O RAMAIAH" → "LALITHA DEVI"
    s = _RE_WO_SO.sub("", s).strip()
    # Multiple parties: keep only the first (before & / AND)
    for sep in (" & ", "&", " AND ", " and "):
        if sep in s:
            s = s.split(sep, 1)[0].strip()
    # Alias suffix
    if "@" in s:
        s = s.split("@", 1)[0].strip()
    # Remove trailing role words accidentally captured
    s = re.sub(
        r"\b(appellant|petitioner|respondent|plaintiff|defendant|complainant|opposite\s+party)s?\s*$",
        "", s, flags=re.I,
    ).strip()
    return " ".join(s.split()) or ""


def _build_filename(court: str, year: str, first: str, second: str) -> str:
    """Assemble the final filename string from extracted parts."""
    parts = []
    if court:
        parts.append(court)
    if year:
        parts.append(year)
    if first or second:
        a = _safe(first or "Unknown")
        b = _safe(second or "Unknown")
        parts.append(f"{a} versus {b}")
    return "_".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Regex fast-path extractors
# ---------------------------------------------------------------------------

# Court
_RE_SC   = re.compile(r"\bsupreme\s+court\s+of\s+india\b", re.I)
_RE_HC   = re.compile(
    r"\b(?:"
    r"high\s+court\s+of(?:\s+judicature\s+(?:at|for))?\s+\w[\w\s]*"
    r"|high\s+court\s+of\s+the\s+state"
    r"|(?:in\s+the\s+)?hon(?:'?ble)?\s+high\s+court"
    r")",
    re.I,
)

# Year: "of 2021", "No. 123/2021", "DECIDED ON: 12.01.2021", "DATED: 5 MAY 2023"
_RE_YEAR = re.compile(
    r"""
    (?:
        \bof\s+(20\d{2}|19\d{2})\b          # "of 2021" (case numbers)
      | /\s*(20\d{2}|19\d{2})\b             # "/2021"
      | \b(20\d{2}|19\d{2})\s*(?:$|\s)      # standalone year at end/space
      | decided\s+on\s*:?\s*\d{1,2}[./\-]\d{1,2}[./\-](20\d{2}|19\d{2})  # date line
      | dated\s*:?\s*\d{1,2}\s+\w+\s+(20\d{2}|19\d{2})  # "dated 5 May 2023"
    )
    """,
    re.VERBOSE | re.I,
)

# Party labels (appellant / petitioner / respondent)
# Character class includes '/' so "W/O", "S/O", "M/S" names are captured whole.
_RE_FIRST = re.compile(
    r"([A-Za-z][A-Za-z0-9 ./,''\-&@]{3,79}?)"
    r"\s*\.{0,6}\s*"
    r"(?:Appellant|Petitioner|Plaintiff|Complainant)(?:s|\(s\))?(?:-\w+)?\b",
    re.I,
)
_RE_RESP = re.compile(
    r"([A-Za-z][A-Za-z0-9 ./,''\-&@]{3,79}?)"
    r"\s*\.{0,6}\s*"
    r"(?:Respondent|Opposite\s+Party|Defendant)(?:s|\(s\))?(?:-\w+)?\b",
    re.I,
)
# Multi-line VERSUS…RESPONDENT block: capture the first non-noise line after VERSUS/AND
# Handles "STATE OF TELANGANA\nREP. BY PUBLIC PROSECUTOR    RESPONDENT"
_RE_VERSUS_BLOCK = re.compile(
    r"(?:VERSUS|AGAINST|AND)\s*\n\s*([A-Z][A-Z0-9 ./,''\-&]{3,79})\s*\n",
    re.I | re.MULTILINE,
)
# Title-style: "Ram Kumar v. State of UP"  or  "M/S XYZ Ltd. V. Union of India"
# Allows M/S, S/O, D/O prefixes and slash chars at the start of party names.
_RE_VS_TITLE = re.compile(
    r"^([A-Za-z/][A-Za-z0-9 ./,''\-&]{3,79}?)\s+[Vv](?:\.|/[Ss]\.?|[Ss]\.?)\s+([A-Z][A-Za-z0-9 .,''\-&]{3,79})",
    re.MULTILINE,
)

_BODY_NOISE = re.compile(
    r"^\d+\.\s|between\s+the\b|marriage\b|petitioner\s+and\s+the\b|\bwhereas\b",
    re.I,
)


def _regex_extract(sample: str) -> dict:
    """
    Fast regex extraction.  Returns a dict; any key may be "" if not found.
    Keys: court, year, first, second.
    """
    result = {"court": "", "year": "", "first": "", "second": ""}

    # Court
    if _RE_SC.search(sample):
        result["court"] = "SC"
    elif _RE_HC.search(sample):
        result["court"] = "HC"

    # Year — collect all matches, prefer 'of YYYY' pattern (case numbers)
    year_candidates = []
    for m in _RE_YEAR.finditer(sample):
        for g in m.groups():
            if g and re.fullmatch(r"(?:19|20)\d{2}", g):
                year_candidates.append(g)
    if year_candidates:
        result["year"] = year_candidates[0]  # first occurrence (case number year)

    def _best_line_for_party(raw: str) -> str:
        """
        From a multi-line regex capture, return the single best line for the party name.
        For the FIRST party  → last non-noise line (numbered list: first line).
        For ANY party        → skip lines that look like "REP. BY …", "THROUGH …" etc.
        Returns the raw capture collapsed to one line.
        """
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return raw.strip()
        # Numbered list: 1. Name / 2. Name → take first
        if _RE_NUMBERED_PREFIX.match(lines[0]):
            return lines[0]
        # Multi-line: prefer the first line that isn't a "REP. BY" fragment
        for ln in lines:
            if not _RE_BAD_PARTY.match(ln):
                return ln
        return lines[0]

    # First party
    for m in _RE_FIRST.finditer(sample):
        raw = m.group(1).strip()
        chosen = _best_line_for_party(raw)
        if len(chosen) > _MAX_PARTY_LEN or _BODY_NOISE.search(chosen):
            continue
        name = _clean_party(chosen)
        if name:
            result["first"] = name
            break

    # Respondent — same logic; "STATE OF TELANGANA\nREP. BY PUBLIC PROSECUTOR" → "STATE OF TELANGANA"
    for m in _RE_RESP.finditer(sample):
        raw = m.group(1).strip()
        chosen = _best_line_for_party(raw)
        if len(chosen) > _MAX_PARTY_LEN or _BODY_NOISE.search(chosen):
            continue
        name = _clean_party(chosen)
        if name:
            result["second"] = name
            break

    # If respondent name looks like a representative fragment ("REP. BY …"),
    # try the VERSUS-block pattern which captures the first line after VERSUS
    if result["second"] and _RE_BAD_PARTY.match(result["second"]):
        for m in _RE_VERSUS_BLOCK.finditer(sample):
            candidate = m.group(1).strip()
            if not _RE_BAD_PARTY.match(candidate) and not _BODY_NOISE.search(candidate):
                name = _clean_party(candidate)
                if name:
                    result["second"] = name
                    break

    # If neither label found, try "X v. Y" title line
    if not result["first"] and not result["second"]:
        m = _RE_VS_TITLE.search(sample)
        if m:
            result["first"]  = _clean_party(m.group(1))
            result["second"] = _clean_party(m.group(2))

    return result


# ---------------------------------------------------------------------------
# LLM extractor — primary path
# ---------------------------------------------------------------------------

# Few-shot examples: deliberately varied to cover the messy real-world formats
# seen in Indian courts (SC civil/criminal, HC Telangana/Andhra/others, aliases,
# multi-party, writ petitions, consumer forums, title-only captions, etc.)
_LLM_EXAMPLES = """
--- EXAMPLE 1 (Supreme Court, civil appeal, multiple appellants) ---
IN THE SUPREME COURT OF INDIA
CIVIL APPELLATE JURISDICTION
CIVIL APPEAL NO. 3456 OF 2021
HDFC BANK LIMITED & ANR.           ....APPELLANTS
VERSUS
ARUN KUMAR SINGH & ORS.            ....RESPONDENTS
{"court":"SC","year":"2021","first":"HDFC BANK LIMITED","second":"ARUN KUMAR SINGH"}

--- EXAMPLE 2 (Supreme Court, criminal appeal, alias name) ---
IN THE SUPREME COURT OF INDIA
CRIMINAL APPELLATE JURISDICTION
CRIMINAL APPEAL NO. 567 OF 2018
RAJU @ RAJESH VERMA                APPELLANT
VERSUS
STATE OF RAJASTHAN                 RESPONDENT
{"court":"SC","year":"2018","first":"RAJU","second":"STATE OF RAJASTHAN"}

--- EXAMPLE 3 (Telangana High Court, writ petition) ---
IN THE HIGH COURT OF TELANGANA
AT HYDERABAD
WRIT PETITION No. 12345 OF 2020
BETWEEN:
LALITHA DEVI W/O RAMAIAH           ...PETITIONER
AND
STATE OF TELANGANA
REPRESENTED BY ITS CHIEF SECRETARY ...RESPONDENT
{"court":"HC","year":"2020","first":"LALITHA DEVI","second":"STATE OF TELANGANA"}

--- EXAMPLE 4 (High Court of Judicature at Hyderabad, criminal appeal) ---
IN THE HIGH COURT OF JUDICATURE AT HYDERABAD
FOR THE STATES OF TELANGANA AND ANDHRA PRADESH
CRIMINAL APPEAL No. 789 OF 2017
T. NARASIMHA REDDY                 APPELLANT
VERSUS
STATE OF TELANGANA
REP. BY PUBLIC PROSECUTOR          RESPONDENT
{"court":"HC","year":"2017","first":"T. NARASIMHA REDDY","second":"STATE OF TELANGANA"}

--- EXAMPLE 5 (Supreme Court, only title line, no party label) ---
IN THE SUPREME COURT OF INDIA
SLP(C) No. 22345/2019
M/S RELIANCE INDUSTRIES LTD. V. UNION OF INDIA
JUDGMENT DATED 15.03.2022
{"court":"SC","year":"2019","first":"M/S RELIANCE INDUSTRIES LTD.","second":"UNION OF INDIA"}

--- EXAMPLE 6 (Punjab & Haryana HC, plaintiff-appellant format) ---
IN THE HIGH COURT OF PUNJAB AND HARYANA
RSA No. 1234 of 2015
SURESH KUMAR                        PLAINTIFF-APPELLANT
VERSUS
RAMESH KUMAR                        DEFENDANT-RESPONDENT
{"court":"HC","year":"2015","first":"SURESH KUMAR","second":"RAMESH KUMAR"}

--- EXAMPLE 7 (Delhi HC, company name with abbreviation) ---
IN THE HIGH COURT OF DELHI AT NEW DELHI
W.P.(C) 5678/2023
AMAZON SELLER SERVICES PVT. LTD.   ...PETITIONER
VERSUS
COMPETITION COMMISSION OF INDIA     ...RESPONDENT
{"court":"HC","year":"2023","first":"AMAZON SELLER SERVICES PVT. LTD.","second":"COMPETITION COMMISSION OF INDIA"}

--- EXAMPLE 8 (Supreme Court, state as appellant) ---
REPORTABLE
IN THE SUPREME COURT OF INDIA
CIVIL APPEAL NO. 999 OF 2016
STATE OF MADHYA PRADESH            APPELLANT
VERSUS
NARMADA BACHAO ANDOLAN & ORS.      RESPONDENTS
{"court":"SC","year":"2016","first":"STATE OF MADHYA PRADESH","second":"NARMADA BACHAO ANDOLAN"}

--- EXAMPLE 9 (Consumer forum, opposite party format) ---
NATIONAL CONSUMER DISPUTES REDRESSAL COMMISSION
REVISION PETITION No. 4321/2019
ABC PHARMA LIMITED                  PETITIONER/COMPLAINANT
VERSUS
NEW INDIA ASSURANCE CO. LTD.        OPPOSITE PARTY/RESPONDENT
{"court":"HC","year":"2019","first":"ABC PHARMA LIMITED","second":"NEW INDIA ASSURANCE CO. LTD."}

--- EXAMPLE 10 (Bombay HC, numbered parties, dots before label) ---
IN THE HIGH COURT OF BOMBAY
ORDINARY ORIGINAL CIVIL JURISDICTION
SUIT NO. 888 OF 2014
1. JAYESH MEHTA
2. PRIYA MEHTA                       ....PLAINTIFFS
VERSUS
1. ICICI BANK LTD.                   ....DEFENDANT
{"court":"HC","year":"2014","first":"JAYESH MEHTA","second":"ICICI BANK LTD."}
"""

_LLM_PROMPT_TEMPLATE = """\
You extract structured metadata from the cover page of an Indian court judgment.

Return ONLY a single JSON object with exactly these four keys:
  "court"  — "SC" (Supreme Court of India) or "HC" (any High Court or Tribunal) or "" if unclear
  "year"   — 4-digit year from the case number OR judgment date, e.g. "2021"; "" if not found
  "first"  — main name of the FIRST PARTY (appellant / petitioner / plaintiff / complainant).
             Use only the primary name — no "& Ors.", no "& Anr.", no alias suffix "@ ...".
             If numbered parties (1. Name 2. Name ...), use the first name only.
  "second" — main name of the RESPONDENT / OPPOSITE PARTY / DEFENDANT (same rules as "first")

Rules:
- Do NOT include titles like "M/s", "Smt.", "Shri", "Dr." unless they are part of a company name.
- If the name contains W/O, S/O, D/O (wife/son/daughter of) keep only the personal name before it.
- State names: keep as-is (e.g. "STATE OF TELANGANA", "UNION OF INDIA").
- If you cannot determine a field with reasonable confidence, return "" for that field.
- Output ONLY the JSON — no explanation, no markdown fences.

Few-shot examples:
{examples}

--- NOW EXTRACT FROM THIS DOCUMENT ---
{text}

JSON:"""


def _llm_extract(sample: str) -> dict:
    """
    Call the LLM to extract court, year, first party, second party.
    Returns dict with keys court/year/first/second; any may be "".
    """
    empty = {"court": "", "year": "", "first": "", "second": ""}
    try:
        from llm.ollama_client import ask_llm
    except ImportError:
        logger.debug("ollama_client not available; skipping LLM extraction")
        return empty

    prompt = _LLM_PROMPT_TEMPLATE.format(
        examples=_LLM_EXAMPLES.strip(),
        text=sample[:_FIRST_PAGE_CHARS],
    )
    try:
        raw = (ask_llm(prompt, timeout=45) or "").strip()
        # Strip markdown fences if the model added them
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S).strip()
        # Extract the first {...} block in case of surrounding text
        m = re.search(r"\{[^{}]+\}", raw, re.S)
        if m:
            raw = m.group(0)
        import json
        obj = json.loads(raw)
        return {
            "court":  str(obj.get("court")  or "").strip().upper(),
            "year":   str(obj.get("year")   or "").strip(),
            "first":  str(obj.get("first")  or "").strip(),
            "second": str(obj.get("second") or "").strip(),
        }
    except Exception as e:
        logger.debug("LLM filename extraction failed: %s", e)
        return empty


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def suggest_case_law_basename(text: str, fallback_title: str = "document") -> str:
    """
    Suggest a filename basename (no extension) for a case law PDF.

    Strategy:
      1. Run regex fast-path on first ~3 000 chars.
      2. If regex gives a fully complete result (court + year + both parties), use it.
      3. Otherwise call LLM; merge LLM result with any regex fields that are already good.
      4. Build filename in format:  COURT_YEAR_FirstParty versus Respondent
         e.g.  SC_2021_HDFC Bank Limited versus Arun Kumar Singh
               HC_2020_Lalitha Devi versus State of Telangana
      5. If still incomplete, fall back to sanitised fallback_title.

    Args:
        text:           First-page text of the judgment (plain text, not HTML).
        fallback_title: Used if extraction fails entirely.

    Returns:
        A filesystem-safe basename string (no extension).
    """
    if not text or not text.strip():
        return _safe(fallback_title) or "document"

    sample = text[:_FIRST_PAGE_CHARS]
    sample = re.sub(r"\r\n", "\n", sample)

    # --- Step 1: regex fast-path ---
    reg = _regex_extract(sample)

    # --- Step 2: is regex result complete enough? ---
    def _complete(d: dict) -> bool:
        return bool(d["court"] and d["year"] and d["first"] and d["second"])

    if _complete(reg):
        logger.debug("case_law_filename: regex fast-path succeeded")
        name = _build_filename(reg["court"], reg["year"], reg["first"], reg["second"])
        if name:
            return name

    # --- Step 3: LLM extraction ---
    llm = _llm_extract(sample)

    # Merge: prefer LLM values; fall back to regex where LLM returned ""
    merged = {
        "court":  llm["court"]  or reg["court"],
        "year":   llm["year"]   or reg["year"],
        "first":  llm["first"]  or reg["first"],
        "second": llm["second"] or reg["second"],
    }

    # Validate year is 4 digits
    if merged["year"] and not re.fullmatch(r"(?:19|20)\d{2}", merged["year"]):
        merged["year"] = ""

    # --- Step 4: build filename ---
    if merged["first"] or merged["second"]:
        name = _build_filename(
            merged["court"], merged["year"], merged["first"], merged["second"]
        )
        if name:
            logger.debug("case_law_filename: result=%s", name[:80])
            return name

    # --- Step 5: final fallback ---
    logger.debug("case_law_filename: extraction failed; using fallback_title")
    return _safe(fallback_title) or "document"
