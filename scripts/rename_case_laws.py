"""
Case law filename generator — LLM-primary, regex fallback.

Target filename format (one of):
  Party A v Party B
  Party A vs Party B
  Party A versus Party B

Examples:
  Punjab & Sind Bank v Baldev Singh.pdf
  Rahul Gandhi v Purnesh Ishwerbhai Modi.pdf
  Ch Kranti v G Ganesh Goud.pdf

When there are multiple appellants or respondents, only the first full name from
each side is used (e.g. "A, B & Ors. v X, Y and Z" → "A v X").

If appellant/petitioner and respondent names cannot be identified, the filename
falls back to the petition/case number from the document (e.g. WRIT PETITION No.22215 of 2022,
W.P.NO.5024 OF 2020, CIVIL APPEAL NO. 1234 OF 2023).

Supported inputs:
  - PDF: text from first 5 pages (pdfplumber, then pypdf fallback if needed).
  - OCR / text: .txt files — first ~12 000 chars used for party detection.

Usage:
  # Dry run (no changes):
  python scripts/rename_case_laws.py
  python scripts/rename_case_laws.py --dir "G:/My Drive/Nyaymalaw/CaseLaws"

  # Apply renames:
  python scripts/rename_case_laws.py --apply
  python scripts/rename_case_laws.py --dir "G:/My Drive/Nyaymalaw/CaseLaws" --apply

  # Force LLM for every file (skip regex fast-path):
  python scripts/rename_case_laws.py --dir "..." --apply --llm-always

  # Skip LLM entirely (regex only):
  python scripts/rename_case_laws.py --dir "..." --apply --no-llm
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from config import CASELAW_DIR
from Ingestion.smart_chunker import extract_text_from_pdf_first_n_pages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Pages to extract from PDF for party detection (cover/title usually in first few)
_PDF_PAGES_FOR_PARTIES = 5
# Chars to read from OCR/text files (enough for cover + first page)
_OCR_TEXT_CHARS = 12000

# ── Filesystem safety ─────────────────────────────────────────────────────────
_UNSAFE_FS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# ── Max lengths ───────────────────────────────────────────────────────────────
_MAX_PARTY_CHARS  = 50    # per party name in filename
_MAX_STEM_CHARS   = 130   # total stem (before extension)

# ── Court detection ───────────────────────────────────────────────────────────
_COURT_SC = re.compile(r"supreme\s+court\s+of\s+india|supreme\s+court", re.IGNORECASE)
_COURT_HC = re.compile(r"high\s+court", re.IGNORECASE)

# ── Year patterns ─────────────────────────────────────────────────────────────
# From case numbers: "CIVIL APPEAL NO. 1234 OF 2023" → 2023
_YEAR_OF  = re.compile(r"\bof\s+(20\d{2}|19\d{2})\b", re.IGNORECASE)
# From dates: "decided on 15 January 2022" / "dated 15.01.2022"
_YEAR_GEN = re.compile(r"\b(20\d{2}|19\d{2})\b")

# ── Citation patterns (also give year) ───────────────────────────────────────
_SCC = re.compile(r"\((\d{4})\)\s*\d+\s*SCC\s*\d+", re.IGNORECASE)
_AIR = re.compile(r"AIR\s+(19\d{2}|20\d{2})\s+SC", re.IGNORECASE)

# ── Petition / case number (fallback when party names missing) ───────────────
# e.g. WRIT PETITION NO: 97090 OF 2023, W.P.NO.5024 OF 2020, WRIT PETITION No.22215 of 2022
_PETITION_NUMBER_PATTERNS = [
    re.compile(
        r"(W\.P\.\s*NO\.?\s*:?\s*\d+\s+OF\s+(?:19|20)\d{2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(WP\s+No\.?\s*\d+\s+OF\s+(?:19|20)\d{2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(WRIT\s+PETITION\s+NO\.?\s*:?\s*\d+\s+OF\s+(?:19|20)\d{2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(CIVIL\s+APPEAL\s+NO\.?\s*:?\s*\d+\s+OF\s+(?:19|20)\d{2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(CRIMINAL\s+APPEAL\s+NO\.?\s*:?\s*\d+\s+OF\s+(?:19|20)\d{2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(SLP\s*\(?\s*C\s*\)?\s*NO\.?\s*:?\s*\d+\s+OF\s+(?:19|20)\d{2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(TRANSFER\s+PETITION\s+NO\.?\s*:?\s*\d+\s+OF\s+(?:19|20)\d{2})",
        re.IGNORECASE,
    ),
]

# ── VERSUS line (SC/HC multi-line format) ─────────────────────────────────────
# Matches "VERSUS" or "V S" or "VS" on its own line
_VERSUS_LINE = re.compile(
    r"^\s*(?:V\s*E\s*R\s*S\s*U\s*S|V\s*S\.?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# ── Inline "X v. Y" or "X versus Y" ─────────────────────────────────────────
_INLINE_VS = re.compile(
    r"((?:(?:M/s|Smt|Sri|Dr|Mr|Mrs|The|Union|State)\.?\s+)?[A-Z][A-Za-z0-9\s\.\,\'\-\(\)&]{4,55}?)"
    r"\s+(?:v\.?s?\.?|versus)\s+"
    r"((?:(?:M/s|Smt|Sri|Dr|Mr|Mrs|The|Union|State)\.?\s+)?[A-Z][A-Za-z0-9\s\.\,\'\-\(\)&]{4,55})",
    re.IGNORECASE,
)

# ── Annotation lines ("...Appellant", ".. Petitioner", "...Respondent(s)") ───
# Allow 1–3 dots (OCR variation) and optional (s) in role
_ANN_PET = re.compile(
    r"^([\w#\$][^\n]{3,75}?)\s*\.{1,3}\s*(?:Appellant|Petitioner|Plaintiff|Applicant)s?\b",
    re.IGNORECASE | re.MULTILINE,
)
_ANN_RES = re.compile(
    r"^([\w#\$][^\n]{3,75}?)\s*\.{1,3}\s*(?:Respondent|Defendant|Accused|Opposite\s+Party)s?\b",
    re.IGNORECASE | re.MULTILINE,
)

# ── PETITIONER: X Vs. RESPONDENT: Y (explicit labels) ─────────────────────────
_PET_VS_RES = re.compile(
    r"PETITIONER\s*:\s*([^\n]{2,80}?)\s+Vs?\.?\s+RESPONDENT\s*:\s*([^\n]{2,80})",
    re.IGNORECASE,
)

# ── "Between:" ... "...PETITIONER" then "AND" ... first respondent ───────────
# Capture line immediately before "...PETITIONER" and first meaningful line after "AND"
_BETWEEN_PET = re.compile(
    r"Between\s*:\s*(?:[\s\S]*?\n)?([^\n]{3,75})\s*\.{1,3}\s*PETITIONER",
    re.IGNORECASE,
)
_AND_FIRST_RES = re.compile(
    r"AND\s+(?:\d+\.\s*)?([^\n]{3,80}?)(?:\s*\.{1,3}\s*Respondent|\n|$)",
    re.IGNORECASE,
)

# ── "And" on its own line separating petitioner block from respondent block ───
_AND_LINE = re.compile(r"\n\s*And\s*\n", re.IGNORECASE)

# ── Lines to reject as party names ───────────────────────────────────────────
_NOISE_LINE = re.compile(
    r"^\s*("
    r"judgment|order|coram|bench|hon.?ble|before|decided|dated|"
    r"civil\s+(appeal|appellate)|criminal\s+(appeal|appellate)|"
    r"writ\s+petition|letters\s+patent|slp|transfer\s+petition|"
    r"with\b|and\b|in\s+the\s+(matter|case)|"
    r"counsel\s+for\s+(the\s+)?|standing\s+counsel\s+for\s*|"
    r"district\.?\s*$|"
    r"sentenced\s+to\b|"
    r"through\s+chief\s+secretary|through\s+secretary\s+to\s+|"
    r"the\s+court\s+made|registration\s+certificate\s+of\s+the|"
    r"contempt\s+has\s+been|injunction\s+against\s+the|"
    r"months\s+old\s+child\s+with\s+the|and\s+had\s+allowed\s+the|"
    r"mere\s+injunction|petition\s+on\s+the\s+ground\s+of|"
    r"workmen\s+represented\s+by|port\s+of\s+the\s+said\s+contention|"
    r"tion\.\s+by\s+judgment\s+dated|"
    r"\d+[\s\.\-]"          # starts with digit (case number fragment)
    r")",
    re.IGNORECASE,
)

# ── Reject whole party name if it looks like generic/narrative/address ───────
# (Applied after cleaning; matches full string or key phrases.)
_BAD_PARTY_PATTERNS = [
    re.compile(r"^respondent\.?\s*$", re.IGNORECASE),
    re.compile(r"^petitioner\.?\s*$", re.IGNORECASE),
    re.compile(r"^counsel\s+for(\s+the)?\s*$", re.IGNORECASE),
    re.compile(r"^counsel\s+for\s+(?:the\s+)?respondent", re.IGNORECASE),
    re.compile(r"heard\s+learned\s+counsel\s+for", re.IGNORECASE),
    re.compile(r"^standing\s+counsel\s+for\s*$", re.IGNORECASE),
    re.compile(r"^district\.?\s*$", re.IGNORECASE),
    re.compile(r"district\.\s*$", re.IGNORECASE),
    re.compile(r"district\.\s*-?\s*\d", re.IGNORECASE),  # "District. -501 301"
    re.compile(r"the\s+honourable\s+(?:the\s+)?(?:chief\s+)?justice\b", re.IGNORECASE),
    re.compile(r"the\s+honourable\s+sri\.?\s*justice\b", re.IGNORECASE),
    re.compile(r"authorised\s+signatory\s+sd\b", re.IGNORECASE),
    re.compile(r"administration\s+and\s+urban\s+development\s+(?:department\s*)?at\.?\s*$", re.IGNORECASE),
    re.compile(r"present\s+a\s+quantitative\s+data", re.IGNORECASE),
    re.compile(r"an\s+offence\s+under\s+section\b", re.IGNORECASE),
    re.compile(r"sentenced\s+to\b", re.IGNORECASE),
    re.compile(r"^secretariat\.?\s*$", re.IGNORECASE),
    re.compile(r"secretariat\s+buildings\s*$", re.IGNORECASE),
    re.compile(r"^\.+$"),   # only dots
    re.compile(r"^r\s+o\s+h\.no\.?\s*\d", re.IGNORECASE),  # "R o H.no.1-12"
    re.compile(r"^y\s+irr\s*$", re.IGNORECASE),
    re.compile(r"^no\.\s*\d+\s*&\s*\d+\s*$", re.IGNORECASE),  # "No. 1 & 2"
    re.compile(r"through\s+chief\s+secretary\.?\s*$", re.IGNORECASE),
    re.compile(r"through\s+secretary\s+to\s*$", re.IGNORECASE),
    re.compile(r"^&\s*Anr\.?\s*$", re.IGNORECASE),
    re.compile(r"^&\s*Ors\.?\s*$", re.IGNORECASE),
    re.compile(r"^\.?\s*petitioners?\.?\s*$", re.IGNORECASE),
    re.compile(r"^\.?\s*respondents?\.?\s*$", re.IGNORECASE),
    re.compile(r"^srl\.\s*v\.?\s*$", re.IGNORECASE),
    re.compile(r"the\s+court\s+made\s+the", re.IGNORECASE),
    re.compile(r"-\s*\d{5,6}\s*$"),
    re.compile(r"^\d{5,6}\s*$"),
    re.compile(r"^[a-z]+\s*-\s*\d{5,6}\s*$", re.IGNORECASE),  # "Hyderabad-500073" style address
    re.compile(r"^[a-z]+\s*-\s*\d{5,6}\s+and\s+others", re.IGNORECASE),  # "Hyderabad-500073 and Others"
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(text: str) -> str:
    """Strip filesystem-unsafe chars and normalise whitespace."""
    # M/s → Ms anywhere (/ not valid in filenames).
    # Use lookahead/lookbehind instead of \b because \b fails after "_" (word char).
    s = re.sub(r"(?<![A-Za-z0-9])M/[Ss](?![A-Za-z0-9])", "Ms", text)
    s = _UNSAFE_FS.sub(" ", s)
    return " ".join(s.split()).strip()


def _is_bad_party_name(party: str) -> bool:
    """Return True if the string is a known bad/generic party (e.g. 'Counsel for', 'Respondent', 'District.')."""
    if not party or len(party) < 2:
        return True
    p = party.strip()
    for pat in _BAD_PARTY_PATTERNS:
        if pat.search(p):
            return True
    return False


def _extract_petition_number(text: str) -> str:
    """
    Extract petition/case number from document text for use as fallback filename
    when party names are missing. Returns first match (sanitized) or "".
    e.g. "WRIT PETITION NO: 97090 OF 2023" → "WRIT PETITION NO 97090 OF 2023"
    """
    if not (text or "").strip():
        return ""
    sample = text[:8000]
    for pat in _PETITION_NUMBER_PATTERNS:
        m = pat.search(sample)
        if m:
            raw = m.group(1).strip()
            stem = _safe(raw)
            if len(stem) > _MAX_STEM_CHARS:
                stem = stem[:_MAX_STEM_CHARS].rsplit(" ", 1)[0]
            return stem if stem else ""
    return ""


def _filename_is_petition_number(stem: str) -> bool:
    """Return True if stem looks like a petition/case number (e.g. W.P. No.5024 of 2020)."""
    return bool(
        re.search(r"NO\.?\s*:?\s*\d+\s+OF\s+(?:19|20)\d{2}", stem, re.IGNORECASE)
        and re.search(
            r"^(?:W\.P\.|WP\s+No\.?|WRIT\s+PETITION|CIVIL\s+APPEAL|CRIMINAL\s+APPEAL|SLP|TRANSFER\s+PETITION)",
            stem,
            re.IGNORECASE,
        )
    )


def _title(name: str) -> str:
    """Title-case a party name, keeping common lowercase connectives."""
    name = re.sub(r"\bM\s*/\s*[Ss]\.?\b", "M/s", " ".join(name.split()))
    lowers = {"of", "and", "the", "in", "at", "by", "for", "a", "an", "v", "vs"}
    words = name.split()
    out = []
    for i, w in enumerate(words):
        if re.match(r"^M/s\.?$", w, re.IGNORECASE):
            out.append("M/s")
        elif re.match(r"^(smt|sri|dr|mr|mrs)\.?$", w, re.IGNORECASE):
            out.append(w.capitalize().rstrip(".") + ".")
        elif i > 0 and w.lower() in lowers:
            out.append(w.lower())
        else:
            out.append(w.capitalize())
    return " ".join(out)


def _shorten(name: str, max_chars: int = _MAX_PARTY_CHARS) -> str:
    """Trim to max_chars at a word boundary, stripping trailing punctuation."""
    name = name.strip()
    # Drop "& Ors", "& Others", "and Others" suffixes
    name = re.sub(r"\s*[&,]\s*(Ors\.?|Others?|Anr\.?|Another)\s*$", "", name, flags=re.IGNORECASE).strip()
    if len(name) <= max_chars:
        return name
    return name[:max_chars].rsplit(" ", 1)[0].rstrip(",.&")


def _first_party_only(party: str) -> str:
    """
    When there are multiple parties on one side (e.g. "A, B and C" or "X & Ors."),
    return only the first full name. For aliases (e.g. "RANA NAHID @ RESHMA @ SANA & ANr.")
    take the segment before " @ ". Then drop anything after the first "," or "(" —
    but keep "." as part of the name (e.g. "Dr. A. B. Rao" stays intact).
    """
    if not party or not isinstance(party, str):
        return ""
    # Aliases: "X @ Y @ Z & Anr." → take "X"
    party = re.split(r"\s*@\s*", party, maxsplit=1)[0].strip()
    # Take substring before first comma or "(", whichever comes first (if present)
    cut_idx = len(party)
    for ch in [",", "("]:
        idx = party.find(ch)
        if idx != -1 and idx < cut_idx:
            cut_idx = idx
    first_segment = party[:cut_idx].strip()
    return _shorten(first_segment) if first_segment else ""


def _clean_party(raw: str) -> str:
    """Title-case + shorten, returning "" if result looks poor. Strips 'represented by', OCR symbols, bad parties."""
    raw = raw.strip()
    # Trailing dots (e.g. "Dr. Ahmed Ali ...." from annotation)
    raw = re.sub(r"\s*\.{2,}\s*$", "", raw).strip()
    # OCR artifacts: leading # or $ before party name (e.g. "# KARVY STOCK BROKING LIMITED")
    raw = re.sub(r"^[#\$]\s+", "", raw).strip()
    # Leading citation/court prefix in captured name (e.g. "S.c.r. a Abhilasha" → "Abhilasha", "Insc 506 Mohd." → "Mohd.")
    raw = re.sub(r"^(?:S\.c\.r\.\s*(?:a\s*)?\d*\s*|Insc\s*\d+\s*|HC\s*\d{4}\s*_?\s*|SC\s*\d{4}\s*_?\s*)\s*", "", raw, flags=re.IGNORECASE).strip()
    # Strip "represented by ..." / "rep. by ..." (take entity name only)
    raw = re.sub(r"\s*,?\s*rep\.?\s*by\s+.+$", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"\s+,?\s+represented\s+by\s+.+$", "", raw, flags=re.IGNORECASE).strip()
    # "AND FIVE OTHERS" / "AND X OTHERS" at end
    raw = re.sub(r"\s+and\s+(?:five|several|\d+)\s+others?\.?\s*$", "", raw, flags=re.IGNORECASE).strip()
    # "(d) by Lrs." / "(d) by Lrs. &" (deceased by legal representatives)
    raw = re.sub(r"\s*\(d\)\s+by\s+Lrs\.?\s*(?:&?\s*.*)?$", "", raw, flags=re.IGNORECASE).strip()
    # " Through Secretary" / " Through Chief Secretary" at end
    raw = re.sub(r"\s+through\s+(?:chief\s+)?secretary\s*(?:,?.*)?$", "", raw, flags=re.IGNORECASE).strip()
    # Trailing role labels: " Petitioner(s)", " . Respondent(s)", " Appellant(s)" etc.
    raw = re.sub(r"\s*\.?\s*(?:Petitioner|Respondent|Appellant|Plaintiff|Defendant)s?\s*\.?\s*$", "", raw, flags=re.IGNORECASE).strip()
    # Strip annotation tails that slipped through ("… Appellant")
    raw = re.sub(
        r"\s*\.{1,3}\s*(?:Appellant|Respondent|Petitioner|Plaintiff|Defendant|Applicant)s?\b.*",
        "", raw, flags=re.IGNORECASE,
    ).strip()
    # Strip citation suffixes that greedy _INLINE_VS group 2 may have captured.
    raw = re.sub(r"\s*\((?:19|20)\d{2}\)\s*.*$", "", raw)
    raw = re.sub(r"\s+(?:AIR|SCC|SCR|MANU|SCJ)\s+.*$", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+In\s+the\s+(?:Supreme|High)\s+.*$", "", raw, flags=re.IGNORECASE)
    raw = re.sub(
        r"\s+(?:Civil|Criminal|Writ|Transfer|Special|Original)\s+(?:Appeal|Petition|Suit|Leave|Application).*$",
        "", raw, flags=re.IGNORECASE,
    )
    raw = re.sub(r"\s+No\.\s+\d+.*$", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw or len(raw) < 3:
        return ""
    if _NOISE_LINE.match(raw):
        return ""
    if len(raw.split()) > 10:
        return ""

    result = _shorten(_title(raw))
    if _is_bad_party_name(result):
        return ""
    return result


def _detect_court(text: str) -> str:
    sample = text[:2000]
    if _COURT_SC.search(sample):
        return "SC"
    if _COURT_HC.search(sample):
        return "HC"
    return ""


def _detect_year(text: str) -> str:
    """Best-effort year extraction — prefers citation year, then case-number year."""
    sample = text[:3000]
    # 1) SCC / AIR citation year
    for pat in (_SCC, _AIR):
        m = pat.search(sample)
        if m:
            return m.group(1)
    # 2) "... of 20XX" (case number)
    m = _YEAR_OF.search(sample)
    if m:
        return m.group(1)
    # 3) First 4-digit year in first 2000 chars
    m = _YEAR_GEN.search(text[:2000])
    if m:
        return m.group(1)
    return ""


def _filename_already_clean(stem: str) -> bool:
    """
    Return True if the filename already follows our target convention:
    "Party A v Party B" (or vs/versus), or a petition/case number (e.g. W.P. No.5024 of 2020).
    """
    # Petition number used as fallback when party names missing
    if _filename_is_petition_number(stem):
        return True
    # Party A v Party B format
    parts = re.split(r"\s+v\s+|\s+vs\.?\s+|\s+versus\s+", stem, flags=re.IGNORECASE, maxsplit=1)
    if len(parts) != 2:
        return False
    left, right = parts[0].strip(), parts[1].strip()
    return len(left) >= 2 and len(right) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Regex fast-path extractor
# ─────────────────────────────────────────────────────────────────────────────

def _regex_extract(text: str) -> dict:
    """
    Try to extract {court, year, first, second} using pure regex.
    Returns a dict; any missing field is an empty string.
    'first' = petitioner/appellant, 'second' = respondent.
    """
    raw = text[:6000]
    flat = " ".join(text.split())[:5000]

    court = _detect_court(raw)
    year  = _detect_year(raw)

    first = second = ""

    def _best_party_from_block(block: str, take_last: bool) -> str:
        """Strip annotation suffixes, filter noise lines, return best candidate."""
        block = re.sub(
            r"\s*\.{1,3}\s*(?:Appellant|Petitioner|Plaintiff|Complainant|Applicant|"
            r"Respondent|Defendant|Accused|Opposite\s+Party)s?\b.*",
            "", block, flags=re.IGNORECASE,
        )
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        lines = [l for l in lines if not _NOISE_LINE.match(l)
                 and not re.match(r"^\(?\d{4}\)?", l)]
        if not lines:
            return ""
        candidate = lines[-1] if take_last else lines[0]
        return _clean_party(candidate)

    # ── 1) VERSUS on its own line (Appellant above, Respondent below) ───────
    vm = _VERSUS_LINE.search(raw)
    if vm:
        before = raw[max(0, vm.start() - 400): vm.start()].strip()
        after  = raw[vm.end(): vm.end() + 400].strip()
        first  = _best_party_from_block(before, take_last=True)
        second = _best_party_from_block(after,  take_last=False)

    # ── 2) PETITIONER: X Vs. RESPONDENT: Y (explicit labels) ─────────────────
    if not first or not second:
        m = _PET_VS_RES.search(raw)
        if m:
            p, r = _clean_party(m.group(1)), _clean_party(m.group(2))
            if p and r:
                first, second = p, r

    # ── 3) Between: ... ...PETITIONER and AND ... first respondent ───────────
    if not first or not second:
        bm = _BETWEEN_PET.search(raw)
        am = _AND_FIRST_RES.search(raw)
        if bm:
            first = _clean_party(bm.group(1)) or first
        if am:
            second = _clean_party(am.group(1)) or second

    # ── 4) "And" on its own line (petitioner block above, respondent below) ──
    if not first or not second:
        parts = _AND_LINE.split(raw, maxsplit=1)
        if len(parts) == 2:
            before_and = parts[0].strip()
            after_and = parts[1].strip()
            if not first:
                first = _best_party_from_block(before_and[-500:] if len(before_and) > 500 else before_and, take_last=True)
            if not second:
                second = _best_party_from_block(after_and[:500], take_last=False)

    # ── 5) Inline "X v. Y" (citation-header or SCC Online style) ──────────────
    if not first or not second:
        for m in _INLINE_VS.finditer(flat):
            p = _clean_party(m.group(1))
            r = _clean_party(m.group(2))
            if p and r:
                first, second = p, r
                break

    # ── 6) Annotation scan ("...Appellant" / "...Respondent") ─────────────────
    if not first or not second:
        pm = _ANN_PET.search(raw)
        rm = _ANN_RES.search(raw)
        if pm:
            first = _clean_party(pm.group(1)) or first
        if rm:
            second = _clean_party(rm.group(1)) or second

    # ── 7) Final correction for Between:/AND pattern — prefer individual v State ─
    # If a "Between: <petitioner> ...PETITIONER" block exists, force FIRST to that
    # petitioner, and if possible set SECOND to the first respondent line that
    # clearly contains a State/Union (e.g. "State of Telangana", "Union of India").
    bm_final = _BETWEEN_PET.search(raw)
    if bm_final:
        pet = _clean_party(bm_final.group(1))
        if pet and not _is_bad_party_name(pet):
            first = pet
        # Look after the first "AND" following the Between: block for a State-line
        and_pos = raw.find("AND", bm_final.end())
        if and_pos != -1:
            after_and = raw[and_pos: and_pos + 800]
            for line in after_and.splitlines():
                if not line.strip():
                    continue
                if re.search(r"State of|Telangana State|Union of India", line, re.IGNORECASE):
                    cand = _clean_party(line)
                    if cand and not _is_bad_party_name(cand):
                        second = cand
                        break

    return {"court": court, "year": year, "first": first, "second": second}


# ─────────────────────────────────────────────────────────────────────────────
# LLM extractor
# ─────────────────────────────────────────────────────────────────────────────

_LLM_PROMPT = """\
You are a legal document parser for Indian courts.
Read the cover page of this judgment and return EXACTLY four lines:

COURT: SC          ← "SC" for Supreme Court of India, "HC" for any High Court, else blank
YEAR: 2022         ← 4-digit year from case number or decision date, else blank
FIRST: Vijay Kumar ← appellant / petitioner main name ONLY
SECOND: State of Maharashtra  ← respondent main name ONLY

Rules:
- Output nothing except those four lines.
- If a field cannot be determined, leave it blank after the colon.
- Names: title-case, main party only, max 8 words. Omit "& Ors.", "& Anr.", "represented by...", "rep. by...".
- For multiple parties take the first named only (e.g. "Leela & Ors." → "Leela"; "RANA NAHID @ SANA & Anr." → "Rana Nahid").
- For "State of X" use "State of X" not abbreviations. For companies use short name (e.g. "KARVY STOCK BROKING LIMITED" not the "represented by..." part).

COVER PAGE TEXT:
{snippet}
"""


def _llm_extract(text: str) -> dict:
    """
    Call the LLM on the first ~3 000 chars and parse its structured response.
    Returns {court, year, first, second}; any missing field is "".
    """
    result = {"court": "", "year": "", "first": "", "second": ""}
    try:
        from llm.llm_client import call_llm
        snippet = " ".join(text.split())[:3000]
        prompt  = _LLM_PROMPT.format(snippet=snippet)
        raw_out = (call_llm(prompt) or "").strip()

        for line in raw_out.splitlines():
            line = line.strip()
            if line.upper().startswith("COURT:"):
                val = line.split(":", 1)[1].strip().upper()
                if val in ("SC", "HC"):
                    result["court"] = val
            elif line.upper().startswith("YEAR:"):
                val = line.split(":", 1)[1].strip()
                if re.match(r"^(19|20)\d{2}$", val):
                    result["year"] = val
            elif line.upper().startswith("FIRST:"):
                result["first"] = _clean_party(line.split(":", 1)[1].strip())
            elif line.upper().startswith("SECOND:"):
                result["second"] = _clean_party(line.split(":", 1)[1].strip())

    except Exception as e:
        logger.debug("LLM extract failed: %s", e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Build target filename
# ─────────────────────────────────────────────────────────────────────────────

# Connector for filename: one of " v " | " vs " | " versus "
_FILENAME_CONNECTOR = " v "


def _build_stem(court: str, year: str, first: str, second: str) -> str:
    """
    Assemble the filename stem as "Party A v Party B" (no court/year prefix).
    Returns non-empty only when BOTH parties are present; otherwise "" so caller
    can use petition number or keep current filename.
    """
    first = _first_party_only(first) if first else ""
    second = _first_party_only(second) if second else ""

    # Require both parties — never produce one-party or empty stem
    if not first or not second:
        return ""
    if _is_bad_party_name(first) or _is_bad_party_name(second):
        return ""

    name_part = f"{first}{_FILENAME_CONNECTOR}{second}"
    stem = name_part
    if len(stem) > _MAX_STEM_CHARS:
        stem = stem[:_MAX_STEM_CHARS].rsplit(" ", 1)[0]
    return _safe(stem)


def _target_basename(
    path: str,
    text: str,
    use_llm: bool = True,
    llm_always: bool = False,
) -> str:
    """
    Derive the target filename (stem + extension) for one case law file.

    Strategy:
      1. Try regex fast-path.
      2. If fast-path gives a complete result (first + second + year) → use it.
      3. Otherwise (or if llm_always=True) call the LLM and merge results:
         LLM fills in any field the regex left blank.
      4. Build stem from merged fields (Party A v Party B).
      5. If no party names: use petition/case number from text if present (e.g. WRIT PETITION No.22215 of 2022).
      6. Else: fall back to current filename stem.
    """
    ext = os.path.splitext(path)[1].lower() or ".pdf"

    # Step 1: regex
    rx = _regex_extract(text)
    complete = bool(rx["first"] and rx["second"] and rx["year"])

    # Step 2: maybe call LLM
    lm: dict = {}
    if use_llm and (not complete or llm_always):
        lm = _llm_extract(text)

    # Step 3: merge — LLM fills gaps left by regex
    court  = rx["court"]  or lm.get("court",  "")
    year   = rx["year"]   or lm.get("year",   "")
    first  = rx["first"]  or lm.get("first",  "")
    second = rx["second"] or lm.get("second", "")

    source = "regex" if complete else ("regex+llm" if lm else "regex-only")
    logger.debug("(%s) court=%s year=%s first=%r second=%r", source, court, year, first, second)

    # Step 4: build stem from party names
    stem = _build_stem(court, year, first, second)

    # Step 5: if party names missing, use petition/case number (from doc text, then from filename)
    if not stem:
        petition = _extract_petition_number(text)
        if not petition:
            current_stem = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
            petition = _extract_petition_number(current_stem)
        if petition:
            stem = petition
            logger.debug("Using petition/case number as filename: %s", stem)

    # Step 6: fallback to current filename stem (keep existing name when nothing valid found)
    if not stem:
        stem = _safe(os.path.splitext(os.path.basename(path))[0]
                     .replace("_", " ").replace("-", " "))
        logger.debug("Fallback to filename stem: %s", stem)

    return stem + ext


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction (PDF + OCR/text)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_pdf_pypdf_fallback(pdf_path: str, n: int) -> str:
    """Extract text from first n pages using pypdf (fallback when pdfplumber fails or returns empty)."""
    text = ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            if i >= n:
                break
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception as e:
        logger.debug("pypdf fallback failed for %s: %s", os.path.basename(pdf_path), e)
    return text


def _extract_text_for_rename(file_path: str) -> str:
    """
    Extract text from a case-law file for party identification.
    - PDF: first _PDF_PAGES_FOR_PARTIES pages via pdfplumber; if empty or fail, try pypdf.
    - OCR / .txt: read first _OCR_TEXT_CHARS characters.
    """
    path_lower = file_path.lower()
    if path_lower.endswith(".pdf"):
        text = extract_text_from_pdf_first_n_pages(file_path, n=_PDF_PAGES_FOR_PARTIES)
        if not (text or "").strip():
            text = _extract_text_pdf_pypdf_fallback(file_path, _PDF_PAGES_FOR_PARTIES)
        return text or ""
    if path_lower.endswith(".txt"):
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                return (f.read(_OCR_TEXT_CHARS) or "").strip()
        except Exception as e:
            logger.warning("Failed to read OCR/text file %s: %s", os.path.basename(file_path), e)
            return ""
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Directory scanning
# ─────────────────────────────────────────────────────────────────────────────

def _scan(directory: str) -> list[tuple[str, str]]:
    """
    Scan directory for case-law files (PDF and OCR/text .txt), extract text from each,
    and return (filepath, extracted_text). Used as the first step of the rename pipeline.
    """
    results = []
    if not os.path.isdir(directory):
        return results
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        low = name.lower()
        if not (low.endswith(".pdf") or low.endswith(".txt")):
            continue
        try:
            text = _extract_text_for_rename(path)
            results.append((path, text or ""))
        except Exception as e:
            logger.warning("Skip %s: %s", name, e)
    return results


def _resolve_collisions(renames: list[tuple[str, str]]) -> list[tuple[str, str]]:
    used: dict[str, int] = {}
    resolved = []
    for src, tgt in renames:
        key = tgt.lower()
        if key in used:
            used[key] += 1
            stem, ext = os.path.splitext(tgt)
            tgt = f"{stem}_{used[key]}{ext}"
        else:
            used[key] = 1
        resolved.append((src, tgt))
    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan case-law files (PDF and OCR/text .txt), identify parties from content, "
            "and rename to 'Party A v Party B.pdf' (first party only per side).\n"
            "Default mode is a DRY RUN — no files are changed.\n"
            "Pass --apply to actually rename."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir", metavar="FOLDER", default=None,
        help=(
            "Folder containing case-law PDF and OCR/text (.txt) files. "
            f"Defaults to CASELAW_DIR from config.py (currently: {CASELAW_DIR})."
        ),
    )
    parser.add_argument("--apply", action="store_true",
                        help="Apply the renames (default: dry run).")
    parser.add_argument("--no-llm", action="store_true",
                        help="Regex only — never call the LLM.")
    parser.add_argument("--llm-always", action="store_true",
                        help="Always call the LLM even when regex gives a full result.")
    args = parser.parse_args()

    target_dir = os.path.normpath(args.dir) if args.dir else CASELAW_DIR

    if not os.path.isdir(target_dir):
        logger.error(
            "Directory not found: %s\n"
            "Pass the full path with --dir, e.g.:\n"
            '  python scripts/rename_case_laws.py --dir "G:/My Drive/Nyaymalaw/CaseLaws" --apply',
            target_dir,
        )
        sys.exit(1)

    use_llm    = not args.no_llm
    llm_always = args.llm_always

    logger.info("Scanning %s for PDF and OCR/text case-law files...", target_dir)
    logger.info(
        "LLM mode: %s",
        "disabled" if not use_llm else ("always" if llm_always else "fallback only"),
    )
    if not args.apply:
        logger.info("DRY RUN — no files will be changed. Pass --apply to rename.")

    scan = _scan(target_dir)
    if not scan:
        logger.info("No PDF or OCR/text (.txt) files found in %s.", target_dir)
        return

    n_pdf = sum(1 for p, _ in scan if p.lower().endswith(".pdf"))
    n_txt = len(scan) - n_pdf
    logger.info("Found %d case-law file(s) (%d PDF, %d OCR/text). Identifying parties from content...", len(scan), n_pdf, n_txt)

    raw_renames: list[tuple[str, str]] = []
    skipped_clean = 0

    for path, text in scan:
        current      = os.path.basename(path)
        current_stem = os.path.splitext(current)[0]

        if _filename_already_clean(current_stem):
            logger.debug("SKIP (already clean): %s", current)
            skipped_clean += 1
            continue

        proposed = _target_basename(path, text, use_llm=use_llm, llm_always=llm_always)
        logger.info("  %s  →  %s", current[:55] + ("..." if len(current) > 55 else ""), proposed)
        if current != proposed:
            raw_renames.append((path, proposed))

    if skipped_clean:
        logger.info("Skipped %d file(s) that already have clean names.", skipped_clean)

    if not raw_renames:
        logger.info("All %d file(s) already have clean names. Nothing to rename.", len(scan))
        return

    renames = _resolve_collisions(raw_renames)

    logger.info(
        "\n%s RENAMES PLANNED (%d files):",
        "DRY RUN —" if not args.apply else "APPLYING",
        len(renames),
    )
    for src, tgt in renames:
        logger.info("  %-60s  →  %s", os.path.basename(src)[:60], tgt)

    if not args.apply:
        logger.info(
            "\nTo apply:\n  python scripts/rename_case_laws.py%s --apply",
            f' --dir "{target_dir}"' if args.dir else "",
        )
        return

    ok = err = 0
    for src, tgt_base in renames:
        tgt_path = os.path.join(target_dir, tgt_base)
        try:
            if os.path.abspath(src) == os.path.abspath(tgt_path):
                continue
            if os.path.isfile(tgt_path):
                logger.warning("  SKIP (exists): %s", tgt_base)
                err += 1
                continue
            os.rename(src, tgt_path)
            ok += 1
        except OSError as e:
            logger.warning("  FAILED %s: %s", os.path.basename(src), e)
            err += 1

    logger.info("\nDone: %d renamed, %d failed/skipped.", ok, err)


if __name__ == "__main__":
    main()
