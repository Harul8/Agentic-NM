"""
Indian legal citation normalization: map variant citations to a canonical ID.

e.g. AIR 1978 SC 597, (1978) 1 SCC 248, 1978 SCR 621 → same case.
When party names are present: "Maneka Gandhi v. Union of India (1978) 1 SCC 248"
→ MANEKA_GANDHI_V_UNION_OF_INDIA_1978
When only reporter is present: "AIR 1978 SC 597" → AIR_1978_SC_597
"""

import re
import logging

logger = logging.getLogger(__name__)

# Reporter abbreviations for fallback canonical (reporter_year_num)
_REPORTER_PAT = re.compile(
    r"\b(AIR|SCC|SCR|SCRA|CRI\.?\s*L\.?J\.?|ILR|MANU|Cri\s*LJ)\s*"
    r"(?:\(?\s*)?(\d{4})\s*\)?\s*(?:\(\d+\)\s*)?(\d+)?",
    re.I,
)
# Party v Party before citation: "X v. Y" or "X v Y" or "X versus Y"
_PARTIES_PAT = re.compile(
    r"([A-Za-z0-9\s\.\',&]+?)\s+v(?:\.|ersus)?\s+([A-Za-z0-9\s\.\',&]+?)"
    r"(?:\s*[\(\[]?\s*(?:19|20)\d{2}|\s*$)",
    re.I,
)
_YEAR_PAT = re.compile(r"(19|20)\d{2}")


def _safe_id_part(s: str) -> str:
    """Uppercase, alphanumeric + underscore only, collapse spaces to single underscore."""
    if not s:
        return ""
    s = re.sub(r"\s+", " ", str(s).upper().strip())
    s = re.sub(r"[^\w]", "", s.replace(" ", "_"))
    return re.sub(r"_+", "_", s).strip("_")[:80]


def citation_to_canonical_id(citation_text: str) -> str:
    """
    Return a canonical ID for an Indian case citation for precedent linking.
    - If party names and year can be extracted: PARTY_A_V_PARTY_B_YEAR
    - Else: REPORTER_YEAR_NUM (e.g. AIR_1978_SC_597) so same citation string maps to same ID.
    """
    if not (citation_text or "").strip():
        return ""
    text = " ".join(citation_text.split()).strip()
    year = ""
    for m in _YEAR_PAT.finditer(text):
        year = m.group(0)
        break
    # Try party names first (X v Y)
    m = _PARTIES_PAT.search(text)
    if m:
        left = _safe_id_part(m.group(1))
        right = _safe_id_part(m.group(2))
        if left and right:
            y = year or "UNK"
            return f"{left}_V_{right}_{y}"
    # Fallback: reporter + year + number
    mm = _REPORTER_PAT.search(text)
    if mm:
        rep = _safe_id_part(mm.group(1))
        yr = mm.group(2)
        num = mm.group(3) or "0"
        return f"{rep}_{yr}_{num}"
    # Last resort: slug from first 60 chars
    slug = _safe_id_part(text[:60])
    return slug or "UNKNOWN"


def normalize_cited_cases_list(cited_cases: list) -> list:
    """Return list of dicts {raw, canonical_id} for each cited case string."""
    result = []
    seen = set()
    for raw in (cited_cases or []):
        if not (raw and isinstance(raw, str)):
            continue
        cid = citation_to_canonical_id(raw.strip())
        if cid and cid not in seen:
            seen.add(cid)
            result.append({"raw": raw.strip(), "canonical_id": cid})
    return result
