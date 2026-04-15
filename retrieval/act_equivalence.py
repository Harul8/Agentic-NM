"""
retrieval/act_equivalence.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Old Indian act → new act (BNS/BNSS/BSA) equivalence helpers.

IPC   → Bharatiya Nyaya Sanhita, 2023 (BNS)         [in force 1 Jul 2024]
CrPC  → Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)
IEA   → Bharatiya Sakshya Adhiniyam, 2023 (BSA)

Old acts are NOT in the bare-acts index. These utilities:
  1. Expand queries containing old-act references so stage-1 can identify BNS/BNSS/BSA.
  2. Provide inline notes for the response generator ("IPC 498A → BNS Section 85").
  3. Map old act name strings to canonical new act names.

Full mapping is loaded lazily from bareacts_v2_chunks.json at first use.
A smaller hardcoded subset covers the most-litigated sections immediately.
"""

from __future__ import annotations
import logging
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Act-name aliases: old full names / short codes → canonical index names
# ---------------------------------------------------------------------------
OLD_ACT_NAME_ALIASES: dict[str, str] = {
    # IPC variants
    "indian penal code": "Bharatiya Nyaya Sanhita, 2023",
    "indian penal code, 1860": "Bharatiya Nyaya Sanhita, 2023",
    "ipc": "Bharatiya Nyaya Sanhita, 2023",
    "i.p.c": "Bharatiya Nyaya Sanhita, 2023",
    # CrPC variants
    "code of criminal procedure": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "code of criminal procedure, 1973": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "crpc": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "cr.p.c": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    # IEA variants
    "indian evidence act": "Bharatiya Sakshya Adhiniyam, 2023",
    "indian evidence act, 1872": "Bharatiya Sakshya Adhiniyam, 2023",
    "iea": "Bharatiya Sakshya Adhiniyam, 2023",
    "evidence act": "Bharatiya Sakshya Adhiniyam, 2023",
    # New act short codes
    "bns": "Bharatiya Nyaya Sanhita, 2023",
    "bnss": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "bsa": "Bharatiya Sakshya Adhiniyam, 2023",
}

# ---------------------------------------------------------------------------
# Core hardcoded equivalences — most commonly litigated sections
# Format: (old_act_code, old_section) → (new_short, new_section, description)
# ---------------------------------------------------------------------------
_CORE_MAP: dict[tuple[str, str], tuple[str, str, str]] = {
    # ── IPC → BNS ──────────────────────────────────────────────────────────
    ("IPC", "302"):   ("BNS", "103",  "murder"),
    ("IPC", "304"):   ("BNS", "105",  "culpable homicide not amounting to murder"),
    ("IPC", "304A"):  ("BNS", "106",  "death by negligence"),
    ("IPC", "304B"):  ("BNS", "80",   "dowry death"),
    ("IPC", "306"):   ("BNS", "108",  "abetment of suicide"),
    ("IPC", "307"):   ("BNS", "109",  "attempt to murder"),
    ("IPC", "308"):   ("BNS", "110",  "attempt to commit culpable homicide"),
    ("IPC", "312"):   ("BNS", "88",   "causing miscarriage"),
    ("IPC", "313"):   ("BNS", "89",   "causing miscarriage without consent"),
    ("IPC", "323"):   ("BNS", "115",  "voluntarily causing hurt"),
    ("IPC", "324"):   ("BNS", "117",  "voluntarily causing hurt by dangerous weapons"),
    ("IPC", "325"):   ("BNS", "116",  "voluntarily causing grievous hurt"),
    ("IPC", "326"):   ("BNS", "118",  "voluntarily causing grievous hurt by dangerous weapons"),
    ("IPC", "354"):   ("BNS", "74",   "assault on woman with intent to outrage modesty"),
    ("IPC", "354A"):  ("BNS", "75",   "sexual harassment"),
    ("IPC", "354B"):  ("BNS", "76",   "assault with intent to disrobe"),
    ("IPC", "354C"):  ("BNS", "77",   "voyeurism"),
    ("IPC", "354D"):  ("BNS", "78",   "stalking"),
    ("IPC", "363"):   ("BNS", "137",  "kidnapping"),
    ("IPC", "364"):   ("BNS", "140",  "kidnapping to murder or for ransom"),
    ("IPC", "366"):   ("BNS", "143",  "kidnapping to compel marriage"),
    ("IPC", "375"):   ("BNS", "63",   "rape"),
    ("IPC", "376"):   ("BNS", "64",   "punishment for rape"),
    ("IPC", "376A"):  ("BNS", "66",   "rape causing death or persistent vegetative state"),
    ("IPC", "376B"):  ("BNS", "67",   "sexual intercourse by husband upon his wife during separation"),
    ("IPC", "376C"):  ("BNS", "68",   "sexual intercourse by person in authority"),
    ("IPC", "376D"):  ("BNS", "70",   "gang rape"),
    ("IPC", "379"):   ("BNS", "303",  "theft"),
    ("IPC", "380"):   ("BNS", "305",  "theft in a dwelling house"),
    ("IPC", "384"):   ("BNS", "308",  "extortion"),
    ("IPC", "392"):   ("BNS", "309",  "robbery"),
    ("IPC", "395"):   ("BNS", "310",  "dacoity"),
    ("IPC", "396"):   ("BNS", "310",  "dacoity with murder"),
    ("IPC", "397"):   ("BNS", "311",  "robbery or dacoity with attempt to cause death"),
    ("IPC", "406"):   ("BNS", "316",  "criminal breach of trust"),
    ("IPC", "409"):   ("BNS", "316",  "criminal breach of trust by public servant"),
    ("IPC", "415"):   ("BNS", "317",  "cheating"),
    ("IPC", "420"):   ("BNS", "318",  "cheating and dishonestly inducing delivery of property"),
    ("IPC", "425"):   ("BNS", "324",  "mischief"),
    ("IPC", "435"):   ("BNS", "324",  "mischief by fire"),
    ("IPC", "436"):   ("BNS", "325",  "mischief by fire or explosive on house"),
    ("IPC", "447"):   ("BNS", "329",  "criminal trespass"),
    ("IPC", "451"):   ("BNS", "333",  "house trespass to commit offence"),
    ("IPC", "452"):   ("BNS", "333",  "house trespass after preparation for hurt"),
    ("IPC", "467"):   ("BNS", "336",  "forgery of valuable security"),
    ("IPC", "468"):   ("BNS", "336",  "forgery for purpose of cheating"),
    ("IPC", "471"):   ("BNS", "340",  "using forged document as genuine"),
    ("IPC", "489A"):  ("BNS", "178",  "counterfeiting currency notes"),
    ("IPC", "489B"):  ("BNS", "179",  "using counterfeit currency"),
    ("IPC", "489C"):  ("BNS", "180",  "possession of counterfeit currency"),
    ("IPC", "498A"):  ("BNS", "85",   "cruelty by husband or relatives"),
    ("IPC", "499"):   ("BNS", "356",  "defamation"),
    ("IPC", "500"):   ("BNS", "356",  "punishment for defamation"),
    ("IPC", "504"):   ("BNS", "352",  "intentional insult to provoke breach of peace"),
    ("IPC", "506"):   ("BNS", "351",  "criminal intimidation"),
    ("IPC", "509"):   ("BNS", "79",   "words or gestures to insult modesty of woman"),
    ("IPC", "34"):    ("BNS", "3",    "acts done by several persons in furtherance of common intention"),
    ("IPC", "120B"):  ("BNS", "61",   "criminal conspiracy"),
    ("IPC", "149"):   ("BNS", "190",  "unlawful assembly"),

    # ── CrPC → BNSS ────────────────────────────────────────────────────────
    ("CRPC", "41"):   ("BNSS", "35",  "when police may arrest without warrant"),
    ("CRPC", "57"):   ("BNSS", "58",  "person arrested not to be detained more than 24 hours"),
    ("CRPC", "91"):   ("BNSS", "94",  "summons to produce document"),
    ("CRPC", "125"):  ("BNSS", "144", "maintenance of wives, children and parents"),
    ("CRPC", "154"):  ("BNSS", "173", "FIR in cognizable cases"),
    ("CRPC", "156"):  ("BNSS", "175", "police power to investigate cognizable case"),
    ("CRPC", "161"):  ("BNSS", "180", "examination of witnesses by police"),
    ("CRPC", "162"):  ("BNSS", "181", "statements to police"),
    ("CRPC", "164"):  ("BNSS", "183", "recording of confessions and statements"),
    ("CRPC", "167"):  ("BNSS", "187", "procedure when investigation cannot be completed in 24 hours"),
    ("CRPC", "173"):  ("BNSS", "193", "report of police officer on completion of investigation"),
    ("CRPC", "197"):  ("BNSS", "218", "prosecution of judges and public servants"),
    ("CRPC", "204"):  ("BNSS", "227", "issue of process"),
    ("CRPC", "207"):  ("BNSS", "230", "supply of copies of documents to accused"),
    ("CRPC", "239"):  ("BNSS", "262", "when accused shall be discharged"),
    ("CRPC", "300"):  ("BNSS", "337", "double jeopardy — person once convicted not to be tried again"),
    ("CRPC", "313"):  ("BNSS", "351", "power to examine accused"),
    ("CRPC", "320"):  ("BNSS", "359", "compounding of offences"),
    ("CRPC", "357"):  ("BNSS", "395", "compensation to victims"),
    ("CRPC", "374"):  ("BNSS", "415", "appeals from convictions"),
    ("CRPC", "378"):  ("BNSS", "419", "appeal in case of acquittal"),
    ("CRPC", "389"):  ("BNSS", "430", "suspension of sentence pending appeal"),
    ("CRPC", "395"):  ("BNSS", "436", "reference to High Court"),
    ("CRPC", "401"):  ("BNSS", "442", "High Court powers of revision"),
    ("CRPC", "406"):  ("BNSS", "446", "Supreme Court power to transfer cases"),
    ("CRPC", "432"):  ("BNSS", "473", "power to suspend or remit sentences"),
    ("CRPC", "436"):  ("BNSS", "478", "bail in bailable offences"),
    ("CRPC", "437"):  ("BNSS", "480", "bail in non-bailable offences"),
    ("CRPC", "438"):  ("BNSS", "482", "anticipatory bail"),
    ("CRPC", "439"):  ("BNSS", "483", "special powers of HC/Sessions Court for bail"),
    ("CRPC", "451"):  ("BNSS", "497", "custody of property pending trial"),
    ("CRPC", "482"):  ("BNSS", "528", "inherent powers of High Court"),

    # ── IEA → BSA ──────────────────────────────────────────────────────────
    ("IEA", "25"):    ("BSA", "23",   "confession to police officer"),
    ("IEA", "27"):    ("BSA", "23",   "discovery of facts from accused in custody"),
    ("IEA", "32"):    ("BSA", "26",   "dying declaration"),
    ("IEA", "45"):    ("BSA", "39",   "opinions of experts"),
    ("IEA", "65B"):   ("BSA", "63",   "admissibility of electronic records"),
    ("IEA", "73"):    ("BSA", "73",   "comparison of handwriting"),
    ("IEA", "101"):   ("BSA", "104",  "burden of proof"),
    ("IEA", "102"):   ("BSA", "105",  "on whom burden of proof lies"),
    ("IEA", "106"):   ("BSA", "109",  "burden of proving fact especially in knowledge of accused"),
    ("IEA", "113A"):  ("BSA", "116",  "presumption as to abetment of suicide by married woman"),
    ("IEA", "113B"):  ("BSA", "117",  "presumption as to dowry death"),
    ("IEA", "114"):   ("BSA", "119",  "court may presume existence of certain facts"),
    ("IEA", "118"):   ("BSA", "124",  "competency of witnesses"),
    ("IEA", "126"):   ("BSA", "132",  "professional communications — lawyer-client privilege"),
    ("IEA", "145"):   ("BSA", "148",  "cross-examination of witnesses"),
    ("IEA", "154"):   ("BSA", "157",  "hostile witness"),
    ("IEA", "155"):   ("BSA", "158",  "impeaching credit of witness"),
}

# Full map loaded lazily from chunks (787 entries from index)
_full_map: Optional[dict[tuple[str, str], tuple[str, str, str]]] = None
_full_map_lock = threading.Lock()


def _load_full_map() -> dict[tuple[str, str], tuple[str, str, str]]:
    """Load the complete mapping from bareacts_v2_chunks.json (one-time)."""
    global _full_map
    with _full_map_lock:
        if _full_map is not None:
            return _full_map

        try:
            import json
            from config import BARE_CHUNKS_V2
            with open(BARE_CHUNKS_V2) as f:
                chunks = json.load(f)

            NEW_ACTS = {
                "Bharatiya Nyaya Sanhita, 2023": "BNS",
                "Bharatiya Nagarik Suraksha Sanhita, 2023": "BNSS",
                "Bharatiya Sakshya Adhiniyam, 2023": "BSA",
            }
            PATTERN = re.compile(
                r"[Ss]imilar to [Ss]ection\s+([\w]+(?:[A-Z])?)\s+(?:from|of)\s+Old\s+(IPC|CrPC|Indian Evidence Act|IEA)",
                re.IGNORECASE,
            )

            result: dict[tuple[str, str], tuple[str, str, str]] = dict(_CORE_MAP)
            for v in chunks.values():
                short = NEW_ACTS.get(v.get("act_name", ""))
                if not short:
                    continue
                ft = v.get("full_text") or ""
                sec = str(v.get("section_number", ""))
                title = (v.get("section_title") or "").strip()
                for m in PATTERN.finditer(ft):
                    old_sec = m.group(1).strip().upper()
                    old_act_raw = m.group(2).strip().upper()
                    old_act = "IEA" if "EVIDENCE" in old_act_raw else old_act_raw
                    key = (old_act, old_sec)
                    if key not in result:
                        result[key] = (short, sec, title)

            _full_map = result
            logger.debug("act_equivalence: loaded %d old→new section mappings", len(result))
        except Exception as exc:
            logger.warning("act_equivalence: failed to load full map (%s), using core only", exc)
            _full_map = dict(_CORE_MAP)

        return _full_map


def lookup(old_act: str, old_section: str) -> Optional[tuple[str, str, str]]:
    """Return (new_short, new_section, description) or None.

    old_act: 'IPC', 'CRPC', 'IEA'  (case-insensitive)
    old_section: '498A', '125', '65B'
    """
    key = (old_act.upper().replace(".", ""), old_section.upper().replace(" ", ""))
    # Try core map first (no I/O); fall back to full map
    result = _CORE_MAP.get(key)
    if result:
        return result
    return _load_full_map().get(key)


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
# Patterns: "IPC 498A", "Section 438 CrPC", "s.65B Evidence Act", "498A IPC"
_OLD_SEC_PATTERNS = [
    # "IPC 498A", "IPC Section 498A", "IPC s.498A"
    re.compile(r"\b(IPC|CrPC|I\.P\.C|Cr\.P\.C)\s+(?:[Ss]ection\s+|[Ss]\.)?(\w+)", re.IGNORECASE),
    # "498A IPC" / "438 CrPC"
    re.compile(r"\b(\w+)\s+(IPC|CrPC|I\.P\.C|Cr\.P\.C)\b", re.IGNORECASE),
    # "Section 498A IPC" / "Section 65B Evidence Act"
    re.compile(r"\b[Ss]ection\s+(\w+)\s+(IPC|CrPC|I\.P\.C|Cr\.P\.C|Evidence Act|Indian Evidence Act|IEA)\b", re.IGNORECASE),
    # "s.65B IEA", "s65B Evidence Act"
    re.compile(r"\b[Ss]\.?\s*(\w+)\s+(IEA|Evidence Act|Indian Evidence Act)\b", re.IGNORECASE),
]

_ACT_NORMALISER = {
    "IPC": "IPC", "I.P.C": "IPC",
    "CRPC": "CRPC", "CR.P.C": "CRPC",
    "IEA": "IEA", "EVIDENCE ACT": "IEA", "INDIAN EVIDENCE ACT": "IEA",
}


def expand_query(query: str) -> str:
    """Append BNS/BNSS/BSA equivalents for any old-act references in the query.

    E.g. "IPC 498A cruelty husband" → "IPC 498A cruelty husband BNS Section 85"
    Returns the original query unchanged if no old-act refs are found.
    """
    additions: list[str] = []
    seen: set[tuple] = set()

    for pat in _OLD_SEC_PATTERNS:
        for m in pat.finditer(query):
            groups = m.groups()
            if len(groups) == 2:
                g0, g1 = groups[0].upper().replace(".", ""), groups[1].upper().replace(".", "")
                # Determine which group is act and which is section
                if g0 in ("IPC", "CRPC", "IEA", "EVIDENCEACT", "INDIANEVIDENCEACT"):
                    act_raw, sec = g0, g1
                elif g1 in ("IPC", "CRPC", "IEA", "EVIDENCEACT", "INDIANEVIDENCEACT"):
                    act_raw, sec = g1, g0
                else:
                    # Second group is act name with spaces
                    act_raw = _ACT_NORMALISER.get(g1, g1)
                    sec = g0
                act = _ACT_NORMALISER.get(act_raw, act_raw)
                key = (act, sec)
                if key in seen:
                    continue
                seen.add(key)
                hit = lookup(act, sec)
                if hit:
                    new_short, new_sec, desc = hit
                    additions.append(f"{new_short} Section {new_sec} {desc}".strip())

    if not additions:
        return query
    expansion = " ".join(additions)
    logger.debug("act_equivalence.expand_query: added %r", expansion)
    return f"{query} {expansion}"


def new_act_name_for(old_name: str) -> Optional[str]:
    """Return canonical new act name for an old act name/abbreviation, or None."""
    return OLD_ACT_NAME_ALIASES.get(old_name.lower().strip())


def equivalence_note(old_act: str, old_section: str) -> str:
    """Return a short inline note like 'IPC Section 498A → BNS Section 85 (cruelty by husband or relatives)'.
    Returns empty string if no mapping found."""
    hit = lookup(old_act, old_section)
    if not hit:
        return ""
    new_short, new_sec, desc = hit
    note = f"IPC Section {old_section}" if old_act == "IPC" else f"{old_act} Section {old_section}"
    new_note = f"{new_short} Section {new_sec}"
    if desc:
        new_note += f" ({desc})"
    return f"{note} → {new_note}"
