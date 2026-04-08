"""
core/citations.py — Citation graph: case → interprets → section, case → cites → case.
"""

import hashlib
import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded singleton
_graph: Optional[dict] = None


def _normalize_case_name_key(case_name: str) -> str:
    """Normalized case name for keying: lower, collapse spaces."""
    if not case_name or not isinstance(case_name, str):
        return ""
    return " ".join((case_name or "").split()).strip().lower()[:200]


def _stable_case_id(case_name: str, year: str = "") -> str:
    """Stable id for a case: hash(normalized_name + year) so same case+year = one node."""
    norm = _normalize_case_name_key(case_name)
    if not norm:
        return ""
    key = f"{norm}|{str(year).strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _normalize_case_id(case_name: str, court: str = "", year: str = "") -> str:
    """Stable id (same as _stable_case_id). Kept for backward compatibility."""
    return _stable_case_id(case_name, year)


def _normalize_cited_case_name(name: str) -> str:
    """Normalize a cited case name for matching to graph nodes."""
    return _normalize_case_name_key(name)


def _clean_cited_name(name: str) -> str:
    """
    Clean OCR / extraction artifacts from a cited case name before normalizing.
    - Collapses newlines and excess whitespace (OCR line-breaks mid-name)
    - Strips leading/trailing punctuation noise
    """
    if not name or not isinstance(name, str):
        return ""
    # Collapse any whitespace including newlines into a single space
    cleaned = " ".join(name.split())
    # Strip leading/trailing punctuation that isn't part of a name
    cleaned = cleaned.strip(".,;:()\"'")
    return cleaned


def _normalize_for_cites_lookup(name: str) -> str:
    """
    Extended normalization for cited-case name lookup.
    Converts 'vs.' / 'versus' → 'v' so short extracted names can match indexed forms.
    """
    norm = _normalize_case_name_key(name)
    # Canonicalize separator: 'vs.' / 'vs' / 'versus' → 'v'
    norm = re.sub(r'\bvs\.?\b|\bversus\b', 'v', norm)
    # Collapse any double spaces introduced above
    norm = " ".join(norm.split())
    return norm


def _is_bad_case_name(name: str) -> bool:
    """True if case name should be excluded from graph (OCR, headings, non-case)."""
    try:
        from core.chunker import _is_likely_bad_case_name
        return _is_likely_bad_case_name(name)
    except Exception:
        return False


def _compute_pagerank(case_info: dict, cites: list, damping: float = 0.85, max_iters: int = 50) -> dict:
    """
    PageRank on citation graph: out-edge = "cites". Being cited increases authority.
    Returns dict case_id -> score (higher = more cited / more authoritative).
    """
    nodes = list(case_info.keys())
    if not nodes or not cites:
        return {}
    n = len(nodes)
    out_degree = {c: 0 for c in nodes}
    for e in cites:
        out_degree[e["from_case_id"]] = out_degree.get(e["from_case_id"], 0) + 1
    # in-edges: for each node, who cites it
    in_edges = {c: [] for c in nodes}
    for e in cites:
        to_id = e.get("to_case_id")
        if to_id in in_edges:
            in_edges[to_id].append(e["from_case_id"])
    score = {c: 1.0 / n for c in nodes}
    for _ in range(max_iters):
        new_score = {}
        for c in nodes:
            inc = (1.0 - damping) / n
            for from_id in in_edges[c]:
                od = out_degree.get(from_id, 1)
                inc += damping * score.get(from_id, 0) / max(od, 1)
            new_score[c] = inc
        score = new_score
    return score


def build_citation_graph_from_chunks(chunks_dict: dict, out_path: str) -> dict:
    """
    Build citation graph from case-law chunk store (id → chunk).
    Uses stable case IDs (hash of normalized_name + year), filters bad/truncated nodes and edges.
    Edge targets (to_case_id) are resolved to stable_id when the cited case exists in the graph.
    Runs PageRank and stores authority_score per case.
    """
    case_info = {}
    name_to_ids = {}  # normalized_name -> [stable_id] for lookup

    # Pass 1: build nodes (case_info, name_to_ids)
    # Register both the raw-normalised key AND a vs→v-normalised key so that
    # short extracted references ("A v B") can resolve to full indexed names
    # ("A vs The B ...") in Pass 2.
    for _idx, chunk in chunks_dict.items():
        if not isinstance(chunk, dict) or chunk.get("doc_type") != "case_law":
            continue
        case_name = (chunk.get("case_name") or "").strip()
        court = (chunk.get("court") or "").strip()
        year = str(chunk.get("year") or "").strip()
        if not case_name or _is_bad_case_name(case_name):
            continue
        cid = _stable_case_id(case_name, year)
        if cid not in case_info:
            case_info[cid] = {"case_name": case_name, "court": court, "year": year}
            norm_name = _normalize_case_name_key(case_name)
            if norm_name:
                name_to_ids.setdefault(norm_name, []).append(cid)
            # Also register v-normalised variant so short cited references can match
            norm_v = _normalize_for_cites_lookup(case_name)
            if norm_v and norm_v != norm_name:
                name_to_ids.setdefault(norm_v, []).append(cid)

    # Pass 2: build interprets and cites (resolve to_case_id to stable_id)
    interprets = []
    cites = []
    for _idx, chunk in chunks_dict.items():
        if not isinstance(chunk, dict) or chunk.get("doc_type") != "case_law":
            continue
        case_name = (chunk.get("case_name") or "").strip()
        court = (chunk.get("court") or "").strip()
        year = str(chunk.get("year") or "").strip()
        if not case_name or _is_bad_case_name(case_name):
            continue
        cid = _stable_case_id(case_name, year)
        for sec in (chunk.get("sections_cited") or []):
            if sec and isinstance(sec, str):
                interprets.append({"case_id": cid, "section_id": sec.strip()})
        for cited in (chunk.get("cited_cases") or []):
            if not cited or not isinstance(cited, str):
                continue

            # ── Step 1: clean OCR/extraction artifacts (newlines, stray punctuation)
            cited_clean = _clean_cited_name(cited)
            if not cited_clean or _is_bad_case_name(cited_clean):
                continue

            # ── Step 2: try exact-normalised lookup first
            to_norm = _normalize_cited_case_name(cited_clean)
            if len(to_norm) < 10:
                continue
            to_id = name_to_ids.get(to_norm, [None])[0] if to_norm in name_to_ids else None

            # ── Step 3: if not found, try vs→v-normalised variant
            if to_id is None:
                to_norm_v = _normalize_for_cites_lookup(cited_clean)
                if to_norm_v and to_norm_v != to_norm:
                    to_id = name_to_ids.get(to_norm_v, [None])[0] if to_norm_v in name_to_ids else None

            # ── Step 4: still not found → create a lightweight stub node so the
            #    edge is never silently dropped.  Stub nodes carry is_stub=True so
            #    callers can choose whether to surface them.
            if to_id is None:
                stub_id = _stable_case_id(cited_clean, "")
                if stub_id not in case_info:
                    case_info[stub_id] = {
                        "case_name": cited_clean,
                        "court": "",
                        "year": "",
                        "is_stub": True,
                    }
                    name_to_ids.setdefault(to_norm, []).append(stub_id)
                to_id = stub_id

            if to_id == cid:
                continue
            cites.append({"from_case_id": cid, "to_case_id": to_id})

    # Dedupe interprets and cites
    interprets = list({(e["case_id"], e["section_id"]): e for e in interprets}.values())
    cites = list({(e["from_case_id"], e["to_case_id"]): e for e in cites}.values())

    # PageRank and store authority_score in case_info
    pagerank = _compute_pagerank(case_info, cites)
    for cid, info in case_info.items():
        if cid in pagerank:
            info["authority_score"] = round(pagerank[cid], 6)

    payload = {
        "case_info": case_info,
        "name_to_ids": name_to_ids,
        "interprets": interprets,
        "cites": cites,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(
        "Citation graph built: %d cases, %d interprets, %d cites → %s",
        len(case_info), len(interprets), len(cites), out_path,
    )
    return payload


def load_graph(path: str) -> Optional[dict]:
    """Load graph from JSON. Returns None if missing or invalid."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Citation graph load failed: %s", e)
        return None


def get_graph() -> Optional[dict]:
    """Lazy-load and return the citation graph (singleton). Uses config.CITATION_GRAPH_PATH."""
    global _graph
    if _graph is not None:
        return _graph
    try:
        from config import CITATION_GRAPH_PATH
        _graph = load_graph(CITATION_GRAPH_PATH)
    except Exception as e:
        logger.debug("Citation graph not available: %s", e)
        _graph = None
    return _graph


def get_case_authority_score(case_name: str, year: str = "") -> float:
    """
    Return PageRank-derived authority score for a case (0 if not in graph or no score).
    Used in retrieval to boost landmark / highly cited cases.
    """
    graph = get_graph()
    if not graph:
        return 0.0
    case_info = graph.get("case_info") or {}
    cid = _stable_case_id(case_name, year)
    info = case_info.get(cid, {})
    return float(info.get("authority_score", 0.0))


def invalidate_graph() -> None:
    """Clear the in-memory graph cache so the next get_graph() reloads from disk (e.g. after indexing)."""
    global _graph
    _graph = None


def _case_ids_from_names(case_names: list, graph: dict) -> set:
    """Resolve display case names to graph case_ids (stable_ids). Uses name_to_ids if present, else builds from case_info."""
    case_info = graph.get("case_info") or {}
    name_to_ids = graph.get("name_to_ids") or {}
    if not name_to_ids and case_info:
        for cid, info in case_info.items():
            name = (info.get("case_name") or "").strip()
            if name:
                key = _normalize_case_name_key(name)
                if key:
                    name_to_ids.setdefault(key, []).append(cid)
    result = set()
    for name in (case_names or []):
        if not name or not isinstance(name, str):
            continue
        key = _normalize_case_name_key(name)
        if key and key in name_to_ids:
            result.update(name_to_ids[key])
    return result


def get_sections_interpreted_by_cases(case_names: list) -> list:
    """
    Return section_ids (e.g. "IPC 302") that any of the given cases interpret.
    case_names: list of display case names from retrieval chunks.
    """
    graph = get_graph()
    if not graph:
        return []
    cids = _case_ids_from_names(case_names, graph)
    if not cids:
        return []
    interprets = graph.get("interprets") or []
    sections = []
    seen = set()
    for e in interprets:
        if e.get("case_id") in cids:
            sec = (e.get("section_id") or "").strip()
            if sec and sec not in seen:
                seen.add(sec)
                sections.append(sec)
    return sections


def get_cases_cited_by(case_names: list, include_stubs: bool = False) -> list:
    """
    Return (case_id, display_name) of cases that the given cases cite.
    include_stubs: if False (default) only return fully-indexed cases (is_stub not set).
    Useful to expand retrieval with precedent.
    """
    graph = get_graph()
    if not graph:
        return []
    cids = _case_ids_from_names(case_names, graph)
    if not cids:
        return []
    cites = graph.get("cites") or []
    case_info = graph.get("case_info") or {}
    out_ids = set()
    for e in cites:
        if e.get("from_case_id") in cids:
            to_id = (e.get("to_case_id") or "").strip()
            if to_id:
                out_ids.add(to_id)
    result = []
    for cid in out_ids:
        info = case_info.get(cid, {})
        if not include_stubs and info.get("is_stub"):
            continue
        result.append((cid, (info.get("case_name") or cid)))
    return result


def get_cases_citing(case_names: list, include_stubs: bool = False) -> list:
    """
    Return (case_id, display_name) of cases that cite any of the given cases (reverse edges).
    include_stubs: if False (default) only return fully-indexed cases (is_stub not set).
    """
    graph = get_graph()
    if not graph:
        return []
    cids = _case_ids_from_names(case_names, graph)
    if not cids:
        return []
    cites = graph.get("cites") or []
    case_info = graph.get("case_info") or {}
    out_ids = set()
    for e in cites:
        if e.get("to_case_id") in cids:
            from_id = (e.get("from_case_id") or "").strip()
            if from_id:
                out_ids.add(from_id)
    result = []
    for cid in out_ids:
        info = case_info.get(cid, {})
        if not include_stubs and info.get("is_stub"):
            continue
        result.append((cid, (info.get("case_name") or cid)))
    return result


def expand_case_names_by_precedent(case_names: list, max_extra: int = 15) -> tuple[list, list]:
    """
    Given case names from retrieval, return (extra_case_names, sections_interpreted).
    extra_case_names: names of cases to add (cited by or citing), capped at max_extra.
    sections_interpreted: section_ids those cases interpret (for bare-act boost or display).
    Stubs (unindexed cited cases) are excluded from expansion since they have no chunks.
    """
    sections = get_sections_interpreted_by_cases(case_names)
    cited = get_cases_cited_by(case_names, include_stubs=False)
    citing = get_cases_citing(case_names, include_stubs=False)
    extra = []
    seen = set(_normalize_case_id(n) for n in case_names)
    for _cid, name in cited + citing:
        key = _normalize_case_id(name) if isinstance(name, str) else _normalize_case_id(str(_cid))
        if key and key not in seen and len(extra) < max_extra:
            seen.add(key)
            extra.append(name if isinstance(name, str) else _cid)
    return extra[:max_extra], sections


def get_chunks_by_case_names(chunks_dict: dict, case_names: list, max_total: int = 10) -> list:
    """
    From a chunk store (id -> chunk), return chunks whose case_name normalizes to one of
    case_names. Used to add precedent-expanded chunks at retrieval. Each returned chunk
    gets _citation_expansion=True and _rerank_score=0 so they sort after main results.
    """
    if not case_names or not chunks_dict:
        return []
    want = {_normalize_case_name_key(n) for n in case_names if n and isinstance(n, str)}
    if not want:
        return []
    out = []
    seen_keys = set()
    for _idx, chunk in chunks_dict.items():
        if not isinstance(chunk, dict) or chunk.get("doc_type") != "case_law":
            continue
        name = (chunk.get("case_name") or "").strip()
        if not name:
            continue
        if _normalize_case_name_key(name) not in want:
            continue
        cid = chunk.get("chunk_id") or _idx
        if cid in seen_keys:
            continue
        seen_keys.add(cid)
        c = dict(chunk)
        c["_citation_expansion"] = True
        c["_rerank_score"] = c.get("_rerank_score", 0.0)
        if "text" not in c or not (c.get("text") or "").strip():
            c["text"] = (c.get("search_text") or c.get("full_text") or "").strip()
        out.append(c)
        if len(out) >= max_total:
            break
    return out
