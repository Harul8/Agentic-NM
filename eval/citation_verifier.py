"""
Citation Verifier — Extracts all case names and section numbers from system output
and verifies each against Indian Kanoon and the local FAISS index.

Catches hallucinated citations — case names the LLM invented that don't exist.

Usage:
    python -m eval.citation_verifier --results eval/results/batch_full_pipeline_*.json
    python -m eval.citation_verifier --text "In Kesavananda Bharati v. State of Kerala (1973)..."

Verification sources:
1. Local FAISS/BM25 index (fast, checks if case exists in local DB)
2. Indian Kanoon search API (online, comprehensive)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("eval.citation_verifier")


# ---------------------------------------------------------------------------
# Citation Extraction
# ---------------------------------------------------------------------------

# Patterns for Indian case citations
_CASE_NAME_PATTERN = re.compile(
    r"(?:"
    r"([A-Z][A-Za-z\.\s]+(?:\s+(?:v\.\s*|v/s\s*|vs\.?\s*|versus\s+)[A-Z][A-Za-z\.\s&,]+))"  # X v. Y
    r")"
    r"(?:\s*\((\d{4})\))?"  # optional year in parens
    r"(?:\s*\[?(\d+\s*(?:SCC|AIR|SCR|Cr\.?L\.?J|Bom\.?L\.?R)\s*\d*)\]?)?"  # optional citation
)

_SECTION_PATTERN = re.compile(
    r"(?:Section|Sec\.?|S\.?)\s*(\d+[A-Za-z]*)"
    r"(?:\s+(?:of|,)\s+(?:the\s+)?([A-Z][A-Za-z\s,]+?(?:Act|Code|Ordinance|Bill|Rules)(?:\s*,?\s*\d{4})?))?",
    re.IGNORECASE,
)

_ACT_PATTERN = re.compile(
    r"\b((?:Indian\s+)?[A-Z][A-Za-z\s]+(?:Act|Code|Ordinance|Rules)(?:\s*,?\s*\d{4})?)\b"
)


def extract_case_citations(text: str) -> list:
    """Extract case name citations from text."""
    if not text:
        return []

    citations = []
    seen = set()

    for match in _CASE_NAME_PATTERN.finditer(text):
        name = match.group(1).strip() if match.group(1) else ""
        year = match.group(2) or ""
        reporter = match.group(3) or ""

        if not name or len(name) < 5:
            continue

        # Normalize
        name_clean = re.sub(r"\s+", " ", name).strip()
        key = name_clean.lower()
        if key in seen:
            continue
        seen.add(key)

        citations.append({
            "case_name": name_clean,
            "year": year,
            "reporter": reporter.strip(),
            "raw_match": match.group(0).strip(),
        })

    return citations


def extract_section_citations(text: str) -> list:
    """Extract bare act section references from text."""
    if not text:
        return []

    citations = []
    seen = set()

    for match in _SECTION_PATTERN.finditer(text):
        section = match.group(1).strip()
        act = (match.group(2) or "").strip()

        key = f"{act}|{section}".lower()
        if key in seen:
            continue
        seen.add(key)

        citations.append({
            "section_number": section,
            "act_name": act,
            "raw_match": match.group(0).strip(),
        })

    return citations


# ---------------------------------------------------------------------------
# Verification Against Local Index
# ---------------------------------------------------------------------------

def verify_case_in_local_index(case_name: str) -> dict:
    """Check if a case name exists in the local FAISS/chunks store."""
    from config import CASE_CHUNKS_V2, CASE_CHUNKS
    from core.retriever import load_chunks

    result = {"found": False, "source": "local_index", "matches": []}

    name_lower = case_name.lower().strip()
    # Extract key party names for fuzzy match
    parties = [p.strip() for p in re.split(r"\s+v\.?\s+|\s+v/s\s+|\s+vs\.?\s+", name_lower) if p.strip()]

    for chunks_path in (CASE_CHUNKS_V2, CASE_CHUNKS):
        chunks = load_chunks(chunks_path)
        if not chunks:
            continue

        for key, chunk in chunks.items():
            chunk_name = (chunk.get("case_name") or chunk.get("source") or "").lower()
            if not chunk_name or chunk_name == "unknown":
                continue

            # Exact substring match
            if name_lower in chunk_name or chunk_name in name_lower:
                result["found"] = True
                result["matches"].append(chunk_name[:100])
                break

            # Party name match
            if parties:
                chunk_parties = [p.strip() for p in re.split(r"\s+v\.?\s+|\s+v/s\s+|\s+vs\.?\s+", chunk_name)]
                for party in parties:
                    if any(party in cp for cp in chunk_parties if len(party) > 3):
                        result["found"] = True
                        result["matches"].append(chunk_name[:100])
                        break
            if result["found"]:
                break
        if result["found"]:
            break

    return result


def verify_section_in_local_index(act_name: str, section: str) -> dict:
    """Check if a bare act section exists in the local index."""
    from config import BARE_CHUNKS_V2, BARE_CHUNKS
    from core.retriever import load_chunks

    result = {"found": False, "source": "local_index", "matches": []}

    act_lower = (act_name or "").lower().strip()
    sec_lower = (section or "").lower().strip()

    for chunks_path in (BARE_CHUNKS_V2, BARE_CHUNKS):
        chunks = load_chunks(chunks_path)
        if not chunks:
            continue

        for key, chunk in chunks.items():
            chunk_act = (chunk.get("act_name") or "").lower()
            chunk_sec = (chunk.get("section_number") or "").lower()

            if not chunk_act or not chunk_sec:
                continue

            act_match = act_lower in chunk_act or chunk_act in act_lower if act_lower else True
            sec_match = sec_lower == chunk_sec or sec_lower in chunk_sec

            if act_match and sec_match:
                result["found"] = True
                result["matches"].append(f"{chunk.get('act_name', '')} Section {chunk.get('section_number', '')}")
                break
        if result["found"]:
            break

    return result


# ---------------------------------------------------------------------------
# Verification via Indian Kanoon (online)
# ---------------------------------------------------------------------------

def verify_case_on_indian_kanoon(case_name: str) -> dict:
    """Search Indian Kanoon for a case name. Requires internet access."""
    import requests

    result = {"found": False, "source": "indian_kanoon", "url": "", "matches": []}

    try:
        # Indian Kanoon search API (public, no auth required)
        search_url = "https://api.indiankanoon.org/search/"
        params = {"formInput": case_name, "pagenum": 0}
        headers = {"Authorization": "Token YOUR_TOKEN_HERE"}  # Replace with actual token if available

        # Fallback: scrape search results page
        search_page = f"https://indiankanoon.org/search/?formInput={requests.utils.quote(case_name)}"
        resp = requests.get(search_page, timeout=15, headers={"User-Agent": "Nyaymalaw-Eval/1.0"})

        if resp.ok:
            text = resp.text.lower()
            name_parts = case_name.lower().split()
            # Check if key words appear in results
            key_words = [w for w in name_parts if len(w) > 3 and w not in ("the", "state", "union", "india")]
            matches = sum(1 for w in key_words if w in text)
            if matches >= len(key_words) * 0.5:
                result["found"] = True
                result["url"] = search_page

        time.sleep(1)  # Rate limiting
    except Exception as e:
        logger.warning(f"Indian Kanoon search failed for '{case_name}': {e}")
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Full Verification Pipeline
# ---------------------------------------------------------------------------

def verify_all_citations(text: str, check_online: bool = False) -> dict:
    """
    Extract and verify all citations in a piece of text.

    Returns:
        {
            "case_citations": [...],
            "section_citations": [...],
            "summary": {
                "total_cases": N,
                "verified_cases": N,
                "hallucinated_cases": N,
                "hallucination_rate": float,
                "total_sections": N,
                "verified_sections": N,
            }
        }
    """
    cases = extract_case_citations(text)
    sections = extract_section_citations(text)

    # Verify cases
    verified_cases = 0
    for case in cases:
        local = verify_case_in_local_index(case["case_name"])
        case["local_verified"] = local["found"]
        case["local_matches"] = local.get("matches", [])

        if check_online and not local["found"]:
            online = verify_case_on_indian_kanoon(case["case_name"])
            case["online_verified"] = online["found"]
            case["online_url"] = online.get("url", "")
        else:
            case["online_verified"] = None

        case["verified"] = local["found"] or (case.get("online_verified") or False)
        if case["verified"]:
            verified_cases += 1

    # Verify sections
    verified_sections = 0
    for sec in sections:
        local = verify_section_in_local_index(sec["act_name"], sec["section_number"])
        sec["local_verified"] = local["found"]
        sec["verified"] = local["found"]
        if sec["verified"]:
            verified_sections += 1

    total_cases = len(cases)
    total_sections = len(sections)
    hallucinated = total_cases - verified_cases

    return {
        "case_citations": cases,
        "section_citations": sections,
        "summary": {
            "total_cases": total_cases,
            "verified_cases": verified_cases,
            "hallucinated_cases": hallucinated,
            "hallucination_rate": round(hallucinated / max(total_cases, 1), 4),
            "total_sections": total_sections,
            "verified_sections": verified_sections,
        },
    }


def verify_batch_results(results_path: str, check_online: bool = False) -> dict:
    """Verify all citations across a batch of results."""
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    all_verifications = []
    totals = {"total_cases": 0, "verified_cases": 0, "hallucinated_cases": 0,
              "total_sections": 0, "verified_sections": 0}

    for result in results:
        final = result.get("final_response") or {}
        explanation = final.get("explanation", "")

        # Also check case law titles
        case_titles = " ".join(
            c.get("title", "") + " " + c.get("case_name", "")
            for c in final.get("case_laws", [])
        )
        full_text = f"{explanation}\n{case_titles}"

        verification = verify_all_citations(full_text, check_online)
        verification["query_id"] = result.get("query_id", "")
        all_verifications.append(verification)

        for key in totals:
            totals[key] += verification["summary"].get(key, 0)

    totals["hallucination_rate"] = round(
        totals["hallucinated_cases"] / max(totals["total_cases"], 1), 4
    )

    return {
        "source_file": results_path,
        "overall": totals,
        "per_query": all_verifications,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Verify citations in Nyaymalaw output")
    parser.add_argument("--results", help="Batch results JSON to verify")
    parser.add_argument("--text", help="Direct text to verify")
    parser.add_argument("--online", action="store_true", help="Also check Indian Kanoon (slower)")
    parser.add_argument("--out", help="Output file")
    args = parser.parse_args()

    if args.text:
        result = verify_all_citations(args.text, args.online)
        print(json.dumps(result, indent=2))
    elif args.results:
        report = verify_batch_results(args.results, args.online)
        output = json.dumps(report, indent=2, default=str)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(output)
            logger.info(f"Report saved to {args.out}")
        else:
            print(output)
    else:
        parser.print_help()
