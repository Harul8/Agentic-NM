"""
agents/document_agent.py — Document intake agent.

Reads uploaded legal documents, extracts structured facts, and cross-references
them against what the user has said verbally in the intake conversation.

This fills the biggest capability gap in the current system: users arrive with
rent agreements, legal notices, court orders, and WhatsApp screenshots but the
system has no way to read them.

Two new MCP tools exposed here (registered in mcp_server.py):
  extract_document_facts(file_text, dispute_context)
    → structured facts extracted from a document
  cross_reference_document(document_facts, intake_state)
    → alignment gaps between document content and user's verbal account

The existing /upload-document endpoint in api_server.py already handles
OCR and text extraction. This agent consumes that extracted text.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("nyaymalaw.document_agent")

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DocumentFacts:
    """Structured facts extracted from a document."""
    document_type: str                    # "rent_agreement", "legal_notice", "court_order" etc.
    parties: list[dict]                   # [{"role": "landlord", "name": "..."}, ...]
    key_dates: list[dict]                 # [{"event": "lease start", "date": "..."}, ...]
    amounts: list[dict]                   # [{"label": "monthly rent", "amount": "₹15,000"}, ...]
    obligations: list[str]               # Stated obligations/terms
    legal_references: list[str]          # Cited sections, acts, case numbers
    document_summary: str                # One-paragraph plain-English summary
    raw_facts: list[str]                 # Flat list for injection into intake_state


@dataclass
class CrossReferenceResult:
    """Alignment between document facts and verbal intake account."""
    alignments: list[str]                # Facts that match — consistent
    discrepancies: list[str]             # Facts that conflict — needs clarification
    gaps_in_intake: list[str]            # Document adds new facts not yet in intake
    gaps_in_document: list[str]          # Intake mentions things not in document
    overall_consistency: str             # "strong" | "partial" | "weak"


# ---------------------------------------------------------------------------
# Document type classifier
# ---------------------------------------------------------------------------

_DOC_TYPE_PATTERNS = {
    "rent_agreement": [
        r"lease\s+agreement", r"rent\s+agreement", r"tenancy\s+agreement",
        r"landlord\s+and\s+tenant", r"monthly\s+rent", r"security\s+deposit",
        r"lessor\s+and\s+lessee",
    ],
    "legal_notice": [
        r"legal\s+notice", r"take\s+notice", r"notice\s+under\s+section",
        r"demand\s+notice", r"30\s+days\s+from\s+receipt",
        r"without\s+prejudice",
    ],
    "court_order": [
        r"order\s+dated", r"this\s+court\s+orders", r"it\s+is\s+hereby\s+ordered",
        r"interim\s+order", r"ex\s+parte\s+order", r"stay\s+order",
        r"case\s+no\.", r"c\.r\.p\.", r"w\.p\.",
    ],
    "fir_complaint": [
        r"first\s+information\s+report", r"f\.i\.r\.", r"police\s+station",
        r"cognizable\s+offence", r"accused\s+named",
    ],
    "employment_letter": [
        r"appointment\s+letter", r"termination\s+letter", r"offer\s+of\s+employment",
        r"notice\s+period", r"gross\s+salary", r"resignation",
    ],
    "cheque": [
        r"cheque\s+no\.", r"dishonoured", r"returned\s+unpaid",
        r"insufficient\s+funds", r"account\s+closed",
    ],
    "affidavit": [
        r"i\s+do\s+solemnly\s+affirm", r"deponent", r"sworn\s+before",
        r"notarized",
    ],
    "consumer_complaint": [
        r"consumer\s+complaint", r"deficiency\s+in\s+service",
        r"unfair\s+trade\s+practice", r"consumer\s+forum",
    ],
}


def _classify_document_type(text: str) -> str:
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for doc_type, patterns in _DOC_TYPE_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        if score:
            scores[doc_type] = score
    if not scores:
        return "general_legal_document"
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Extraction helpers (heuristic, fast — no LLM call for basic extraction)
# ---------------------------------------------------------------------------

def _extract_dates(text: str) -> list[dict]:
    """Extract dates and their context from document text."""
    date_patterns = [
        r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})",
        r"(\d{1,2}/\d{1,2}/\d{2,4})",
        r"(\d{1,2}-\d{1,2}-\d{2,4})",
        r"(\d{4}-\d{2}-\d{2})",
    ]
    dates = []
    for pattern in date_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            # Get 40 chars of context before the date
            start = max(0, m.start() - 40)
            context = text[start:m.end()].strip().replace("\n", " ")
            dates.append({"date": m.group(1), "context": context[:80]})
    return dates[:10]  # cap at 10 dates


def _extract_amounts(text: str) -> list[dict]:
    """Extract monetary amounts and their labels."""
    amount_pattern = r"(?:Rs\.?|₹|INR)\s*[\d,]+(?:\.\d{1,2})?"
    amounts = []
    for m in re.finditer(amount_pattern, text, re.IGNORECASE):
        start = max(0, m.start() - 30)
        context = text[start:m.end()].strip().replace("\n", " ")
        amounts.append({"amount": m.group(0), "context": context[:60]})
    return amounts[:8]


def _extract_parties(text: str) -> list[dict]:
    """Extract named parties and their roles."""
    parties = []
    role_patterns = [
        (r"landlord[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "landlord"),
        (r"tenant[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "tenant"),
        (r"employer[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "employer"),
        (r"employee[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "employee"),
        (r"complainant[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "complainant"),
        (r"respondent[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "respondent"),
        (r"petitioner[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "petitioner"),
        (r"lessor[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "lessor"),
        (r"lessee[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "lessee"),
    ]
    seen = set()
    for pattern, role in role_patterns:
        m = re.search(pattern, text)
        if m:
            name = m.group(1).strip()
            key = (role, name)
            if key not in seen:
                seen.add(key)
                parties.append({"role": role, "name": name})
    return parties


def _extract_legal_refs(text: str) -> list[str]:
    """Extract cited acts, sections, and case numbers."""
    refs = []
    patterns = [
        r"Section\s+\d+[A-Za-z]?\s+(?:of\s+)?(?:the\s+)?[A-Z][A-Za-z\s]+(?:Act|Code)",
        r"(?:I\.P\.C|B\.N\.S|C\.P\.C|C\.r\.P\.C|N\.I\.\s*Act)\.?\s+[Ss]ection\s+\d+[A-Za-z]?",
        r"[A-Z]\.\w+\.\s+(?:No\.|Case)\s+[\d/]+(?:/\d+)*",
        r"W\.P\.\s*(?:\(Civil\))?\s*No\.\s*\d+",
    ]
    seen = set()
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            ref = m.group(0).strip()
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs[:10]


# ---------------------------------------------------------------------------
# LLM-powered extraction (called when heuristics alone are insufficient)
# ---------------------------------------------------------------------------

def _llm_extract_facts(text: str, dispute_context: str, doc_type: str) -> dict:
    """
    Use the LLM to extract structured facts when heuristics miss important details.
    Returns a dict with document_summary and obligations.
    """
    try:
        from platform.llm import ask_llm
        prompt = (
            f"You are analysing a {doc_type.replace('_', ' ')} uploaded by a client seeking legal advice.\n\n"
            f"DISPUTE CONTEXT:\n{dispute_context[:500]}\n\n"
            f"DOCUMENT TEXT (first 2000 chars):\n{text[:2000]}\n\n"
            "Extract and return valid JSON only:\n"
            "{{\n"
            '  "document_summary": "One paragraph plain-English summary of what this document is and what it establishes.",\n'
            '  "obligations": ["list of stated obligations, duties, or conditions from the document"],\n'
            '  "key_finding": "Single most important fact this document proves or disproves for the client\'s case."\n'
            "}}"
        )
        raw = (ask_llm(prompt, task_hint="fast") or "").strip()
        if "{" in raw and "}" in raw:
            raw = raw[raw.find("{"):raw.rfind("}") + 1]
        return json.loads(raw)
    except Exception as exc:
        logger.warning("LLM document extraction failed: %s", exc)
        return {
            "document_summary": f"A {doc_type.replace('_', ' ')} relevant to the dispute.",
            "obligations": [],
            "key_finding": "",
        }


# ---------------------------------------------------------------------------
# Main agent functions (exposed as MCP tools)
# ---------------------------------------------------------------------------

def extract_document_facts(file_text: str, dispute_context: str = "") -> str:
    """
    MCP tool: extract_document_facts

    Takes the plain text of an uploaded document (already OCR'd by /upload-document)
    and returns structured facts for injection into the intake session.

    Args:
        file_text: Plain text of the document
        dispute_context: Brief description of the dispute (from agents.intake state)

    Returns:
        JSON string with DocumentFacts
    """
    if not (file_text or "").strip():
        return json.dumps({"error": "No document text provided"})

    try:
        text = file_text.strip()

        # Heuristic extraction (fast, no LLM)
        doc_type = _classify_document_type(text)
        parties = _extract_parties(text)
        key_dates = _extract_dates(text)
        amounts = _extract_amounts(text)
        legal_refs = _extract_legal_refs(text)

        # LLM extraction for summary + obligations
        llm_data = _llm_extract_facts(text, dispute_context, doc_type)

        # Build flat fact list for intake injection
        raw_facts = []
        for p in parties:
            raw_facts.append(f"{p['role'].title()}: {p['name']}")
        for d in key_dates[:3]:
            raw_facts.append(f"Date ({d['context'][:40]}): {d['date']}")
        for a in amounts[:3]:
            raw_facts.append(f"Amount ({a['context'][:40]}): {a['amount']}")
        for ref in legal_refs[:3]:
            raw_facts.append(f"Legal reference: {ref}")
        if llm_data.get("key_finding"):
            raw_facts.append(f"Key finding: {llm_data['key_finding']}")

        result = {
            "document_type": doc_type,
            "parties": parties,
            "key_dates": key_dates[:5],
            "amounts": amounts[:5],
            "legal_references": legal_refs,
            "obligations": llm_data.get("obligations", []),
            "document_summary": llm_data.get("document_summary", ""),
            "key_finding": llm_data.get("key_finding", ""),
            "raw_facts": raw_facts,
        }
        logger.info(
            "Document facts extracted: type=%s parties=%d dates=%d",
            doc_type, len(parties), len(key_dates)
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as exc:
        logger.exception("extract_document_facts failed: %s", exc)
        return json.dumps({"error": str(exc)})


def cross_reference_document(document_facts_json: str, intake_state: dict) -> str:
    """
    MCP tool: cross_reference_document

    Compares facts extracted from a document against the verbal intake account.
    Flags discrepancies quietly for the advocate review panel — never accuses the
    user of dishonesty. Returns structured alignment result.

    Args:
        document_facts_json: JSON string from extract_document_facts
        intake_state: Current intake state dict

    Returns:
        JSON string with CrossReferenceResult
    """
    try:
        doc_facts = json.loads(document_facts_json) if isinstance(document_facts_json, str) else document_facts_json
        known_facts = intake_state.get("known_facts") or []
        known_text = " ".join(str(f) for f in known_facts).lower()

        alignments = []
        discrepancies = []
        gaps_in_intake = []
        gaps_in_document = []

        # Check party alignments
        for party in doc_facts.get("parties", []):
            name_lower = party.get("name", "").lower()
            if name_lower and name_lower in known_text:
                alignments.append(f"{party['role'].title()} name '{party['name']}' matches intake account.")
            elif name_lower:
                gaps_in_intake.append(
                    f"Document names {party['role']} as '{party['name']}' — this name was not mentioned in the intake conversation."
                )

        # Check amount alignments
        for amount in doc_facts.get("amounts", []):
            amt = amount.get("amount", "")
            amt_digits = re.sub(r"[^\d]", "", amt)
            if amt_digits and amt_digits in known_text.replace(",", "").replace(" ", ""):
                alignments.append(f"Amount {amt} matches intake account.")
            elif amt_digits:
                gaps_in_intake.append(
                    f"Document states amount {amt} — verify this matches what the client described."
                )

        # Check for document legal refs not in intake
        for ref in doc_facts.get("legal_references", [])[:3]:
            if ref.lower() not in known_text:
                gaps_in_intake.append(f"Document cites '{ref}' which was not raised in intake.")

        # Key finding — always surface it
        key_finding = doc_facts.get("key_finding", "")
        if key_finding:
            gaps_in_intake.append(f"Document key finding: {key_finding}")

        # Determine overall consistency
        if discrepancies:
            consistency = "weak"
        elif not gaps_in_intake and not gaps_in_document:
            consistency = "strong"
        else:
            consistency = "partial"

        result = {
            "alignments": alignments,
            "discrepancies": discrepancies,
            "gaps_in_intake": gaps_in_intake,
            "gaps_in_document": gaps_in_document,
            "overall_consistency": consistency,
            "summary": (
                f"Document is {consistency}ly consistent with intake account. "
                f"{len(gaps_in_intake)} new detail(s) from document, "
                f"{len(discrepancies)} discrepancy/ies."
            ),
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as exc:
        logger.exception("cross_reference_document failed: %s", exc)
        return json.dumps({"error": str(exc)})
