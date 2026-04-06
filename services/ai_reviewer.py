"""
ai_reviewer.py
──────────────
AI Gate: reads a logged Feedback Log row by Case ID, asks the configured LLM
to review the model output against the Golden Rules (R001–R045), and writes
results back into columns I–M (AI Gate 5) and S–T, W (Final Feedback: Issues, What should differ, Rating).

23-column layout:
  A–B Identity (Timestamp, Case ID)  C–H User Input + Model Output
  I–M AI Gate (5): G Disputes, H Sections, Additional info, I Case Laws, J Legal Opinion
  N–R Human Gate (5)  S–W Final Feedback (5)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

# ── Golden Rules snapshot (kept in code so AI Gate can check without reading
#    the Excel file — update when new rules are added to the workbook) ────────
GOLDEN_RULES: dict[str, str] = {
    "R001": "User input MUST be decomposed into one or more discrete dispute components before retrieval.",
    "R002": "Each dispute component must be independently resolvable using Indian law.",
    "R003": "Disputes must be labelled clearly (e.g. 'Dispute 1: Non-payment of rent').",
    "R004": "A follow-up question must be asked when the facts are ambiguous or incomplete.",
    "R005": "Follow-up questions must directly affect the legal analysis (not cosmetic).",
    "R006": "Only one follow-up question per response to avoid overwhelming the user.",
    "R007": "Act identification must be based on dispute keywords, not broad guesses.",
    "R008": "The top 3 most relevant acts must be identified before section retrieval.",
    "R009": "Act identification must prefer specific acts over general acts when both apply.",
    "R010": "Act name must match the official Indian statute title exactly.",
    "R011": "Section retrieval must be scoped to the identified acts only.",
    "R012": "Each retrieved section must directly address at least one identified dispute.",
    "R013": "Section number shown to the user must match the section text verbatim.",
    "R014": "Verbatim bare act text must be trimmed to the single most relevant section.",
    "R015": "Verbatim text must not exceed 1200 characters unless the section is shorter.",
    "R016": "Case laws must be retrieved per-dispute, not globally.",
    "R017": "Case law citations must include: case name, court, and year.",
    "R018": "Each case law must be linked to the section it interprets.",
    "R019": "Landmark/Supreme Court cases preferred over lower court decisions.",
    "R020": "Case laws must not be fabricated; cite only verified sources.",
    "R021": "Legal opinion must open with a brief restatement of the key dispute(s).",
    "R022": "Legal opinion must address every identified dispute component.",
    "R023": "Legal opinion must cite sections and case laws inline, not in a footnote-only list.",
    "R024": "Legal opinion must state the applicable legal standard or test.",
    "R025": "Legal opinion must include a practical recommendation or next step.",
    "R026": "Legal opinion must not give financial advice (quantum of damages, valuations).",
    "R027": "Legal opinion must flag jurisdiction if it varies across states.",
    "R028": "Legal opinion must distinguish between civil and criminal remedies where both apply.",
    "R029": "Legal opinion must not promise a specific legal outcome.",
    "R030": "Legal opinion must close with a disclaimer that it is not a substitute for legal counsel.",
    "R031": "Re-scoring of case laws using cross-encoder after retrieval is prohibited.",
    "R032": "Case laws must be matched to sections via dispute-tag + text matching only.",
    "R033": "Minimum cross-encoder rerank score for bare acts: 0.5.",
    "R034": "High-quality bare act threshold: cross-encoder score ≥ 2.0.",
    "R035": "High-quality case law threshold: cross-encoder score ≥ 5.0.",
    "R036": "sci.gov.in URLs must only be used for judgment/order paths, not general pages.",
    "R037": "Act profile index must skip corrupt act-name fragments (e.g. single-letter prefixes).",
    "R038": "Verbatim section text must match section number shown (no metadata mismatch).",
    "R039": "Embedding and cross-encoder models must be pre-warmed at server startup.",
    "R040": "Cross-log hints must be validated against the allowed_acts set before use.",
    "R041": "Feedback log must be auto-filled (Identity + Input + Output) on every interaction.",
    "R042": "AI Gate must run before Human Gate marks a row as Approved.",
    "R043": "Human Gate override notes must be recorded before changing AI Gate verdict.",
    "R044": "Rule violations found by AI Gate must use the standard R-code format.",
    "R045": "Any new rule derived from feedback must be added to the Golden Rules sheet.",
}

# ── prompt template ──────────────────────────────────────────────────────────
_REVIEW_PROMPT = """\
You are a senior Indian legal AI quality auditor reviewing a Nyaymalaw model response.

## TASK
Review the model output below against the Golden Rules. Identify rule violations, suggest corrections,
and provide a quality rating. Respond ONLY with valid JSON (no markdown fences).

## GOLDEN RULES (selected relevant subset)
{rules_json}

## INTERACTION TO REVIEW

**User Facts:**
{facts}

**Follow-up Question Asked by Model:**
{followup}

**Disputes Identified by Model:**
{disputes}

**Sections Retrieved:**
{sections}

**Case Laws Retrieved:**
{case_laws}

**Legal Opinion:**
{opinion}

## RESPONSE FORMAT (return exactly this JSON, fill all fields):
{{
  "gate_status": "Pass|Fail|Review",
  "rule_violations": ["R0XX", ...],
  "suggested_rating": 1,
  "better_followup": "<improved follow-up question, or empty string if current is fine>",
  "actual_disputes": "<corrected/confirmed dispute list>",
  "actual_sections": "<corrected/confirmed section refs>",
  "actual_case_laws": "<corrected/confirmed case citations>",
  "ai_legal_opinion": "<concise corrected legal opinion, or empty string if opinion is correct>",
  "issues_in_opinion": "<enumerated issues, or empty string>",
  "what_should_differ": "<corrective guidance, or empty string>"
}}

Rules for gate_status:
- Pass  : 0 violations, rating ≥ 4
- Fail  : any critical violation (R013, R014, R020, R022, R023, R030) OR rating ≤ 2
- Review: 1–2 minor violations OR rating = 3

suggested_rating scale: 1=Poor 2=Below average 3=Acceptable 4=Good 5=Excellent
"""

# ── column index map (1-based) — 26-column layout ────────────────────────────
# A=Timestamp B=Case ID  C=Facts D=Disputes E=Sections F=Additional info
# G=Case Laws H=Opinion I=Router classification
# J–O=AI Gate (6)  P–U=Human Gate (6)  V–Z=Final Feedback (5)
_COL = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "J": 10,
    "K": 11,
    "L": 12,
    "M": 13,
    "N": 14,
    "O": 15,
    "P": 16,
    "Q": 17,
    "R": 18,
    "S": 19,
    "T": 20,
    "U": 21,
    "V": 22,
    "W": 23,
    "X": 24,
    "Y": 25,
    "Z": 26,
}


def _resolve_path(override: Optional[str] = None) -> Path:
    p = override or os.environ.get("FEEDBACK_LOG_PATH", "")
    if not p:
        try:
            from config import FEEDBACK_LOG_PATH as _cp
            p = _cp
        except Exception:
            pass
    if not p:
        raise ValueError("FEEDBACK_LOG_PATH not configured.")
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(f"Feedback log not found: {path}")
    return path


def _find_row(ws, case_id: str) -> Optional[int]:
    for row in ws.iter_rows(min_row=4, min_col=2, max_col=2):
        if row[0].value == case_id:
            return row[0].row
    return None


def _call_llm(prompt: str) -> str:
    """
    Call OpenAI for the AI Gate review.

    Returns the raw text content of the response.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Cannot call OpenAI API."
        )
    try:
        import requests as _req
        resp = _req.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=120,
        )
        if resp.ok:
            return resp.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"OpenAI call failed: HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error("OpenAI call failed: %s", e)
        raise RuntimeError(f"OpenAI call failed: {e}") from e


def _parse_json(raw: str) -> dict:
    """Extract the first JSON object from raw LLM output."""
    raw = raw.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Find first { … }
    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in LLM output: {raw[:200]}")
    return json.loads(raw[start:end+1])


def run_ai_review(
    case_id: str,
    *,
    feedback_log_path: Optional[str] = None,
    rules_subset: Optional[list[str]] = None,
) -> dict:
    """
    Run AI Gate review for a given Case ID.

    Parameters
    ----------
    case_id          : e.g. 'NM-20260301-004'
    feedback_log_path: Override path to the .xlsx file.
    rules_subset     : Limit which rule codes to include in the prompt.
                       Defaults to all 45 rules.

    Returns the parsed review dict (same structure as JSON response).
    Raises on error; caller should handle.
    """
    path = _resolve_path(feedback_log_path)

    with _LOCK:
        try:
            from openpyxl import load_workbook
            from openpyxl.styles import Font, Alignment
        except ImportError:
            raise ImportError("openpyxl required: pip install openpyxl")

        wb = load_workbook(str(path))
        if "Feedback Log" not in wb.sheetnames:
            raise ValueError("No 'Feedback Log' sheet.")
        ws = wb["Feedback Log"]

        row = _find_row(ws, case_id)
        if row is None:
            raise KeyError(f"Case ID '{case_id}' not found in Feedback Log.")

        def _cell(col_letter: str):
            return ws.cell(row=row, column=_COL[col_letter]).value or ""

        facts      = _cell("C")
        followup   = _cell("F")
        disputes   = _cell("D")
        sections   = _cell("E")
        case_laws  = _cell("G")
        opinion    = _cell("H")

    # ── select rules ─────────────────────────────────────────────────────
    if rules_subset:
        rules = {k: v for k, v in GOLDEN_RULES.items() if k in rules_subset}
    else:
        rules = GOLDEN_RULES

    prompt = _REVIEW_PROMPT.format(
        rules_json=json.dumps(rules, indent=2),
        facts=facts,
        followup=followup,
        disputes=disputes,
        sections=sections,
        case_laws=case_laws,
        opinion=opinion[:4000],  # guard against giant opinions
    )

    logger.info("AI Gate: reviewing %s …", case_id)
    raw = _call_llm(prompt)
    review = _parse_json(raw)

    # ── write results back ────────────────────────────────────────────────
    with _LOCK:
        wb = load_workbook(str(path))
        ws = wb["Feedback Log"]

        row = _find_row(ws, case_id)
        if row is None:
            raise KeyError(f"Case ID '{case_id}' disappeared from log during review.")

        today_str = date.today().strftime("%Y-%m-%d")
        violations_str = ", ".join(review.get("rule_violations", []))
        rating = str(review.get("suggested_rating", ""))
        gate   = review.get("gate_status", "Review")

        # 26-column layout — AI Gate occupies J–O (10–15):
        #   J=G Disputes, K=H Sections, L=Additional info, M=I Case Laws, N=J Legal Opinion, O=AI Router classification
        # Final Feedback V–Z (22–26): V=Issues, W=What should differ, X=Rule, Y=Actioned, Z=Rating
        updates: dict[str, str] = {
            "J": review.get("actual_disputes", ""),
            "K": review.get("actual_sections", ""),
            "L": review.get("better_followup", ""),
            "M": review.get("actual_case_laws", ""),
            "N": review.get("ai_legal_opinion", ""),
            "V": review.get("issues_in_opinion", ""),
            "W": review.get("what_should_differ", ""),
            "Z": rating,
        }

        body_font = Font(name="Arial", size=9)
        wrap_align = Alignment(wrap_text=True, vertical="top")

        for col_letter, value in updates.items():
            cell = ws.cell(row=row, column=_COL[col_letter])
            cell.value     = value
            cell.font      = body_font
            cell.alignment = wrap_align

        wb.save(str(path))
        logger.info(
            "AI Gate complete: %s → %s | violations: %s | rating: %s | cols J–N + V,W,Z written",
            case_id,
            gate,
            violations_str or "none",
            rating,
        )

    return review
