"""
agents/verification_agent.py — Post-draft verification agent.

Extends the existing grounding guard (nm_platform/guard.py) into a full
checking loop that runs after draft_opinion produces output.

Three checks run in parallel:
  1. Citation check  — every cited section exists in retrieved bare acts
  2. Evidence gap    — relief claimed is supportable on stated facts
  3. Limitation flag — any mentioned dates imply a limitation risk

Returns a VerificationResult with:
  - passed: bool
  - issues: list of human-readable issues
  - safe_draft: the original draft if passed, or a patched/fallback draft
  - warnings: non-blocking observations for the advocate review panel
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from retrieval.guard import check_response_safety

logger = logging.getLogger("nyaymalaw.verification")

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    safe_draft: str = ""
    citation_check: dict = field(default_factory=dict)
    evidence_gap: dict = field(default_factory=dict)
    limitation_flag: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Check 1 — Citation verifier
# ---------------------------------------------------------------------------

def _check_citations(draft: str, retrieved_sections: list[dict]) -> dict:
    """
    Verify that every statutory citation in the draft appears in retrieved sections.

    A citation is flagged if the draft mentions "Section X of <Act>" but no
    retrieved section matches that act + number combination.

    Returns:
        {"passed": bool, "unsupported": [str], "supported": [str]}
    """
    # Extract cited sections from the draft
    # Pattern: "Section 14 of the PWDVA" | "s. 138 NI Act" | "Section 85 BNS"
    citation_patterns = [
        r"[Ss]ection\s+(\d+[A-Za-z]?)\s+(?:of\s+)?(?:the\s+)?([A-Z][A-Za-z\s,]+?(?:Act|Code|Rules|Order))",
        r"\bs\.?\s*(\d+[A-Za-z]?)\s+([A-Z]{2,}(?:\s+[A-Z][a-z]+)*)",
    ]
    cited: list[tuple[str, str]] = []  # (section_num, act_hint)
    for pattern in citation_patterns:
        for m in re.finditer(pattern, draft):
            sec_num = m.group(1).strip()
            act_hint = m.group(2).strip().rstrip(",.")
            cited.append((sec_num, act_hint))

    if not cited:
        return {"passed": True, "unsupported": [], "supported": [], "note": "No statutory citations detected"}

    # Build lookup from retrieved sections
    retrieved_index: set[tuple[str, str]] = set()
    for sec in (retrieved_sections or []):
        act = (sec.get("act_name") or "").lower()
        num = str(sec.get("section_number") or "").strip()
        if act and num:
            retrieved_index.add((num, act))

    supported = []
    unsupported = []
    for sec_num, act_hint in cited:
        act_lower = act_hint.lower()
        # Check if any retrieved section matches
        matched = any(
            num == sec_num and act_lower in idx_act
            for num, idx_act in retrieved_index
        )
        label = f"s.{sec_num} ({act_hint})"
        if matched:
            supported.append(label)
        else:
            unsupported.append(label)

    passed = len(unsupported) == 0
    return {
        "passed": passed,
        "unsupported": unsupported,
        "supported": supported,
    }


# ---------------------------------------------------------------------------
# Check 2 — Evidence gap detector
# ---------------------------------------------------------------------------

def _check_evidence_gaps(draft: str, intake_state: dict) -> dict:
    """
    Flag gaps between claimed relief and stated evidence posture.

    Uses simple heuristics — no LLM call, keeps this fast.
    The orchestrator can call the LLM for a deeper review if needed.

    Returns:
        {"passed": bool, "gaps": [str], "note": str}
    """
    if not intake_state:
        return {"passed": True, "gaps": [], "note": "No intake state available"}

    evidence_posture = (intake_state.get("evidence_posture") or "").lower()
    known_facts = intake_state.get("known_facts") or []
    relief_posture = (intake_state.get("relief_posture") or "").lower()

    gaps = []

    # Check: injunction/stay claimed but no urgency stated
    injunction_terms = ("injunction", "stay order", "interim relief", "urgent")
    urgency = (intake_state.get("urgency_signal") or "").lower()
    if any(t in draft.lower() for t in injunction_terms):
        if urgency not in ("high", "urgent", "immediate"):
            gaps.append(
                "Draft mentions interim relief but intake urgency signal is not high — "
                "verify with client whether immediate protection is actually needed."
            )

    # Check: criminal complaint mentioned but no FIR/prior action noted
    criminal_terms = ("FIR", "police complaint", "Section 498", "domestic violence", "cruelty")
    if any(t in draft for t in criminal_terms):
        prior_action = (intake_state.get("prior_actions_taken") or "").lower()
        if "fir" not in prior_action and "police" not in prior_action and "complaint" not in prior_action:
            gaps.append(
                "Draft references criminal complaint but no prior FIR or police complaint "
                "is recorded in intake facts — confirm with client."
            )

    # Check: evidence posture is weak but draft claims strong documentary support
    strong_doc_terms = ("documentary evidence", "supported by documents", "in writing")
    if any(t in draft.lower() for t in strong_doc_terms):
        if "no document" in evidence_posture or "oral only" in evidence_posture:
            gaps.append(
                "Draft implies documentary support but evidence posture indicates primarily "
                "oral evidence — review what documents the client actually holds."
            )

    passed = len(gaps) == 0
    return {"passed": passed, "gaps": gaps}


# ---------------------------------------------------------------------------
# Check 3 — Limitation period flagging
# ---------------------------------------------------------------------------

_LIMITATION_PATTERNS = [
    # "3 years ago", "two years back", "last year" etc.
    (r"(\d+)\s+years?\s+ago", lambda m: int(m.group(1))),
    (r"(two|three|four|five|six)\s+years?\s+(?:ago|back|earlier)", {
        "two": 2, "three": 3, "four": 4, "five": 5, "six": 6
    }),
]

# Rough limitation windows for common dispute types (years)
_LIMITATION_WINDOWS = {
    "contract":   3,
    "cheque":     1,   # NI Act s.138 — 1 month from 15 days after demand notice
    "property":   12,
    "tort":       3,
    "rent":       3,
    "employment": 3,
    "consumer":   2,   # Consumer Protection Act 2019
    "domestic":   None,  # No strict bar — but promptness matters
}


def _check_limitation(draft: str, intake_state: dict) -> dict:
    """
    Flag if the facts suggest a limitation period may have expired or is close.

    Returns:
        {"passed": bool, "risk_level": "none"|"low"|"high", "message": str}
    """
    if not intake_state:
        return {"passed": True, "risk_level": "none", "message": ""}

    category = (intake_state.get("category") or "").lower()
    years_elapsed: int | None = None

    # Try to find years mentioned in the draft
    for pattern, handler in _LIMITATION_PATTERNS:
        m = re.search(pattern, draft.lower())
        if m:
            if callable(handler):
                years_elapsed = handler(m)
            elif isinstance(handler, dict):
                word = m.group(1).lower()
                years_elapsed = handler.get(word)
            break

    if years_elapsed is None:
        return {"passed": True, "risk_level": "none", "message": ""}

    window = None
    for key, w in _LIMITATION_WINDOWS.items():
        if key in category:
            window = w
            break

    if window is None:
        return {"passed": True, "risk_level": "none", "message": ""}

    if years_elapsed >= window:
        return {
            "passed": False,
            "risk_level": "high",
            "message": (
                f"Facts suggest {years_elapsed} year(s) have elapsed. "
                f"Standard limitation for {category} matters is {window} year(s). "
                "Advise client to seek urgent legal counsel on whether the matter is time-barred."
            )
        }
    if years_elapsed >= window - 1:
        return {
            "passed": True,
            "risk_level": "low",
            "message": (
                f"Facts suggest {years_elapsed} year(s) have elapsed against a {window}-year window. "
                "Limitation is close — advise the client to act promptly."
            )
        }
    return {"passed": True, "risk_level": "none", "message": ""}


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

class VerificationAgent:
    """
    Runs three parallel checks on a draft opinion and returns a VerificationResult.

    Usage:
        verifier = VerificationAgent()
        result = verifier.verify(draft_text, retrieved_sections, intake_state)

        if not result.passed:
            # Use result.safe_draft — either patched or fallback
            ...
        if result.warnings:
            # Show to advocate review panel
            ...
    """

    def verify(
        self,
        draft: str,
        retrieved_sections: list[dict] | None = None,
        intake_state: dict | None = None,
    ) -> VerificationResult:
        """
        Run all three checks concurrently. Returns VerificationResult.
        """
        if not draft:
            return VerificationResult(passed=False, issues=["Empty draft"], safe_draft="")

        # ── 0. Existing response safety guard ────────────────────────
        guard_result = check_response_safety(draft)
        if not guard_result["safe"]:
            return VerificationResult(
                passed=False,
                issues=[f"Safety guard: {guard_result['reason']}"],
                safe_draft=self._fallback_draft(intake_state),
            )

        # ── 1–3. Run checks in parallel ───────────────────────────────
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_citation  = executor.submit(_check_citations, draft, retrieved_sections or [])
            f_evidence  = executor.submit(_check_evidence_gaps, draft, intake_state or {})
            f_limitation = executor.submit(_check_limitation, draft, intake_state or {})

            citation_result   = f_citation.result(timeout=10)
            evidence_result   = f_evidence.result(timeout=10)
            limitation_result = f_limitation.result(timeout=10)

        issues: list[str] = []
        warnings: list[str] = []

        # Citation failures → blocking issue
        if not citation_result["passed"]:
            for label in citation_result.get("unsupported", []):
                issues.append(f"Unsupported citation: {label} — not found in retrieved materials.")

        # Evidence gaps → warnings for advocate, not blocking
        for gap in evidence_result.get("gaps", []):
            warnings.append(gap)

        # Limitation → high risk is blocking; low is a warning
        lim = limitation_result
        if lim.get("risk_level") == "high":
            issues.append(lim["message"])
        elif lim.get("risk_level") == "low":
            warnings.append(lim["message"])

        passed = len(issues) == 0
        safe_draft = draft if passed else self._patch_draft(draft, issues) or self._fallback_draft(intake_state)

        return VerificationResult(
            passed=passed,
            issues=issues,
            warnings=warnings,
            safe_draft=safe_draft,
            citation_check=citation_result,
            evidence_gap=evidence_result,
            limitation_flag=limitation_result,
        )

    # ------------------------------------------------------------------
    # Draft repair helpers
    # ------------------------------------------------------------------

    def _patch_draft(self, draft: str, issues: list[str]) -> str:
        """
        Attempt to patch the draft by appending a grounded disclaimer
        when citations could not be verified.

        This keeps the substantive content but signals to the reader that
        some citations need human verification before reliance.
        """
        if not any("Unsupported citation" in i for i in issues):
            return draft  # Nothing to patch

        disclaimer = (
            "\n\n---\n"
            "**Advocate Note:** One or more statutory citations in this draft could not be "
            "verified against the retrieved materials. Please verify the following before "
            "relying on this opinion:\n"
        )
        for issue in issues:
            if "Unsupported citation" in issue:
                disclaimer += f"- {issue}\n"

        return draft + disclaimer

    def _fallback_draft(self, intake_state: dict | None) -> str:
        """
        Produce a minimal safe fallback when the draft is fully rejected.
        Uses the intake state to give at least a factual summary.
        """
        if not intake_state:
            return (
                "Based on the facts provided, this matter requires further review. "
                "Please consult a qualified legal professional for detailed advice."
            )

        facts = intake_state.get("known_facts") or []
        category = intake_state.get("category") or "legal"
        facts_summary = "; ".join(str(f) for f in facts[:5]) if facts else "as described"

        return (
            f"Based on the facts shared ({facts_summary}), this appears to be a {category} matter. "
            "The available legal materials support further analysis, but a full grounded opinion "
            "requires verification of the applicable statutory provisions. "
            "Please consult a qualified advocate who can review the supporting documents and "
            "provide advice grounded in the verified applicable law."
        )
