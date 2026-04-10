"""
agents/forum_agent.py — Forum identification and limitation period agent.

Identifies the correct Indian legal forum for a dispute and checks whether
limitation periods are still open. Currently this logic is buried inside
draft generation — making it an explicit agent unlocks two things:

  1. The orchestrator can surface forum advice early (before the full draft)
  2. Limitation warnings can be flagged during intake, not just in the draft

Two new MCP tools:
  identify_forum(intake_state)
    → recommended forum(s) with filing requirements and jurisdiction notes
  check_limitation(dispute_type, key_dates)
    → limitation period status with urgency flag

Indian legal forum landscape covered:
  Civil court hierarchy          → Munsiff/Sub-Judge → District Court → High Court
  Family Court                   → matrimonial, custody, maintenance
  Labour Court / CGIT            → employment disputes, retrenchment
  Consumer Forum (DCDRC/SCDRC/NCDRC) → consumer protection
  Rent Tribunal / RERA           → tenancy, real estate
  DRT / DRAT                     → debt recovery > ₹20 lakh
  NCLT / NCLAT                   → company law, insolvency
  Protection Officer / Magistrate → domestic violence (PWDVA)
  Police / Magistrate            → criminal matters
  High Court / Supreme Court     → constitutional, writ, exceptional cases
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger("nyaymalaw.forum_agent")

# ---------------------------------------------------------------------------
# Forum definitions
# ---------------------------------------------------------------------------

# Each entry: {forum_name, jurisdiction, typical_relief, pecuniary_limit, notes}
_FORUM_RULES: list[dict] = [
    {
        "id": "protection_officer",
        "name": "Protection Officer / Magistrate (PWDVA)",
        "categories": ["domestic_violence", "domestic violence", "pwdva", "protection order"],
        "typical_relief": "Protection order, residence order, monetary relief, custody order",
        "pecuniary_limit": None,
        "limitation_years": None,
        "notes": "File complaint with Protection Officer or directly before Magistrate. Interim relief available ex parte.",
        "urgency_applicable": True,
    },
    {
        "id": "family_court",
        "name": "Family Court",
        "categories": ["divorce", "maintenance", "custody", "matrimonial", "alimony", "cruelty", "498a"],
        "typical_relief": "Divorce decree, maintenance, custody, injunction",
        "pecuniary_limit": None,
        "limitation_years": 3,
        "notes": "Family Courts have exclusive jurisdiction in matrimonial matters in cities. District courts elsewhere.",
        "urgency_applicable": False,
    },
    {
        "id": "labour_court",
        "name": "Labour Court / Industrial Tribunal",
        "categories": ["employment", "retrenchment", "termination", "dismissal", "wages", "gratuity", "epf", "esi"],
        "typical_relief": "Reinstatement, back wages, gratuity, PF dues",
        "pecuniary_limit": None,
        "limitation_years": 3,
        "notes": "Raise industrial dispute within 3 years of cause. Government servant? Central Administrative Tribunal instead.",
        "urgency_applicable": False,
    },
    {
        "id": "consumer_forum",
        "name": "District Consumer Disputes Redressal Commission (DCDRC)",
        "categories": ["consumer", "deficiency", "product", "service", "refund", "insurance claim", "builder delay", "ecommerce"],
        "typical_relief": "Refund, compensation, replacement, interest",
        "pecuniary_limit": "Up to ₹50 lakh",
        "limitation_years": 2,
        "notes": "File within 2 years of cause of action. State commission for ₹50L–₹2Cr; NCDRC above ₹2Cr.",
        "urgency_applicable": False,
    },
    {
        "id": "rent_tribunal",
        "name": "Rent Controller / Rent Tribunal",
        "categories": ["rent", "tenancy", "eviction", "landlord", "lease", "deposit refund", "lockout"],
        "typical_relief": "Eviction, deposit refund, rent fixation, restoration of possession",
        "pecuniary_limit": None,
        "limitation_years": 3,
        "notes": "State-specific rent control acts. RERA for real estate builder disputes. Check state jurisdiction.",
        "urgency_applicable": True,
    },
    {
        "id": "rera",
        "name": "Real Estate Regulatory Authority (RERA)",
        "categories": ["rera", "builder", "flat", "possession delay", "real estate", "developer"],
        "typical_relief": "Possession, refund with interest, compensation",
        "pecuniary_limit": None,
        "limitation_years": 3,
        "notes": "File before State RERA. Builder must be RERA-registered. Appellate Tribunal available.",
        "urgency_applicable": False,
    },
    {
        "id": "civil_court",
        "name": "Civil Court (District Court / Sub-Judge)",
        "categories": ["property", "contract", "money recovery", "cheque bounce", "fraud", "defamation", "trespass"],
        "typical_relief": "Injunction, specific performance, damages, declaration",
        "pecuniary_limit": "Jurisdiction varies by state (District Court: usually up to ₹3Cr+)",
        "limitation_years": 3,
        "notes": "General civil jurisdiction. Cheque bounce under NI Act s.138 — file before Magistrate.",
        "urgency_applicable": True,
    },
    {
        "id": "magistrate_cheque",
        "name": "Judicial Magistrate (NI Act s.138 — Cheque Bounce)",
        "categories": ["cheque", "dishonour", "138", "ni act", "bounced cheque"],
        "typical_relief": "Fine up to 2× cheque amount, imprisonment up to 2 years",
        "pecuniary_limit": None,
        "limitation_years": None,  # 30-day window from 15 days after demand notice
        "notes": "Send demand notice within 30 days of dishonour. File complaint within 30 days of expiry of notice period.",
        "urgency_applicable": True,
    },
    {
        "id": "police_criminal",
        "name": "Police Station / Magistrate (Criminal)",
        "categories": ["assault", "threat", "criminal intimidation", "harassment", "cheating", "fraud", "extortion", "stalking"],
        "typical_relief": "FIR, arrest, bail conditions, protection",
        "pecuniary_limit": None,
        "limitation_years": None,
        "notes": "Lodge FIR at local police station. If police refuse, file complaint before Magistrate under CrPC/BNSS.",
        "urgency_applicable": True,
    },
    {
        "id": "drt",
        "name": "Debt Recovery Tribunal (DRT)",
        "categories": ["debt recovery", "loan default", "bank recovery", "npa", "sarfaesi"],
        "typical_relief": "Recovery certificate, attachment, sale of secured assets",
        "pecuniary_limit": "Above ₹20 lakh",
        "limitation_years": 3,
        "notes": "Banks/financial institutions file here. Borrower can contest and file counter-claim.",
        "urgency_applicable": False,
    },
    {
        "id": "high_court_writ",
        "name": "High Court (Writ Jurisdiction)",
        "categories": ["fundamental right", "arbitrary government action", "writ", "mandamus", "certiorari", "habeas corpus"],
        "typical_relief": "Writ orders, stay of government action, directions",
        "pecuniary_limit": None,
        "limitation_years": None,
        "notes": "File writ petition when other remedies are inadequate. Laches can bar writ petitions — act promptly.",
        "urgency_applicable": True,
    },
]

# ---------------------------------------------------------------------------
# Limitation periods by dispute type
# ---------------------------------------------------------------------------

_LIMITATION_DATA = {
    "cheque_bounce": {
        "window_description": "30 days from expiry of 15-day demand notice period",
        "years": None,
        "days": 45,  # 30-day notice + 15-day window
        "notes": "File complaint within 30 days of the end of the 15-day notice period. Missing this is fatal.",
        "critical": True,
    },
    "consumer": {
        "window_description": "2 years from date of cause of action",
        "years": 2,
        "days": None,
        "notes": "Condonation of delay allowed if sufficient cause shown.",
        "critical": False,
    },
    "contract": {
        "window_description": "3 years from date of breach",
        "years": 3,
        "days": None,
        "notes": "Running from date of breach, not date of loss.",
        "critical": False,
    },
    "property": {
        "window_description": "12 years for possession suits; 3 years for injunction",
        "years": 12,
        "days": None,
        "notes": "Adverse possession ripens after 12 years. Declaration suits: 3 years from knowledge.",
        "critical": False,
    },
    "employment": {
        "window_description": "3 years from date of dismissal / last accrual",
        "years": 3,
        "days": None,
        "notes": "Reference to Labour Court must be made while dispute is alive.",
        "critical": False,
    },
    "domestic_violence": {
        "window_description": "No strict limitation — but promptness strengthens the case",
        "years": None,
        "days": None,
        "notes": "Courts may consider delay in assessing credibility, but there is no statutory bar.",
        "critical": False,
    },
    "tort": {
        "window_description": "3 years from date of knowledge of injury",
        "years": 3,
        "days": None,
        "notes": "Discovery rule applies — runs from when injury was known or should have been known.",
        "critical": False,
    },
}


# ---------------------------------------------------------------------------
# Forum identification logic
# ---------------------------------------------------------------------------

def _score_forums(intake_state: dict) -> list[tuple[float, dict]]:
    """Score all forums against the intake state. Returns sorted (score, forum) list."""
    category = (
        intake_state.get("category")
        or intake_state.get("primary_issue_cluster")
        or ""
    )
    category = str(category).lower()
    issue_summary = str(intake_state.get("issue_summary") or "").lower()
    known_fact_parts = []
    for fact in (intake_state.get("known_facts") or []):
        if isinstance(fact, dict):
            known_fact_parts.append(str(fact.get("fact") or ""))
        else:
            known_fact_parts.append(str(fact))
    known_facts = " ".join(known_fact_parts).lower()
    case_file = intake_state.get("case_file") or {}
    case_summary = str(case_file.get("summary") or "").lower()
    relief_hint = " ".join(
        str(x or "")
        for x in [
            (case_file.get("case_theory") or {}).get("immediate_relief"),
            (case_file.get("case_theory") or {}).get("long_term_relief"),
            intake_state.get("assessed_remedy"),
            intake_state.get("client_goal_initial"),
        ]
    ).lower()
    combined = f"{category} {issue_summary} {known_facts} {case_summary} {relief_hint}"

    scored = []
    for forum in _FORUM_RULES:
        score = 0.0
        for keyword in forum["categories"]:
            if keyword in combined:
                score += 1.0
                # Boost for category match (more specific)
                if keyword in category:
                    score += 0.5
        if score > 0:
            scored.append((score, forum))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def identify_forum(intake_state_json: str) -> str:
    """
    MCP tool: identify_forum

    Takes the current intake state and returns recommended forum(s) with
    jurisdiction notes, filing requirements, and urgency assessment.

    Args:
        intake_state_json: JSON string of intake_state (or dict serialized)

    Returns:
        JSON string with forum recommendations
    """
    try:
        if isinstance(intake_state_json, str):
            intake_state = json.loads(intake_state_json)
        else:
            intake_state = intake_state_json or {}

        if not intake_state:
            return json.dumps({"error": "No intake state provided. Complete intake first."})

        scored = _score_forums(intake_state)

        if not scored:
            # Fallback — civil court is the catch-all
            scored = [(0.5, next(f for f in _FORUM_RULES if f["id"] == "civil_court"))]

        # Primary + up to one alternative
        recommendations = []
        for score, forum in scored[:2]:
            recommendations.append({
                "forum": forum["name"],
                "confidence": "high" if score >= 2 else "moderate" if score >= 1 else "low",
                "typical_relief": forum["typical_relief"],
                "pecuniary_limit": forum["pecuniary_limit"],
                "limitation_years": forum["limitation_years"],
                "filing_notes": forum["notes"],
                "urgency_applicable": forum["urgency_applicable"],
            })

        # Urgency assessment
        urgency_signal = (intake_state.get("urgency_signal") or "").lower()
        urgent_forums = [r for r in recommendations if r["urgency_applicable"]]
        urgency_note = ""
        if urgency_signal in ("high", "urgent", "immediate") and urgent_forums:
            urgency_note = (
                f"Urgency detected. {urgent_forums[0]['forum']} allows interim/ex parte relief. "
                "Consider applying for interim orders before the primary hearing."
            )

        result = {
            "primary_forum": recommendations[0] if recommendations else None,
            "alternative_forum": recommendations[1] if len(recommendations) > 1 else None,
            "urgency_note": urgency_note,
            "category_detected": intake_state.get("category") or intake_state.get("primary_issue_cluster", ""),
            "all_forums_considered": len(_FORUM_RULES),
        }
        logger.info(
            "Forum identified: primary=%s confidence=%s",
            result.get("primary_forum", {}).get("forum", "unknown"),
            result.get("primary_forum", {}).get("confidence", "unknown"),
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as exc:
        logger.exception("identify_forum failed: %s", exc)
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Limitation check
# ---------------------------------------------------------------------------

def check_limitation(dispute_type: str, key_dates: list[str]) -> str:
    """
    MCP tool: check_limitation

    Checks whether the limitation period for a dispute is still open given the
    key dates extracted from the intake or document.

    Args:
        dispute_type: One of the keys in _LIMITATION_DATA
                      (e.g. "cheque_bounce", "consumer", "employment")
        key_dates: List of date strings (YYYY-MM-DD or DD/MM/YYYY format)
                   representing the event dates (breach, dishonour, etc.)

    Returns:
        JSON string with limitation status and urgency flag
    """
    try:
        # Normalize dispute type
        dispute_lower = (dispute_type or "").lower().replace(" ", "_")
        limitation = None
        for key in _LIMITATION_DATA:
            if key in dispute_lower or dispute_lower in key:
                limitation = _LIMITATION_DATA[key]
                break

        if limitation is None:
            # Default to 3-year contract limitation as a safe fallback
            limitation = _LIMITATION_DATA["contract"]
            dispute_type = "general (3-year default)"

        # Parse dates
        parsed_dates: list[date] = []
        for d_str in (key_dates or []):
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%B %d, %Y"):
                try:
                    parsed_dates.append(datetime.strptime(d_str.strip(), fmt).date())
                    break
                except ValueError:
                    continue

        today = date.today()
        result = {
            "dispute_type": dispute_type,
            "window_description": limitation["window_description"],
            "notes": limitation["notes"],
            "critical": limitation["critical"],
            "status": "unknown",
            "urgency": "none",
            "message": "",
        }

        if not parsed_dates:
            result["status"] = "cannot_determine"
            result["message"] = (
                "No parseable dates provided. Please provide key event dates "
                "(e.g. date of cheque dishonour, date of termination) to assess limitation."
            )
            return json.dumps(result, ensure_ascii=False, indent=2)

        # Use earliest date as trigger date
        trigger_date = min(parsed_dates)

        if limitation["days"]:
            deadline = trigger_date + timedelta(days=limitation["days"])
            days_left = (deadline - today).days
        elif limitation["years"]:
            deadline = date(trigger_date.year + limitation["years"], trigger_date.month, trigger_date.day)
            days_left = (deadline - today).days
        else:
            result["status"] = "open"
            result["message"] = limitation["notes"]
            return json.dumps(result, ensure_ascii=False, indent=2)

        if days_left < 0:
            result["status"] = "expired"
            result["urgency"] = "critical"
            result["message"] = (
                f"The limitation period appears to have expired {abs(days_left)} day(s) ago "
                f"(trigger date: {trigger_date}, deadline: {deadline}). "
                "The client should urgently consult an advocate about whether condonation of delay is possible."
            )
        elif days_left <= 30:
            result["status"] = "expiring_soon"
            result["urgency"] = "high"
            result["message"] = (
                f"Only {days_left} day(s) remain before the limitation deadline ({deadline}). "
                "File immediately or seek interim protection."
            )
        elif days_left <= 90:
            result["status"] = "open"
            result["urgency"] = "moderate"
            result["message"] = (
                f"Limitation period is still open — {days_left} day(s) remain (deadline: {deadline}). "
                "Advise the client to act within the next few weeks."
            )
        else:
            result["status"] = "open"
            result["urgency"] = "low"
            result["message"] = (
                f"Limitation period is open — approximately {days_left} day(s) remain (deadline: {deadline})."
            )

        result["trigger_date"] = str(trigger_date)
        result["deadline"] = str(deadline)
        result["days_remaining"] = days_left

        logger.info(
            "Limitation check: type=%s status=%s urgency=%s days_left=%d",
            dispute_type, result["status"], result["urgency"], days_left
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as exc:
        logger.exception("check_limitation failed: %s", exc)
        return json.dumps({"error": str(exc)})
