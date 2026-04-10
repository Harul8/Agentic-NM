"""
Workflow evaluation runner for the senior-advocate intake and drafting flow.

Usage:
    python -m eval.legal_workflow_eval
    python -m eval.legal_workflow_eval --case intake_repeat_bundle
    python -m eval.legal_workflow_eval --cases eval/data/legal_workflow_cases_v1.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.intake.casefile import ensure_case_file_structure, make_case_file_template  # noqa: E402

logger = logging.getLogger("eval.legal_workflow_eval")


def _deep_merge(base: dict, patch: dict) -> dict:
    result = json.loads(json.dumps(base))
    for key, value in (patch or {}).items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _walk_path(obj, path: str):
    current = obj
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except Exception:
                return None
        else:
            return None
    return current


def _resolve_path(payload: dict, path: str):
    actual = _walk_path(payload, path)
    if actual is not None:
        return actual
    mode = str(payload.get("mode") or "").strip().lower()
    if mode == "draft" and not path.startswith("result."):
        return _walk_path(payload, f"result.{path}")
    if mode == "intake" and not path.startswith("last_result."):
        return _walk_path(payload, f"last_result.{path}")
    return actual


def _combined_text(payload: dict) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False).lower()
    except Exception:
        return str(payload).lower()


def _build_session() -> dict:
    return {"history": [], "intake_state": None}


def _run_intake_case(case: dict) -> dict:
    from agents.intake.stage1_opening import process_turn

    session = _build_session()
    results = []
    last = {}
    start = time.perf_counter()
    for turn in case.get("turns") or []:
        user_message = str((turn or {}).get("user") or "").strip()
        last = process_turn(session, user_message)
        results.append(last)
        session["intake_state"] = last.get("intake_state") or session.get("intake_state")
        session["history"].append({"role": "user", "content": user_message})
        session["history"].append({"role": "assistant", "content": last.get("reply", "")})
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "mode": "intake",
        "elapsed_ms": round(elapsed_ms, 1),
        "last_result": last,
        "session": session,
        "results": results,
    }


def _run_draft_case(case: dict) -> dict:
    from agents.intake.stage5_draft import build_legal_draft

    payload = case.get("input") or {}
    intake_state = _deep_merge({"case_file": make_case_file_template()}, payload.get("intake_state_patch") or {})
    intake_state["case_file"] = ensure_case_file_structure(intake_state.get("case_file"))
    start = time.perf_counter()
    result = build_legal_draft(
        intake_state=intake_state,
        bare_act_sections=payload.get("bare_act_sections") or [],
        facts_summary=str(payload.get("facts_summary") or ""),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "mode": "draft",
        "elapsed_ms": round(elapsed_ms, 1),
        "result": result,
        "intake_state": intake_state,
    }


def _evaluate_expectations(case: dict, payload: dict) -> tuple[str, list[str]]:
    expect = case.get("expect") or {}
    reasons: list[str] = []
    failures = 0

    if case.get("mode") == "intake":
        last_result = payload.get("last_result") or {}
        intake_state = (last_result.get("intake_state") or {})
        combined = _combined_text(last_result)
    else:
        result = payload.get("result") or {}
        combined = _combined_text(result)
        intake_state = ((result.get("advocate_review") or {}).get("case_file") or {})

    for needle in expect.get("must_contain_any") or []:
        if str(needle).lower() in combined:
            reasons.append(f"contains expected marker: {needle}")
            break
    else:
        if expect.get("must_contain_any"):
            failures += 1
            reasons.append("missing all required markers")

    for needle in expect.get("must_not_contain_any") or []:
        if str(needle).lower() in combined:
            failures += 1
            reasons.append(f"forbidden marker present: {needle}")

    if "advance_to_stage2" in expect:
        actual = bool((payload.get("last_result") or {}).get("advance_to_stage2"))
        if actual != bool(expect.get("advance_to_stage2")):
            failures += 1
            reasons.append(f"advance_to_stage2 expected {expect.get('advance_to_stage2')} got {actual}")

    if "missing_detail_groups_min" in expect:
        actual = len(((payload.get("last_result") or {}).get("intake_state") or {}).get("missing_detail_groups") or [])
        if actual < int(expect.get("missing_detail_groups_min")):
            failures += 1
            reasons.append(f"missing_detail_groups expected at least {expect.get('missing_detail_groups_min')} got {actual}")

    if "missing_detail_groups_max" in expect:
        actual = len(((payload.get("last_result") or {}).get("intake_state") or {}).get("missing_detail_groups") or [])
        if actual > int(expect.get("missing_detail_groups_max")):
            failures += 1
            reasons.append(f"missing_detail_groups expected at most {expect.get('missing_detail_groups_max')} got {actual}")

    if "contradictions_min" in expect:
        contradictions = (((payload.get("last_result") or {}).get("intake_state") or {}).get("case_file") or {}).get("contradictions") or []
        actual = len(contradictions)
        if actual < int(expect.get("contradictions_min")):
            failures += 1
            reasons.append(f"contradictions expected at least {expect.get('contradictions_min')} got {actual}")

    for path, wanted in (expect.get("path_equals") or {}).items():
        actual = _resolve_path(payload, path)
        if actual != wanted:
            failures += 1
            reasons.append(f"path {path} expected {wanted!r} got {actual!r}")

    for path in (expect.get("path_nonempty") or []):
        actual = _resolve_path(payload, path)
        if actual in (None, "", [], {}):
            failures += 1
            reasons.append(f"path {path} expected non-empty value")

    for path, minimum in (expect.get("quality_signal_min") or {}).items():
        actual = _resolve_path(payload, path)
        try:
            actual_num = float(actual or 0)
        except Exception:
            actual_num = 0.0
        if actual_num < float(minimum):
            failures += 1
            reasons.append(f"path {path} expected >= {minimum} got {actual}")

    if "research_packets_min" in expect:
        packets = _walk_path(payload, "result.advocate_review.research_packets") or []
        if len(packets) < int(expect.get("research_packets_min")):
            failures += 1
            reasons.append(f"research_packets expected at least {expect.get('research_packets_min')} got {len(packets)}")

    expected_layers = expect.get("pipeline_layer_status") or {}
    if expected_layers:
        layers = {
            str(item.get("layer")): str(item.get("status"))
            for item in (_walk_path(payload, "result.advocate_review.pipeline_layers") or [])
            if isinstance(item, dict)
        }
        for layer, status in expected_layers.items():
            if layers.get(layer) != status:
                failures += 1
                reasons.append(f"layer {layer} expected status {status!r} got {layers.get(layer)!r}")

    if failures == 0:
        verdict = "PASS"
    elif failures <= 2:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    return verdict, reasons


def _build_dashboard(results: list[dict]) -> dict:
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"PASS": 0, "PARTIAL": 0, "FAIL": 0})
    failure_counter: Counter = Counter()
    signal_totals: Counter = Counter()
    for item in results:
        by_category[item["category"]][item["verdict"]] += 1
        for reason in item.get("reasons") or []:
            if item["verdict"] != "PASS":
                failure_counter[reason] += 1
        payload = item.get("payload") or {}
        review_signals = (
            _walk_path(payload, "result.advocate_review.quality_review_signals")
            or _walk_path(payload, "last_result.intake_state.case_file.audit_metadata.quality_review_signals")
            or {}
        )
        if isinstance(review_signals, dict):
            for key, value in review_signals.items():
                if isinstance(value, bool):
                    signal_totals[key] += int(value)
                elif isinstance(value, (int, float)):
                    signal_totals[key] += value
                elif isinstance(value, list):
                    signal_totals[key] += len(value)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(results),
        "pass_count": sum(1 for x in results if x["verdict"] == "PASS"),
        "partial_count": sum(1 for x in results if x["verdict"] == "PARTIAL"),
        "fail_count": sum(1 for x in results if x["verdict"] == "FAIL"),
        "by_category": by_category,
        "common_failures": failure_counter.most_common(12),
        "junior_or_robotic_signal_totals": dict(signal_totals),
    }


def run_cases(cases_path: str, only_case: str = "") -> dict:
    with open(cases_path, "r", encoding="utf-8") as fh:
        cases = json.load(fh)

    selected = [c for c in cases if not only_case or c.get("id") == only_case]
    results = []
    for case in selected:
        mode = case.get("mode")
        if mode == "intake":
            payload = _run_intake_case(case)
        elif mode == "draft":
            payload = _run_draft_case(case)
        else:
            raise ValueError(f"Unknown mode for case {case.get('id')}: {mode}")
        verdict, reasons = _evaluate_expectations(case, payload)
        results.append({
            "id": case.get("id"),
            "category": case.get("category"),
            "mode": mode,
            "verdict": verdict,
            "reasons": reasons,
            "payload": payload,
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "results": results,
        "dashboard": _build_dashboard(results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Workflow eval runner for the legal intake and draft pipeline")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "eval" / "data" / "legal_workflow_cases_v1.json"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "eval" / "results" / "workflow"))
    parser.add_argument("--case", default="", help="Run only one case by id")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    report = run_cases(args.cases, only_case=args.case)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_path = out_dir / f"workflow_eval_{ts}.json"
    dashboard_path = out_dir / "workflow_dashboard_latest.json"

    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    with open(dashboard_path, "w", encoding="utf-8") as fh:
        json.dump(report["dashboard"], fh, ensure_ascii=False, indent=2)

    print(json.dumps(report["dashboard"], ensure_ascii=False, indent=2))
    print(f"\nSaved detailed report to: {result_path}")
    print(f"Saved dashboard to: {dashboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
