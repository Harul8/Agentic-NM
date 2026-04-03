"""
Progress Tracker — Tracks detailed search progress for UI display.

Tracks:
- Start time and elapsed time
- Steps grouped by "Internal Search" and "Web Search"
- Document scans with scores and inclusion decisions
- Counts: total searched, passed threshold, included
"""

import copy
import time
from typing import Optional, Dict, List, Any
from datetime import datetime


class ProgressTracker:
    """Tracks search progress with timestamps and detailed step information."""

    def __init__(self):
        self.start_time = time.time()
        self.groups: List[Dict[str, Any]] = []
        self.current_group: Optional[Dict[str, Any]] = None
        # Persisted in snapshots / final response: query expansion, hybrid traces, retrieved doc labels
        self.retrieval_diagnostics: Dict[str, Any] = {}

    def update_retrieval_diagnostics(self, patch: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Merge UI-facing retrieval metadata (replaces keys in patch)."""
        merged = {**(patch or {}), **kwargs}
        for k, v in merged.items():
            self.retrieval_diagnostics[k] = copy.deepcopy(v)

    def start_group(self, group_name: str, description: str = ""):
        """Start a new progress group (e.g., 'Internal Search', 'Web Search')."""
        if self.current_group:
            self.groups.append(self.current_group)
        self.current_group = {
            "name": group_name,
            "description": description,
            "steps": [],
            "stats": {
                "total_searched": 0,
                "passed_threshold": 0,
                "included": 0,
                "ignored": 0,
            },
        }

    def add_step(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a progress step to the current group."""
        if not self.current_group:
            self.start_group("Default", "Progress tracking")
        elapsed = time.time() - self.start_time
        step = {
            "timestamp": round(elapsed, 2),
            "message": message,
            "metadata": metadata or {},
        }
        self.current_group["steps"].append(step)

    def update_last_step(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Update the last step's message (and optionally metadata) in place for live progress (e.g. (n/total))."""
        if not self.current_group or not self.current_group["steps"]:
            self.add_step(message, metadata)
            return
        elapsed = time.time() - self.start_time
        last = self.current_group["steps"][-1]
        last["message"] = message
        last["timestamp"] = round(elapsed, 2)
        if metadata is not None:
            last["metadata"] = {**(last.get("metadata") or {}), **metadata}

    def update_last_step_with_prefix(self, message_prefix: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Update the last step whose message starts with message_prefix (so we don't overwrite a doc-scan step)."""
        if not self.current_group or not self.current_group["steps"]:
            self.add_step(message, metadata)
            return
        elapsed = time.time() - self.start_time
        steps = self.current_group["steps"]
        for i in range(len(steps) - 1, -1, -1):
            if steps[i].get("message", "").startswith(message_prefix):
                steps[i]["message"] = message
                steps[i]["timestamp"] = round(elapsed, 2)
                if metadata is not None:
                    steps[i]["metadata"] = {**(steps[i].get("metadata") or {}), **metadata}
                return
        self.add_step(message, metadata)

    def add_document_scan(
        self,
        doc_name: str,
        score: float,
        included: bool,
        threshold: float = 2.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a document scan with score and inclusion decision."""
        if not self.current_group:
            self.start_group("Default", "Progress tracking")
        
        elapsed = time.time() - self.start_time
        passed_threshold = score >= threshold
        
        # Update stats
        self.current_group["stats"]["total_searched"] += 1
        if passed_threshold:
            self.current_group["stats"]["passed_threshold"] += 1
        if included:
            self.current_group["stats"]["included"] += 1
        else:
            self.current_group["stats"]["ignored"] += 1

        step = {
            "timestamp": round(elapsed, 2),
            "message": f"Document '{doc_name}' scanned, score = {score:.2f}, {'included' if included else 'ignored'}",
            "metadata": {
                "doc_name": doc_name,
                "score": round(score, 2),
                "included": included,
                "passed_threshold": passed_threshold,
                "threshold": threshold,
                **(metadata or {}),
            },
        }
        self.current_group["steps"].append(step)

    def finish_group(self):
        """Finish the current group and add it to groups list."""
        if self.current_group:
            self.groups.append(self.current_group)
            self.current_group = None

    def get_progress(self) -> Dict[str, Any]:
        """Get the complete progress data structure (finishes current group)."""
        # Finish current group if any
        if self.current_group:
            self.finish_group()
        
        elapsed = time.time() - self.start_time
        return {
            "start_time": datetime.utcnow().isoformat() + "Z",
            "elapsed_seconds": round(elapsed, 2),
            "groups": self.groups,
            "retrieval_diagnostics": copy.deepcopy(self.retrieval_diagnostics),
        }

    def get_progress_snapshot(self) -> Dict[str, Any]:
        """Get current progress without modifying state (for streaming/live updates)."""
        elapsed = time.time() - self.start_time
        groups = copy.deepcopy(self.groups)
        if self.current_group:
            groups = groups + [copy.deepcopy(self.current_group)]
        return {
            "start_time": datetime.utcnow().isoformat() + "Z",
            "elapsed_seconds": round(elapsed, 2),
            "groups": groups,
            "retrieval_diagnostics": copy.deepcopy(self.retrieval_diagnostics),
        }

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
