"""
feedback_logger.py
──────────────────
Thread-safe logger that appends one row per Nyaymalaw interaction to:
  1. Nyaymalaw_Feedback_Log_v3.xlsx  (Excel workbook)
  2. Nyaymalaw_Feedback_Log_v3.html  (browser-viewable table, same name/dir)

Auto-filled columns (Identity + User Input + Model Output):
  A  Timestamp (12hr)        B  Case ID (NM-YYYYMMDD-NNN)
  C  Facts Entered           D  Disputes Identified
  E  Sections Retrieved      F  Additional information requested
  G  Case Laws Retrieved     H  Legal Opinion

Columns K–Z (Ground Truth, AI Gate, Human Gate, Final Feedback)
are left blank for the AI Gate / Human Gate to fill later.

File-lock handling
──────────────────
If either file is open in Excel / a browser and locked, the logger retries
up to MAX_RETRIES times, sleeping RETRY_DELAY seconds between attempts, and
prints a prominent console message asking the user to close the file.
"""

from __future__ import annotations

import html as _html_mod
import logging
import os
import re
import threading
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Global lock – prevents concurrent threads corrupting the files ────────────
_LOCK = threading.Lock()

# ── Auto-increment counter per calendar day ───────────────────────────────────
_COUNTER: dict[str, int] = {}

# ── Retry config for locked files ─────────────────────────────────────────────
MAX_RETRIES  = 8      # total attempts
RETRY_DELAY  = 4      # seconds between retries


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_case_id(today_str: str) -> str:
    _COUNTER[today_str] = _COUNTER.get(today_str, 0) + 1
    return f"NM-{today_str}-{_COUNTER[today_str]:03d}"


def _highest_existing_seq(ws, today_str: str) -> int:
    prefix = f"NM-{today_str}-"
    max_seq = 0
    for row in ws.iter_rows(min_row=4, min_col=2, max_col=2, values_only=True):
        val = row[0]
        if isinstance(val, str) and val.startswith(prefix):
            try:
                seq = int(val[len(prefix):])
                max_seq = max(max_seq, seq)
            except ValueError:
                pass
    return max_seq


def _timestamp_12hr(iso_or_date_str: str) -> str:
    """Convert ISO or YYYY-MM-DD string to 'Mon DD, YYYY H:MM AM/PM'."""
    if not iso_or_date_str or not isinstance(iso_or_date_str, str):
        return ""
    s = iso_or_date_str.strip()
    if not s:
        return ""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            raw = s.replace("Z", "").strip()[:19]
            dt = datetime.strptime(raw, fmt.replace("Z", "").strip()[:19])
            part = dt.strftime("%b %d, %Y %I:%M %p")
            if len(part) > 6 and part[4] == "0" and part[5] != " ":
                part = part[:4] + part[5:]
            if " 0" in part and ":" in part:
                part = part.replace(" 0", " ", 1)
            return part
        except ValueError:
            continue
    try:
        dt = datetime.strptime(s[:10], "%Y-%m-%d")
        return dt.strftime("%b %d, %Y")
    except ValueError:
        return s


def _truncate(text: str, max_chars: int = 32000) -> str:
    if not text:
        return ""
    return text[:max_chars] + "…" if len(text) > max_chars else text


def _esc(text: str) -> str:
    """HTML-escape a string for safe insertion into the HTML log."""
    return _html_mod.escape(str(text or ""))


def _write_with_retry(write_fn, file_path: Path, label: str) -> bool:
    """
    Call write_fn(); retry on PermissionError (file locked by another app).
    Logs a prominent warning on first lock, asking the user to close the file.
    Returns True on success, False if all retries exhausted.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            write_fn()
            return True
        except PermissionError:
            if attempt == 1:
                logger.warning(
                    "\n"
                    "╔══════════════════════════════════════════════════════╗\n"
                    "║  ⚠  FEEDBACK LOG FILE IS LOCKED                      ║\n"
                    "║                                                        ║\n"
                    "║  %s\n"
                    "║  is open in another application (Excel / browser).    ║\n"
                    "║                                                        ║\n"
                    "║  ➜  Please CLOSE that file.                           ║\n"
                    "║     The system will retry automatically every %ds.   ║\n"
                    "╚══════════════════════════════════════════════════════╝",
                    str(file_path)[:52].ljust(52),
                    RETRY_DELAY,
                )
            else:
                logger.info(
                    "Waiting for %s to be released… (attempt %d/%d)",
                    label, attempt, MAX_RETRIES,
                )
            time.sleep(RETRY_DELAY)
        except Exception as exc:
            logger.error("Unexpected error writing %s: %s", label, exc)
            return False

    logger.error(
        "❌ Could not write to %s after %d attempts.\n"
        "   Please close the file and trigger a manual log or restart the server.",
        label, MAX_RETRIES,
    )
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Excel writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_excel(path: Path, row_values: list, case_id: str) -> bool:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    SECTION_LIGHT = {
        range(1, 5):   "FFFFFF",
        range(5, 7):   "FFFFFF",
        range(7, 11):  "FFFFFF",
        range(11, 15): "FFFFFF",
        range(15, 19): "FFFFFF",
        range(19, 22): "FFFFFF",
        range(22, 27): "FFFFFF",
    }

    def _do_write():
        wb = load_workbook(str(path))
        if "Feedback Log" not in wb.sheetnames:
            raise ValueError("Workbook has no 'Feedback Log' sheet.")
        ws = wb["Feedback Log"]

        next_row = max(ws.max_row + 1, 4)

        body_font  = Font(name="Arial", size=9)
        wrap_align = Alignment(wrap_text=True, vertical="top")
        thin       = Side(style="thin", color="CCCCCC")
        border     = Border(left=thin, right=thin, top=thin, bottom=thin)

        for col_idx, value in enumerate(row_values, start=1):
            cell           = ws.cell(row=next_row, column=col_idx, value=value)
            cell.font      = body_font
            cell.alignment = wrap_align
            cell.border    = border
            cell.fill      = PatternFill("solid", fgColor="FFFFFF")

        ws.row_dimensions[next_row].height = 60
        wb.save(str(path))
        logger.info("Excel log updated: %s → row %d", case_id, next_row)

    return _write_with_retry(_do_write, path, f"Excel ({path.name})")


# ─────────────────────────────────────────────────────────────────────────────
# HTML writer
# ─────────────────────────────────────────────────────────────────────────────

_HTML_INSERTION_MARKER = "<!-- ADD NEW ROWS BELOW THIS LINE -->"


def _build_html_row(
    case_id: str,
    timestamp: str,
    facts: str,
    disputes: str,
    sections: str,
    additional_info: str,
    case_laws: str,
    opinion: str,
) -> str:
    """Return an HTML <tr> string for one new log entry (22-column structure).

    Column layout:
      Identity (1) · User Input (1) · Model Output (5)
      AI Gate (5) · Human Gate (5) · Final Feedback (5)
    """
    cid = _esc(case_id)
    opinion_short = _esc(opinion[:600] + "…" if len(opinion) > 600 else opinion)

    return f"""
          <!-- DATA ROW: {case_id} -->
          <tr class="data" data-id="{cid}">
            <!-- Identity (1) -->
            <td>{_esc(timestamp)}</td>
            <!-- User Input (1) -->
            <td class="wrap">{_esc(facts[:400])}</td>
            <!-- Model Output (5) -->
            <td class="wrap">{_esc(disputes)}</td>
            <td class="wrap">{_esc(sections)}</td>
            <td class="wrap">{_esc(additional_info)}</td>
            <td class="wrap">{_esc(case_laws)}</td>
            <td class="wrap">{opinion_short}</td>
            <!-- AI Gate (5): same as Model Output -->
            <td></td><td></td><td></td><td></td><td></td>
            <!-- Human Gate (5): same as Model Output, editable -->
            <td class="editable wrap">
              <div contenteditable="true" data-id="{cid}" data-field="hg-disputes" data-placeholder="Your disputes…" oninput="markDirty()"></div>
            </td>
            <td class="editable wrap">
              <div contenteditable="true" data-id="{cid}" data-field="hg-sections" data-placeholder="Your sections…" oninput="markDirty()"></div>
            </td>
            <td class="editable wrap">
              <div contenteditable="true" data-id="{cid}" data-field="hg-followup" data-placeholder="Your additional info…" oninput="markDirty()"></div>
            </td>
            <td class="editable wrap">
              <div contenteditable="true" data-id="{cid}" data-field="hg-caselaws" data-placeholder="Your case laws…" oninput="markDirty()"></div>
            </td>
            <td class="editable wrap">
              <div contenteditable="true" data-id="{cid}" data-field="hg-opinion" data-placeholder="Your legal opinion…" oninput="markDirty()"></div>
            </td>
            <!-- Final Feedback (5): all editable -->
            <td class="editable wrap">
              <div contenteditable="true" data-id="{cid}" data-field="ff-issues" data-placeholder="Issues…" oninput="markDirty()"></div>
            </td>
            <td class="editable wrap">
              <div contenteditable="true" data-id="{cid}" data-field="ff-diff" data-placeholder="What should differ…" oninput="markDirty()"></div>
            </td>
            <td class="editable wrap">
              <div contenteditable="true" data-id="{cid}" data-field="ff-rule" data-placeholder="Rule extracted…" oninput="markDirty()"></div>
            </td>
            <td class="editable">
              <select data-id="{cid}" data-field="ff-actioned" onchange="markDirty()">
                <option value="">—</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                <option value="In Progress">In Progress</option>
              </select>
            </td>
            <td class="editable">
              <select data-id="{cid}" data-field="ff-rating" onchange="markDirty()">
                <option value="">—</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
              </select>
            </td>
          </tr>"""


def _write_html(html_path: Path, new_row_html: str) -> bool:
    """Insert new_row_html before the insertion marker in the HTML file."""

    def _do_write():
        content = html_path.read_text(encoding="utf-8")
        if _HTML_INSERTION_MARKER not in content:
            raise ValueError(
                f"Insertion marker not found in {html_path.name}. "
                "Has the HTML file been manually edited?"
            )
        updated = content.replace(
            _HTML_INSERTION_MARKER,
            new_row_html + "\n          " + _HTML_INSERTION_MARKER,
        )
        html_path.write_text(updated, encoding="utf-8")
        logger.info("HTML log updated: %s", html_path.name)

    return _write_with_retry(_do_write, html_path, f"HTML ({html_path.name})")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def log_interaction(
    *,
    facts: str,
    followup_question: str,
    disputes: list[str],
    sections: list[str],
    case_laws: list[str],
    legal_opinion: str,
    session_ref: str = "",
    feedback_log_path: Optional[str] = None,
) -> str:
    """
    Append one interaction row to both the Excel workbook and the HTML log.

    Returns the Case ID assigned (e.g. 'NM-20260301-004').
    Both files are updated; each retries independently on PermissionError
    (file locked). Raises ValueError / FileNotFoundError on config errors.
    """
    # ── Resolve paths ─────────────────────────────────────────────────────────
    if feedback_log_path is None:
        feedback_log_path = os.environ.get("FEEDBACK_LOG_PATH", "")
    if not feedback_log_path:
        try:
            from config import FEEDBACK_LOG_PATH as _cfg_path
            feedback_log_path = _cfg_path
        except Exception:
            pass
    if not feedback_log_path:
        raise ValueError(
            "feedback_log_path not set. Pass it explicitly, set FEEDBACK_LOG_PATH "
            "env var, or add FEEDBACK_LOG_PATH to config.py"
        )

    xlsx_path = Path(feedback_log_path)
    html_path = xlsx_path.with_suffix(".html")

    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel feedback log not found: {xlsx_path}")

    # ── Serialize all file access ─────────────────────────────────────────────
    with _LOCK:
        try:
            from openpyxl import load_workbook as _lw
        except ImportError:
            raise ImportError("openpyxl is required: pip install openpyxl")

        # Determine Case ID by scanning existing Excel rows
        wb_temp = _lw(str(xlsx_path))
        ws_temp = wb_temp["Feedback Log"]
        today_str   = date.today().strftime("%Y%m%d")
        existing_max = _highest_existing_seq(ws_temp, today_str)
        _COUNTER[today_str] = max(_COUNTER.get(today_str, 0), existing_max)
        case_id  = _next_case_id(today_str)
        now_iso  = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # ── Build row data ────────────────────────────────────────────────────
        disputes_str  = "; ".join(disputes)  if disputes  else ""
        sections_str  = "; ".join(sections)  if sections  else ""
        case_laws_str = "; ".join(case_laws) if case_laws else ""

        # 23-column layout: 0=Timestamp, 1=Case ID, 2=Facts, 3=Disputes, 4=Sections, 5=Addl info, 6=Case Laws, 7=Opinion,
        # 8–12=AI Gate (5), 13–17=Human Gate (5), 18–22=Final Feedback (5)
        row_values = [""] * 23
        row_values[0]  = _timestamp_12hr(now_iso)
        row_values[1]  = case_id
        row_values[2]  = _truncate(facts)
        row_values[3]  = disputes_str
        row_values[4]  = sections_str
        row_values[5]  = _truncate(followup_question)
        row_values[6]  = case_laws_str
        row_values[7]  = _truncate(legal_opinion)
        # cols 8–22 remain blank

        # ── 1. Write Excel (with retry) ───────────────────────────────────────
        _write_excel(xlsx_path, row_values, case_id)

        # ── 2. Write HTML (with retry) ────────────────────────────────────────
        if html_path.exists():
            ts_12 = _timestamp_12hr(now_iso)
            new_row = _build_html_row(
                case_id          = case_id,
                timestamp        = ts_12,
                facts            = _truncate(facts, 2000),
                disputes         = disputes_str,
                sections         = sections_str,
                additional_info  = _truncate(followup_question, 500),
                case_laws        = case_laws_str,
                opinion          = _truncate(legal_opinion, 800),
            )
            _write_html(html_path, new_row)
        else:
            logger.warning(
                "HTML log not found at %s — only Excel updated. "
                "Run rebuild_feedback_log.py to regenerate.",
                html_path,
            )

    logger.info("Feedback logged: %s (Excel + HTML)", case_id)
    return case_id
