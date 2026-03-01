"""
Populate the Feedback Log from chat history (DB used by the UI left pane).

Reads all chats from the app database (config.DB_PATH). The DB path comes from
config.DATA_ROOT: if you set NYAYMALAW_DATA_ROOT to your Google Drive folder
(e.g. G:\\My Drive\\Nyaymalaw or C:\\Users\\YourName\\Google Drive\\Nyaymalaw),
the script uses chat_history/app.db inside that folder — same as the UI.

For each chat, fills one feedback-log row with:
  Identity: Timestamp (from chat created_at)
  User Input: Facts (user messages concatenated)
  Model Output: Disputes, Sections, Additional information requested, Case Laws, Legal Opinion
  AI Gate, Human Gate, Final Feedback: left empty

Usage:
  Set NYAYMALAW_DATA_ROOT to your data folder (e.g. Google Drive Nyaymalaw folder), then:
  python scripts/fill_feedback_log_from_chats.py

Requires: openpyxl. Excel/HTML paths from config.FEEDBACK_LOG_PATH.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
except ImportError:
    print("Install: pip install openpyxl")
    sys.exit(1)


def _timestamp_12hr(iso_or_date_str: str) -> str:
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


def _escape_html(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _extract_facts(messages_json: str) -> str:
    """Concatenate all user message content into one 'facts' string."""
    try:
        messages = json.loads(messages_json or "[]")
    except Exception:
        return ""
    parts = []
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c.strip())
        elif isinstance(c, dict):
            parts.append((c.get("opinionText") or c.get("text") or c.get("summary") or "").strip())
    return "\n".join(p for p in parts if p)


def _parse_retrieved(retrieved_json: str) -> tuple:
    """Return (sections_list, case_laws_list) from stored retrieved array."""
    try:
        arr = json.loads(retrieved_json or "[]")
    except Exception:
        return [], []
    sections = []
    case_laws = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        act = item.get("act_name") or item.get("act")
        sec = item.get("section_number") or item.get("section") or item.get("section_number_display")
        if act or sec:
            sections.append(f"{act or '?'} § {sec or '?'}")
            continue
        name = item.get("case_name") or item.get("title") or item.get("citation") or item.get("signature")
        if name:
            case_laws.append(str(name))
    return sections, case_laws


def _disputes_from_messages_or_title(messages_json: str, title: str) -> str:
    """Single dispute line from title or first user message (we don't store disputes in chat)."""
    if title and str(title).strip() and str(title).strip() != "Untitled chat":
        return _truncate(str(title).strip(), 200)
    facts = _extract_facts(messages_json)
    first_line = (facts.split("\n")[0] or facts).strip()
    return _truncate(first_line, 200) if first_line else ""


def main() -> None:
    os.chdir(ROOT)
    try:
        from config import DB_PATH, FEEDBACK_LOG_PATH
    except Exception as e:
        print("Could not load config (DATA_ROOT / DB_PATH):", e)
        print("Set NYAYMALAW_DATA_ROOT to your Google Drive path if chat history is there (e.g. G:\\My Drive\\Nyaymalaw).")
        sys.exit(1)

    db_path = Path(DB_PATH)
    if not db_path.exists():
        print("Chat database not found:", db_path)
        print("Set NYAYMALAW_DATA_ROOT to the folder that contains chat_history (e.g. your Google Drive Nyaymalaw folder).")
        sys.exit(1)

    xlsx_path = Path(FEEDBACK_LOG_PATH)
    html_path = xlsx_path.with_suffix(".html")
    if not xlsx_path.exists():
        print("Feedback log Excel not found:", xlsx_path)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, title, messages_json, opinion_text, retrieved_json, created_at FROM chats ORDER BY created_at ASC"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No chats found in the database.")
        return

    print("Found", len(rows), "chat(s) at", db_path)
    print("Feedback log:", xlsx_path)

    new_excel_rows = []
    new_html_rows = []
    for r in rows:
        chat_id = r["id"]
        created_at = r["created_at"] or ""
        title = (r["title"] or "").strip()
        messages_json = r["messages_json"] or "[]"
        opinion_text = (r["opinion_text"] or "").strip()
        retrieved_json = r["retrieved_json"] or "[]"

        facts = _extract_facts(messages_json)
        disputes = _disputes_from_messages_or_title(messages_json, title)
        sections_list, case_laws_list = _parse_retrieved(retrieved_json)
        sections_str = "; ".join(sections_list) if sections_list else ""
        case_laws_str = "; ".join(case_laws_list) if case_laws_list else ""
        additional_info = ""

        ts_12 = _timestamp_12hr(created_at)
        case_id = "NM-CHAT-" + str(chat_id)

        row_vals = [
            ts_12, case_id, _truncate(facts, 32000), disputes, sections_str,
            additional_info, case_laws_str, _truncate(opinion_text, 32000),
        ]
        row_vals += [""] * (23 - len(row_vals))
        new_excel_rows.append(row_vals[:23])

        opinion_short = _truncate(opinion_text, 600)
        new_html_rows.append(_html_row(
            case_id, ts_12, facts, disputes, sections_str, additional_info, case_laws_str, opinion_short
        ))

    wb = load_workbook(str(xlsx_path))
    ws = wb.active
    body_font = Font(name="Arial", size=9)
    wrap_align = Alignment(wrap_text=True, vertical="top")
    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    start_row = ws.max_row + 1
    for i, row_vals in enumerate(new_excel_rows):
        row_num = start_row + i
        for col_idx, value in enumerate(row_vals, start=1):
            cell = ws.cell(row=row_num, column=col_idx, value=value or "")
            cell.font = body_font
            cell.alignment = wrap_align
            cell.border = border
            cell.fill = PatternFill("solid", fgColor="FFFFFF")
        ws.row_dimensions[row_num].height = 60
    wb.save(str(xlsx_path))
    print("Appended", len(new_excel_rows), "row(s) to Excel.")

    if html_path.exists():
        marker = "<!-- ADD NEW ROWS BELOW THIS LINE -->"
        content = html_path.read_text(encoding="utf-8")
        if marker in content:
            inserted = "\n".join(new_html_rows) + "\n          " + marker
            content = content.replace(marker, inserted)
            html_path.write_text(content, encoding="utf-8")
            print("Appended", len(new_html_rows), "row(s) to HTML.")
        else:
            print("HTML insertion marker not found; skipping HTML update.")
    else:
        print("HTML log not found; only Excel was updated.")


def _html_row(case_id, timestamp, facts, disputes, sections, additional_info, case_laws, opinion):
    cid = _escape_html(case_id)
    return (
        '          <!-- DATA ROW from chat: ' + cid + ' -->\n'
        '          <tr class="data" data-id="' + cid + '">\n'
        '            <td>' + _escape_html(timestamp) + '</td>\n'
        '            <td class="wrap">' + _escape_html(_truncate(facts, 400)) + '</td>\n'
        '            <td class="wrap">' + _escape_html(disputes) + '</td>\n'
        '            <td class="wrap">' + _escape_html(sections) + '</td>\n'
        '            <td class="wrap">' + _escape_html(additional_info) + '</td>\n'
        '            <td class="wrap">' + _escape_html(case_laws) + '</td>\n'
        '            <td class="wrap">' + _escape_html(opinion) + '</td>\n'
        '            <td></td><td></td><td></td><td></td><td></td>\n'
        '            <td class="editable wrap"><div contenteditable="true" data-id="' + cid + '" data-field="hg-disputes" data-placeholder="Your disputes…" oninput="markDirty()"></div></td>\n'
        '            <td class="editable wrap"><div contenteditable="true" data-id="' + cid + '" data-field="hg-sections" data-placeholder="Your sections…" oninput="markDirty()"></div></td>\n'
        '            <td class="editable wrap"><div contenteditable="true" data-id="' + cid + '" data-field="hg-followup" data-placeholder="Your additional info…" oninput="markDirty()"></div></td>\n'
        '            <td class="editable wrap"><div contenteditable="true" data-id="' + cid + '" data-field="hg-caselaws" data-placeholder="Your case laws…" oninput="markDirty()"></div></td>\n'
        '            <td class="editable wrap"><div contenteditable="true" data-id="' + cid + '" data-field="hg-opinion" data-placeholder="Your legal opinion…" oninput="markDirty()"></div></td>\n'
        '            <td class="editable wrap"><div contenteditable="true" data-id="' + cid + '" data-field="ff-issues" data-placeholder="Issues…" oninput="markDirty()"></div></td>\n'
        '            <td class="editable wrap"><div contenteditable="true" data-id="' + cid + '" data-field="ff-diff" data-placeholder="What should differ…" oninput="markDirty()"></div></td>\n'
        '            <td class="editable wrap"><div contenteditable="true" data-id="' + cid + '" data-field="ff-rule" data-placeholder="Rule extracted…" oninput="markDirty()"></div></td>\n'
        '            <td class="editable"><select data-id="' + cid + '" data-field="ff-actioned" onchange="markDirty()"><option value="">—</option><option value="Yes">Yes</option><option value="No">No</option><option value="In Progress">In Progress</option></select></td>\n'
        '            <td class="editable"><select data-id="' + cid + '" data-field="ff-rating" onchange="markDirty()"><option value="">—</option><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option></select></td>\n'
        '          </tr>'
    )


if __name__ == "__main__":
    main()
