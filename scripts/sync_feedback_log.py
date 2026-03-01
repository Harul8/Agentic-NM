"""
Nyaymalaw Feedback Log v3 — Excel ↔ HTML sync and column layout.

Layout:
  Identity (1): Timestamp (12hr)
  User Input (1): E · Facts Entered
  Model Output (5): G Disputes, H Sections, Additional information requested, I Case Laws, J Legal Opinion
  AI Gate (5), Human Gate (5), Final Feedback (5) — 23 columns total in Excel (incl. Case ID in col B); 22 visible tds in HTML.

- Run:
  - python scripts/sync_feedback_log.py html-to-excel   → replace Excel with content from HTML (same columns/names)
  - python scripts/sync_feedback_log.py excel-to-html   → read Excel, overwrite HTML table
  - python scripts/sync_feedback_log.py normalize-excel → one-time: rewrite Excel to 23-col layout

Usage:
  html-to-excel : Replace Excel with HTML format (headers and column names match HTML exactly). Run this to sync Excel to HTML.
  normalize-excel: Read Excel, output 23-col layout.
  excel-to-html : Read Excel (23 cols), generate full HTML tbody (22 visible tds; Case ID in data-id).
"""

import os
import re
import sys
from datetime import datetime

try:
    import pandas as pd
    from openpyxl import load_workbook, Workbook
    from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
except ImportError:
    print("Install: pip install pandas openpyxl")
    sys.exit(1)

# Paths relative to project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_PATH = os.path.join(ROOT, "Nyaymalaw_Feedback_Log_v3.xlsx")
HTML_PATH = os.path.join(ROOT, "Nyaymalaw_Feedback_Log_v3.html")

# Column headers matching HTML exactly (23 cols: Timestamp, Case ID, then 21 content headers)
EXCEL_HEADERS_23 = [
    "Timestamp", "Case ID",
    "E · Facts Entered", "G · Disputes Identified", "H · Sections Retrieved",
    "Additional information requested", "I · Case Laws Retrieved", "J · Legal Opinion",
    "AI · G Disputes", "AI · H Sections", "AI · Additional information requested", "AI · I Case Laws", "AI · J Legal Opinion",
    "HG · G Disputes ✎", "HG · H Sections ✎", "HG · Additional information requested ✎", "HG · I Case Laws ✎", "HG · J Legal Opinion ✎",
    "Issues in Opinion ✎", "What Should Have Been Different ✎", "Rule Extracted ✎", "Actioned ✎", "Rating ✎",
]


def _format_timestamp_12hr(val):
    """Convert ISO or date-like string to 'Mon DD, YYYY H:MM AM/PM'."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if not s:
        return ""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            raw = s.replace("Z", "").strip()[:19]
            dt = datetime.strptime(raw, fmt.replace("Z", "").strip()[:19])
            part = dt.strftime("%b %d, %Y %I:%M %p")
            if part[4] == "0" and part[5] != " ":  # day
                part = part[:4] + part[5:]
            if " 0" in part and ":" in part:  # hour 01-09
                part = part.replace(" 0", " ", 1)
            return part
        except ValueError:
            continue
    try:
        dt = datetime.strptime(s[:10], "%Y-%m-%d")
        part = dt.strftime("%b %d, %Y")
        if len(part) > 6 and part[4] == "0":
            part = part[:4] + part[5:]
        return part
    except ValueError:
        return s


def normalize_excel():
    """Rewrite Excel to 23-col layout: Timestamp, Case ID, Facts, Disputes, Sections, Addl info, Case Laws, Opinion, AI Gate (5), Human Gate (5), Final Feedback (5)."""
    df = pd.read_excel(EXCEL_PATH, sheet_name=0, header=None)
    ncols = df.shape[1]
    if ncols < 8:
        print("Excel has fewer than 8 columns, skipping.")
        return
    # Target: 23 cols. If source has 30: map AI Gate 9→5 (first 5), Human Gate 8→5 (assessment 5), FF 5→5.
    new_rows = []
    hdr1 = EXCEL_HEADERS_23
    notes_row = ["e.g. Mar 1, 2026 10:15 AM", "", "Raw user query", "Disputes from model", "Act § No", "Bullet list of info requested", "Case citations", "Full legal opinion",
                 "AI's disputes", "AI's sections", "AI's additional info", "AI's case laws", "AI's legal opinion",
                 "Your disputes", "Your sections", "Your additional info", "Your case laws", "Your legal opinion",
                 "Enumerated issues", "Corrective guidance", "New rule derived", "Yes/No/In Progress", "1–5"]
    for i in range(len(df)):
        row = df.iloc[i]
        if i == 0:
            new_rows.append([""] * 23)
        elif i == 1:
            new_rows.append(hdr1)
        elif i == 2:
            new_rows.append(notes_row)
        else:
            if ncols >= 30:
                # Map from 30-col: 0-7 same, AI Gate take 8-12, HG take 17-21, FF take 25-29
                part1 = [row[k] if k < len(row) and not pd.isna(row[k]) else "" for k in range(8)]
                if len(part1) > 0 and part1[0] and "T" in str(part1[0]):
                    part1[0] = _format_timestamp_12hr(part1[0])
                ai5 = [row[k] if k < len(row) and not pd.isna(row[k]) else "" for k in range(8, 13)]
                hg5 = [row[k] if k < len(row) and not pd.isna(row[k]) else "" for k in range(17, 22)]
                ff5 = [row[k] if k < len(row) and not pd.isna(row[k]) else "" for k in range(25, 30)]
                new_rows.append(part1 + ai5 + hg5 + ff5)
            else:
                part1 = [row[k] if k < ncols and not pd.isna(row[k]) else "" for k in range(min(8, ncols))]
                rest = list(row[8:8 + 15]) if ncols > 8 else []
                rest = rest + [""] * (15 - len(rest))
                new_rows.append((part1 + rest)[:23])
    new_df = pd.DataFrame(new_rows)
    new_df = new_df.iloc[:, :23]
    new_df.to_excel(EXCEL_PATH, index=False, header=False)
    print("Normalized Excel: 23 columns (AI Gate 5, Human Gate 5).")


def excel_to_html():
    """Read Excel (23 cols) and regenerate the HTML file table body. 22 visible tds per row (skip Excel col 1 Case ID)."""
    df = pd.read_excel(EXCEL_PATH, sheet_name=0, header=None)
    if df.shape[1] < 3:
        print("Excel has too few columns.")
        return
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    # Excel: 0=Timestamp, 1=Case ID, 2=Facts, ..., 22=Rating (23 total). HTML: 22 tds (skip col 1).
    NOTES_ROW_LEN = 22
    notes = df.iloc[2] if len(df) > 2 else []
    lines = []
    lines.append("          <tr class=\"notes\">")
    for j in range(NOTES_ROW_LEN):
        excel_col = 0 if j == 0 else (j + 1)
        if excel_col >= len(notes):
            lines.append("            <td></td>")
        else:
            val = notes[excel_col]
            lines.append(f"            <td>{_escape(str(val if not pd.isna(val) else ''))}</td>")
    lines.append("          </tr>")
    for r in range(3, len(df)):
        row = df.iloc[r]
        case_id = str(row[1]) if len(row) > 1 and not pd.isna(row[1]) else f"NM-{r:03d}"
        lines.append(f'          <tr class="data" data-id="{_escape(case_id)}">')
        for j in range(NOTES_ROW_LEN):
            excel_col = 0 if j == 0 else (j + 1)
            if excel_col >= len(row):
                lines.append("            <td></td>")
            else:
                val = row[excel_col]
                if pd.isna(val):
                    val = ""
                lines.append(f"            <td>{_escape(str(val))}</td>")
        lines.append("          </tr>")
    tbody_new = "\n".join(lines)
    pattern = r"(<tbody>\s*)(.*?)(\s*</tbody>)"
    if re.search(pattern, html, re.DOTALL):
        html = re.sub(pattern, r"\1\n" + tbody_new + r"\n        \3", html, flags=re.DOTALL)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print("Regenerated HTML from Excel.")


def _escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _td_text(td_html: str) -> str:
    """Extract displayable text from a <td>...</td> fragment (strip tags, normalize whitespace)."""
    import re
    # Remove script/style
    s = re.sub(r"<script[^>]*>.*?</script>", "", td_html, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<style[^>]*>.*?</style>", "", s, flags=re.DOTALL | re.IGNORECASE)
    # For <select>, take the selected option text if any
    sel = re.search(r'<option[^>]*selected[^>]*>([^<]*)</option>', s, re.IGNORECASE)
    if sel:
        return sel.group(1).strip()
    # For contenteditable div, take its text
    div = re.search(r'<div[^>]*contenteditable[^>]*>([^<]*(?:<[^/][^>]*>[^<]*)*)</div>', s, re.IGNORECASE)
    if div:
        return re.sub(r"<[^>]+>", "", div.group(1)).replace("&nbsp;", " ").strip()
    # For input
    inp = re.search(r'<input[^>]*value=["\']([^"\']*)["\']', s, re.IGNORECASE)
    if inp:
        return inp.group(1).strip()
    # Else strip all tags
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    return " ".join(s.split()).strip()


def html_to_excel():
    """Replace Excel with content from HTML. Column names and order match the HTML table exactly (22 cols in HTML; Excel gets 23 with Case ID as col B)."""
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    # Extract second header row (column names) from thead of the main log table (id="logTable")
    table_match = re.search(r'<table[^>]*id="logTable"[^>]*>(.*?)</table>', html, re.DOTALL)
    if not table_match:
        print("Could not find #logTable in HTML.")
        return
    table_html = table_match.group(1)
    header_row2 = re.search(r"<thead>\s*<tr>.*?</tr>\s*<tr>(.*?)</tr>\s*</thead>", table_html, re.DOTALL)
    if not header_row2:
        print("Could not find thead second row in HTML.")
        return
    ths = re.findall(r"<th[^>]*>([^<]+)</th>", header_row2.group(1))
    if len(ths) != 22:
        print(f"Expected 22 column headers, got {len(ths)}.")
        return

    # Excel header row: Timestamp, Case ID, then the rest (same as HTML)
    excel_headers = [ths[0], "Case ID"] + list(ths[1:])

    # Extract notes row and data rows from the same table's tbody
    tbody_match = re.search(r"<tbody>(.*?)</tbody>", table_html, re.DOTALL)
    tbody_html = tbody_match.group(1) if tbody_match else html
    notes_tr = re.search(r'<tr class="notes">(.*?)</tr>', tbody_html, re.DOTALL)
    notes_cells = []
    if notes_tr:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", notes_tr.group(1), re.DOTALL)
        notes_cells = [_td_text(td) for td in tds]
    # Build notes row for Excel: 23 cols (insert blank for Case ID at index 1)
    if len(notes_cells) >= 22:
        notes_row = [notes_cells[0], ""] + list(notes_cells[1:22])
    else:
        notes_row = (notes_cells + [""] * 22)[:22]
        notes_row = [notes_row[0], ""] + notes_row[1:] if len(notes_row) > 1 else [notes_row[0], ""] + [""] * 21
    notes_row = (notes_row + [""] * 23)[:23]

    # Extract data rows from tbody
    data_rows = []
    for tr in re.finditer(r'<tr class="data" data-id="([^"]*)">(.*?)</tr>', tbody_html, re.DOTALL):
        case_id = tr.group(1).strip()
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr.group(2), re.DOTALL)
        cells = [_td_text(td) for td in tds]
        if len(cells) >= 22:
            row = [cells[0], case_id] + list(cells[1:22])
        else:
            row = (cells + [""] * 22)[:22]
            row = [row[0], case_id] + row[1:] if len(row) > 1 else [row[0], case_id] + [""] * 21
        row = (row + [""] * 23)[:23]
        data_rows.append(row)

    # Build workbook: row 1 = headers, row 2 = notes, row 3+ = data
    wb = Workbook()
    ws = wb.active
    ws.title = "Feedback Log"

    body_font = Font(name="Arial", size=9)
    header_font = Font(name="Arial", size=10, bold=True)
    wrap_align = Alignment(wrap_text=True, vertical="top")
    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Row 1: column headers (exact match to HTML)
    for col_idx, val in enumerate(excel_headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=val)
        cell.font = header_font
        cell.alignment = wrap_align
        cell.border = border
    # Row 2: notes
    for col_idx, val in enumerate(notes_row, start=1):
        cell = ws.cell(row=2, column=col_idx, value=val)
        cell.font = body_font
        cell.alignment = wrap_align
        cell.border = border
    # Row 3+: data
    for r_idx, row_vals in enumerate(data_rows, start=3):
        for col_idx, val in enumerate(row_vals, start=1):
            cell = ws.cell(row=r_idx, column=col_idx, value=val or "")
            cell.font = body_font
            cell.alignment = wrap_align
            cell.border = border
            cell.fill = PatternFill("solid", fgColor="FFFFFF")
        ws.row_dimensions[r_idx].height = 60

    wb.save(EXCEL_PATH)
    print(f"Replaced Excel with HTML format: sheet 'Feedback Log', 23 columns, 1 header row, 1 notes row, {len(data_rows)} data rows.")


if __name__ == "__main__":
    cmd = (sys.argv[1] or "").strip().lower()
    if cmd == "html-to-excel":
        html_to_excel()
    elif cmd == "normalize-excel":
        normalize_excel()
    elif cmd == "excel-to-html":
        excel_to_html()
    else:
        print("Usage: python sync_feedback_log.py html-to-excel | normalize-excel | excel-to-html")
        sys.exit(1)
