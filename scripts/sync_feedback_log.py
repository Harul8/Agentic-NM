"""
Nyaymalaw Feedback Log v3 — Excel ↔ HTML sync and column layout.

Layout:
  Identity (1): Timestamp (12hr)
  User Input (1): E · Facts Entered
  Model Output (5): G Disputes, H Sections, Additional information requested, I Case Laws, J Legal Opinion
  AI Gate (9), Human Gate (8), Final Feedback (5) — 30 columns total in Excel (incl. Case ID in col B).

- Run:
  - python scripts/sync_feedback_log.py excel-to-html   → read Excel, overwrite HTML table
  - python scripts/sync_feedback_log.py normalize-excel → one-time: rewrite Excel to new layout (drop Date, drop F, add Additional info col)

Usage:
  normalize-excel  : Read Excel, output 30-col layout (Timestamp, Case ID, Facts, Disputes, Sections, Additional info, Case Laws, Opinion, gates).
  excel-to-html   : Read Excel (30 cols), generate full HTML tbody (29 visible cols; Case ID in data-id).
"""

import os
import re
import sys
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    print("Install: pip install pandas openpyxl")
    sys.exit(1)

# Paths relative to project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_PATH = os.path.join(ROOT, "Nyaymalaw_Feedback_Log_v3.xlsx")
HTML_PATH = os.path.join(ROOT, "Nyaymalaw_Feedback_Log_v3.html")


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
    """Rewrite Excel to 30-col layout: Timestamp (12hr), Case ID, Facts, Disputes, Sections, Additional info, Case Laws, Opinion, then gates. Drops Date and separate Follow-up Q column."""
    df = pd.read_excel(EXCEL_PATH, sheet_name=0, header=None)
    ncols = df.shape[1]
    if ncols < 9:
        print("Excel has fewer than 9 columns, skipping.")
        return
    # Old layout (example): 0=Date, 1=Timestamp, 2=Case ID, 3=Facts, 4=Follow-up Q, 5=Disputes, 6=Sections, 7=Case Laws, 8=Opinion, 9+=gates
    new_rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        if i == 0:
            new_rows.append(["Identity", "User Input", "Model Output", "", "", "", "", "", ""] + [""] * 21)
        elif i == 1:
            new_rows.append(["Timestamp", "Case ID", "E · Facts Entered", "G · Disputes", "H · Sections", "Additional information requested", "I · Case Laws", "J · Legal Opinion"] + [""] * 22)
        elif i == 2:
            new_rows.append(["e.g. Mar 1, 2026 10:15 AM", "", "Raw user query", "Disputes from model", "Act § No", "Bullet list of info requested", "Case citations", "Full legal opinion"] + [""] * 22)
        else:
            ts = _format_timestamp_12hr(row[1]) if ncols > 1 else ""
            case_id = row[2] if ncols > 2 else ""
            facts = row[3] if ncols > 3 else ""
            followup = row[4] if ncols > 4 else ""
            disputes = row[5] if ncols > 5 else ""
            sections = row[6] if ncols > 6 else ""
            case_laws = row[7] if ncols > 7 else ""
            opinion = row[8] if ncols > 8 else ""
            rest = list(row[9:9 + 22]) if ncols > 9 else []
            rest = rest + [""] * (22 - len(rest))
            new_rows.append([ts, case_id, facts, disputes, sections, followup, case_laws, opinion] + rest)
    new_df = pd.DataFrame(new_rows)
    new_df = new_df.iloc[:, :30]
    new_df.to_excel(EXCEL_PATH, index=False, header=False)
    print("Normalized Excel: 30 columns (Timestamp, Case ID, Facts, Disputes, Sections, Additional info, Case Laws, Opinion, gates).")


def excel_to_html():
    """Read Excel (30 cols) and regenerate the HTML file table body. Identity = 1 col (Timestamp); Case ID in data-id only; 29 visible tds per row."""
    df = pd.read_excel(EXCEL_PATH, sheet_name=0, header=None)
    if df.shape[1] < 3:
        print("Excel has too few columns.")
        return
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    # Excel: 0=Timestamp, 1=Case ID, 2=Facts, 3=Disputes, 4=Sections, 5=Additional info, 6=Case Laws, 7=Opinion, 8+=gates (30 total)
    NOTES_ROW_LEN = 29
    notes = df.iloc[2] if len(df) > 2 else []
    lines = []
    # Notes row: 29 tds — Excel col 0 → td0, Excel col 2→td1, 3→td2, ... 29→td28 (skip Excel col 1 Case ID)
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
            excel_col = j if j == 0 else (j + 1)
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


if __name__ == "__main__":
    cmd = (sys.argv[1] or "").strip().lower()
    if cmd == "normalize-excel":
        normalize_excel()
    elif cmd == "excel-to-html":
        excel_to_html()
    else:
        print("Usage: python sync_feedback_log.py normalize-excel | excel-to-html")
        sys.exit(1)
