"""
Library / document-viewer routes.

/bareacts/library           GET
/bareacts/list              GET
/bareacts/debug             GET
/bareacts/json              GET
/bareacts/view              GET  (HTML)
/bareacts/download          GET
/caselaws/list              GET
/caselaws/library           GET
/caselaws/json              GET
/caselaws/view              GET  (HTML)
/caselaws/download          GET
/caselaws/most_cited        GET  (HTML)
/library/refresh-cache      POST
"""
import json
import os
from html import escape as _html_escape
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from api.deps import (
    _BARE_ACTS_DIR,
    _CASELAW_DIR,
    _VECTOR_STORE,
    _USE_LEGAL_DATABASE,
    _LEGAL_DB_JSON_OUTPUT,
    _LEGAL_DB_BAREACTS_DIR,
    _LEGAL_DB_CASELAWS_DIR,
    _bareacts_library_cache,
    _caselaws_library_cache,
    logger,
)

# deps module-level caches are dicts — we mutate via module attribute access
import api.deps as _deps

router = APIRouter()


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------

def _iter_bareacts_json_files():
    if not _USE_LEGAL_DATABASE:
        return
    if os.path.isdir(_LEGAL_DB_BAREACTS_DIR):
        for root, _, files in os.walk(_LEGAL_DB_BAREACTS_DIR):
            rel = os.path.relpath(root, _LEGAL_DB_BAREACTS_DIR)
            for name in files:
                if not name.lower().endswith(".json"):
                    continue
                base = os.path.splitext(name)[0]
                if base:
                    yield rel, base
        return
    if os.path.isdir(_LEGAL_DB_JSON_OUTPUT):
        for name in os.listdir(_LEGAL_DB_JSON_OUTPUT):
            if not name.lower().endswith(".json"):
                continue
            base = os.path.splitext(name)[0]
            if base:
                yield "", base


def _list_bare_acts_from_json_output() -> list[str]:
    if not _USE_LEGAL_DATABASE:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for _rel, base in _iter_bareacts_json_files():
        if base in seen:
            continue
        seen.add(base)
        out.append(base)
    return sorted(out)


def _list_bare_acts_from_disk() -> list[str]:
    if not os.path.isdir(_BARE_ACTS_DIR):
        return []
    out = []
    for f in os.listdir(_BARE_ACTS_DIR):
        path = os.path.join(_BARE_ACTS_DIR, f)
        if os.path.isfile(path) and (f.lower().endswith(".pdf") or f.lower().endswith(".txt")):
            out.append(f)
    return sorted(out)


def _list_bare_acts_from_vector_store() -> list[str]:
    if _USE_LEGAL_DATABASE:
        return _list_bare_acts_from_json_output()
    return _list_bare_acts_from_disk()


def _iter_caselaws_json_files():
    if not _USE_LEGAL_DATABASE:
        return
    if os.path.isdir(_LEGAL_DB_CASELAWS_DIR):
        for root, _, files in os.walk(_LEGAL_DB_CASELAWS_DIR):
            for name in files:
                if not name.lower().endswith(".json"):
                    continue
                base = os.path.splitext(name)[0]
                if base:
                    yield os.path.join(root, name), base
        return
    if os.path.isdir(_LEGAL_DB_JSON_OUTPUT):
        for name in os.listdir(_LEGAL_DB_JSON_OUTPUT):
            if not name.lower().endswith(".json"):
                continue
            base = os.path.splitext(name)[0]
            if base:
                yield os.path.join(_LEGAL_DB_JSON_OUTPUT, name), base


def _list_case_laws_from_json_output() -> list[str]:
    if not _USE_LEGAL_DATABASE:
        return []
    bases: set[str] = set()
    for _path, base in _iter_caselaws_json_files():
        bases.add(base)
    return sorted(bases)


def _list_case_laws_from_disk() -> list[str]:
    if not os.path.isdir(_CASELAW_DIR):
        return []
    out = []
    for f in os.listdir(_CASELAW_DIR):
        path = os.path.join(_CASELAW_DIR, f)
        if os.path.isfile(path) and (f.lower().endswith(".pdf") or f.lower().endswith(".txt")):
            out.append(f)
    return sorted(out)


def _list_case_laws_from_disk_or_json() -> list[str]:
    if _USE_LEGAL_DATABASE:
        return _list_case_laws_from_json_output()
    return _list_case_laws_from_disk()


def _load_act_names_from_summaries() -> dict[str, str]:
    vs_path = os.path.join(_VECTOR_STORE, "act_summaries_v2_chunks.json")
    if not os.path.exists(vs_path):
        return {}
    try:
        with open(vs_path, encoding="utf-8") as f:
            chunks = json.load(f)
        mapping: dict[str, str] = {}
        for chunk in chunks.values():
            sf = chunk.get("source_file", "").replace("\\", "/")
            name = chunk.get("act_name", "").strip()
            if sf and name:
                mapping[sf] = name
        return mapping
    except Exception:
        return {}


def _group_bare_acts_by_jurisdiction() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    if not _USE_LEGAL_DATABASE or not os.path.isdir(_BARE_ACTS_DIR):
        return groups
    act_name_map = _load_act_names_from_summaries()
    raw_data_dir = os.path.dirname(_BARE_ACTS_DIR)
    for entry in os.scandir(_BARE_ACTS_DIR):
        if not entry.is_dir():
            continue
        jurisdiction = entry.name
        acts: list[dict] = []
        for root, _, files in os.walk(entry.path):
            for fname in files:
                if not fname.lower().endswith(".txt"):
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, raw_data_dir).replace("\\", "/")
                act_name = act_name_map.get(rel, "") or os.path.splitext(fname)[0]
                acts.append({"name": act_name, "file": rel})
        if acts:
            acts.sort(key=lambda a: a["name"])
            groups[jurisdiction] = acts
    return groups


def _group_case_laws_by_court() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    if not _USE_LEGAL_DATABASE or not os.path.isdir(_CASELAW_DIR):
        return groups
    vs_path = os.path.join(_VECTOR_STORE, "case_summaries_v2_chunks.json")
    case_title_map: dict[str, str] = {}
    if os.path.exists(vs_path):
        try:
            with open(vs_path, encoding="utf-8") as f:
                summaries = json.load(f)
            for chunk in summaries.values():
                sf = chunk.get("source_file", "").replace("\\", "/")
                title = (chunk.get("title") or chunk.get("case_name") or "").strip()
                if sf and title:
                    case_title_map[sf] = title
        except Exception:
            pass
    raw_data_dir = os.path.dirname(_CASELAW_DIR)
    for entry in os.scandir(_CASELAW_DIR):
        if not entry.is_dir():
            continue
        court_label = entry.name
        cases: list[dict] = []
        for root, _, files in os.walk(entry.path):
            for fname in files:
                if not fname.lower().endswith(".txt"):
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, raw_data_dir).replace("\\", "/")
                title = case_title_map.get(rel, "") or os.path.splitext(fname)[0]
                cases.append({"name": title, "file": rel})
        if cases:
            cases.sort(key=lambda c: c["name"])
            groups[court_label] = cases
    return groups


# ---------------------------------------------------------------------------
# JSON lookup helpers
# ---------------------------------------------------------------------------

def _get_json_from_legal_db(base: str, is_statute: bool) -> dict | None:
    if not _USE_LEGAL_DATABASE:
        return None
    base = (base or "").strip()
    if not base:
        return None
    if base.lower().endswith(".json"):
        base = base[:-5]
    if is_statute:
        search_root = _LEGAL_DB_BAREACTS_DIR if os.path.isdir(_LEGAL_DB_BAREACTS_DIR) else _LEGAL_DB_JSON_OUTPUT
    else:
        search_root = _LEGAL_DB_CASELAWS_DIR if os.path.isdir(_LEGAL_DB_CASELAWS_DIR) else _LEGAL_DB_JSON_OUTPUT
    if not os.path.isdir(search_root):
        return None
    candidate = os.path.join(search_root, base + ".json")
    path = candidate if os.path.isfile(candidate) else ""
    if not path:
        target = base.lower()
        for root, _, files in os.walk(search_root):
            for name in files:
                if not name.lower().endswith(".json"):
                    continue
                if os.path.splitext(name)[0].lower() == target:
                    path = os.path.join(root, name)
                    break
            if path:
                break
        if not path:
            return None
    try:
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        if is_statute and not (data.get("act_id") or data.get("sections")):
            return None
        if not is_statute and data.get("act_id"):
            return None
        return data
    except Exception:
        return None


def _resolve_bare_act_path(base: str):
    base = (base or "").strip()
    if not base:
        return None
    lookup = base if base.lower().endswith((".pdf", ".txt")) else base + ".pdf"
    path = os.path.join(_BARE_ACTS_DIR, lookup)
    if os.path.isfile(path):
        return os.path.abspath(path)
    if os.path.isdir(_BARE_ACTS_DIR):
        for f in os.listdir(_BARE_ACTS_DIR):
            if f.lower() == lookup.lower():
                return os.path.abspath(os.path.join(_BARE_ACTS_DIR, f))
    return None


def _resolve_case_law_path(base: str):
    base = (base or "").strip()
    if not base:
        return None
    lookup = base if base.lower().endswith((".pdf", ".txt")) else base + ".pdf"
    path = os.path.join(_CASELAW_DIR, lookup)
    if os.path.isfile(path):
        return os.path.abspath(path)
    if os.path.isdir(_CASELAW_DIR):
        for f in os.listdir(_CASELAW_DIR):
            if f.lower() == lookup.lower():
                return os.path.abspath(os.path.join(_CASELAW_DIR, f))
    return None


# ---------------------------------------------------------------------------
# HTML formatters
# ---------------------------------------------------------------------------

def _serve_raw_txt_as_html(file_rel: str, title: str) -> HTMLResponse:
    raw_data_dir = os.path.dirname(_BARE_ACTS_DIR)
    raw_path = os.path.abspath(os.path.join(raw_data_dir, file_rel.replace("/", os.sep)))
    if not raw_path.startswith(os.path.abspath(raw_data_dir)):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not os.path.isfile(raw_path):
        raise HTTPException(status_code=404, detail="File not found")
    content = open(raw_path, encoding="utf-8", errors="replace").read()
    is_case = file_rel.startswith("CaseLaws/") or "/CaseLaws/" in file_rel
    return HTMLResponse(content=_format_legal_html(content, title, is_case=is_case))


def _format_legal_html(text: str, title: str, is_case: bool = False) -> str:
    import html as _h
    import re as _re

    body_parts: list[str] = []
    if is_case:
        body_parts.append(f'<pre class="case-raw">{_h.escape(text)}</pre>')
    else:
        lines = [l.rstrip() for l in text.splitlines()]
        SEC_RE = _re.compile(r'^(\d+[A-Z]?)\.\s*$')
        SUB_RE = _re.compile(r'^(\([0-9a-zA-Z]+\))\s*$')
        HEAD_RE = _re.compile(r'^(PART|CHAPTER|SCHEDULE)\b', _re.I)
        BRACKET_LINE_RE = _re.compile(r'^\[.*\]\s*$')

        def next_nonempty(idx):
            j = idx + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            return j

        i = 0
        while i < len(lines):
            raw = lines[i]
            line = raw.strip()
            if not line:
                i += 1
                continue
            m = SEC_RE.match(line)
            if m:
                num = m.group(1)
                j = next_nonempty(i)
                heading = _h.escape(lines[j].strip()) if j < len(lines) else ""
                body_parts.append(
                    f'<div class="section"><span class="sec-num">{_h.escape(num)}.</span> '
                    f'<span class="sec-title">{heading}</span></div>'
                )
                i = j + 1 if j < len(lines) else i + 1
                continue
            m = SUB_RE.match(line)
            if m:
                marker = m.group(1)
                j = next_nonempty(i)
                sub_text = _h.escape(lines[j].strip()) if j < len(lines) else ""
                body_parts.append(
                    f'<p class="subsection"><span class="sub-marker">{_h.escape(marker)}</span> {sub_text}</p>'
                )
                i = j + 1 if j < len(lines) else i + 1
                continue
            if HEAD_RE.match(line) or (line.isupper() and 4 < len(line) < 80 and not BRACKET_LINE_RE.match(line)):
                body_parts.append(f'<h2 class="chapter">{_h.escape(line)}</h2>')
                i += 1
                continue
            if BRACKET_LINE_RE.match(line):
                body_parts.append(f'<p class="editorial">{_h.escape(line)}</p>')
                i += 1
                continue
            body_parts.append(f'<p>{_h.escape(line)}</p>')
            i += 1

    body_html = "\n".join(body_parts)
    t = _h.escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{t}</title>
<style>
  body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 860px; margin: 36px auto; padding: 0 28px 60px; line-height: 1.8; color: #1a1a1a; font-size: 15px; }}
  h1 {{ font-size: 1.25em; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 24px; }}
  h2.chapter {{ font-size: 1em; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 28px; margin-bottom: 4px; color: #444; }}
  div.section {{ margin-top: 20px; margin-bottom: 4px; }}
  .sec-num {{ font-weight: bold; font-size: 1em; color: #111; }}
  .sec-title {{ font-weight: bold; }}
  p {{ margin: 4px 0; }}
  p.subsection {{ margin: 4px 0 4px 1.6em; }}
  .sub-marker {{ font-weight: 600; min-width: 2em; display: inline-block; }}
  p.editorial {{ color: #666; font-style: italic; font-size: 0.9em; margin: 2px 0 2px 1.6em; }}
  pre.case-raw {{ white-space: pre-wrap; word-break: break-word; font-family: Georgia, 'Times New Roman', serif; font-size: 15px; line-height: 1.8; margin: 0; }}
  #dl-bar {{ position: sticky; top: 0; background: #f8f8f8; border-bottom: 1px solid #ddd; padding: 8px 0; margin-bottom: 20px; display: flex; align-items: center; gap: 12px; z-index: 100; }}
  #dl-bar h1 {{ margin: 0; border: none; padding: 0; font-size: 1em; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  #pdf-btn {{ background: #1a56db; color: white; border: none; border-radius: 5px; padding: 6px 16px; font-size: 0.88em; cursor: pointer; white-space: nowrap; flex-shrink: 0; }}
  #pdf-btn:hover {{ background: #1648c0; }}
  @media print {{ #dl-bar {{ display: none; }} body {{ margin: 0; padding: 16px; }} }}
</style>
</head>
<body>
<div id="dl-bar">
  <h1>{t}</h1>
  <button id="pdf-btn" onclick="window.print()">⬇ Download PDF</button>
</div>
{body_html}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/bareacts/library")
def bareacts_library():
    if not _USE_LEGAL_DATABASE:
        acts = _list_bare_acts_from_vector_store()
        return {"jurisdictions": [{"name": "All", "acts": acts}]}
    if _deps._bareacts_library_cache is None:
        _deps._bareacts_library_cache = _group_bare_acts_by_jurisdiction()
    jurisdictions = [
        {"name": name, "acts": acts}
        for name, acts in sorted(_deps._bareacts_library_cache.items(), key=lambda kv: kv[0].lower())
    ]
    return {"jurisdictions": jurisdictions}


@router.get("/bareacts/list")
def bareacts_list():
    acts = _list_bare_acts_from_vector_store()
    return {"acts": acts}


@router.get("/bareacts/debug")
def bareacts_debug():
    from api.deps import _BASE_DIR
    return {
        "base_dir": _BASE_DIR,
        "bare_acts_dir": _BARE_ACTS_DIR,
        "dir_exists": os.path.isdir(_BARE_ACTS_DIR),
        "files": sorted(os.listdir(_BARE_ACTS_DIR)) if os.path.isdir(_BARE_ACTS_DIR) else [],
    }


@router.get("/bareacts/json")
def bareacts_json(name: str = Query(..., description="Base name of the act JSON")):
    if not _USE_LEGAL_DATABASE:
        raise HTTPException(status_code=400, detail="JSON endpoint requires NYAYMALAW_DATA_SOURCE=legal_database")
    data = _get_json_from_legal_db(name, is_statute=True)
    if not data:
        raise HTTPException(status_code=404, detail="Act not found")
    return data


@router.get("/caselaws/json")
def caselaws_json(name: str = Query(..., description="Base name of the case JSON")):
    if not _USE_LEGAL_DATABASE:
        raise HTTPException(status_code=400, detail="JSON endpoint requires NYAYMALAW_DATA_SOURCE=legal_database")
    data = _get_json_from_legal_db(name, is_statute=False)
    if not data:
        raise HTTPException(status_code=404, detail="Case not found")
    return data


@router.get("/bareacts/view", response_class=HTMLResponse)
def bareacts_view(
    file: str = Query(None, description="Path relative to raw_data/ e.g. BareActs/Telangana/1948_0.txt"),
    name: str = Query(None, description="Legacy: act name"),
):
    if not _USE_LEGAL_DATABASE:
        raise HTTPException(status_code=400, detail="Requires NYAYMALAW_DATA_SOURCE=legal_database")
    if file:
        return _serve_raw_txt_as_html(file, name or file.split("/")[-1])
    data = _get_json_from_legal_db(name or "", is_statute=True)
    if not data:
        raise HTTPException(status_code=404, detail="Act not found")

    title = (data.get("act_name") or name or "").strip() or "Bare Act"
    year = data.get("year") or (data.get("act_summary") or {}).get("year")
    sections = data.get("sections") or []
    sections_by_num: dict[str, list[dict]] = {}
    for sec in sections:
        num = str(sec.get("section_number") or "").strip()
        if not num:
            continue
        sections_by_num.setdefault(num, []).append(sec)

    def _sec_sort_key(num_str: str) -> tuple[int, str]:
        import re as _re
        m = _re.match(r"(\d+)", num_str)
        if m:
            return (int(m.group(1)), num_str[m.end():].strip())
        return (10**9, num_str)

    def _sub_sort_key(sec: dict) -> tuple[int, int, str]:
        import re as _re
        sub = (sec.get("sub_section") or "").strip()
        if not sub:
            return (0, 0, "")
        m = _re.match(r"\(?(\d+)\)?\s*([A-Za-z]*)", sub)
        if m:
            return (1, int(m.group(1)) if m.group(1) else 0, m.group(2) or "")
        return (1, 0, sub)

    def _render_text_block(text: str) -> str:
        import re as _re
        t = text or ""
        lines = t.splitlines()
        joined_lines: list[str] = []
        i = 0
        sec_pat = _re.compile(r"^\s*\d+[A-Za-z]*\.\s*$")
        sub_pat = _re.compile(r"^\s*\([A-Za-z0-9ivxIVX]+\)\s*$")
        while i < len(lines):
            line = lines[i]
            if (sec_pat.match(line) or sub_pat.match(line)) and i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.strip():
                    joined_lines.append(line.strip() + " " + next_line.lstrip())
                    i += 2
                    continue
            joined_lines.append(line)
            i += 1
        normalised = "\n".join(joined_lines)
        paras = [s for s in normalised.split("\n\n") if s.strip()]
        if not paras:
            return ""
        return "\n".join(f"<p>{_html_escape(para).replace(chr(10), '<br />')}</p>" for para in paras)

    ordered_nums = sorted(sections_by_num.keys(), key=_sec_sort_key)
    sections_html: list[str] = []
    for num in ordered_nums:
        group = sorted(sections_by_num[num], key=_sub_sort_key)
        first = group[0]
        sec_title = (
            (first.get("section_title") or first.get("title") or "").strip()
            or (first.get("text") or "").split("\n", 1)[0].strip()
        )
        header = f"Section {num}" + (f". {sec_title}" if sec_title else "")
        body_text = "\n\n".join((sec.get("text") or "").strip() for sec in group if (sec.get("text") or "").strip())
        body_html = _render_text_block(body_text)
        sections_html.append(
            f"<section class='bare-section'><h2>{_html_escape(header)}</h2>{body_html}</section>"
        )

    sections_joined = "\n".join(sections_html) if sections_html else "<p>No sections found in this act.</p>"
    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>{_html_escape(title)}</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 16px; background: #f7f7f8; color: #222; }}
      h1 {{ font-size: 1.4rem; margin-bottom: 12px; }}
      h2 {{ font-size: 1.05rem; margin-top: 18px; margin-bottom: 6px; }}
      .meta {{ margin-bottom: 16px; font-size: 0.9rem; color: #555; }}
      .bare-section {{ padding-bottom: 12px; border-bottom: 1px solid #e0e0e0; margin-bottom: 12px; }}
      .bare-section:last-of-type {{ border-bottom: none; }}
      p {{ font-size: 0.92rem; line-height: 1.5; }}
    </style>
  </head>
  <body>
    <h1>{_html_escape(title)}</h1>
    <div class="meta">Source: legal_database/json_output (bare act JSON){"&nbsp;•&nbsp;Year: " + _html_escape(str(year)) if year else ""}</div>
    {sections_joined}
  </body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/caselaws/view", response_class=HTMLResponse)
def caselaws_view(
    file: str = Query(None, description="Path relative to raw_data/"),
    name: str = Query(None, description="Legacy: case name"),
):
    if not _USE_LEGAL_DATABASE:
        raise HTTPException(status_code=400, detail="Requires NYAYMALAW_DATA_SOURCE=legal_database")
    if file:
        return _serve_raw_txt_as_html(file, name or file.split("/")[-1])
    data = _get_json_from_legal_db(name or "", is_statute=False)
    if not data:
        raise HTTPException(status_code=404, detail="Case not found")

    title = (data.get("case_name") or name or "").strip() or "Case law"
    court = (data.get("court") or "").strip()
    date = (data.get("date_of_judgment") or "").strip()
    judges = data.get("judges") or []
    bench_type = (data.get("bench_type") or "").strip()
    citations = data.get("equivalent_citations") or data.get("reporter_citations") or []
    paragraphs = data.get("paragraphs") or []
    paragraphs_sorted = sorted(paragraphs, key=lambda p: int(p.get("paragraph_id") or 0))

    def _render_para_text(text: str) -> str:
        parts = [t for t in (text or "").split("\n\n") if t.strip()]
        if not parts:
            return ""
        return "\n".join(f"<p>{_html_escape(part).replace(chr(10), '<br />')}</p>" for part in parts)

    paras_html: list[str] = []
    for p in paragraphs_sorted:
        pid = p.get("paragraph_id")
        body_html = _render_para_text(p.get("text") or "")
        if not body_html:
            continue
        label = f"¶ {pid}" if pid is not None else "¶"
        paras_html.append(
            f"<article class='case-paragraph'>"
            f"<div class='para-label'>{_html_escape(label)}</div>"
            f"<div class='para-body'>{body_html}</div>"
            f"</article>"
        )

    paras_joined = "\n".join(paras_html) if paras_html else "<p>No paragraphs available for this judgment.</p>"
    judges_str = ", ".join(judges) if judges else ""
    citations_str = "; ".join(citations) if citations else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>{_html_escape(title)}</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 16px; background: #f7f7f8; color: #222; }}
      h1 {{ font-size: 1.4rem; margin-bottom: 12px; }}
      .meta {{ margin-bottom: 16px; font-size: 0.9rem; color: #555; }}
      .meta b {{ font-weight: 600; }}
      .case-paragraph {{ display: grid; grid-template-columns: auto 1fr; gap: 8px 12px; padding: 8px 0; border-bottom: 1px solid #e0e0e0; }}
      .case-paragraph:last-of-type {{ border-bottom: none; }}
      .para-label {{ font-size: 0.8rem; color: #777; min-width: 48px; }}
      .para-body p {{ margin: 0 0 6px 0; font-size: 0.95rem; line-height: 1.5; }}
      .para-body p:last-child {{ margin-bottom: 0; }}
    </style>
  </head>
  <body>
    <h1>{_html_escape(title)}</h1>
    <div class="meta">
      {f"<b>Court:</b> {_html_escape(court)}<br />" if court else ""}
      {f"<b>Date:</b> {_html_escape(date)}<br />" if date else ""}
      {f"<b>Bench:</b> {_html_escape(judges_str)}" if judges_str else ""}
      {f" &nbsp;&nbsp;({_html_escape(bench_type)})" if bench_type else ""}
      {f"<br /><b>Citations:</b> {_html_escape(citations_str)}" if citations_str else ""}
    </div>
    {paras_joined}
  </body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/bareacts/download")
def bareacts_download(
    name: str = Query(...),
    inline: bool = Query(False),
):
    base = os.path.basename(name).strip() if name else ""
    if not base or ".." in base or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = _resolve_bare_act_path(base)
    if not path:
        raise HTTPException(status_code=404, detail="File not found")
    filename = os.path.basename(path)
    media_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
    try:
        response = FileResponse(path, filename=filename, media_type=media_type)
        response.headers["Content-Disposition"] = f'{"inline" if inline else "attachment"}; filename="{filename}"'
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not serve file: {e!s}")


@router.get("/caselaws/list")
def caselaws_list():
    files = _list_case_laws_from_disk_or_json()
    return {"cases": files}


@router.get("/caselaws/library")
def caselaws_library():
    if not _USE_LEGAL_DATABASE:
        files = _list_case_laws_from_disk_or_json()
        return {"courts": [{"name": "TG HC", "cases": files}]}
    if _deps._caselaws_library_cache is None:
        _deps._caselaws_library_cache = _group_case_laws_by_court()
    courts = [
        {"name": name, "cases": cases}
        for name, cases in sorted(_deps._caselaws_library_cache.items(), key=lambda kv: kv[0].lower())
    ]
    return {"courts": courts}


@router.get("/caselaws/download")
def caselaws_download(
    name: str = Query(...),
    inline: bool = Query(False),
):
    base = os.path.basename(name).strip() if name else ""
    if not base or ".." in base or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = _resolve_case_law_path(base)
    if not path:
        raise HTTPException(status_code=404, detail="File not found")
    filename = os.path.basename(path)
    media_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
    try:
        response = FileResponse(path, filename=filename, media_type=media_type)
        response.headers["Content-Disposition"] = f'{"inline" if inline else "attachment"}; filename="{filename}"'
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/caselaws/most_cited", response_class=HTMLResponse)
def caselaws_most_cited():
    from config import LEGAL_DATABASE_DIR as _LEGAL_DATABASE_DIR
    path = os.path.join(_LEGAL_DATABASE_DIR, "top_200_cited_cases.jsonl")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Most-cited cases file not found")
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not rows:
        raise HTTPException(status_code=404, detail="No most-cited cases available")

    columns = list(rows[0].keys())

    def esc(val) -> str:
        return _html_escape(str(val)) if val is not None else ""

    header_cells_parts = []
    for col in columns:
        col_label = esc(col)
        if col == "case_name":
            header_cells_parts.append(f"<th class='col-case-name'>{col_label}</th>")
        elif col in ("disputes", "key_arguments", "reasoning"):
            header_cells_parts.append(f"<th class='col-long'>{col_label}</th>")
        else:
            header_cells_parts.append(f"<th class='col-meta'>{col_label}</th>")
    header_cells = "".join(header_cells_parts)

    body_rows: list[str] = []
    for idx, r in enumerate(rows):
        cells: list[str] = []
        for col in columns:
            val = r.get(col)
            if col == "case_name":
                base = str(r.get("json_base") or r.get("case_id") or "").strip()
                if base:
                    href = f"/caselaws/view?name={base}"
                    cells.append(f"<td class='col-case-name'><a href='{esc(href)}' target='_blank' rel='noopener noreferrer'>{esc(val)}</a></td>")
                else:
                    cells.append(f"<td class='col-case-name'>{esc(val)}</td>")
            elif col in ("disputes", "key_arguments", "reasoning"):
                text = esc(val) if val is not None else ""
                cells.append(
                    f"<td class='col-long'><div class='cell-text truncated' data-row='{idx}' data-col='{esc(col)}'>{text}</div>"
                    f"<button type='button' class='expand-btn' data-row='{idx}' aria-label='Expand row'>⤢</button></td>"
                )
            else:
                cells.append(f"<td class='col-meta'>{esc(val)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"<table><thead><tr>{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"

    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" /><title>Most cited case laws</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 16px; background: #f7f7f8; color: #222; }}
      h1 {{ font-size: 1.4rem; margin-bottom: 12px; }}
      table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; table-layout: fixed; }}
      th, td {{ border: 1px solid #ddd; padding: 4px 6px; vertical-align: top; }}
      th {{ background: #f0f0f3; position: sticky; top: 0; z-index: 1; text-align: left; }}
      tr:nth-child(even) td {{ background: #fafafa; }}
      .col-meta {{ width: 90px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
      .col-case-name {{ width: 220px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
      .col-long {{ width: 28%; }}
      .cell-text {{ display: block; line-height: 1.35; }}
      .cell-text.truncated {{ max-height: 4.05em; overflow: hidden; }}
      .expand-btn {{ margin-top: 4px; padding: 0 4px; font-size: 0.7rem; border: 1px solid #ccc; border-radius: 3px; background: #fff; cursor: pointer; }}
      .expand-btn:hover {{ background: #f0f0f3; }}
      a {{ color: #0b5fff; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
    </style>
  </head>
  <body>
    <h1>Most cited case laws (Top 200)</h1>
    {table_html}
    <script>
      (function () {{
        let expandedRow = null;
        function setRowState(rowId, expand) {{
          document.querySelectorAll(".cell-text[data-row='" + rowId + "']").forEach(function(el) {{
            expand ? el.classList.remove("truncated") : el.classList.add("truncated");
          }});
          document.querySelectorAll(".expand-btn[data-row='" + rowId + "']").forEach(function(btn) {{
            btn.textContent = expand ? "⤡" : "⤢";
          }});
        }}
        document.addEventListener("click", function (e) {{
          var btn = e.target.closest(".expand-btn");
          if (!btn) return;
          var rowId = btn.getAttribute("data-row");
          if (!rowId) return;
          if (expandedRow !== null && expandedRow !== rowId) setRowState(expandedRow, false);
          if (expandedRow === rowId) {{ setRowState(rowId, false); expandedRow = null; }}
          else {{ setRowState(rowId, true); expandedRow = rowId; }}
        }});
      }})();
    </script>
  </body>
</html>"""
    return HTMLResponse(content=html)


@router.post("/library/refresh-cache")
def library_refresh_cache():
    """Force-refresh the in-memory library caches."""
    _deps._caselaws_library_cache = None
    _deps._bareacts_library_cache = None
    return {"status": "ok", "message": "Library caches cleared — will reload on next request"}
