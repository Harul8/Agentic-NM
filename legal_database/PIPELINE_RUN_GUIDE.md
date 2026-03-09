# Pipeline Re-Run Guide — After P2–P7 Improvements

## What Changed (Summary)

| Priority | Change | Expected Impact |
|----------|--------|----------------|
| P2 | Widened metadata scan window (lines[:80], page-2 fallback for date/judges) | `date_of_judgment`: 2% → ~70%+, `judges`: 19% → ~60%+ |
| P3 | Fixed oversized bare-act chunks (1800-char limit, sub-section splitting) | 12.7% oversized chunks → near 0% |
| P4 | Citation graph: `_clean_party_name()` strips verb tails from cited cases | Fewer garbage citation edges |
| P5 | Strengthened paragraph classifier (5 new regex patterns, 3000-char window) | Better `facts/reasoning/ratio/order` labels in chunks |
| P6 | `run_extraction_missing_only()` added — handles 13 missing PDFs | 13 new JSON files created |
| P7 | `build_act_summary()` now includes preamble, chapters with section ranges, total_sections | Richer act summaries in vector store |

---

## Baseline Quality (Before Re-Run)

**Case laws (36 files):**
- `date_of_judgment`: **2%** ⚠️ (1/36)
- `judges`: **19%** ⚠️ (7/36)
- `outcome`: **19%** ⚠️ (7/36)
- `case_name` / `court`: **80%** ✅
- `year` / `paragraphs`: **100%** ✅

**Bare acts (31 files):**
- All core fields: **100%** ✅
- Sections over 1800 chars: **12.7%** (556/4372) ⚠️
- Largest blob: 192,171 chars (BNSS §531) — will be sub-chunked

**Missing PDFs (13 case laws with no JSON output):**
```
1959 near the Ahari Payin he saw the...pdf
2009_Punjab & Sind Bank versus Baldev Singh.pdf
2019_Prasenjit Bose Petitioner versus...pdf
Counsel for the versus Counsel for the.pdf
SC_2001_Agencies and Anr. versus...pdf
SC_2001_Mandal Revenue Officer...pdf
SC_2014_Ms S.e. Graphites Private Limited...pdf
SC_2015_Chairman and Managing Director Fci...pdf
SC_2016_(now State of Telangana) versus...pdf (×2)
SC_2016_Shayara Bano...pdf
WRIT_PETITION_512_OF_2018_1.pdf
and had allowed the versus the instigation of.pdf
```

---

## How to Run

### Option A — Full re-extraction (all PDFs, recommended)

```python
# In your Python environment on Windows (fitz/PyMuPDF must be installed):
import sys
sys.path.insert(0, r"C:\path\to\Nyaymalaw 4.0\legal_database")

import pipeline
pipeline.run_extraction_only()          # re-extracts all PDFs → json_output/
```

Then rebuild vector store:
```bash
cd "Nyaymalaw 4.0/legal_database"
python build_indexes.py                  # rebuilds FAISS + BM25 + citation graph
```

### Option B — Missing PDFs only (faster, preserves existing JSONs)

```python
import pipeline
pipeline.run_extraction_missing_only()   # only processes the 13 missing PDFs
```

Then rebuild indexes:
```bash
python build_indexes.py
```

### Option C — Re-extract a single PDF

```python
import pipeline
result = pipeline.process_pdf(r"path\to\file.pdf")
print(result.keys())
```

---

## After Re-Run: Verify Quality

Run the audit script to compare against this baseline:

```bash
cd "Nyaymalaw 4.0/legal_database"
python ../audit_review.py
# → writes JSON_PDF_Audit_Report.md
```

**Target metrics after re-run:**
- `date_of_judgment`: > 70%
- `judges`: > 60%
- `outcome`: > 40%
- Bare act sections over 1800 chars: < 2%
- Missing PDFs with JSON: 13 new files

---

## Dependencies Required on Windows

```
pip install pymupdf pdfplumber pytesseract tqdm
# Optional: paddleocr paddlepaddle (for scanned PDFs)
```

Also ensure Tesseract OCR is installed and on PATH.

---

## File Locations

```
legal_database/
├── pipeline.py              ← All P2–P7 changes here (2557 lines)
├── json_output/             ← 67 existing JSONs; re-run adds/updates
├── raw_data/
│   ├── CaseLaws/            ← 49 PDFs
│   └── BareActs/            ← 31 PDFs
└── PIPELINE_RUN_GUIDE.md    ← This file
```
