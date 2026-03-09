# JSON Output Audit Report — Legal Database Pipeline
**Date:** 6 March 2026
**Scope:** `json_output/` vs `pdf_input/` (BareActs + CaseLaws)

---

## Summary of Problems Found

The JSON outputs have **five distinct categories of failures**. Each is explained below with the affected files, root cause, and exact fix needed.

---

## 1. Garbled Text from Corrupted Embedded Text Layers (Most Impactful)

### Affected Files
| JSON File | Root Cause |
|---|---|
| `HC_2024_UNKNOWN.json` | Embedded font encoding broken — page 1 text quality: **19%** |
| `HC_2025_UNKNOWN.json` | Same issue |
| `1_V_DIST_PEDDAPALLI.json` | Same — page 1 quality: **24%** |
| `WIFE_V_I.json` | Same — party names extracted from garbled characters |
| `NO.1_-9-27_81_49LCL...json` | Same — petitioner's address mis-parsed as case name |

### What You See in the JSON
```json
"text": "ryr\":ll\"ltft ,Y?r%,,,#Blvfo%\"nf idSfi {ii\"?f t?;Y:if..."
"case_name": "Wife v I\\"   ← extracted from garbage characters
```

### Root Cause
These PDFs have a text layer embedded but the **font encoding table is broken** — a common artefact when a scanned PDF was processed with a poor OCR tool that embedded garbled text. `pdfplumber` faithfully extracts this garbage.

The current `is_scanned_pdf()` function only checks total character count:
```python
def is_scanned_pdf(pdf_path):
    text = ""
    for page in doc:
        text += page.get_text()
    return len(text.strip()) < 500   # ← passes if total chars > 500, even if all garbage
```
HC_2024_UNKNOWN has 10,457 characters, so it passes the check and never gets OCR'd — but 81% of the "words" on page 1 contain no vowels (`rtJ`, `ltft`, `toBtffE`).

### Fix Required: Add a Text Quality Gate

Replace the `is_scanned_pdf()` check with a **two-stage quality check**:

```python
def _text_quality_score(text: str) -> float:
    """Returns fraction of tokens that look like real English words (have vowels, 3+ chars)."""
    words = re.findall(r'[a-zA-Z]+', text)
    if not words:
        return 0.0
    good = sum(1 for w in words if len(w) >= 3 and re.search(r'[aeiouAEIOU]', w))
    return good / len(words)

def is_scanned_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    page_texts = []
    for page in doc:
        t = page.get_text()
        page_texts.append(t)
        full_text += t

    # Stage 1: no text at all
    if len(full_text.strip()) < 500:
        return True

    # Stage 2: text exists but is garbled (broken font encoding)
    # Check the first content page (page 0 or page 1 if page 0 is blank)
    for t in page_texts[:3]:
        if len(t.strip()) > 100:
            if _text_quality_score(t) < 0.45:   # less than 45% real words → treat as scanned
                return True
            break

    return False
```

**Threshold rationale:** Clean PDFs score 65–70% on page 1. Garbled ones score 19–24%. A threshold of 0.45 cleanly separates them with headroom.

---

## 2. Blank First Page — Metadata on Page 2+ Not Checked

### Affected Files
- `UNK_0000_2009_PUNJAB_SIND_BANK_BALDEV_SINGH.json` — `case_name: null`, `court: null`, `year: null`
- `UNK_0000_1959_NEAR_THE_AHARI_PAYIN...json` — same
- Any file whose page 1 is a blank cover page

### What's in the PDF
Page 1 = blank/title image (0 characters). Page 2 contains:
```
CIVIL APPEAL NO. 1945 OF 2009
Punjab & Sind Bank & Ors. .. Appellants
Versus
Baldev Singh ..Respondent
```

### Root Cause
`extract_case_metadata()` has a hard rule: `header_source = pages[0]`. If page 1 is blank, all metadata returns `None`.

### Fix Required

```python
# In extract_case_metadata(): find first non-blank page (up to page 3)
header_source = ""
for p in pages[:3]:
    if p and len(p.strip()) > 80:
        header_source = p
        break
```

Also add the `.. Appellants` / `..Respondent` party pattern that appears in Supreme Court formatting:

```python
# Add to party extraction: "Name .. Appellants\nVersus\nName ..Respondent" pattern
sc_pattern = re.compile(
    r'^([A-Z][A-Za-z0-9\s\.\,&]+)\s*\.\.\s*Appellants?\s*\n.*?Versus\s*\n([A-Z][A-Za-z0-9\s\.\,&]+)\s*\.\.',
    re.MULTILINE | re.DOTALL | re.I
)
```

---

## 3. Missing Fields in the JSON Schema

### Current Schema (Case Law)
The output only captures: `case_id`, `case_name`, `court`, `year`, `paragraphs[]`

### What Is Clearly Visible in Every PDF but Not Extracted

| Field | Example Value | Where Found |
|---|---|---|
| `judges` | `"B.V. Nagarathna, Augustine George Masih JJ."` | Page 1, after court name |
| `date_of_judgment` | `"10 July 2024"` | Page 1, after judges |
| `case_number` | `"Criminal Appeal No. 2842 of 2024"` | Page 1 |
| `petitioner` | `"Mohd. Abdul Samad"` | Page 1 |
| `respondent` | `"The State of Telangana"` | Page 1 |
| `outcome` | `"allowed"` | Last paragraph |
| `reporter_citations` | `"[2024] 7 S.C.R. 1236 : 2024 INSC 506"` | Page 1 header |
| `bench_type` | `"division"` / `"single"` | Based on number of judges |

### Extraction Patterns to Add to `extract_case_metadata()`

```python
# Reporter citations (AIR, SCC, SCR, INSC)
citations_pat = re.compile(
    r'(?:AIR|SCC|SCR|INSC|SLT)\s*[\[\(]?\d{4}[\]\)]?\s*\d+(?:\s*INSC\s*\d+)?',
    re.I
)

# Judges: line after "PRESENT" or after "BEFORE:"
judges_pat = re.compile(
    r'(?:PRESENT|BEFORE)[:\s]*\n(.+?)\n(?:Between|BETWEEN|Between:)',
    re.DOTALL | re.I
)

# Date: spelled-out dates like "TENTH DAY OF SEPTEMBER TWO THOUSAND AND TWENTY FIVE"
# or numeric dates like "10 July 2024", "July 10, 2024"
date_pat = re.compile(
    r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
    re.I
)

# Case number
case_num_pat = re.compile(
    r'(?:WRIT PETITION|CRIMINAL PETITION|CIVIL APPEAL|CRIMINAL APPEAL|WP|CRL\.P|CMA|SLP)[^:]*(?:NO\.?S?\.?|NOS?\.?)\s*:?\s*([\d\s,ANDand]+OF\s+\d{4})',
    re.I
)

# Outcome from final paragraphs
outcome_pat = re.compile(
    r'\b(?:appeal|petition|application)\s+is\s+(allowed|dismissed|disposed\s+of|partly\s+allowed)',
    re.I
)
```

### Updated Return Structure
```python
return {
    "case_id": case_id,
    "case_name": case_name,
    "normalized_case_name": normalized_case_name,
    "court": court,
    "year": year,
    "case_number": case_number,       # NEW
    "date_of_judgment": date,         # NEW
    "judges": judges_list,            # NEW  e.g. ["B.V. Nagarathna J.", "Augustine George Masih J."]
    "petitioner": party_a,            # NEW (currently buried in case_name)
    "respondent": party_b,            # NEW
    "bench_type": bench_type,         # NEW "single" / "division" / "constitution"
    "reporter_citations": citations,  # NEW  e.g. ["2024 INSC 506", "[2024] 7 SCR 1236"]
    "outcome": outcome,               # NEW "allowed" / "dismissed" / "disposed"
    "paragraphs": paragraphs,
}
```

---

## 4. OCR Path: No Image Preprocessing

### When This Triggers
When `is_scanned_pdf()` returns `True` (after the fix above, this will now also cover the garbled-font PDFs).

### Current Code
```python
def extract_text_ocr(pdf_path):
    for page_num in range(len(doc)):
        pix = page.get_pixmap(dpi=300)
        img = Image.open(BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)   # ← raw, no preprocessing
```

### Problems
- No deskewing (rotated pages give terrible results)
- No denoising (speckles confuse character boundaries)
- No binarisation optimised for text (default PIL is suboptimal)
- No page segmentation mode specified for dense legal text
- `opencv-python` is already installed and unused

### Fix: Add OpenCV Preprocessing Pipeline

```python
import cv2
import numpy as np

def _preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Deskew + denoise + adaptive threshold for cleaner Tesseract input."""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)

    # 1. Deskew using Hough lines
    coords = np.column_stack(np.where(img < 128))
    if len(coords) > 50:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.3:   # only rotate if tilt is meaningful
            (h, w) = img.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

    # 2. Denoise
    img = cv2.fastNlMeansDenoising(img, h=10)

    # 3. Adaptive threshold (better than Otsu for uneven lighting)
    img = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 11)
    return Image.fromarray(img)


def extract_text_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.open(BytesIO(pix.tobytes("png")))
        img = _preprocess_for_ocr(img)           # ← ADD THIS
        text = pytesseract.image_to_string(
            img,
            config='--psm 3 --oem 1 -l eng'     # ← ADD CONFIG
        )
        pages.append(text)
    return pages
```

**Config flags explained:**
- `--psm 3` = fully automatic page segmentation (best for mixed layouts)
- `--oem 1` = LSTM neural net engine (higher accuracy than legacy)
- `-l eng` = English language model

---

## 5. `clean_text()` Does Not Clean OCR Noise

### Current Code
```python
def clean_text(text):
    patterns = [r"Page\s+\d+", r"\s+"]
    text = text.replace("\t", " ")
    return text.strip()   # ← patterns defined but never applied!
```

There is a bug: the `patterns` list is defined but the regex substitutions are **never called**. The patterns are dead code.

### Fix

```python
def clean_text(text: str) -> str:
    text = text.replace("\t", " ")
    # Remove standalone page numbers
    text = re.sub(r"(?m)^\s*Page\s+\d+\s*$", "", text)
    # Remove header/footer repeated lines (e.g. case number repeated on every page)
    text = re.sub(r"(?m)^\s*\[\s*\d{3,5}\s*\]\s*$", "", text)
    # Collapse multiple spaces (but preserve newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
```

---

## 6. Required Libraries Not Installed

The `requirements_ingestion.txt` lists `pymupdf`, `paddleocr`, and `unstructured` — but these are **not installed** in the current Python 3.10 environment.

### Current State
| Library | Required In | Installed? | Impact |
|---|---|---|---|
| `pymupdf` (fitz) | `pipeline.py` line 7 | **NO** | Pipeline crashes on `import fitz` in Python 3.10 |
| `paddleocr` | `requirements_ingestion.txt` | **NO** | Not used in pipeline yet |
| `unstructured` | `requirements_ingestion.txt` | **NO** | Not used in pipeline yet |
| `opencv-python` | — | YES | Available but unused |
| `pytesseract 4.1.1` | `pipeline.py` | YES | Installed but no preprocessing |
| `pdfplumber 0.11.9` | `pipeline.py` | YES | Working |
| `camelot-py` | — | YES | Available but unused (good for table extraction) |

> Note: The `.pyc` files in `__pycache__` are compiled with Python **3.11**, meaning the pipeline was working in a Python 3.11 virtual environment on Windows. The VM is Python 3.10 only.

### Recommended Library Changes

#### Install (High Priority)
```
pymupdf>=1.24.3        # fitz — required for current pipeline to even run
```

#### Install (Medium Priority — Better OCR)
```
paddleocr>=2.7.3       # Significantly better than tesseract for Indian legal docs
paddlepaddle-cpu       # CPU-only PaddlePaddle backend
```
PaddleOCR handles mixed fonts, bleed-through, and low-contrast text much better than Tesseract 4.x for Indian legal documents, especially High Court orders that are often printed and re-scanned.

#### Already Installed — Start Using
```
opencv-python          # Use for image preprocessing (deskew, denoise, threshold)
camelot-py             # Use for table extraction from Bare Acts (schedules, tariff tables)
tabula-py              # Alternative for table extraction
```

#### Optional (Low Priority)
```
surya-ocr              # State-of-the-art layout-aware OCR — excellent for complex page layouts
                       # but heavy (requires torch); use only if PaddleOCR is insufficient
```

---

## Priority Action Plan

| Priority | Change | Effort | Impact |
|---|---|---|---|
| 🔴 Critical | Install `pymupdf` — pipeline can't run without `fitz` | `pip install pymupdf` | Unblocks everything |
| 🔴 Critical | Add text quality gate to `is_scanned_pdf()` | ~15 lines | Fixes HC_2024, HC_2025, 1_V_PEDDAPALLI, WIFE_V_I |
| 🔴 Critical | Fix `extract_case_metadata()` to scan pages[0:3] not just pages[0] | ~5 lines | Fixes Punjab Sind Bank, UNK_0000_1959 |
| 🟠 High | Add OpenCV preprocessing in `extract_text_ocr()` | ~25 lines | Improves all OCR'd PDFs |
| 🟠 High | Fix `clean_text()` bug (patterns never applied) | ~5 lines | Cleaner text for all files |
| 🟠 High | Add missing schema fields (judges, date, case_number, outcome, citations) | ~60 lines | Richer structured data |
| 🟠 High | Add `.. Appellants / Versus / ..Respondent` party pattern | ~10 lines | Fixes Supreme Court party extraction |
| 🟡 Medium | Add Tesseract `--psm 3 --oem 1` config | 1 line | Better accuracy immediately |
| 🟡 Medium | Install and integrate `paddleocr` as fallback for Tesseract | ~40 lines | Significant OCR quality gain |
| 🟢 Low | Use `camelot-py` for table extraction in Bare Acts | ~30 lines | Captures schedules, penalties tables |

---

## Files That Will Be Fixed by Each Change

| Fix | Files Corrected |
|---|---|
| Text quality gate | HC_2024_UNKNOWN, HC_2025_UNKNOWN, 1_V_DIST_PEDDAPALLI, WIFE_V_I, NO.1_-9-27_81... |
| Scan pages[0:3] for metadata | UNK_0000_2009_PUNJAB_SIND, UNK_0000_1959_NEAR_THE_AHARI |
| SC party pattern `..Appellants` | UNK_0000_2009_PUNJAB_SIND, SC_2018_UNKNOWN |
| Schema fields added | All 18 case law files |
| clean_text() bug fix | All files |
| OCR preprocessing | Any file going through OCR path |

---

*Generated by audit of `json_output/` vs `pdf_input/` — 6 March 2026*
