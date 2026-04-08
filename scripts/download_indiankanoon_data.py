"""
Download judgment text from Indian Kanoon URLs in links_Supreme_Court.json.
Saves under legal_database/raw_data/caselaws/<year>/<month>/ (e.g. 1950/JAN/).
General download: only saves a doc when BOTH Cites and Cited by are >= limit.
Stop-month rule: only Cited by is considered; if at least MIN_LOW_CITED_BY_TO_STOP_MONTH
links in a batch have Cited by < limit, the scraper stops that month.
Usage: python download_indiankanoon_data.py [path_to_links_json]
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request
from http.client import IncompleteRead
from pathlib import Path

from bs4 import BeautifulSoup

# Paths relative to project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LINKS_JSON = Path(__file__).parent / "links_Supreme_Court.json"
OUTPUT_BASE = PROJECT_ROOT / "legal_database" / "raw_data" / "caselaws"
CATEGORY = "Supreme Court of India"
SLEEP_BETWEEN_REQUESTS = 2
SLEEP_EVERY_N_DOCS = 100
SLEEP_EVERY_N_SECS = 2
# Browser-like User-Agent and retries for 403 / 5xx server errors
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
RETRY_ON_ERROR_WAIT = 10  # seconds before retry on 403, 502, 500, etc.
MAX_ERROR_RETRIES = 2

# For general download: both Cites and Cited by must be >= this to save the file.
CITES_AND_CITED_BY_MIN = 0
# For stop-month/year rule: if this many links in a batch have Cited by below threshold, stop (default threshold = CITES_AND_CITED_BY_MIN).
MIN_LOW_CITED_BY_TO_STOP_MONTH = 0

# Regex to extract Cites and Cited by (match on normalized text; use \s+ for flexible whitespace)
CITES_RE = re.compile(r"Cites\s*:?\s*(\d+(?:,\d+)*)", re.I)
CITED_BY_RE = re.compile(r"Cited\s+by\s*:?\s*(\d+(?:,\d+)*)", re.I)


def _parse_cites_citedby(text: str) -> tuple[int, int, bool, bool] | None:
    """
    Parse Cites and Cited by numbers from text (use normalized page text, not raw HTML).
    Returns (cites, cited_by, found_cites, found_cited_by).
    Returns None if neither pattern found (unknown format -> don't download).
    """
    cites = 0
    cited_by = 0
    found_cites = False
    found_cited_by = False
    m = CITES_RE.search(text)
    if m:
        cites = int(m.group(1).replace(",", ""))
        found_cites = True
    m = CITED_BY_RE.search(text)
    if m:
        cited_by = int(m.group(1).replace(",", ""))
        found_cited_by = True
    if not found_cites and not found_cited_by:
        return None  # Can't parse either -> don't download
    return cites, cited_by, found_cites, found_cited_by


def _strip_indiankanoon_banner(text: str) -> str:
    """
    Remove the Indian Kanoon PRISM/banner block that appears at the top of
    some documents, e.g.:
      - Tools for analyzing structure and cite text of judgments
      - Unlock Advanced Research with PRISM AI ...
      - Document Options / Get in PDF / Print it!
    """
    if not text:
        return text
    lower = text.lower()
    # Try to locate the banner start using robust markers
    start = lower.find("tools for analyzing structure and cite text of judgments")
    if start == -1:
        start = lower.find("unlock advanced research with")
    if start == -1:
        return text
    # End marker: "Print it!" preferred; fallback to "Document Options"
    end = lower.find("print it!", start)
    if end != -1:
        end += len("print it!")
    else:
        end = lower.find("document options", start)
        if end == -1:
            return text
        # extend to end of that line
        newline = text.find("\n", end)
        end = newline if newline != -1 else end
    # Splice out the banner block
    prefix = text[:start].rstrip()
    suffix = text[end:].lstrip()
    if prefix and suffix:
        return prefix + "\n\n" + suffix
    return prefix or suffix


def get_judgment_text(url: str, cited_by_stop_threshold: int | None = None) -> tuple[str | None, str | None]:
    """
    Fetch URL and return (text, skip_reason).
    General download: save only when BOTH Cites and Cited by >= CITES_AND_CITED_BY_MIN.
    cited_by_stop_threshold: if provided, count as "low_cited_by" only when Cited by < this
    (for stop-month/year rule). Default None = use CITES_AND_CITED_BY_MIN.
    - (text, None): both above limit, text from div.judgments to save.
    - (None, "low_cited_by"): Cited by < threshold (counts for stop-month/year rule).
    - (None, "low_cites"): Cites < limit or missing (does not count for stop-month).
    - (None, None): fetch error, no judgments div, or could not parse both Cites and Cited by.
    Retries on 403, 502, 500.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    html = None
    for attempt in range(MAX_ERROR_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read()
            break
        except urllib.error.HTTPError as e:
            retryable = e.code in (403, 500, 502, 503) and attempt < MAX_ERROR_RETRIES
            if retryable:
                time.sleep(RETRY_ON_ERROR_WAIT)
                continue
            print(f"  Error fetching {url}: HTTP {e.code}")
            return (None, None)
        except (OSError, IncompleteRead) as e:
            if attempt < MAX_ERROR_RETRIES:
                time.sleep(RETRY_ON_ERROR_WAIT)
                continue
            print(f"  Error fetching {url}: {e}")
            return (None, None)
    if html is None:
        return (None, None)
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    try:
        # Parse Cites/Cited by from normalized text so we don't miss when HTML splits "Cited by 2" across tags
        page_text = soup.get_text(separator=" ", strip=True)
        parsed = _parse_cites_citedby(page_text)
        if parsed is None:
            return (None, None)  # Can't parse both -> don't download
        cites, cited_by, found_cites, found_cited_by = parsed
        # Need both Cites and Cited by to be present and above limit to download.
        if not found_cites or not found_cited_by:
            return (None, "low_cites")  # Missing one or both -> don't download; don't count for stop_month
        threshold = cited_by_stop_threshold if cited_by_stop_threshold is not None else CITES_AND_CITED_BY_MIN
        if cited_by < CITES_AND_CITED_BY_MIN:
            # Only count as "low_cited_by" when below the stop threshold (e.g. 5 for Telangana, 10 default)
            if cited_by < threshold:
                return (None, "low_cited_by")  # Counts for stop-month/year rule
            return (None, "low_cites")  # Don't download; don't count for stop
        if cites < CITES_AND_CITED_BY_MIN:
            return (None, "low_cites")  # Don't download; don't count for stop_month

        data_html = soup.find("div", attrs={"class": "judgments"})
        if data_html is None:
            return (None, None)
        raw_text = data_html.get_text(separator="\n", strip=True)
        clean_text = _strip_indiankanoon_banner(raw_text)
        return (clean_text, None)
    except Exception:
        return (None, None)  # Any parsing/other error -> skip this doc, don't crash


def get_doc_text_any(url: str) -> str | None:
    """
    Fetch any Indian Kanoon doc URL and return main content as text (no cites/cited-by filter).
    Used for bare acts (andhra-act, etc.) where we want to download everything.
    Tries div.judgments, then div with class containing 'doc' or 'content', then body.
    Returns None on fetch error or if no content found.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    html = None
    for attempt in range(MAX_ERROR_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (403, 500, 502, 503) and attempt < MAX_ERROR_RETRIES:
                time.sleep(RETRY_ON_ERROR_WAIT)
                continue
            print(f"  Error fetching {url}: HTTP {e.code}")
            return None
        except (OSError, IncompleteRead) as e:
            if attempt < MAX_ERROR_RETRIES:
                time.sleep(RETRY_ON_ERROR_WAIT)
                continue
            print(f"  Error fetching {url}: {e}")
            return None
    if html is None:
        return None
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    # Prefer judgments div (acts/judgments often use same layout), then doc/content divs, then body
    for selector in ["div.judgments", "div.doc_content", "div[class*='doc']", "div[class*='content']"]:
        el = soup.select_one(selector)
        if el:
            raw = el.get_text(separator="\n", strip=True)
            text = _strip_indiankanoon_banner(raw)
            if text and len(text) > 100:
                return text
    body = soup.find("body")
    if body:
        raw = body.get_text(separator="\n", strip=True)
        text = _strip_indiankanoon_banner(raw)
        if text and len(text) > 100:
            return text
    return None


def get_doc_text_with_citedby_min(url: str, min_cited_by: int) -> str | None:
    """
    Fetch an Indian Kanoon doc URL and return main content as text,
    but only when Cited by >= min_cited_by.

    Used for union-act bare acts to avoid downloading sparsely cited acts.
    Falls back to 0 when Cited by cannot be parsed or is missing.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    html = None
    for attempt in range(MAX_ERROR_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (403, 500, 502, 503) and attempt < MAX_ERROR_RETRIES:
                time.sleep(RETRY_ON_ERROR_WAIT)
                continue
            print(f"  Error fetching {url}: HTTP {e.code}")
            return None
        except (OSError, IncompleteRead) as e:
            if attempt < MAX_ERROR_RETRIES:
                time.sleep(RETRY_ON_ERROR_WAIT)
                continue
            print(f"  Error fetching {url}: {e}")
            return None
    if html is None:
        return None
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Decide based on Cited by count from normalized page text
    cited_by = 0
    try:
        page_text = soup.get_text(separator=" ", strip=True)
        parsed = _parse_cites_citedby(page_text)
        if parsed is not None:
            _, cited_by_val, _, found_cited_by = parsed
            if found_cited_by:
                cited_by = cited_by_val
    except Exception:
        pass
    if cited_by < min_cited_by:
        return None

    # Reuse the main-content extraction logic from get_doc_text_any
    for selector in ["div.judgments", "div.doc_content", "div[class*='doc']", "div[class*='content']"]:
        el = soup.select_one(selector)
        if el:
            raw = el.get_text(separator="\n", strip=True)
            text = _strip_indiankanoon_banner(raw)
            if text and len(text) > 100:
                return text
    body = soup.find("body")
    if body:
        raw = body.get_text(separator="\n", strip=True)
        text = _strip_indiankanoon_banner(raw)
        if text and len(text) > 100:
            return text
    return None


def download_month_urls(
    urls: list[str],
    year: str,
    month: str,
    start_index: int = 0,
    *,
    output_base: Path | None = None,
    cited_by_stop_threshold: int | None = None,
    min_low_to_stop: int | None = None,
) -> tuple[int, bool]:
    """
    Download judgment text for a list of URLs and save under (output_base or OUTPUT_BASE)/year/month/.
    When month is "ALL", save under (output_base or OUTPUT_BASE)/year/ with filenames year_0.txt, year_1.txt, ...
    start_index: starting file index (e.g. 10 for second batch in same month).
    output_base: if set, use this instead of OUTPUT_BASE (e.g. for search script: CaseLaws/Supreme Court).
    cited_by_stop_threshold: count link as "low cited by" when Cited by < this (default: CITES_AND_CITED_BY_MIN).
    min_low_to_stop: stop month/year when this many links in batch are "low cited by" (default: MIN_LOW_CITED_BY_TO_STOP_MONTH).
    Returns (saved_count, stop_month, low_cited_by_count). stop_month is True when this batch has
    at least min_low_to_stop links with Cited by < cited_by_stop_threshold. low_cited_by_count
    is how many in this batch had Cited by below threshold (caller may use for cumulative stop).
    """
    if not urls:
        return (0, False, 0)
    base = output_base if output_base is not None else OUTPUT_BASE
    flat_year = month == "ALL"
    try:
        if flat_year:
            out_dir = base / year
        else:
            out_dir = base / year / month
        out_dir.mkdir(parents=True, exist_ok=True)
        stop_count = min_low_to_stop if min_low_to_stop is not None else MIN_LOW_CITED_BY_TO_STOP_MONTH
        saved = 0
        low_cited_by_count = 0
        for i, url in enumerate(urls):
            try:
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                text, skip_reason = get_judgment_text(url, cited_by_stop_threshold=cited_by_stop_threshold)
                if skip_reason == "low_cited_by":
                    low_cited_by_count += 1
                    continue
                if text is None:
                    continue
                if flat_year:
                    out_path = out_dir / f"{year}_{start_index + i}.txt"
                else:
                    out_path = out_dir / f"{year}_{month}_{start_index + i}.txt"
                out_path.write_text(text, encoding="utf-8")
                saved += 1
                if saved % SLEEP_EVERY_N_DOCS == 0:
                    time.sleep(SLEEP_EVERY_N_SECS)
            except Exception as e:
                print(f"  Skip doc {url}: {e}")
                continue
        stop_month = low_cited_by_count >= stop_count
        return (saved, stop_month, low_cited_by_count)
    except Exception as e:
        print(f"  Error in batch ({year}/{month}): {e}")
        return (0, False, 0)


def iter_year_month_links(links_json: dict):
    """
    Yield (year, month, url).
    Supports: court -> year -> month -> [urls] (year/month grouped)
             court -> year -> [urls] (year-only), or court -> [urls] (flat).
    """
    data = links_json.get(CATEGORY)
    if data is None:
        raise KeyError(f'Category "{CATEGORY}" not found in JSON')
    if isinstance(data, list):
        for url in data:
            yield "all", "", url
        return
    if not isinstance(data, dict):
        raise TypeError("JSON value for category must be a dict")

    for year, year_val in sorted(data.items()):
        if isinstance(year_val, dict):
            for month, urls in sorted(year_val.items()):
                if not isinstance(urls, list):
                    continue
                for url in urls:
                    yield str(year), str(month), url
        elif isinstance(year_val, list):
            for url in year_val:
                yield str(year), "", url


def main():
    links_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LINKS_JSON
    if not links_path.is_file():
        print(f"Links file not found: {links_path}")
        sys.exit(1)

    with open(links_path, encoding="utf-8") as f:
        links_json = json.load(f)

    total = 0
    for _ in iter_year_month_links(links_json):
        total += 1
    print(f"Started download from {links_path} with total docs = {total}\n")

    ndocs = 0
    current_year = None
    current_month = None
    month_index = 0

    for year, month, url in iter_year_month_links(links_json):
        if year != current_year or month != current_month:
            current_year = year
            current_month = month
            month_index = 0
            if month:
                out_dir = OUTPUT_BASE / year / month
            else:
                out_dir = OUTPUT_BASE / year
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"  {year}" + (f" / {month}" if month else "") + f" — saving to {out_dir}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)
        text, _ = get_judgment_text(url)
        if text is None:
            continue

        if month:
            out_path = out_dir / f"{year}_{month}_{month_index}.txt"
        else:
            out_path = out_dir / f"{year}_{month_index}.txt"
        out_path.write_text(text, encoding="utf-8")
        ndocs += 1
        month_index += 1

        if ndocs % SLEEP_EVERY_N_DOCS == 0:
            time.sleep(SLEEP_EVERY_N_SECS)
            print(f"  Downloaded {ndocs} docs so far ...")

    print(f"\nDone. Total documents saved: {ndocs}")


if __name__ == "__main__":
    main()
