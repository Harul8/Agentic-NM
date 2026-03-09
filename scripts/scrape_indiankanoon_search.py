"""
Scrape case law document URLs from Indian Kanoon SEARCH page (formInput + pagenum).
URL format: https://indiankanoon.org/search/?formInput=...&pagenum=N
Flow: fetch one search page -> download those docs -> next page. Stop when page has < 10 links.
Set START_YEAR, END_YEAR at top. Results are stored per year under legal_database/raw_data/CaseLaws/Supreme Court/<year>/ (no ALL subfolder).
Usage: python scrape_indiankanoon_search.py [--test]  # --test: START_YEAR only, 2 pages
"""
import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

# Output: save under legal_database/raw_data/CaseLaws/Supreme Court/<year>/ (no ALL subfolder)
_scripts_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _scripts_dir.parent
SEARCH_OUTPUT_BASE = PROJECT_ROOT / "legal_database" / "raw_data" / "CaseLaws" / "Supreme Court"

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Quick test: START_YEAR only, 2 pages")
args = parser.parse_args()

# Date range (inclusive). Process order: newest first.
START_YEAR = 1991
END_YEAR = 1950

# Ensure scripts/ is on path when run from project root
import sys
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import requests
from bs4 import BeautifulSoup
from download_indiankanoon_data import download_month_urls

base = "https://indiankanoon.org"
search_base = f"{base}/search/"
court_name = "Supreme Court of India"

# Search query template: citation + doctypes: supremecourt + year: YYYY
FORMINPUT_TEMPLATE = "citation   doctypes: supremecourt year: {year}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
SEARCH_RETRY_WAIT = 5  # base wait (seconds); 403 uses exponential backoff
SEARCH_MAX_RETRIES = 3  # retry 403/5xx more times before stopping
SEARCH_PAGE_DELAY = 0.5  # seconds before fetching next search page
BETWEEN_YEAR_DELAY = 0.5
ERROR_RETRY_WAIT = 2

# Stop-year rule: pass to download_month_urls. Stop current year when this many links in batch have Cited by < threshold.
CITED_BY_STOP_THRESHOLD = 20  # count as "low" when Cited by < this
MIN_LOW_TO_STOP = 5  # stop year when this many in batch are "low"

# Month key for search-by-year results (no month breakdown)
MONTH_KEY = "ALL"

links = {court_name: {}}
outfile_path = Path(__file__).parent / "links_Supreme_Court_search.json"

years = list(range(END_YEAR, START_YEAR + 1))
years.reverse()  # newest first
if args.test:
    years = [START_YEAR]

print(f"Date range: {START_YEAR} — {END_YEAR} (search URL format)\n")

session = requests.Session()
session.headers.update(HEADERS)

for year in years:
    try:
        print(f"{year} Year Started .....\n")
        links[court_name][str(year)] = {MONTH_KEY: []}

        form_input = FORMINPUT_TEMPLATE.format(year=year)
        page_in = 0
        prev_search_url = search_base

        while True:
            time.sleep(SEARCH_PAGE_DELAY)
            query = quote(form_input, safe="")
            search_url = f"{search_base}?formInput={query}&pagenum={page_in}"

            headers = {**HEADERS, "Referer": prev_search_url}
            page = None
            for attempt in range(SEARCH_MAX_RETRIES + 1):
                try:
                    resp = session.get(search_url, headers=headers, timeout=30)
                except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
                    if attempt < SEARCH_MAX_RETRIES:
                        wait = SEARCH_RETRY_WAIT * (2 ** attempt)
                        print(f"[{year}] Search pagenum={page_in} connection error ({type(e).__name__}), waiting {wait}s and retrying ({attempt + 1}/{SEARCH_MAX_RETRIES + 1}) ...")
                        time.sleep(wait)
                        continue
                    print(f"[{year}] Search pagenum={page_in} connection error after retries, stopping year.")
                    page = None
                    break
                if resp.status_code == 200:
                    page = resp
                    break
                retryable = resp.status_code == 403 or (500 <= resp.status_code < 600)
                if retryable and attempt < SEARCH_MAX_RETRIES:
                    wait = SEARCH_RETRY_WAIT * (2 ** attempt)  # exponential backoff for 403
                    print(f"[{year}] Search pagenum={page_in} got {resp.status_code}, waiting {wait}s and retrying ({attempt + 1}/{SEARCH_MAX_RETRIES + 1}) ...")
                    time.sleep(wait)
                    continue
                break

            if page is None:
                print(f"[{year}] Search pagenum={page_in} failed (non-200), stopping year.")
                break

            try:
                soup = BeautifulSoup(page.content, "html.parser")
                doc_links = soup.find_all("a", href=re.compile(r"^/doc/\d"))
                page_urls = []
                for a in doc_links:
                    href = a.get("href")
                    if not href:
                        continue
                    full_url = base + href if href.startswith("/") else base + "/" + href
                    page_urls.append(full_url)

                if len(page_urls) < 10:
                    treat_as_last = True
                    if page_in >= 2 and len(links[court_name][str(year)][MONTH_KEY]) >= 10 and len(page_urls) < 10:
                        for retry_n in range(1):
                            print(f"[{year}] Page {page_in} returned {len(page_urls)} (< 10, possible rate limit), waiting {ERROR_RETRY_WAIT}s and retrying ({retry_n + 1}/1) ...")
                            time.sleep(ERROR_RETRY_WAIT)
                            try:
                                resp = session.get(search_url, headers=headers, timeout=30)
                            except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError):
                                continue
                            if resp.status_code == 200:
                                soup = BeautifulSoup(resp.content, "html.parser")
                                doc_links = soup.find_all("a", href=re.compile(r"^/doc/\d"))
                                page_urls = []
                                for a in doc_links:
                                    h = a.get("href")
                                    if h:
                                        page_urls.append(base + h if h.startswith("/") else base + "/" + h)
                                if len(page_urls) >= 10:
                                    treat_as_last = False
                                    print(f"[{year}] Retry succeeded: got {len(page_urls)} links.")
                                    break

                    if treat_as_last:
                        new_urls = [u for u in page_urls if u not in links[court_name][str(year)][MONTH_KEY]]
                        start_idx = len(links[court_name][str(year)][MONTH_KEY])
                        links[court_name][str(year)][MONTH_KEY].extend(new_urls)
                        if new_urls:
                            saved, stop_year, _ = download_month_urls(
                                new_urls, str(year), MONTH_KEY, start_idx,
                                output_base=SEARCH_OUTPUT_BASE,
                                cited_by_stop_threshold=CITED_BY_STOP_THRESHOLD, min_low_to_stop=MIN_LOW_TO_STOP,
                            )
                            if stop_year:
                                print(f"[{year}] At least {MIN_LOW_TO_STOP} links in batch below cited-by limit ({CITED_BY_STOP_THRESHOLD}); stopping year.")
                            print(f"[{year}] Page {page_in}: {len(new_urls)} new links, {saved} downloaded.")
                        with open(outfile_path, "w") as outfile:
                            outfile.write(json.dumps(links, indent=4))
                        print(f"[{year}] Year done ({len(links[court_name][str(year)][MONTH_KEY])} URLs total).\n")
                        break

                if len(page_urls) >= 10:
                    new_urls = [u for u in page_urls if u not in links[court_name][str(year)][MONTH_KEY]]
                    start_idx = len(links[court_name][str(year)][MONTH_KEY])
                    links[court_name][str(year)][MONTH_KEY].extend(new_urls)
                    with open(outfile_path, "w") as outfile:
                        outfile.write(json.dumps(links, indent=4))
                    print(f"[{year}] Page {page_in}: {len(page_urls)} links ({len(new_urls)} new), downloading ...")
                    saved, stop_year, _ = download_month_urls(
                        new_urls, str(year), MONTH_KEY, start_idx,
                        output_base=SEARCH_OUTPUT_BASE,
                        cited_by_stop_threshold=CITED_BY_STOP_THRESHOLD, min_low_to_stop=MIN_LOW_TO_STOP,
                    )
                    print(f"[{year}] Page {page_in}: {saved} docs saved.")
                    if stop_year:
                        print(f"[{year}] At least {MIN_LOW_TO_STOP} links in batch below cited-by limit ({CITED_BY_STOP_THRESHOLD}); stopping year and moving to next.")
                        with open(outfile_path, "w") as outfile:
                            outfile.write(json.dumps(links, indent=4))
                        print(f"[{year}] Year done ({len(links[court_name][str(year)][MONTH_KEY])} URLs total).\n")
                        break

                    prev_search_url = search_url
                    page_in += 1
                    if args.test and page_in >= 2:
                        break
            except Exception as e:
                print(f"[{year}] Batch pagenum={page_in} error (skipping batch): {e}")
                page_in += 1
                continue
    except Exception as e:
        print(f"[{year}] Error (skipping to next year): {e}")
    if year != years[-1]:
        time.sleep(BETWEEN_YEAR_DELAY)
    print(f"\n{year} Year Completed.\n")
