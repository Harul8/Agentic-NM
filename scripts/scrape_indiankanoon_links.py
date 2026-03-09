"""
Scrape case law document URLs from Indian Kanoon browse page (by court, year, month).
Flow: fetch one search page -> download those docs -> next page. Stop when page has < 10 links.
Set START_YEAR, START_MONTH, END_YEAR, END_MONTH at top to define date range.
Usage: python scrape_indiankanoon_links.py [--test]  # --test: START_YEAR only, 2 pages/month
"""
import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Quick test: START_YEAR only, 2 pages per month")
args = parser.parse_args()

# Date range to scrape (inclusive). Process order: newest first (START down to END)
START_YEAR = 2016
START_MONTH = "JAN"  # First month to scrape
END_YEAR = 2011
END_MONTH = "DEC"  # Last month to scrape

# Ensure scripts/ is on path when run from project root
_scripts_dir = Path(__file__).resolve().parent
import sys
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import requests
from bs4 import BeautifulSoup
from download_indiankanoon_data import download_month_urls

# Start directly at Supreme Court browse (main browse page no longer uses browselist)
base = "https://indiankanoon.org"
court_name = "Supreme Court of India"
court_url = f"{base}/browse/supremecourt/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
SEARCH_RETRY_WAIT = 10  # base wait (seconds); 403 uses exponential backoff
SEARCH_MAX_RETRIES = 4  # retry 403/5xx more times before stopping
SEARCH_PAGE_DELAY = 0.5  # seconds before fetching next search page
BETWEEN_MONTH_DELAY = 0.5  # seconds before starting next month
ERROR_RETRY_WAIT = 2  # wait on "0 results" (possible rate limit) before retrying

# Stop-month rule: pass to download_month_urls. Stop current month when this many links in batch have Cited by < threshold.
CITED_BY_STOP_THRESHOLD = 10  # count as "low" when Cited by < this
MIN_LOW_TO_STOP = 5  # stop month when this many in batch are "low"

MONTH_ORDER = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def month_in_range(year: int, month_key: str) -> bool:
    """True if (year, month) is within [END_YEAR/END_MONTH, START_YEAR/START_MONTH]."""
    if year < END_YEAR or year > START_YEAR:
        return False
    if year == START_YEAR and month_key in MONTH_ORDER:
        return MONTH_ORDER.index(month_key) >= MONTH_ORDER.index(START_MONTH)
    if year == END_YEAR and month_key in MONTH_ORDER:
        return MONTH_ORDER.index(month_key) <= MONTH_ORDER.index(END_MONTH)
    return True

page = requests.get(court_url, headers=HEADERS)
soup = BeautifulSoup(page.content, "html.parser")
result_new = soup.find_all(class_="browselist")

# Month name as on site -> folder/key (YYYY \ JAN format)
MONTH_TO_ABBREV = {
    "January": "JAN", "February": "FEB", "March": "MAR", "April": "APR",
    "May": "MAY", "June": "JUN", "July": "JUL", "August": "AUG",
    "September": "SEP", "October": "OCT", "November": "NOV", "December": "DEC",
}

links = {court_name: {}}
outfile_path = Path(__file__).parent / "links_Supreme_Court.json"

# Initial browse page - retry on connection error so we don't exit immediately
result_new = []
for _ in range(SEARCH_MAX_RETRIES + 1):
    try:
        page = requests.get(court_url, headers=HEADERS, timeout=30)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, "html.parser")
        result_new = soup.find_all(class_="browselist")
        break
    except (requests.exceptions.RequestException, Exception) as e:
        print(f"Initial browse failed ({e}), retrying ...")
        time.sleep(SEARCH_RETRY_WAIT)
else:
    print("Initial browse failed after retries, skipping scrape.")

year_entries = []
for link_new in result_new:
    year_el = link_new.find("a")
    if not year_el:
        continue
    try:
        y = int(year_el.text)
    except (ValueError, TypeError):
        continue
    if y < END_YEAR or y > START_YEAR:
        continue
    if args.test and y != START_YEAR:
        continue
    year_entries.append((y, link_new))

year_entries.sort(key=lambda x: x[0], reverse=True)  # START_YEAR down to END_YEAR

print(f"Date range: {START_YEAR} {START_MONTH} — {END_YEAR} {END_MONTH}\n")

for year, link_new in year_entries:
    try:
        year_el = link_new.find("a")
        print(f"{year} Year Started .....\n")
        links[court_name][str(year)] = {}
        year_url = base + year_el["href"]
        session = requests.Session()
        session.headers.update(HEADERS)
        try:
            page = session.get(year_url, headers=HEADERS, timeout=30)
            page.raise_for_status()
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"[{year}] Year page failed ({e}), skipping year.")
            continue
        soup = BeautifulSoup(page.content, "html.parser")
        result_new2 = soup.find_all(class_="browselist")
        for link_new2 in result_new2:
            try:
                month_a = link_new2.find("a")
                if not month_a or "href" not in month_a.attrs:
                    continue
                month_name = (month_a.text or "").strip()
                month_key = MONTH_TO_ABBREV.get(month_name)  # Skip e.g. "Entire Year"
                if not month_key:
                    continue
                if not month_in_range(year, month_key):
                    continue

                prefix = f"[{year} {month_key}]"
                if link_new2 != result_new2[0] or year != year_entries[0][0]:
                    time.sleep(BETWEEN_MONTH_DELAY)

                links[court_name][str(year)][month_key] = []
                page_in = 0
                prev_search_url = year_url

                while True:
                    time.sleep(SEARCH_PAGE_DELAY)
                    raw_href = month_a["href"]
                    encoded_query = quote(raw_href.split("?", 1)[1], safe="=&") if "?" in raw_href else ""
                    search_url = base + (raw_href.split("?")[0] + "?" + encoded_query if encoded_query else raw_href)
                    search_url = search_url + ("&" if "?" in search_url else "?") + f"pagenum={page_in}"

                    headers = {**HEADERS, "Referer": prev_search_url}
                    page = None
                    for attempt in range(SEARCH_MAX_RETRIES + 1):
                        try:
                            resp = session.get(search_url, headers=headers, timeout=30)
                        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
                            if attempt < SEARCH_MAX_RETRIES:
                                wait = SEARCH_RETRY_WAIT * (2 ** attempt)
                                print(f"{prefix} Search pagenum={page_in} connection error ({type(e).__name__}), waiting {wait}s and retrying ({attempt + 1}/{SEARCH_MAX_RETRIES + 1}) ...")
                                time.sleep(wait)
                                continue
                            print(f"{prefix} Search pagenum={page_in} connection error after retries, stopping month.")
                            page = None
                            break
                        if resp.status_code == 200:
                            page = resp
                            break
                        retryable = resp.status_code == 403 or (500 <= resp.status_code < 600)
                        if retryable and attempt < SEARCH_MAX_RETRIES:
                            wait = SEARCH_RETRY_WAIT * (2 ** attempt)  # exponential backoff for 403
                            print(f"{prefix} Search pagenum={page_in} got {resp.status_code}, waiting {wait}s and retrying ({attempt + 1}/{SEARCH_MAX_RETRIES + 1}) ...")
                            time.sleep(wait)
                            continue
                        break

                    if page is None:
                        print(f"{prefix} Search pagenum={page_in} failed (non-200), stopping month.")
                        break

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
                        # Last page or empty. If page 2+ and few results, might be rate limit - retry
                        treat_as_last = True
                        if page_in >= 2 and len(links[court_name][str(year)][month_key]) >= 10 and len(page_urls) < 10:
                            for retry_n in range(1):
                                print(f"{prefix} Page {page_in} returned {len(page_urls)} (< 10, possible rate limit), waiting {ERROR_RETRY_WAIT}s and retrying ({retry_n + 1}/1) ...")
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
                                        print(f"{prefix} Retry succeeded: got {len(page_urls)} links.")
                                    break

                        if treat_as_last:
                            # Last page - add any links we got, then move to next month
                            new_urls = [u for u in page_urls if u not in links[court_name][str(year)][month_key]]
                            start_idx = len(links[court_name][str(year)][month_key])
                            links[court_name][str(year)][month_key].extend(new_urls)
                            if new_urls:
                                saved, stop_month, _ = download_month_urls(
                                    new_urls, str(year), month_key, start_idx,
                                    cited_by_stop_threshold=CITED_BY_STOP_THRESHOLD, min_low_to_stop=MIN_LOW_TO_STOP,
                                )
                                if stop_month:
                                    print(f"{prefix} At least {MIN_LOW_TO_STOP} links in batch below cited-by limit ({CITED_BY_STOP_THRESHOLD}); stopping month.")
                                print(f"{prefix} Page {page_in}: {len(new_urls)} new links, {saved} downloaded.")
                            with open(outfile_path, "w") as outfile:
                                outfile.write(json.dumps(links, indent=4))
                            print(f"{prefix} Month done ({len(links[court_name][str(year)][month_key])} URLs total).\n")
                            break

                    if len(page_urls) >= 10:
                        new_urls = [u for u in page_urls if u not in links[court_name][str(year)][month_key]]
                        start_idx = len(links[court_name][str(year)][month_key])
                        links[court_name][str(year)][month_key].extend(new_urls)
                        with open(outfile_path, "w") as outfile:
                            outfile.write(json.dumps(links, indent=4))
                        print(f"{prefix} Page {page_in}: {len(page_urls)} links ({len(new_urls)} new), downloading ...")
                        saved, stop_month, _ = download_month_urls(
                            new_urls, str(year), month_key, start_idx,
                            cited_by_stop_threshold=CITED_BY_STOP_THRESHOLD, min_low_to_stop=MIN_LOW_TO_STOP,
                        )
                        print(f"{prefix} Page {page_in}: {saved} docs saved.")
                        if stop_month:
                            print(f"{prefix} At least {MIN_LOW_TO_STOP} links in batch below cited-by limit ({CITED_BY_STOP_THRESHOLD}); stopping month and moving to next.")
                            with open(outfile_path, "w") as outfile:
                                outfile.write(json.dumps(links, indent=4))
                            print(f"{prefix} Month done ({len(links[court_name][str(year)][month_key])} URLs total).\n")
                            break

                        prev_search_url = search_url
                        page_in += 1
                        if args.test and page_in >= 2:
                            break
            except Exception as e:
                print(f"[{year}] Month error (skipping to next month): {e}")
                continue
    except Exception as e:
        print(f"[{year}] Error (skipping to next year): {e}")
    print(f"\n{year} Year Completed.\n")

