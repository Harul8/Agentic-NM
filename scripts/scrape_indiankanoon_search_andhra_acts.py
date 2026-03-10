"""
Scrape bare acts from Indian Kanoon search: Telangana state acts and/or Union acts.
URL templates:
  Telangana: .../search/?formInput=doctypes:telengana-act%20year:YYYY&pagenum=N
  Union:     .../search/?formInput=doctypes%3A%20union-act%20year%3A%20YYYY&pagenum=N
Flow: fetch search page by year -> collect all /doc/ links -> download each (no cites/cited-by filter).
Saves (no year subfolders):
  Telangana -> legal_database/raw_data/BareActs/Telangana/<year>_<index>.txt
  Union     -> legal_database/raw_data/BareActs/Union of India/<year>_<index>.txt
Usage: python scrape_indiankanoon_search_andhra_acts.py [--test] [--type telangana|union|both]
"""
import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

_scripts_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _scripts_dir.parent
BAREACTS_BASE = PROJECT_ROOT / "legal_database" / "raw_data" / "BareActs"

# (doctype query, category label for links JSON, output subdir under BareActs, links filename)
# Indian Kanoon URL uses "telengana-act" (as in formInput=doctypes:telengana-act%20year:2020)
ACT_TYPES = [
    ("doctypes:telengana-act", "Telangana - Act", "Telangana", "links_Telangana_Acts_search.json"),
    ("doctypes: union-act", "Union of India - Act", "Union of India", "links_Union_Acts_search.json"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Quick test: START_YEAR only, 2 pages")
parser.add_argument("--type", choices=("telangana", "union", "both"), default="both",
                    help="Which acts to scrape: telangana, union, or both (default)")
args = parser.parse_args()

# Year range (inclusive). Process order: newest first.
START_YEAR = 2015
END_YEAR = 1947
import sys
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import requests
from bs4 import BeautifulSoup
from download_indiankanoon_data import get_doc_text_any, get_doc_text_with_citedby_min

base = "https://indiankanoon.org"
search_base = f"{base}/search/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
SEARCH_RETRY_WAIT = 5
SEARCH_MAX_RETRIES = 3
SEARCH_PAGE_DELAY = 0.5
BETWEEN_YEAR_DELAY = 0.5
BETWEEN_ACT_TYPE_DELAY = 1.0
ERROR_RETRY_WAIT = 2
SLEEP_BETWEEN_DOCS = 2

years = list(range(END_YEAR, START_YEAR + 1))
years.reverse()
if args.test:
    years = [START_YEAR]

# Select act types to run
if args.type == "telangana":
    selected = [ACT_TYPES[0]]
elif args.type == "union":
    selected = [ACT_TYPES[1]]
else:
    selected = ACT_TYPES

session = requests.Session()
session.headers.update(HEADERS)

for type_idx, (form_suffix, category_name, output_subdir, links_filename) in enumerate(selected):
    output_base = BAREACTS_BASE / output_subdir
    outfile_path = Path(__file__).parent / links_filename
    links = {category_name: {}}
    FORMINPUT_TEMPLATE = f"{form_suffix} year: {{year}}"
    is_union = "union-act" in form_suffix

    print(f"\n{'='*60}")
    print(f"Acts: {category_name} | years {START_YEAR} - {END_YEAR} | save to {output_base}")
    print("="*60)

    for year in years:
        try:
            print(f"[{category_name}] {year} Year Started .....\n")
            links[category_name][str(year)] = []

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
                            print(f"[{year}] Search pagenum={page_in} connection error ({type(e).__name__}), waiting {wait}s ...")
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
                        wait = SEARCH_RETRY_WAIT * (2 ** attempt)
                        print(f"[{year}] Search pagenum={page_in} got {resp.status_code}, waiting {wait}s ...")
                        time.sleep(wait)
                        continue
                    break

                if page is None:
                    print(f"[{year}] Search pagenum={page_in} failed, stopping year.")
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
                        if page_in >= 1 and len(links[category_name][str(year)]) >= 10 and len(page_urls) < 10:
                            time.sleep(ERROR_RETRY_WAIT)
                            try:
                                resp = session.get(search_url, headers=headers, timeout=30)
                                if resp.status_code == 200:
                                    soup = BeautifulSoup(resp.content, "html.parser")
                                    doc_links = soup.find_all("a", href=re.compile(r"^/doc/\d"))
                                    page_urls = [base + (a.get("href") or "") for a in doc_links if a.get("href")]
                                    if len(page_urls) >= 10:
                                        treat_as_last = False
                            except Exception:
                                pass

                        if treat_as_last:
                            new_urls = [u for u in page_urls if u not in links[category_name][str(year)]]
                            links[category_name][str(year)].extend(new_urls)
                            out_dir = output_base
                            out_dir.mkdir(parents=True, exist_ok=True)
                            start_idx = len(links[category_name][str(year)]) - len(new_urls)
                            saved = 0
                            for i, url in enumerate(new_urls):
                                time.sleep(SLEEP_BETWEEN_DOCS)
                                if is_union:
                                    text = get_doc_text_with_citedby_min(url, min_cited_by=5)
                                else:
                                    text = get_doc_text_any(url)
                                if text:
                                    out_path = out_dir / f"{year}_{start_idx + i}.txt"
                                    out_path.write_text(text, encoding="utf-8")
                                    saved += 1
                            print(f"[{year}] Page {page_in}: {len(new_urls)} new links, {saved} downloaded.")
                            with open(outfile_path, "w", encoding="utf-8") as outfile:
                                json.dump(links, outfile, indent=4)
                            print(f"[{year}] Year done ({len(links[category_name][str(year)])} URLs total).\n")
                            break

                    if len(page_urls) >= 10:
                        new_urls = [u for u in page_urls if u not in links[category_name][str(year)]]
                        links[category_name][str(year)].extend(new_urls)
                        with open(outfile_path, "w", encoding="utf-8") as outfile:
                            json.dump(links, outfile, indent=4)
                        out_dir = output_base
                        out_dir.mkdir(parents=True, exist_ok=True)
                        start_idx = len(links[category_name][str(year)]) - len(new_urls)
                        saved = 0
                        print(f"[{year}] Page {page_in}: {len(page_urls)} links ({len(new_urls)} new), downloading ...")
                        for i, url in enumerate(new_urls):
                            time.sleep(SLEEP_BETWEEN_DOCS)
                            if is_union:
                                text = get_doc_text_with_citedby_min(url, min_cited_by=5)
                            else:
                                text = get_doc_text_any(url)
                            if text:
                                out_path = out_dir / f"{year}_{start_idx + i}.txt"
                                out_path.write_text(text, encoding="utf-8")
                                saved += 1
                        print(f"[{year}] Page {page_in}: {saved} docs saved.")
                        prev_search_url = search_url
                        page_in += 1
                        if args.test and page_in >= 2:
                            break
                except Exception as e:
                    print(f"[{year}] Batch pagenum={page_in} error: {e}")
                    page_in += 1
                    continue
        except Exception as e:
            print(f"[{year}] Error (skipping to next year): {e}")
        if year != years[-1]:
            time.sleep(BETWEEN_YEAR_DELAY)
        print(f"\n[{category_name}] {year} Year Completed.\n")

    if type_idx < len(selected) - 1:
        time.sleep(BETWEEN_ACT_TYPE_DELAY)

print("\nDone.")
