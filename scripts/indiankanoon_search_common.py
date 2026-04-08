"""
Shared utilities for scraping Indian Kanoon search result pages for case laws.

This module centralises the search/pagination logic so individual scrapers
only need to define their configuration (years, URL template, court, etc.)
at the top.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from download_indiankanoon_data import download_month_urls


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

SEARCH_RETRY_WAIT = 5        # base wait (seconds); 403 uses exponential backoff
SEARCH_MAX_RETRIES = 3       # retry 403/5xx more times before stopping
SEARCH_PAGE_DELAY = 0.5      # seconds before fetching next search page
BETWEEN_YEAR_DELAY = 0.5
ERROR_RETRY_WAIT = 2

BASE_URL = "https://indiankanoon.org"
SEARCH_BASE = f"{BASE_URL}/search/"


@dataclass
class CaseSearchJob:
    court_name: str
    forminput_template: str
    start_year: int
    end_year: int
    month_key: str
    cited_by_stop_threshold: int
    min_low_to_stop: int
    links_path: Path
    output_base: Path | None = None  # where download_month_urls saves docs


def run_case_search(job: CaseSearchJob, *, test: bool = False) -> None:
    """
    Run a search-based scraper for a single court configuration.
    """
    years = list(range(job.end_year, job.start_year + 1))
    years.reverse()  # newest first
    if test:
        years = [job.start_year]

    links: dict[str, dict] = {job.court_name: {}}
    outfile_path = job.links_path

    print(f"Date range: {job.start_year} — {job.end_year} ({job.court_name} search)\n")

    session = requests.Session()
    session.headers.update(HEADERS)

    for year in years:
        try:
            print(f"{year} Year Started .....\n")
            links[job.court_name][str(year)] = {job.month_key: []}

            form_input = job.forminput_template.format(year=year)
            page_in = 0
            prev_search_url = SEARCH_BASE

            while True:
                time.sleep(SEARCH_PAGE_DELAY)
                query = quote(form_input, safe="")
                search_url = f"{SEARCH_BASE}?formInput={query}&pagenum={page_in}"

                headers = {**HEADERS, "Referer": prev_search_url}
                page = None
                for attempt in range(SEARCH_MAX_RETRIES + 1):
                    try:
                        resp = session.get(search_url, headers=headers, timeout=30)
                    except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
                        if attempt < SEARCH_MAX_RETRIES:
                            wait = SEARCH_RETRY_WAIT * (2 ** attempt)
                            print(
                                f"[{year}] Search pagenum={page_in} connection error "
                                f"({type(e).__name__}), waiting {wait}s and retrying "
                                f"({attempt + 1}/{SEARCH_MAX_RETRIES + 1}) ..."
                            )
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
                        print(
                            f"[{year}] Search pagenum={page_in} got {resp.status_code}, "
                            f"waiting {wait}s and retrying ({attempt + 1}/{SEARCH_MAX_RETRIES + 1}) ..."
                        )
                        time.sleep(wait)
                        continue
                    break

                if page is None:
                    print(f"[{year}] Search pagenum={page_in} failed (non-200), stopping year.")
                    break

                try:
                    soup = BeautifulSoup(page.content, "html.parser")
                    doc_links = soup.find_all("a", href=re.compile(r"^/doc/\d"))
                    page_urls: list[str] = []
                    for a in doc_links:
                        href = a.get("href")
                        if not href:
                            continue
                        full_url = BASE_URL + href if href.startswith("/") else BASE_URL + "/" + href
                        page_urls.append(full_url)

                    # Last-page handling: < 10 links (with one retry heuristic)
                    if len(page_urls) < 10:
                        treat_as_last = True
                        if page_in >= 2 and len(links[job.court_name][str(year)][job.month_key]) >= 10 and len(page_urls) < 10:
                            for retry_n in range(1):
                                print(
                                    f"[{year}] Page {page_in} returned {len(page_urls)} "
                                    f"(< 10, possible rate limit), waiting {ERROR_RETRY_WAIT}s "
                                    f"and retrying ({retry_n + 1}/1) ..."
                                )
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
                                            page_urls.append(BASE_URL + h if h.startswith("/") else BASE_URL + "/" + h)
                                    if len(page_urls) >= 10:
                                        treat_as_last = False
                                        print(f"[{year}] Retry succeeded: got {len(page_urls)} links.")
                                        break

                        if treat_as_last:
                            new_urls = [
                                u
                                for u in page_urls
                                if u not in links[job.court_name][str(year)][job.month_key]
                            ]
                            start_idx = len(links[job.court_name][str(year)][job.month_key])
                            links[job.court_name][str(year)][job.month_key].extend(new_urls)
                            if new_urls:
                                saved, stop_year, low_count = download_month_urls(
                                    new_urls,
                                    str(year),
                                    job.month_key,
                                    start_idx,
                                    output_base=job.output_base,
                                    cited_by_stop_threshold=job.cited_by_stop_threshold,
                                    min_low_to_stop=job.min_low_to_stop,
                                )
                                print(
                                    f"[{year}] Page {page_in}: {len(new_urls)} new links, {saved} downloaded "
                                    f"({low_count} with cited by < {job.cited_by_stop_threshold})."
                                )
                                if stop_year:
                                    print(
                                        f"[{year}] {job.min_low_to_stop}+ links in batch with cited by < "
                                        f"{job.cited_by_stop_threshold}; stopping year."
                                    )
                                    with open(outfile_path, "w", encoding="utf-8") as outfile:
                                        json.dump(links, outfile, indent=4)
                                    print(
                                        f"[{year}] Year done "
                                        f"({len(links[job.court_name][str(year)][job.month_key])} URLs total).\n"
                                    )
                                    break
                            with open(outfile_path, "w", encoding="utf-8") as outfile:
                                json.dump(links, outfile, indent=4)
                            print(
                                f"[{year}] Year done "
                                f"({len(links[job.court_name][str(year)][job.month_key])} URLs total).\n"
                            )
                            break

                    # Normal page: 10+ links
                    if len(page_urls) >= 10:
                        new_urls = [
                            u
                            for u in page_urls
                            if u not in links[job.court_name][str(year)][job.month_key]
                        ]
                        start_idx = len(links[job.court_name][str(year)][job.month_key])
                        links[job.court_name][str(year)][job.month_key].extend(new_urls)
                        with open(outfile_path, "w", encoding="utf-8") as outfile:
                            json.dump(links, outfile, indent=4)
                        print(
                            f"[{year}] Page {page_in}: {len(page_urls)} links "
                            f"({len(new_urls)} new), downloading ..."
                        )
                        saved, stop_year, low_count = download_month_urls(
                            new_urls,
                            str(year),
                            job.month_key,
                            start_idx,
                            output_base=job.output_base,
                            cited_by_stop_threshold=job.cited_by_stop_threshold,
                            min_low_to_stop=job.min_low_to_stop,
                        )
                        print(
                            f"[{year}] Page {page_in}: {saved} docs saved "
                            f"({low_count} with cited by < {job.cited_by_stop_threshold})."
                        )
                        if stop_year:
                            print(
                                f"[{year}] {job.min_low_to_stop}+ links in batch with cited by < "
                                f"{job.cited_by_stop_threshold}; stopping year and moving to next."
                            )
                            with open(outfile_path, "w", encoding="utf-8") as outfile:
                                json.dump(links, outfile, indent=4)
                            print(
                                f"[{year}] Year done "
                                f"({len(links[job.court_name][str(year)][job.month_key])} URLs total).\n"
                            )
                            break

                        prev_search_url = search_url
                        page_in += 1
                        if test and page_in >= 2:
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


# ---------------------------------------------------------------------------
# Built-in jobs / CLI entrypoint
# ---------------------------------------------------------------------------

# These targets let you run the scraper directly from this module instead of
# having separate launcher scripts.

# SC: Supreme Court of India
_scripts_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _scripts_dir.parent

SC_SEARCH_OUTPUT_BASE = PROJECT_ROOT / "legal_database" / "raw_data" / "CaseLaws" / "Supreme Court"

SC_START_YEAR = 2026
SC_END_YEAR = 2014

SC_JOB = CaseSearchJob(
    court_name="Supreme Court of India",
    forminput_template="citation   doctypes: supremecourt year: {year}",
    start_year=SC_START_YEAR,
    end_year=SC_END_YEAR,
    month_key="ALL",
    cited_by_stop_threshold=0,
    min_low_to_stop=11,
    links_path=_scripts_dir / "links_Supreme_Court_search.json",
    output_base=SC_SEARCH_OUTPUT_BASE,
)


# Telangana HC (Andhra HC pre-Telangana)
TEL_SEARCH_OUTPUT_BASE = PROJECT_ROOT / "legal_database" / "raw_data" / "CaseLaws" / "Telangana HC"

TEL_START_YEAR = 1994
TEL_END_YEAR = 1950

TEL_JOB = CaseSearchJob(
    court_name="Andhra HC (Pre-Telangana)",
    forminput_template="citation   doctypes: andhra year: {year}",
    start_year=TEL_START_YEAR,
    end_year=TEL_END_YEAR,
    month_key="ALL_AndhraHC",
    cited_by_stop_threshold=3,
    min_low_to_stop=8,
    links_path=_scripts_dir / "links_Andhra_HC_search.json",
    output_base=TEL_SEARCH_OUTPUT_BASE,
)


def main() -> None:
    """
    CLI entrypoint.

    Examples:
      python indiankanoon_search_common.py --target sc
      python indiankanoon_search_common.py --target telangana
      python indiankanoon_search_common.py --target both
      python indiankanoon_search_common.py --target sc --test
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=("sc", "telangana", "both"),
        default="sc",
        help="Which court to scrape (Supreme Court, Telangana HC, or both).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test: only START_YEAR, max 2 pages per court.",
    )
    args = parser.parse_args()

    if args.target in ("sc", "both"):
        run_case_search(SC_JOB, test=args.test)
    if args.target in ("telangana", "both"):
        run_case_search(TEL_JOB, test=args.test)


if __name__ == "__main__":
    main()

