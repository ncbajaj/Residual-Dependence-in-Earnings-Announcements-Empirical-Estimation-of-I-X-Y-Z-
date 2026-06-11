"""
Fetch 8-K filings (Item 2.02 — Results of Operations) from the SEC EDGAR
full-text search API and store raw text per ticker per filing date as JSON.

SEC fair-access policy: max 10 requests/second. We enforce this with a
token-bucket rate limiter shared across all calls in a process.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter — 10 req/s per EDGAR fair-access policy
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Thread-safe token bucket: refills at `rate` tokens/second up to `capacity`."""

    def __init__(self, rate: float = 10.0, capacity: float = 10.0):
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last = time.monotonic()
        self._lock = Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens < 1.0:
                sleep_for = (1.0 - self._tokens) / self._rate
                time.sleep(sleep_for)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


_BUCKET = _TokenBucket(rate=10.0, capacity=10.0)

# ---------------------------------------------------------------------------
# EDGAR HTTP session
# ---------------------------------------------------------------------------

# SEC requires a descriptive User-Agent with contact info.
_HEADERS = {
    "User-Agent": "residual-dependence-research neil.chopra.bajaj@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


def _get(url: str, params: Optional[dict] = None, retries: int = 3) -> requests.Response:
    """GET with rate limiting and retry on transient 5xx / 429."""
    for attempt in range(retries):
        _BUCKET.acquire()
        try:
            resp = requests.get(url, headers=_HEADERS, params=params, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** attempt
                logger.warning("HTTP %d from %s — retrying in %ds", resp.status_code, url, wait)
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            logger.warning("Request error (%s) — retrying (%d/%d)", exc, attempt + 1, retries)
            time.sleep(2 ** attempt)
    # unreachable, but satisfies type checkers
    raise RuntimeError(f"All retries exhausted for {url}")


# ---------------------------------------------------------------------------
# CIK lookup
# ---------------------------------------------------------------------------

_cik_cache: dict[str, str] = {}


def ticker_to_cik(ticker: str) -> str:
    """
    Return zero-padded 10-digit CIK for a ticker using the EDGAR company tickers
    JSON. Raises ValueError if the ticker is not found.
    """
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    resp = _get("https://www.sec.gov/files/company_tickers.json")
    resp.raise_for_status()
    data = resp.json()

    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            cik = str(entry["cik_str"]).zfill(10)
            _cik_cache[ticker] = cik
            return cik

    raise ValueError(f"Ticker '{ticker}' not found in EDGAR company tickers list")


# ---------------------------------------------------------------------------
# Filing discovery
# ---------------------------------------------------------------------------

@dataclass
class FilingRecord:
    ticker: str
    cik: str
    accession_number: str      # formatted as XXXXXXXXXX-YY-ZZZZZZ
    filing_date: str           # YYYY-MM-DD
    period_of_report: str      # YYYY-MM-DD (the quarter/period covered)
    form_type: str             # "8-K" or "8-K/A"
    item_202_present: bool     # True if Item 2.02 text was found
    raw_text: Optional[str] = field(default=None, repr=False)
    parse_error: Optional[str] = field(default=None)


def _has_item_202(text: str) -> bool:
    """Heuristic: check whether the document mentions Item 2.02."""
    # Covers both "Item 2.02" and "Item 2.02." and "ITEM 2.02"
    return bool(re.search(r"item\s+2\.02", text, re.IGNORECASE))


def get_8k_filings(
    ticker: str,
    start_date: str,
    end_date: str,
) -> list[FilingRecord]:
    """
    Return a list of FilingRecord objects for all 8-K filings by `ticker`
    between `start_date` and `end_date` (both inclusive, format YYYY-MM-DD).

    Uses the EDGAR submissions API rather than full-text search so we get
    structured metadata reliably.

    Parse failures are flagged in `FilingRecord.parse_error` and never
    silently dropped — the caller sees every attempted filing.
    """
    try:
        cik = ticker_to_cik(ticker)
    except ValueError as exc:
        logger.error("CIK lookup failed for %s: %s", ticker, exc)
        return [FilingRecord(
            ticker=ticker, cik="", accession_number="", filing_date="",
            period_of_report="", form_type="8-K", item_202_present=False,
            parse_error=f"CIK lookup failed: {exc}",
        )]

    filings = _fetch_submissions_filings(ticker, cik, start_date, end_date)
    return filings


def _scan_filing_batch(
    batch: dict,
    ticker: str,
    cik: str,
    start_date: str,
    end_date: str,
) -> list[FilingRecord]:
    """Extract and fetch 8-K records from a single submissions batch dict."""
    forms = batch.get("form", [])
    dates = batch.get("filingDate", [])
    accessions = batch.get("accessionNumber", [])
    periods = batch.get("reportDate", [])

    records: list[FilingRecord] = []
    for form, date, accession, period in zip(forms, dates, accessions, periods):
        if form not in ("8-K", "8-K/A"):
            continue
        if not (start_date <= date <= end_date):
            continue
        rec = FilingRecord(
            ticker=ticker,
            cik=cik,
            accession_number=accession,
            filing_date=date,
            period_of_report=period or date,
            form_type=form,
            item_202_present=False,
        )
        _fetch_filing_text(rec)
        records.append(rec)
    return records


def _fetch_submissions_filings(
    ticker: str,
    cik: str,
    start_date: str,
    end_date: str,
) -> list[FilingRecord]:
    """
    Pull the submissions JSON for a CIK (plus any archived pages that overlap
    the requested date range), filter to 8-K filings, then fetch full text.

    EDGAR paginates older filings into separate JSON files
    (CIK{n}-submissions-NNN.json). Each archived page has `filingFrom` /
    `filingTo` metadata so we only fetch pages that overlap [start_date,
    end_date].
    """
    url = f"{_SUBMISSIONS_BASE}/CIK{cik}.json"
    try:
        resp = _get(url)
        resp.raise_for_status()
        submissions = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch submissions for %s (CIK %s): %s", ticker, cik, exc)
        return [FilingRecord(
            ticker=ticker, cik=cik, accession_number="", filing_date="",
            period_of_report="", form_type="8-K", item_202_present=False,
            parse_error=f"Submissions fetch failed: {exc}",
        )]

    recent = submissions.get("filings", {}).get("recent", {})
    records = _scan_filing_batch(recent, ticker, cik, start_date, end_date)

    # Paginate through archived batches that overlap [start_date, end_date].
    # Each entry in `files` has filingFrom/filingTo giving the date span of
    # that page, so we skip pages that clearly don't overlap.
    archive_pages = submissions.get("filings", {}).get("files", [])
    for page in archive_pages:
        page_from = page.get("filingFrom", "")
        page_to   = page.get("filingTo",   "")
        # Skip if this page's range doesn't overlap [start_date, end_date]
        if page_to and page_to < start_date:
            continue
        if page_from and page_from > end_date:
            continue
        page_url = f"{_SUBMISSIONS_BASE}/{page['name']}"
        try:
            page_resp = _get(page_url)
            page_resp.raise_for_status()
            page_data = page_resp.json()
        except Exception as exc:
            logger.warning("Failed to fetch archived submissions page %s: %s", page["name"], exc)
            continue
        records.extend(_scan_filing_batch(page_data, ticker, cik, start_date, end_date))

    logger.info("%s: found %d 8-K filings between %s and %s", ticker, len(records), start_date, end_date)
    return records


def _fetch_filing_text(rec: FilingRecord) -> None:
    """
    Fetch the full text of a filing and populate rec.raw_text.
    Sets rec.parse_error on any failure — never raises, never silently drops.
    """
    # Accession number with dashes: XXXXXXXXXX-YY-ZZZZZZ
    # Without dashes (for URL path): XXXXXXXXXXYYZZZZZ
    acc_nodash = rec.accession_number.replace("-", "")
    index_url = (
        f"{_ARCHIVES_BASE}/{rec.cik.lstrip('0')}"
        f"/{acc_nodash}/{rec.accession_number}-index.htm"
    )

    try:
        resp = _get(index_url)
        resp.raise_for_status()
        index_html = resp.text
    except Exception as exc:
        rec.parse_error = f"Index fetch failed: {exc}"
        logger.warning("Index fetch failed for %s %s: %s", rec.ticker, rec.accession_number, exc)
        return

    # Find the primary document (.htm/.html) from the filing index.
    # Two link formats exist in EDGAR:
    #   Standard:  href="/Archives/edgar/data/.../file.htm"
    #   iXBRL viewer: href="/ix?doc=/Archives/edgar/data/.../file.htm"
    # Both resolve to the same underlying document — extract the path only.
    doc_match = re.search(
        r'href="(?:/ix\?doc=)?(/Archives/edgar/data/[^"]+\.htm[l]?)"',
        index_html,
        re.IGNORECASE,
    )
    if not doc_match:
        rec.parse_error = "Primary document link not found in filing index"
        logger.warning("No primary doc link for %s %s", rec.ticker, rec.accession_number)
        return

    doc_url = "https://www.sec.gov" + doc_match.group(1)
    try:
        doc_resp = _get(doc_url)
        doc_resp.raise_for_status()
        raw_text = doc_resp.text
    except Exception as exc:
        rec.parse_error = f"Document fetch failed: {exc}"
        logger.warning("Doc fetch failed for %s %s: %s", rec.ticker, rec.accession_number, exc)
        return

    rec.raw_text = raw_text
    rec.item_202_present = _has_item_202(raw_text)

    if not rec.item_202_present:
        # Not a parse failure — many 8-Ks don't cover earnings results.
        logger.debug(
            "%s %s (%s): Item 2.02 not detected — filing retained but flagged",
            rec.ticker, rec.filing_date, rec.form_type,
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_filings(records: list[FilingRecord], out_dir: Path) -> None:
    """
    Write each filing to <out_dir>/<ticker>/<filing_date>_<accession>.json.
    All records are written, including those with parse errors — the JSON
    will contain a non-null `parse_error` field in those cases.
    """
    for rec in records:
        ticker_dir = out_dir / rec.ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        safe_acc = rec.accession_number.replace("-", "")
        filename = f"{rec.filing_date}_{safe_acc}.json"
        payload = {
            "ticker": rec.ticker,
            "cik": rec.cik,
            "accession_number": rec.accession_number,
            "filing_date": rec.filing_date,
            "period_of_report": rec.period_of_report,
            "form_type": rec.form_type,
            "item_202_present": rec.item_202_present,
            "parse_error": rec.parse_error,
            # raw_text can be very large; store it last for readability
            "raw_text": rec.raw_text,
        }
        (ticker_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if rec.parse_error:
            logger.warning("Saved filing with parse error: %s/%s — %s", rec.ticker, filename, rec.parse_error)
        else:
            logger.debug("Saved %s/%s (Item 2.02: %s)", rec.ticker, filename, rec.item_202_present)


# ---------------------------------------------------------------------------
# CLI entry point for smoke testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fetch 8-K filings from EDGAR")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "JPM"])
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--out", default="raw_filings")
    args = parser.parse_args()

    out_path = Path(args.out)
    for ticker in args.tickers:
        logger.info("=== %s ===", ticker)
        records = get_8k_filings(ticker, args.start, args.end)
        save_filings(records, out_path)
        n_ok = sum(1 for r in records if r.parse_error is None)
        n_202 = sum(1 for r in records if r.item_202_present)
        n_err = sum(1 for r in records if r.parse_error is not None)
        print(f"{ticker}: {len(records)} filings | {n_202} with Item 2.02 | {n_ok} parsed OK | {n_err} errors")
