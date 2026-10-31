"""
Step 1 validation: fetch earnings-surprise data for the 150-ticker cached
dataset and check match quality against the existing Z/X/Y events before
using it as a feature.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("surprise_merge")

from data.earnings_surprise import match_surprise_to_events

data = np.load("/tmp/zy_aligned.npz", allow_pickle=True)
Y_all, tickers = data["Y_all"], data["ticker"]
N = len(Y_all)

# We don't have filing dates in the npz cache (only ticker/X/Z/Y), so
# re-derive events_df the same way scripts_adhoc_zy_link.py did, from the
# raw cached filings, to get (ticker, filing_date) pairs aligned 1:1 with
# Y_all's row order is NOT guaranteed here -- rebuild properly instead.
import json
RAW_DIR = Path("raw_filings_fast")
rows = []
for tdir in sorted(RAW_DIR.iterdir()):
    if not tdir.is_dir():
        continue
    ticker = tdir.name
    for jpath in sorted(tdir.glob("*.json")):
        try:
            rec = json.loads(jpath.read_text())
        except Exception:
            continue
        if not rec.get("item_202_present", False):
            continue
        rows.append({"ticker": ticker, "filing_date": rec["filing_date"]})

events = pd.DataFrame(rows)
events["filing_date"] = pd.to_datetime(events["filing_date"])
events = events[(events["filing_date"] >= "2020-01-01") & (events["filing_date"] <= "2023-12-31")]
logger.info("Rebuilt %d raw (ticker, filing_date) pairs for surprise matching", len(events))

CACHE_DIR = Path("earnings_surprise_cache")
matched = match_surprise_to_events(events, cache_dir=CACHE_DIR)

n_total = len(matched)
n_with_surprise = matched["surprise_pct"].notna().sum()
print(f"\nTotal events: {n_total}")
print(f"Matched to a surprise record: {n_with_surprise} ({100*n_with_surprise/n_total:.1f}%)")
print(f"Unmatched (no surprise within tolerance): {n_total - n_with_surprise}")

print("\nSample of matched rows:")
print(matched[matched["surprise_pct"].notna()].head(10).to_string(index=False))

print("\nSample of UNmatched rows (for inspection):")
print(matched[matched["surprise_pct"].isna()].head(10).to_string(index=False))

matched.to_csv("/tmp/events_with_surprise.csv", index=False)
print("\nSaved merged table to /tmp/events_with_surprise.csv")
