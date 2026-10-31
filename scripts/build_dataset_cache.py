"""
Build the full aligned dataset (ticker, date, X, Z, Y, surprise) ONCE and
cache it to disk, so every diagnostic script in this session stops
re-fetching the same yfinance price data from scratch. That repeated
re-fetching is what caused the step2d regularization-sweep run to stall
for hours on a single slow yf.download() call with no timeout (now
fixed in data/prices.py, but caching avoids re-paying the cost anyway).
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_cache")

from main import stage_price_features, stage_abnormal_returns, stage_align
from features.embeddings import embed_transcripts_batch
from data.earnings_surprise import match_surprise_to_events

RAW_DIR = Path("raw_filings_fast")
EMBED_CACHE_DIR = Path("embeddings_cache_fast")
SURPRISE_CACHE_DIR = Path("earnings_surprise_cache")
START, END = "2020-01-01", "2023-12-31"
OUT_PATH = Path("dataset_cache/surprise_matched_150.npz")


def load_events():
    tickers = Path("/tmp/fast_tickers.txt").read_text().split()
    rows = []
    for ticker in tickers:
        tdir = RAW_DIR / ticker
        if not tdir.exists():
            continue
        for jpath in sorted(tdir.glob("*.json")):
            try:
                rec = json.loads(jpath.read_text())
            except Exception:
                continue
            if not rec.get("item_202_present", False):
                continue
            text = rec.get("raw_text", "") or ""
            if len(text) < 50:
                continue
            rows.append({"ticker": ticker, "filing_date": pd.Timestamp(rec["filing_date"]),
                         "text": text, "accession": rec.get("accession_number", "")})
    df = pd.DataFrame(rows)
    df = df[(df["filing_date"] >= pd.Timestamp(START)) & (df["filing_date"] <= pd.Timestamp(END))].reset_index(drop=True)
    records = [{"ticker": r["ticker"], "filing_date": str(r["filing_date"].date()), "raw_text": r["text"]}
               for _, r in df.iterrows()]
    embeddings = embed_transcripts_batch(records, cache_dir=EMBED_CACHE_DIR)
    df["embedding"] = [embeddings.get((r["ticker"], str(r["filing_date"].date()))) for _, r in df.iterrows()]
    return df[df["embedding"].apply(lambda e: e is not None)].reset_index(drop=True)


def main():
    logger.info("Loading cached events (EDGAR + embeddings, no re-fetch needed)...")
    df = load_events()
    logger.info("Computing price features and forward returns via yfinance (one-time cost)...")
    df = stage_price_features(df)
    df = stage_abnormal_returns(df)
    events_df, X_all, Z_all, Y_all = stage_align(df)
    logger.info("Aligned: N=%d", len(Y_all))

    merged = match_surprise_to_events(events_df, cache_dir=SURPRISE_CACHE_DIR)
    mask = merged["surprise_pct"].notna().values
    events_df = merged[mask].reset_index(drop=True)
    X_all, Z_all, Y_all = X_all[mask], Z_all[mask], Y_all[mask]
    S_all = events_df["surprise_pct"].values.reshape(-1, 1)
    logger.info("Surprise-matched: N=%d", mask.sum())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        X_all=X_all, Z_all=Z_all, Y_all=Y_all, S_all=S_all,
        ticker=events_df["ticker"].values,
        date=events_df["date"].values.astype("datetime64[ns]"),
    )
    logger.info("Saved cached dataset to %s (N=%d events)", OUT_PATH, len(Y_all))


if __name__ == "__main__":
    main()
