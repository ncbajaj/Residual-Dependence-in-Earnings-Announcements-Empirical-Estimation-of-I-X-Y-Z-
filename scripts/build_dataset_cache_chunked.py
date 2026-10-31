"""
Chunked version of build_dataset_cache.py. Splits tickers into small
groups and runs each group's price-feature + abnormal-return computation
in its own fresh subprocess (_price_chunk_worker.py), to avoid the
progressive yfinance slowdown observed when one process makes hundreds
of sequential download calls. Resumable: already-completed chunk output
files are skipped on rerun.
"""
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_cache_chunked")

from main import stage_align
from features.embeddings import embed_transcripts_batch
from data.earnings_surprise import match_surprise_to_events

RAW_DIR = Path("raw_filings_fast")
EMBED_CACHE_DIR = Path("embeddings_cache_fast")
SURPRISE_CACHE_DIR = Path("earnings_surprise_cache")
START, END = "2020-01-01", "2023-12-31"
OUT_PATH = Path("dataset_cache/surprise_matched_150.npz")
CHUNK_DIR = Path("dataset_cache/chunks")
CHUNK_SIZE = 15
PER_CHUNK_TIMEOUT = 300  # seconds; a healthy chunk takes well under this


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
    logger.info("Loading cached events (EDGAR + embeddings, no network needed)...")
    df = load_events()
    tickers = sorted(df["ticker"].unique())
    chunks = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    logger.info("Split %d tickers into %d chunks of <=%d", len(tickers), len(chunks), CHUNK_SIZE)

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    worker = str(Path(__file__).resolve().parent / "_price_chunk_worker.py")

    results = []
    for i, chunk_tickers in enumerate(chunks):
        out_path = CHUNK_DIR / f"chunk_{i:03d}.pkl"
        if out_path.exists():
            logger.info("Chunk %d/%d already done, loading from cache", i + 1, len(chunks))
            results.append(pd.read_pickle(out_path))
            continue

        chunk_df = df[df["ticker"].isin(chunk_tickers)].reset_index(drop=True)
        in_path = CHUNK_DIR / f"chunk_{i:03d}_input.pkl"
        chunk_df.to_pickle(in_path)

        logger.info("Chunk %d/%d: %d tickers, %d events -- launching fresh subprocess",
                    i + 1, len(chunks), len(chunk_tickers), len(chunk_df))
        try:
            subprocess.run(
                [sys.executable, worker, str(in_path), str(out_path)],
                timeout=PER_CHUNK_TIMEOUT, check=True,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            logger.error("Chunk %d TIMED OUT after %ds -- skipping these tickers: %s",
                         i + 1, PER_CHUNK_TIMEOUT, chunk_tickers)
            continue
        except subprocess.CalledProcessError as exc:
            logger.error("Chunk %d FAILED: %s", i + 1, exc.stderr[-2000:] if exc.stderr else exc)
            continue

        results.append(pd.read_pickle(out_path))
        logger.info("Chunk %d/%d done (%d events survived)", i + 1, len(chunks), len(results[-1]))

    full_df = pd.concat(results, ignore_index=True)
    logger.info("Combined all chunks: %d events total", len(full_df))

    events_df, X_all, Z_all, Y_all = stage_align(full_df)
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
