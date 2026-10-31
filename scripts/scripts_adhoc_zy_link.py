"""
Ad-hoc analysis: is transcript content (Z) itself linked to returns (Y)?

The main pipeline never estimates I(Z;Y) directly -- it only estimates
I(X;Z) (price vs. transcript) and I(X;Y|Z) (price's INCREMENTAL value over
transcript). Near-zero I(X;Y|Z) is consistent with two very different
stories:
  (a) Z fully mediates whatever X would tell you about Y (transcripts ARE
      informative, and price features add nothing beyond them), or
  (b) neither X nor Z meaningfully predicts Y at all, so there's nothing
      for either to add.

This script tells (a) apart from (b) by estimating I(Z;Y) directly on the
same 150-ticker, 2020-2023 dataset used for output_fast/report.md, using
the exact same NOCCO estimator already validated in dependence/contingency.py,
plus a label-permutation null to get a p-value.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("zy_link")

from main import stage_price_features, stage_abnormal_returns, stage_align
from data.universe import SP500_TICKERS_2024  # noqa: F401 (not used directly, for reference)
from dependence.contingency import mean_square_contingency

import json
from pathlib import Path
import pandas as pd
from features.embeddings import embed_transcripts_batch

TICKERS_FILE = "/tmp/fast_tickers.txt"
RAW_DIR = Path("raw_filings_fast")
EMBED_CACHE_DIR = Path("embeddings_cache_fast")
START, END = "2020-01-01", "2023-12-31"


def load_cached_events(tickers):
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
            rows.append({
                "ticker": ticker,
                "filing_date": pd.Timestamp(rec["filing_date"]),
                "text": text,
                "accession": rec.get("accession_number", ""),
            })
    df = pd.DataFrame(rows)
    df = df[(df["filing_date"] >= pd.Timestamp(START)) & (df["filing_date"] <= pd.Timestamp(END))]
    df = df.reset_index(drop=True)
    records = [
        {"ticker": r["ticker"], "filing_date": str(r["filing_date"].date()), "raw_text": r["text"]}
        for _, r in df.iterrows()
    ]
    embeddings = embed_transcripts_batch(records, cache_dir=EMBED_CACHE_DIR)
    df["embedding"] = [
        embeddings.get((r["ticker"], str(r["filing_date"].date())))
        for _, r in df.iterrows()
    ]
    df = df[df["embedding"].apply(lambda e: e is not None)].reset_index(drop=True)
    return df


def main():
    tickers = Path(TICKERS_FILE).read_text().split()
    logger.info("Loading cached events for %d tickers", len(tickers))
    df = load_cached_events(tickers)
    logger.info("Loaded %d raw events", len(df))

    df = stage_price_features(df)
    logger.info("After price features: %d events", len(df))
    df = stage_abnormal_returns(df)
    logger.info("After abnormal returns: %d events", len(df))

    events_df, X_all, Z_all, Y_all = stage_align(df)
    N = len(Y_all)
    logger.info("Aligned dataset: N=%d, Z dim=%d, X dim=%d", N, Z_all.shape[1], X_all.shape[1])

    np.savez("/tmp/zy_aligned.npz", X_all=X_all, Z_all=Z_all, Y_all=Y_all,
             ticker=events_df["ticker"].values)
    logger.info("Cached aligned arrays to /tmp/zy_aligned.npz")

    Y_col = Y_all.reshape(-1, 1)

    # --- Direct test: I(Z; Y) ---
    res_zy = mean_square_contingency(Z_all, Y_col, reg=1e-3)
    logger.info("Observed Î(Z;Y) NOCCO = %.6f (SVD cross-check rel_err=%.2e)",
                res_zy.I_XZ_nocco, res_zy.rel_error)

    # --- For comparison: I(X; Y) ---
    res_xy = mean_square_contingency(X_all, Y_col, reg=1e-3)
    logger.info("Observed Î(X;Y) NOCCO = %.6f", res_xy.I_XZ_nocco)

    # --- Permutation null for I(Z;Y): shuffle Y, recompute, repeat ---
    rng = np.random.default_rng(42)
    n_perm = 200
    null_vals = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(N)
        res_null = mean_square_contingency(Z_all, Y_col[perm], reg=1e-3)
        null_vals[i] = res_null.I_XZ_nocco
    p_value = float(np.mean(null_vals >= res_zy.I_XZ_nocco))

    logger.info("Permutation null: mean=%.6f, std=%.6f, 95th pct=%.6f",
                null_vals.mean(), null_vals.std(), np.percentile(null_vals, 95))
    logger.info("Permutation p-value for I(Z;Y) > null: %.4f", p_value)

    print("\n=== Z (transcript) -> Y (forward 5-day abnormal return) ===")
    print(f"N = {N}")
    print(f"Î(Z;Y) observed      = {res_zy.I_XZ_nocco:.6f}")
    print(f"Î(X;Y) observed      = {res_xy.I_XZ_nocco:.6f}  (for comparison)")
    print(f"Permutation null mean = {null_vals.mean():.6f} +/- {null_vals.std():.6f}")
    print(f"p-value (Z;Y vs null) = {p_value:.4f}")
    print(f"Conclusion: {'Z carries detectable info about Y' if p_value < 0.05 else 'No detectable link between Z and Y at this N'}")


if __name__ == "__main__":
    main()
