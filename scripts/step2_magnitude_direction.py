"""
Step 2: magnitude filter + dedicated direction model.

Design (see conversation / README for the "why"):
  1. Fit a KRR magnitude model: predict |Y| from (Z, X, surprise) on the
     training set. This is the signal we already know works well
     (z=4.61 in the earlier exploratory test).
  2. Use it to rank TEST events by predicted magnitude and keep only the
     top fraction -- a "high conviction" subset, analogous to only
     trading when the model says a big move is coming.
  3. Fit a SEPARATE KRR direction model: predict (ticker-raw, not
     demeaned) Y itself from the same features on the training set.
  4. Apply the direction model ONLY to the high-conviction test subset,
     trade sign(prediction), and compare the cost-adjusted Sharpe/hit
     rate against trading everything with the same direction model
     (no magnitude filter) -- isolating the filter's contribution.

LEAKAGE GUARDS
---------------
- Chronological train/test split (train_end), same convention as the
  main pipeline.
- Kernel bandwidths (median heuristic) are computed on TRAINING data
  only and reused (frozen) for the cross-kernel against test points.
- The backtest uses RAW abnormal return Y, not the ticker-demeaned
  version used in the earlier exploratory dependence tests -- demeaning
  there used the full-sample ticker mean (train+test), which would leak
  into a real backtest. A real trading signal must not do that.
- The surprise-match filter (data/earnings_surprise.py) is applied
  before the split, using only information (surprise_pct) that is
  public at announcement time -- same timing as Z, no lookahead.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("step2")

from main import stage_price_features, stage_abnormal_returns, stage_align
from features.embeddings import embed_transcripts_batch
from data.earnings_surprise import match_surprise_to_events
from dependence.kernel_cca import median_heuristic, rbf_kernel

RAW_DIR = Path("raw_filings_fast")
EMBED_CACHE_DIR = Path("embeddings_cache_fast")
SURPRISE_CACHE_DIR = Path("earnings_surprise_cache")
START, END = "2020-01-01", "2023-12-31"
TRAIN_END = "2022-12-31"
TOP_FRACTION = 0.4       # fraction of test events kept as "high conviction"
TC_BPS = 5.0             # one-way transaction cost, matches main pipeline default
KRR_REG = 1e-2           # matches main pipeline default


def load_events():
    tickers_file = "/tmp/fast_tickers.txt"
    tickers = Path(tickers_file).read_text().split()
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


def block_kernels(Z_train, X_train, S_train, Z_other, X_other, S_other):
    """
    Equal-weighted sum of 3 RBF kernels (one per modality), bandwidths
    chosen by median heuristic on TRAINING data only and frozen for the
    cross-kernel. Returns (K_train_train, K_other_train).
    """
    h2_Z = median_heuristic(Z_train)
    h2_X = median_heuristic(X_train)
    h2_S = median_heuristic(S_train)

    K_Z_tt = rbf_kernel(Z_train, h2=h2_Z)
    K_X_tt = rbf_kernel(X_train, h2=h2_X)
    K_S_tt = rbf_kernel(S_train, h2=h2_S)
    K_tt = (K_Z_tt + K_X_tt + K_S_tt) / 3.0

    K_Z_ot = rbf_kernel(Z_other, h2=h2_Z, Y=Z_train)
    K_X_ot = rbf_kernel(X_other, h2=h2_X, Y=X_train)
    K_S_ot = rbf_kernel(S_other, h2=h2_S, Y=S_train)
    K_ot = (K_Z_ot + K_X_ot + K_S_ot) / 3.0

    return K_tt, K_ot


def krr_fit_predict(K_train, y_train, K_cross, reg):
    N = K_train.shape[0]
    alpha = np.linalg.solve(K_train + N * reg * np.eye(N), y_train)
    return K_cross @ alpha


def sharpe(pnl, periods_per_year=365.25 / 5):
    if len(pnl) < 2 or pnl.std() == 0:
        return float("nan")
    return float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year))


def main():
    logger.info("Loading cached events...")
    df = load_events()
    df = stage_price_features(df)
    df = stage_abnormal_returns(df)
    events_df, X_all, Z_all, Y_all = stage_align(df)
    logger.info("Aligned: N=%d", len(Y_all))

    merged = match_surprise_to_events(events_df, cache_dir=SURPRISE_CACHE_DIR)
    mask = merged["surprise_pct"].notna().values
    logger.info("Surprise-matched: %d / %d events retained", mask.sum(), len(mask))

    events_df = merged[mask].reset_index(drop=True)
    X_all = X_all[mask]
    Z_all = Z_all[mask]
    Y_all = Y_all[mask]
    S_all = events_df["surprise_pct"].values.reshape(-1, 1)

    train_mask = (events_df["date"] <= TRAIN_END).values
    test_mask = ~train_mask
    logger.info("Train: %d events, Test: %d events", train_mask.sum(), test_mask.sum())

    Z_tr, X_tr, S_tr, Y_tr = Z_all[train_mask], X_all[train_mask], S_all[train_mask], Y_all[train_mask]
    Z_te, X_te, S_te, Y_te = Z_all[test_mask], X_all[test_mask], S_all[test_mask], Y_all[test_mask]

    # Standardize X, S using TRAIN statistics only (Z embeddings left as-is,
    # already roughly unit-scale from the sentence-transformer).
    x_mean, x_std = X_tr.mean(0), X_tr.std(0) + 1e-8
    s_mean, s_std = S_tr.mean(0), S_tr.std(0) + 1e-8
    X_tr_s, X_te_s = (X_tr - x_mean) / x_std, (X_te - x_mean) / x_std
    S_tr_s, S_te_s = (S_tr - s_mean) / s_std, (S_te - s_mean) / s_std

    K_tt, K_te_tr = block_kernels(Z_tr, X_tr_s, S_tr_s, Z_te, X_te_s, S_te_s)

    # --- Magnitude model: predict |Y| ---
    mag_pred = krr_fit_predict(K_tt, np.abs(Y_tr), K_te_tr, KRR_REG)

    # --- Direction model: predict raw Y (sign of prediction = trade direction) ---
    dir_pred = krr_fit_predict(K_tt, Y_tr, K_te_tr, KRR_REG)

    n_test = len(Y_te)
    n_keep = max(1, int(TOP_FRACTION * n_test))
    high_conv_idx = np.argsort(-mag_pred)[:n_keep]

    def evaluate(idx, label):
        pos = np.sign(dir_pred[idx])
        pos[pos == 0] = 1.0
        gross_ret = pos * Y_te[idx]
        cost = TC_BPS * 2 * 1e-4   # round-trip: entry + exit
        net_ret = gross_ret - cost
        hit_rate = float((np.sign(Y_te[idx]) == pos).mean())
        print(f"\n=== {label} (N={len(idx)}) ===")
        print(f"Hit rate         = {hit_rate:.3f}")
        print(f"Mean net P&L     = {net_ret.mean() * 1e4:.2f} bps/event")
        print(f"Net Sharpe (ann) = {sharpe(net_ret):.3f}")
        print(f"Gross Sharpe     = {sharpe(gross_ret):.3f}")
        return net_ret

    print(f"\nTrain: {train_mask.sum()} events (<= {TRAIN_END})")
    print(f"Test:  {test_mask.sum()} events (> {TRAIN_END})")
    print(f"High-conviction subset: top {TOP_FRACTION:.0%} by predicted magnitude = {n_keep} events")

    evaluate(np.arange(n_test), "Baseline: trade ALL test events")
    evaluate(high_conv_idx, "High-conviction: magnitude-filtered subset")


if __name__ == "__main__":
    main()
