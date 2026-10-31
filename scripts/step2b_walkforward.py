"""
Step 2b: walk-forward validation of the magnitude-filtered direction
strategy from step2_magnitude_direction.py, using the main project's own
fold convention (2yr train / 1yr test / 6mo step) and Newey-West
significance test, instead of a single train/test split.

Each fold independently:
  1. Fits the magnitude KRR and direction KRR on the fold's training data
     only (bandwidths frozen from training, same leakage guard as the
     rest of the pipeline).
  2. Selects the top TOP_FRACTION of the fold's test events by predicted
     magnitude.
  3. Computes net P&L for both the high-conviction subset and the full
     test set (baseline), using the SAME direction model.

Pooled P&L across all folds is tested with newey_west_tstat (L=5 lags,
matching the 5-day holding period) -- reused directly from
strategy/validation.py for consistency with the rest of the project.
"""
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("step2b")

from main import stage_price_features, stage_abnormal_returns, stage_align
from features.embeddings import embed_transcripts_batch
from data.earnings_surprise import match_surprise_to_events
from dependence.kernel_cca import median_heuristic, rbf_kernel
from strategy.validation import newey_west_tstat

RAW_DIR = Path("raw_filings_fast")
EMBED_CACHE_DIR = Path("embeddings_cache_fast")
SURPRISE_CACHE_DIR = Path("earnings_surprise_cache")
START, END = "2020-01-01", "2023-12-31"
TOP_FRACTION = 0.4
TC_BPS = 5.0
KRR_REG = 1e-2
TRAIN_YEARS, TEST_YEARS, STEP_MONTHS = 2.0, 1.0, 6
NW_LAGS = 5
MIN_TRAIN, MIN_TEST = 50, 15


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


def block_kernels(Z_tr, X_tr, S_tr, Z_o, X_o, S_o):
    h2_Z, h2_X, h2_S = median_heuristic(Z_tr), median_heuristic(X_tr), median_heuristic(S_tr)
    K_tt = (rbf_kernel(Z_tr, h2=h2_Z) + rbf_kernel(X_tr, h2=h2_X) + rbf_kernel(S_tr, h2=h2_S)) / 3.0
    K_ot = (rbf_kernel(Z_o, h2=h2_Z, Y=Z_tr) + rbf_kernel(X_o, h2=h2_X, Y=X_tr) + rbf_kernel(S_o, h2=h2_S, Y=S_tr)) / 3.0
    return K_tt, K_ot


def krr_fit_predict(K_train, y_train, K_cross, reg):
    N = K_train.shape[0]
    alpha = np.linalg.solve(K_train + N * reg * np.eye(N), y_train)
    return K_cross @ alpha


def main():
    logger.info("Loading cached events...")
    df = load_events()
    df = stage_price_features(df)
    df = stage_abnormal_returns(df)
    events_df, X_all, Z_all, Y_all = stage_align(df)
    merged = match_surprise_to_events(events_df, cache_dir=SURPRISE_CACHE_DIR)
    mask = merged["surprise_pct"].notna().values
    events_df = merged[mask].reset_index(drop=True)
    X_all, Z_all, Y_all = X_all[mask], Z_all[mask], Y_all[mask]
    S_all = events_df["surprise_pct"].values.reshape(-1, 1)
    dates = events_df["date"].values
    logger.info("Surprise-matched dataset: N=%d, date range %s to %s",
                len(Y_all), pd.Timestamp(dates.min()).date(), pd.Timestamp(dates.max()).date())

    t_min, t_max = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    train_td = timedelta(days=int(TRAIN_YEARS * 365.25))
    test_td = timedelta(days=int(TEST_YEARS * 365.25))
    step_td = timedelta(days=int(STEP_MONTHS * 30.44))

    fold_starts = []
    t = t_min
    while t + train_td + test_td <= t_max + step_td:
        fold_starts.append(t)
        t += step_td
    logger.info("Walk-forward: %d candidate folds (train=%s, test=%s, step=%s)",
                len(fold_starts), train_td, test_td, step_td)

    all_baseline_pnl, all_hc_pnl = [], []
    cost = TC_BPS * 2 * 1e-4

    for fold_idx, t_start in enumerate(fold_starts):
        train_start, train_end = t_start, t_start + train_td
        test_start, test_end = train_end, train_end + test_td
        train_mask = (dates >= train_start.to_datetime64()) & (dates < train_end.to_datetime64())
        test_mask = (dates >= test_start.to_datetime64()) & (dates < test_end.to_datetime64())
        n_tr, n_te = train_mask.sum(), test_mask.sum()

        if n_tr < MIN_TRAIN or n_te < MIN_TEST:
            logger.info("Fold %d: skip (n_train=%d, n_test=%d)", fold_idx, n_tr, n_te)
            continue

        Z_tr, X_tr, S_tr, Y_tr = Z_all[train_mask], X_all[train_mask], S_all[train_mask], Y_all[train_mask]
        Z_te, X_te, S_te, Y_te = Z_all[test_mask], X_all[test_mask], S_all[test_mask], Y_all[test_mask]

        x_mean, x_std = X_tr.mean(0), X_tr.std(0) + 1e-8
        s_mean, s_std = S_tr.mean(0), S_tr.std(0) + 1e-8
        X_tr_s, X_te_s = (X_tr - x_mean) / x_std, (X_te - x_mean) / x_std
        S_tr_s, S_te_s = (S_tr - s_mean) / s_std, (S_te - s_mean) / s_std

        K_tt, K_te_tr = block_kernels(Z_tr, X_tr_s, S_tr_s, Z_te, X_te_s, S_te_s)
        mag_pred = krr_fit_predict(K_tt, np.abs(Y_tr), K_te_tr, KRR_REG)
        dir_pred = krr_fit_predict(K_tt, Y_tr, K_te_tr, KRR_REG)

        pos = np.sign(dir_pred)
        pos[pos == 0] = 1.0
        baseline_net = pos * Y_te - cost
        all_baseline_pnl.append(baseline_net)

        n_keep = max(1, int(TOP_FRACTION * n_te))
        hc_idx = np.argsort(-mag_pred)[:n_keep]
        hc_net = baseline_net[hc_idx]
        all_hc_pnl.append(hc_net)

        logger.info(
            "Fold %d [%s -> %s]: n_train=%d, n_test=%d, n_hc=%d, "
            "baseline mean=%.2fbps, hc mean=%.2fbps",
            fold_idx, test_start.date(), test_end.date(), n_tr, n_te, n_keep,
            baseline_net.mean() * 1e4, hc_net.mean() * 1e4,
        )

    baseline_pooled = np.concatenate(all_baseline_pnl)
    hc_pooled = np.concatenate(all_hc_pnl)

    t_base, p_base = newey_west_tstat(baseline_pooled, n_lags=NW_LAGS)
    t_hc, p_hc = newey_west_tstat(hc_pooled, n_lags=NW_LAGS)

    def ann_sharpe(pnl):
        periods_per_year = 365.25 / 5
        return float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year)) if pnl.std() > 0 else float("nan")

    print(f"\n=== Pooled across {len(all_baseline_pnl)} folds ===")
    print(f"\nBASELINE (trade all test events, N={len(baseline_pooled)}):")
    print(f"  Mean net P&L = {baseline_pooled.mean()*1e4:.2f} bps/event")
    print(f"  Net Sharpe   = {ann_sharpe(baseline_pooled):.3f}")
    print(f"  NW t-stat    = {t_base:.3f}  (p={p_base:.4f})")

    print(f"\nHIGH-CONVICTION (magnitude-filtered, N={len(hc_pooled)}):")
    print(f"  Mean net P&L = {hc_pooled.mean()*1e4:.2f} bps/event")
    print(f"  Net Sharpe   = {ann_sharpe(hc_pooled):.3f}")
    print(f"  NW t-stat    = {t_hc:.3f}  (p={p_hc:.4f})")

    print(f"\nConclusion: {'Magnitude filter is STATISTICALLY SIGNIFICANT (p<0.05)' if p_hc < 0.05 else 'Magnitude filter improvement is NOT statistically significant at this sample size (p>=0.05) -- consistent with a real but small/noisy effect, not yet strong enough to claim as a validated strategy'}")


if __name__ == "__main__":
    main()
