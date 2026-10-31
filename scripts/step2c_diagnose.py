"""
Step 2c: diagnose why the walk-forward magnitude/direction models failed.

For each fold, report:
  - In-sample (training) correlation between predicted and actual
    |Y| / Y -- tells us how well the model fits its own training data.
  - Out-of-sample (test) correlation between predicted and actual
    |Y| / Y -- tells us whether that fit generalizes at all, or even
    flips sign (anti-predictive), which would explain why the magnitude
    filter made things WORSE: it would be scaling up bets on a direction
    model that is actively wrong in that period.
  - Same diagnostics using ONLY X+surprise (4-dim, no 384-dim Z) to test
    whether the high-dimensional transcript embedding is the source of
    the overfitting, by comparing its generalization to a much
    lower-capacity model.
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
logger = logging.getLogger("step2c")

from main import stage_price_features, stage_abnormal_returns, stage_align
from features.embeddings import embed_transcripts_batch
from data.earnings_surprise import match_surprise_to_events
from dependence.kernel_cca import median_heuristic, rbf_kernel

RAW_DIR = Path("raw_filings_fast")
EMBED_CACHE_DIR = Path("embeddings_cache_fast")
SURPRISE_CACHE_DIR = Path("earnings_surprise_cache")
START, END = "2020-01-01", "2023-12-31"
KRR_REG = 1e-2
TRAIN_YEARS, TEST_YEARS, STEP_MONTHS = 2.0, 1.0, 6
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


def krr_fit_predict(K_train, y_train, K_cross_train, K_cross_test, reg):
    """Returns (in_sample_pred, out_of_sample_pred)."""
    N = K_train.shape[0]
    alpha = np.linalg.solve(K_train + N * reg * np.eye(N), y_train)
    return K_cross_train @ alpha, K_cross_test @ alpha


def corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


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

    t_min, t_max = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    train_td = timedelta(days=int(TRAIN_YEARS * 365.25))
    test_td = timedelta(days=int(TEST_YEARS * 365.25))
    step_td = timedelta(days=int(STEP_MONTHS * 30.44))

    fold_starts = []
    t = t_min
    while t + train_td + test_td <= t_max + step_td:
        fold_starts.append(t)
        t += step_td

    for fold_idx, t_start in enumerate(fold_starts):
        train_start, train_end = t_start, t_start + train_td
        test_start, test_end = train_end, train_end + test_td
        train_mask = (dates >= train_start.to_datetime64()) & (dates < train_end.to_datetime64())
        test_mask = (dates >= test_start.to_datetime64()) & (dates < test_end.to_datetime64())
        n_tr, n_te = train_mask.sum(), test_mask.sum()
        if n_tr < MIN_TRAIN or n_te < MIN_TEST:
            continue

        Z_tr, X_tr, S_tr, Y_tr = Z_all[train_mask], X_all[train_mask], S_all[train_mask], Y_all[train_mask]
        Z_te, X_te, S_te, Y_te = Z_all[test_mask], X_all[test_mask], S_all[test_mask], Y_all[test_mask]

        x_mean, x_std = X_tr.mean(0), X_tr.std(0) + 1e-8
        s_mean, s_std = S_tr.mean(0), S_tr.std(0) + 1e-8
        X_tr_s, X_te_s = (X_tr - x_mean) / x_std, (X_te - x_mean) / x_std
        S_tr_s, S_te_s = (S_tr - s_mean) / s_std, (S_te - s_mean) / s_std

        print(f"\n{'='*70}\nFOLD {fold_idx}: train [{train_start.date()} - {train_end.date()}) "
              f"n={n_tr}, test [{test_start.date()} - {test_end.date()}) n={n_te}\n{'='*70}")

        # --- FULL model: Z + X + S (the one that failed) ---
        h2_Z, h2_X, h2_S = median_heuristic(Z_tr), median_heuristic(X_tr_s), median_heuristic(S_tr_s)
        K_tt = (rbf_kernel(Z_tr, h2=h2_Z) + rbf_kernel(X_tr_s, h2=h2_X) + rbf_kernel(S_tr_s, h2=h2_S)) / 3.0
        K_trtr_cross = K_tt  # train-vs-train cross is just K_tt itself for in-sample pred
        K_tetr = (rbf_kernel(Z_te, h2=h2_Z, Y=Z_tr) + rbf_kernel(X_te_s, h2=h2_X, Y=X_tr_s) + rbf_kernel(S_te_s, h2=h2_S, Y=S_tr_s)) / 3.0

        dir_in, dir_out = krr_fit_predict(K_tt, Y_tr, K_trtr_cross, K_tetr, KRR_REG)
        mag_in, mag_out = krr_fit_predict(K_tt, np.abs(Y_tr), K_trtr_cross, K_tetr, KRR_REG)

        print("FULL model (Z+X+surprise, 384+3+1 dims):")
        print(f"  Direction: in-sample corr={corr(dir_in, Y_tr):.3f}   out-of-sample corr={corr(dir_out, Y_te):.3f}")
        print(f"  Magnitude: in-sample corr={corr(mag_in, np.abs(Y_tr)):.3f}   out-of-sample corr={corr(mag_out, np.abs(Y_te)):.3f}")

        # --- LOW-CAPACITY model: X + S only (4-dim, no embedding) ---
        h2_X2, h2_S2 = median_heuristic(X_tr_s), median_heuristic(S_tr_s)
        K_tt_lo = (rbf_kernel(X_tr_s, h2=h2_X2) + rbf_kernel(S_tr_s, h2=h2_S2)) / 2.0
        K_tetr_lo = (rbf_kernel(X_te_s, h2=h2_X2, Y=X_tr_s) + rbf_kernel(S_te_s, h2=h2_S2, Y=S_tr_s)) / 2.0

        dir_in_lo, dir_out_lo = krr_fit_predict(K_tt_lo, Y_tr, K_tt_lo, K_tetr_lo, KRR_REG)
        mag_in_lo, mag_out_lo = krr_fit_predict(K_tt_lo, np.abs(Y_tr), K_tt_lo, K_tetr_lo, KRR_REG)

        print("LOW-CAPACITY model (X+surprise only, 4 dims, no transcript):")
        print(f"  Direction: in-sample corr={corr(dir_in_lo, Y_tr):.3f}   out-of-sample corr={corr(dir_out_lo, Y_te):.3f}")
        print(f"  Magnitude: in-sample corr={corr(mag_in_lo, np.abs(Y_tr)):.3f}   out-of-sample corr={corr(mag_out_lo, np.abs(Y_te)):.3f}")


if __name__ == "__main__":
    main()
