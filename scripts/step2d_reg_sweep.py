"""
Step 2d: is the in-sample/out-of-sample gap (seen in step2c) fixable with
more regularization, or does out-of-sample correlation stay at/below
zero no matter how hard the model is constrained?

If some lambda produces positive, stable out-of-sample correlation while
shrinking the (suspiciously high) in-sample correlation, the earlier
failure was just under-regularization -- fixable.
If out-of-sample correlation never becomes reliably positive at ANY
lambda, the in-sample fit was noise-fitting, not an extractable signal,
regardless of model complexity -- not fixable by regularizing harder.

Uses the LOW-CAPACITY (X + surprise, 4 dims) feature set only, since
step2c already showed it fails identically to the 384-dim model --
cheaper to sweep and the conclusion transfers.

Loads from dataset_cache/surprise_matched_150.npz (built once by
build_dataset_cache_chunked.py) instead of re-fetching from yfinance.
"""
import logging
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("step2d")

from dependence.kernel_cca import median_heuristic, rbf_kernel

CACHE_PATH = Path("dataset_cache/surprise_matched_150.npz")
TRAIN_YEARS, TEST_YEARS, STEP_MONTHS = 2.0, 1.0, 6
MIN_TRAIN, MIN_TEST = 50, 15
REG_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def krr_fit_predict(K_train, y_train, K_cross_train, K_cross_test, reg):
    N = K_train.shape[0]
    alpha = np.linalg.solve(K_train + N * reg * np.eye(N), y_train)
    return K_cross_train @ alpha, K_cross_test @ alpha


def corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    logger.info("Loading cached dataset from %s ...", CACHE_PATH)
    d = np.load(CACHE_PATH, allow_pickle=True)
    X_all, Y_all, S_all, dates = d["X_all"], d["Y_all"], d["S_all"], d["date"]
    logger.info("Loaded N=%d events", len(Y_all))

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

        X_tr, S_tr, Y_tr = X_all[train_mask], S_all[train_mask], Y_all[train_mask]
        X_te, S_te, Y_te = X_all[test_mask], S_all[test_mask], Y_all[test_mask]

        x_mean, x_std = X_tr.mean(0), X_tr.std(0) + 1e-8
        s_mean, s_std = S_tr.mean(0), S_tr.std(0) + 1e-8
        X_tr_s, X_te_s = (X_tr - x_mean) / x_std, (X_te - x_mean) / x_std
        S_tr_s, S_te_s = (S_tr - s_mean) / s_std, (S_te - s_mean) / s_std

        h2_X, h2_S = median_heuristic(X_tr_s), median_heuristic(S_tr_s)
        K_tt = (rbf_kernel(X_tr_s, h2=h2_X) + rbf_kernel(S_tr_s, h2=h2_S)) / 2.0
        K_tetr = (rbf_kernel(X_te_s, h2=h2_X, Y=X_tr_s) + rbf_kernel(S_te_s, h2=h2_S, Y=S_tr_s)) / 2.0

        print(f"\n{'='*78}\nFOLD {fold_idx}: train n={n_tr} [{train_start.date()}-{train_end.date()}), "
              f"test n={n_te} [{test_start.date()}-{test_end.date()})\n{'='*78}")
        print(f"{'lambda':>10} | {'in-sample corr':>16} | {'out-of-sample corr':>20}")
        for reg in REG_GRID:
            in_pred, out_pred = krr_fit_predict(K_tt, Y_tr, K_tt, K_tetr, reg)
            print(f"{reg:>10.4g} | {corr(in_pred, Y_tr):>16.3f} | {corr(out_pred, Y_te):>20.3f}")


if __name__ == "__main__":
    main()
