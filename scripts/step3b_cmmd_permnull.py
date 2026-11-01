"""
Step 3b: step3a showed the raw CMMD point estimate has a finite-sample
noise floor comparable to the real between-year signal -- a same-year
random-split "null" case produced CMMD values as large as genuine
year-over-year comparisons. This calibrates CMMD properly with a
permutation null (same approach used earlier for the I(Z;Y) tests):
pool the two groups, repeatedly reshuffle which points go to "group 1"
vs "group 2" (preserving group sizes), recompute CMMD each time, and
check whether the REAL grouping's CMMD falls outside that null
distribution.

If calibrated CMMD is significant for year-over-year comparisons but
NOT for a same-year random split, the statistic is validated as a
usable drift signal and tracks what the Chow test already found
directly. If not, the naive estimator needs more work (e.g. a bias
correction, different lambda/bandwidth, or the doubly-robust estimator
mentioned in the source paper) before it's worth building a theorem
around.

Loads from dataset_cache/surprise_matched_150.npz -- no network needed.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("step3b")

from dependence.kernel_cca import median_heuristic, rbf_kernel

CACHE_PATH = Path("dataset_cache/surprise_matched_150.npz")
LAMBDA = 1e-2
N_QUERY = 100
N_PERM = 60


def cmmd_mean(X1, Y1, X2, Y2, h2_cov, h2_out, lam, query_X):
    K1 = rbf_kernel(X1, h2=h2_cov)
    K2 = rbf_kernel(X2, h2=h2_cov)
    n1, n2 = len(Y1), len(Y2)
    A1_inv = np.linalg.inv(K1 + lam * n1 * np.eye(n1))
    A2_inv = np.linalg.inv(K2 + lam * n2 * np.eye(n2))

    C11 = rbf_kernel(Y1.reshape(-1, 1), h2=h2_out)
    C22 = rbf_kernel(Y2.reshape(-1, 1), h2=h2_out)
    C12 = rbf_kernel(Y1.reshape(-1, 1), h2=h2_out, Y=Y2.reshape(-1, 1))

    a1 = rbf_kernel(query_X, h2=h2_cov, Y=X1) @ A1_inv
    a2 = rbf_kernel(query_X, h2=h2_cov, Y=X2) @ A2_inv

    term1 = np.einsum("mi,ij,mj->m", a1, C11, a1)
    term2 = np.einsum("mi,ij,mj->m", a1, C12, a2)
    term3 = np.einsum("mi,ij,mj->m", a2, C22, a2)
    cmmd_sq = np.clip(term1 - 2 * term2 + term3, 0, None)
    return float(np.sqrt(cmmd_sq).mean())


def calibrated_test(cov_pool, Y_pool, n1, h2_cov, h2_out, query_X, rng, label):
    """Observed CMMD for the TRUE first-n1/rest split, vs a permutation null."""
    N = len(Y_pool)
    X1, Y1 = cov_pool[:n1], Y_pool[:n1]
    X2, Y2 = cov_pool[n1:], Y_pool[n1:]
    observed = cmmd_mean(X1, Y1, X2, Y2, h2_cov, h2_out, LAMBDA, query_X)

    null_vals = np.empty(N_PERM)
    for i in range(N_PERM):
        perm = rng.permutation(N)
        Xp, Yp = cov_pool[perm], Y_pool[perm]
        null_vals[i] = cmmd_mean(Xp[:n1], Yp[:n1], Xp[n1:], Yp[n1:], h2_cov, h2_out, LAMBDA, query_X)

    p = float(np.mean(null_vals >= observed))
    z = (observed - null_vals.mean()) / null_vals.std() if null_vals.std() > 0 else float("nan")
    print(f"  [{label}] observed={observed:.5f}  null={null_vals.mean():.5f}+/-{null_vals.std():.5f}  "
          f"z={z:.2f}  p={p:.3f}")
    return observed, null_vals, p


def main():
    logger.info("Loading cached dataset from %s ...", CACHE_PATH)
    d = np.load(CACHE_PATH, allow_pickle=True)
    X_all, S_all, Y_all, dates = d["X_all"], d["S_all"], d["Y_all"], d["date"]
    dates_ts = pd.DatetimeIndex(dates)
    cov_all = np.column_stack([X_all, S_all])
    cov_std = (cov_all - cov_all.mean(0)) / (cov_all.std(0) + 1e-8)

    h2_cov = median_heuristic(cov_std)
    h2_out = median_heuristic(Y_all.reshape(-1, 1))
    rng = np.random.default_rng(0)

    years = [2020, 2021, 2022, 2023]
    masks = {y: (dates_ts.year == y) for y in years}

    print(f"\n{'='*70}\nCalibrated CMMD: consecutive year pairs (permutation null, n_perm={N_PERM})\n{'='*70}")
    for y1, y2 in zip(years[:-1], years[1:]):
        idx1, idx2 = np.where(masks[y1])[0], np.where(masks[y2])[0]
        cov_pool = np.vstack([cov_std[idx1], cov_std[idx2]])
        Y_pool = np.concatenate([Y_all[idx1], Y_all[idx2]])
        query = cov_pool[rng.choice(len(cov_pool), size=min(N_QUERY, len(cov_pool)), replace=False)]
        calibrated_test(cov_pool, Y_pool, len(idx1), h2_cov, h2_out, query, rng, f"{y1} vs {y2}")

    print(f"\n{'='*70}\nCalibrated CMMD: same-year random split (expect NOT significant)\n{'='*70}")
    for y in years:
        idx = np.where(masks[y])[0].copy()
        rng.shuffle(idx)  # BUG FIX: dataset row order is ticker-grouped (from
        # chunked fetching), so splitting by raw position without shuffling
        # first compares different SETS OF COMPANIES, not a random split --
        # that confound, not drift, was what the earlier unshuffled version
        # was actually detecting. Must shuffle before cutting.
        cov_pool, Y_pool = cov_std[idx], Y_all[idx]
        n1 = len(idx) // 2
        query = cov_pool[rng.choice(len(cov_pool), size=min(N_QUERY, len(cov_pool)), replace=False)]
        calibrated_test(cov_pool, Y_pool, n1, h2_cov, h2_out, query, rng, f"{y} split A vs B")


if __name__ == "__main__":
    main()
