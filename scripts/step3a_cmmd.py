"""
Step 3a: a first empirical check for the ICML direction (extending the
zero-shot bound with a concept-drift term).

Implements a kernel Conditional MMD (CMMD) statistic -- a kernel
two-sample test for whether P(Y|X) differs between two datasets, as
opposed to a marginal (covariate-shift) MMD which only tests P(X).
Formula (pointwise at a query covariate x), following the
KRR-witness-function construction:

    a_i(x) = (K_i + lambda*I)^{-1} k_i(x)           for dataset i=1,2
    CMMD^2(x) = a1^T C11 a1 - 2 a1^T C12 a2 + a2^T C22 a2

where K_i is the covariate (X,S) kernel matrix within dataset i, k_i(x)
is the vector of covariate-kernel evaluations between query point x and
dataset i's covariates, and C_ij is the output (Y) kernel matrix between
datasets i and j. This is the same "witness function norm-difference"
construction as an unconditional MMD, but projected through a KRR
regression on covariates first -- so it measures how differently Y
behaves, given the SAME covariate value x, across the two datasets.

We don't yet implement the formal finite-sample confidence bound /
hypothesis test from the source paper -- this is a first check: does the
point-estimate CMMD, averaged over query points, track the drift we
already found directly with the Chow test? If it lights up on the same
year-pairs that showed large coefficient changes, it's a validated
proxy worth building the theorem around. If not, better to find out now.

Loads from dataset_cache/surprise_matched_150.npz -- no network needed.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("step3a")

from dependence.kernel_cca import median_heuristic, rbf_kernel

CACHE_PATH = Path("dataset_cache/surprise_matched_150.npz")
LAMBDA = 1e-2
N_QUERY = 150  # subsample of query points per pair, for speed


def cmmd(X1, Y1, X2, Y2, h2_cov, h2_out, lam, query_X, rng):
    """
    Returns (mean_cmmd, cmmd_values_per_query).
    X1, X2: (n1,d), (n2,d) covariates. Y1, Y2: (n1,), (n2,) outputs.
    query_X: (m, d) covariate points to evaluate CMMD at.
    """
    K1 = rbf_kernel(X1, h2=h2_cov)
    K2 = rbf_kernel(X2, h2=h2_cov)
    n1, n2 = len(Y1), len(Y2)
    A1 = K1 + lam * n1 * np.eye(n1)
    A2 = K2 + lam * n2 * np.eye(n2)
    A1_inv = np.linalg.inv(A1)
    A2_inv = np.linalg.inv(A2)

    C11 = rbf_kernel(Y1.reshape(-1, 1), h2=h2_out)
    C22 = rbf_kernel(Y2.reshape(-1, 1), h2=h2_out)
    C12 = rbf_kernel(Y1.reshape(-1, 1), h2=h2_out, Y=Y2.reshape(-1, 1))

    k1_q = rbf_kernel(query_X, h2=h2_cov, Y=X1)  # (m, n1)
    k2_q = rbf_kernel(query_X, h2=h2_cov, Y=X2)  # (m, n2)

    a1 = k1_q @ A1_inv   # (m, n1)
    a2 = k2_q @ A2_inv   # (m, n2)

    term1 = np.einsum("mi,ij,mj->m", a1, C11, a1)
    term2 = np.einsum("mi,ij,mj->m", a1, C12, a2)
    term3 = np.einsum("mi,ij,mj->m", a2, C22, a2)
    cmmd_sq = np.clip(term1 - 2 * term2 + term3, 0, None)
    vals = np.sqrt(cmmd_sq)
    return float(vals.mean()), vals


def main():
    logger.info("Loading cached dataset from %s ...", CACHE_PATH)
    d = np.load(CACHE_PATH, allow_pickle=True)
    X_all, S_all, Y_all, dates = d["X_all"], d["S_all"], d["Y_all"], d["date"]
    dates_ts = pd.DatetimeIndex(dates)
    cov_all = np.column_stack([X_all, S_all])
    cov_std = (cov_all - cov_all.mean(0)) / (cov_all.std(0) + 1e-8)

    years = [2020, 2021, 2022, 2023]
    masks = {y: (dates_ts.year == y) for y in years}
    for y in years:
        logger.info("Year %d: n=%d", y, masks[y].sum())

    h2_cov = median_heuristic(cov_std)
    h2_out = median_heuristic(Y_all.reshape(-1, 1))
    rng = np.random.default_rng(0)

    print(f"\n{'='*60}\nCMMD between consecutive calendar years (covariates: X + surprise)\n{'='*60}")
    for y1, y2 in zip(years[:-1], years[1:]):
        X1, Y1 = cov_std[masks[y1]], Y_all[masks[y1]]
        X2, Y2 = cov_std[masks[y2]], Y_all[masks[y2]]
        pooled_idx = rng.choice(len(X1) + len(X2), size=min(N_QUERY, len(X1) + len(X2)), replace=False)
        query = np.vstack([X1, X2])[pooled_idx]
        mean_cmmd, _ = cmmd(X1, Y1, X2, Y2, h2_cov, h2_out, LAMBDA, query, rng)
        print(f"  {y1} vs {y2}: mean CMMD = {mean_cmmd:.5f}  (n1={len(Y1)}, n2={len(Y2)})")

    print(f"\n{'='*60}\nCMMD, each year vs. pooled 'other three years' (one-vs-rest)\n{'='*60}")
    for y in years:
        X1, Y1 = cov_std[masks[y]], Y_all[masks[y]]
        other_mask = ~masks[y]
        X2, Y2 = cov_std[other_mask], Y_all[other_mask]
        pooled_idx = rng.choice(len(X1) + len(X2), size=min(N_QUERY, len(X1) + len(X2)), replace=False)
        query = np.vstack([X1, X2])[pooled_idx]
        mean_cmmd, _ = cmmd(X1, Y1, X2, Y2, h2_cov, h2_out, LAMBDA, query, rng)
        print(f"  {y} vs rest: mean CMMD = {mean_cmmd:.5f}  (n1={len(Y1)}, n2={len(Y2)})")

    # --- Baseline sanity check: CMMD between two RANDOM halves of the SAME
    # year (should be small/noise-level if the statistic is well-behaved,
    # since there's no real drift within a random split of one year) ---
    print(f"\n{'='*60}\nSanity check: CMMD between two random halves of the SAME year (expect ~noise level)\n{'='*60}")
    for y in years:
        idx = np.where(masks[y])[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        i1, i2 = idx[:half], idx[half:]
        X1, Y1 = cov_std[i1], Y_all[i1]
        X2, Y2 = cov_std[i2], Y_all[i2]
        query = np.vstack([X1, X2])[rng.choice(len(X1) + len(X2), size=min(N_QUERY, len(X1) + len(X2)), replace=False)]
        mean_cmmd, _ = cmmd(X1, Y1, X2, Y2, h2_cov, h2_out, LAMBDA, query, rng)
        print(f"  {y} split A vs split B: mean CMMD = {mean_cmmd:.5f}")


if __name__ == "__main__":
    main()
