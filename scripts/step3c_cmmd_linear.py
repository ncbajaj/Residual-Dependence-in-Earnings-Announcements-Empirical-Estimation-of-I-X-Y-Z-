"""
Step 3c: the bound-matching version of step3b. With a LINEAR output
kernel (kappa(y,y')=yy'), the CMMD construction collapses exactly to
    CMMD_linear(x) = |f1(x) - f2(x)|
where f_i is the KRR regression of Y on covariates, fit on dataset i.
This is the quantity that appears exactly (via (3) in
paper/bound_sketch.md) in the concept-drift term B = E_{Q_X}[CMMD^2] of
the proposed combined bound -- not just a general drift diagnostic, but
the specific plug-in estimator the bound needs.

Same permutation-null calibration as step3b (fixed version): pool two
groups, reshuffle group assignment many times, compare the true
grouping's mean CMMD_linear against that null.

Loads from dataset_cache/surprise_matched_150.npz -- no network needed.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("step3c")

from dependence.kernel_cca import median_heuristic, rbf_kernel

CACHE_PATH = Path("dataset_cache/surprise_matched_150.npz")
LAMBDA = 1e-2
N_QUERY = 100
N_PERM = 60


def krr_predict(X_train, Y_train, X_query, h2, lam):
    K = rbf_kernel(X_train, h2=h2)
    n = len(Y_train)
    alpha = np.linalg.solve(K + lam * n * np.eye(n), Y_train)
    K_q = rbf_kernel(X_query, h2=h2, Y=X_train)
    return K_q @ alpha


def cmmd_linear_mean(X1, Y1, X2, Y2, h2_cov, lam, query_X):
    f1 = krr_predict(X1, Y1, query_X, h2_cov, lam)
    f2 = krr_predict(X2, Y2, query_X, h2_cov, lam)
    return float(np.abs(f1 - f2).mean())


def calibrated_test(cov_pool, Y_pool, n1, h2_cov, query_X, rng, label):
    N = len(Y_pool)
    observed = cmmd_linear_mean(cov_pool[:n1], Y_pool[:n1], cov_pool[n1:], Y_pool[n1:], h2_cov, LAMBDA, query_X)
    null_vals = np.empty(N_PERM)
    for i in range(N_PERM):
        perm = rng.permutation(N)
        Xp, Yp = cov_pool[perm], Y_pool[perm]
        null_vals[i] = cmmd_linear_mean(Xp[:n1], Yp[:n1], Xp[n1:], Yp[n1:], h2_cov, LAMBDA, query_X)
    p = float(np.mean(null_vals >= observed))
    z = (observed - null_vals.mean()) / null_vals.std() if null_vals.std() > 0 else float("nan")
    print(f"  [{label}] observed={observed:.5f}  null={null_vals.mean():.5f}+/-{null_vals.std():.5f}  "
          f"z={z:.2f}  p={p:.3f}")


def main():
    logger.info("Loading cached dataset from %s ...", CACHE_PATH)
    d = np.load(CACHE_PATH, allow_pickle=True)
    X_all, S_all, Y_all, dates = d["X_all"], d["S_all"], d["Y_all"], d["date"]
    dates_ts = pd.DatetimeIndex(dates)
    cov_all = np.column_stack([X_all, S_all])
    cov_std = (cov_all - cov_all.mean(0)) / (cov_all.std(0) + 1e-8)
    h2_cov = median_heuristic(cov_std)
    rng = np.random.default_rng(0)

    years = [2020, 2021, 2022, 2023]
    masks = {y: (dates_ts.year == y) for y in years}

    print(f"\n{'='*70}\nLinear-kernel CMMD (bound-matching |f1(x)-f2(x)|): consecutive years\n{'='*70}")
    for y1, y2 in zip(years[:-1], years[1:]):
        idx1, idx2 = np.where(masks[y1])[0], np.where(masks[y2])[0]
        cov_pool = np.vstack([cov_std[idx1], cov_std[idx2]])
        Y_pool = np.concatenate([Y_all[idx1], Y_all[idx2]])
        query = cov_pool[rng.choice(len(cov_pool), size=min(N_QUERY, len(cov_pool)), replace=False)]
        calibrated_test(cov_pool, Y_pool, len(idx1), h2_cov, query, rng, f"{y1} vs {y2}")

    print(f"\n{'='*70}\nLinear-kernel CMMD: same-year random split (expect NOT significant)\n{'='*70}")
    for y in years:
        idx = np.where(masks[y])[0].copy()
        rng.shuffle(idx)
        cov_pool, Y_pool = cov_std[idx], Y_all[idx]
        n1 = len(idx) // 2
        query = cov_pool[rng.choice(len(cov_pool), size=min(N_QUERY, len(cov_pool)), replace=False)]
        calibrated_test(cov_pool, Y_pool, n1, h2_cov, query, rng, f"{y} split A vs B")


if __name__ == "__main__":
    main()
