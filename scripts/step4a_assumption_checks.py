"""
Step 4a: empirically check the assumptions left open in paper/bound_sketch.md
(Gap 1b, and the Section 6 analogues) rather than just asserting them.

Four checks, each tied to a specific assumption in the sketch:

1. Sup-norm boundedness of eta_rho - eta_P (needed for the additive form of
   Lemma B, eq. 2). Since Y is empirically bounded (returns can't blow up
   arbitrarily), both eta_rho (a KRR fit) and the true eta_P are bounded by
   max|Y| in absolute value *if* the fit doesn't extrapolate outside the
   observed range. We check this directly: does the KRR fit's prediction
   ever exceed the empirical range of Y it was trained on?

2. Covariate support containment (needed for the multiplicative form, eq.
   2', and for Gap 2's requirement that supp(Q_X) subset of S). We check
   whether one year's standardized covariates fall inside the convex hull
   (approximated via per-dimension range, cheap and conservative) of
   another year's -- i.e. whether training data has excursions outside
   deployment data's support, or vice versa.

3. Subgaussianity of the residual distribution (needed for Theorem 4.3 in
   Massiani et al., which the Gap-2 argument relies on). We check excess
   kurtosis of the KRR residuals against the Gaussian benchmark (0) and
   against a few reference heavy-tailed distributions, as a necessary (not
   sufficient) sanity check -- true subgaussianity is a tail-decay
   property, not fully captured by one moment, but a wildly excess kurtosis
   would already rule it out.

4. Stability of the RKHS-norm plug-in estimate S = sqrt(alpha^T K alpha)
   used in step3d. If S changes wildly across random subsamples of the
   same year's data, it isn't a trustworthy stand-in for a fixed population
   quantity. We check via a simple subsample bootstrap.

Loads from dataset_cache/surprise_matched_150.npz -- no network needed.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("step4a")

from dependence.kernel_cca import median_heuristic, rbf_kernel

CACHE_PATH = Path("dataset_cache/surprise_matched_150.npz")
LAMBDA = 1e-2
N_BOOTSTRAP = 30
RNG = np.random.default_rng(0)


def fit_krr(X, Y, h2, lam_reg):
    K = rbf_kernel(X, h2=h2)
    n = len(Y)
    alpha = np.linalg.solve(K + lam_reg * np.eye(n), Y)
    return alpha, K


def main():
    logger.info("Loading cached dataset from %s ...", CACHE_PATH)
    d = np.load(CACHE_PATH, allow_pickle=True)
    X_all, S_all, Y_all, dates = d["X_all"], d["S_all"], d["Y_all"], d["date"]
    dates_ts = pd.DatetimeIndex(dates)
    cov_all = np.column_stack([X_all, S_all])
    cov_std = (cov_all - cov_all.mean(0)) / (cov_all.std(0) + 1e-8)
    h2_cov = median_heuristic(cov_std)

    years = [2020, 2021, 2022, 2023]
    masks = {y: (dates_ts.year == y) for y in years}

    print(f"\n{'='*78}\n1. Sup-norm boundedness check (Gap 1b, additive form)\n{'='*78}")
    y_abs_max_global = float(np.abs(Y_all).max())
    print(f"Global |Y| max (empirical bound on both eta_rho and eta_P): {y_abs_max_global:.4f}")
    for y in years:
        idx = np.where(masks[y])[0]
        X, Y = cov_std[idx], Y_all[idx]
        n = len(Y)
        alpha, K = fit_krr(X, Y, h2_cov, LAMBDA * n)
        f_train = K @ alpha
        pred_abs_max = float(np.abs(f_train).max())
        y_abs_max = float(np.abs(Y).max())
        ok = pred_abs_max <= y_abs_max * 1.5  # generous slack, flag only real blowups
        print(f"  {y}: n={n}  |Y|_max={y_abs_max:.4f}  |pred|_max={pred_abs_max:.4f}  "
              f"{'OK (no extrapolation blowup)' if ok else 'WARNING: prediction exceeds training range'}")

    print(f"\n{'='*78}\n2. Covariate support containment check (eq. 2' / Gap 2's S requirement)\n{'='*78}")
    print("Per-dimension [min,max] range overlap: fraction of each year's points whose")
    print("coordinates all fall within the POOLED range across all other years.")
    for y in years:
        idx_y = np.where(masks[y])[0]
        idx_other = np.where(~masks[y])[0]
        other_min = cov_std[idx_other].min(axis=0)
        other_max = cov_std[idx_other].max(axis=0)
        inside = np.all((cov_std[idx_y] >= other_min) & (cov_std[idx_y] <= other_max), axis=1)
        frac = float(inside.mean())
        print(f"  {y}: {frac*100:.1f}% of points fall within the other years' joint range")

    print(f"\n{'='*78}\n3. Residual subgaussianity check (Massiani et al. Thm 4.3 assumption)\n{'='*78}")
    for y in years:
        idx = np.where(masks[y])[0]
        X, Y = cov_std[idx], Y_all[idx]
        n = len(Y)
        alpha, K = fit_krr(X, Y, h2_cov, LAMBDA * n)
        resid = Y - K @ alpha
        resid_std = resid.std()
        kurt = float(((resid - resid.mean()) ** 4).mean() / resid_std ** 4 - 3)
        verdict = "mild, plausibly subgaussian" if kurt < 2 else ("moderate, borderline" if kurt < 5 else "heavy-tailed, subgaussianity doubtful")
        print(f"  {y}: n={n}  residual excess kurtosis={kurt:.3f}  ({verdict})")

    print(f"\n{'='*78}\n4. RKHS-norm (S) plug-in stability, subsample bootstrap\n{'='*78}")
    for y in years:
        idx = np.where(masks[y])[0]
        X, Y = cov_std[idx], Y_all[idx]
        n = len(Y)
        S_vals = []
        for _ in range(N_BOOTSTRAP):
            sub = RNG.choice(n, size=int(0.8 * n), replace=False)
            Xs, Ys = X[sub], Y[sub]
            ns = len(Ys)
            a, K = fit_krr(Xs, Ys, h2_cov, LAMBDA * ns)
            S_vals.append(float(np.sqrt(max(a @ K @ a, 0.0))))
        S_vals = np.array(S_vals)
        print(f"  {y}: S = {S_vals.mean():.4f} +/- {S_vals.std():.4f}  "
              f"(CV={S_vals.std()/S_vals.mean():.3f})")


if __name__ == "__main__":
    main()
