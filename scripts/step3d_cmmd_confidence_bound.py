"""
Step 3d: numerically instantiate the Gap-2 confidence bound from
paper/bound_sketch.md Section 6 (eq. 3'), using the exact Theorem 4.3/4.4
machinery of Massiani, Fiedler, Haverbeck, Solowjow & Trimpe,
"Kernel conditional tests from learning-theoretic bounds" (arXiv:2506.03898v2).

For scalar-output KRR (our case: Y is scalar, so the vector-valued output
space G = R), their general operator-valued bound reduces to a closed form
in terms of the eigenvalues of the training Gram matrix. Derivation summary
(see commit message / conversation, not re-derived in full here):

  Let K be the n x n training Gram matrix, {mu_i} its eigenvalues, lambda the
  ridge regularization constant actually used in the fit (alpha = (K+lambda I)^-1 Y,
  so lambda = LAMBDA * n under this repo's convention), sigma^2 an assumed/estimated
  subgaussian variance proxy of the residual Y - f(X), and S an assumed/estimated
  RKHS-norm bound on the true regression function. Then, writing r_i = mu_i/(mu_i+lambda):

      Tr(T)        = sigma^2 * sum_i r_i
      ||T||_2 (HS) = sigma^2 * sqrt(sum_i r_i^2)
      ||T||_op     = sigma^2 * max_i r_i

      beta_lambda(D, delta) = S + (1/sqrt(lambda)) * sqrt(
          Tr(T) + 2*sqrt(ln(1/delta))*||T||_2 + 2*ln(1/delta)*||T||_op
      )

  and the pointwise confidence width is B(D, x) = beta_lambda(D, delta) * sigma_{D,lambda}(x),
  with sigma_{D,lambda}(x) = sqrt(max(0, k(x,x) - k_x^T (K+lambda I)^-1 k_x)) the usual
  KRR/GP posterior standard deviation.

  Because Theorem 4.3's coverage is uniform in x over a region S (not pointwise-only),
  squaring and Q_X-averaging CMMD_linear(x) <= B_1(x) + B_2(x) transfers directly to
  a high-probability bound on the population quantity B = E_{Q_X}[CMMD_linear(x)^2]:

      B <= E_{Q_X}[(B_1(x) + B_2(x))^2]   w.p. >= 1 - (delta_1 + delta_2).

This script computes both sides for the same year-pairs as step3c, using the query
sample as a Monte Carlo proxy for Q_X. S and sigma^2 are estimated in-sample from the
fitted KRR model itself (the fitted function's own RKHS norm and residual variance) --
this is a plug-in approximation, not a rigorous a priori bound, and is flagged as such;
see the "assumption-checking" items in paper/bound_sketch.md Section 6.

Loads from dataset_cache/surprise_matched_150.npz -- no network needed.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("step3d")

from dependence.kernel_cca import median_heuristic, rbf_kernel

CACHE_PATH = Path("dataset_cache/surprise_matched_150.npz")
LAMBDA = 1e-2          # same ridge convention as step3c: lambda_reg = LAMBDA * n
N_QUERY = 100
DELTA_PER_SIDE = 0.05  # overall level = 2 * DELTA_PER_SIDE


def fit_krr(X, Y, h2, lam_reg):
    K = rbf_kernel(X, h2=h2)
    n = len(Y)
    alpha = np.linalg.solve(K + lam_reg * np.eye(n), Y)
    resid = Y - K @ alpha
    sigma2 = float(np.mean(resid ** 2))
    S = float(np.sqrt(max(alpha @ K @ alpha, 0.0)))
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.clip(eigvals, 0.0, None)
    return dict(X=X, alpha=alpha, K=K, h2=h2, lam_reg=lam_reg, sigma2=sigma2, S=S, eigvals=eigvals)


def beta_lambda(fit, delta):
    lam, sigma2, S, eig = fit["lam_reg"], fit["sigma2"], fit["S"], fit["eigvals"]
    r = eig / (eig + lam)
    tr_T = sigma2 * r.sum()
    hs_T = sigma2 * np.sqrt((r ** 2).sum())
    op_T = sigma2 * r.max()
    inner = tr_T + 2 * np.sqrt(np.log(1 / delta)) * hs_T + 2 * np.log(1 / delta) * op_T
    return S + np.sqrt(max(inner, 0.0)) / np.sqrt(lam)


def predict_and_sigma(fit, query_X):
    X, alpha, h2, lam = fit["X"], fit["alpha"], fit["h2"], fit["lam_reg"]
    K_q = rbf_kernel(query_X, h2=h2, Y=X)
    f = K_q @ alpha
    K_train = fit["K"]
    A_inv = np.linalg.inv(K_train + lam * np.eye(len(X)))
    k_xx = 1.0  # rbf_kernel(x,x) = 1 for our standardized RBF with h2 from median heuristic
    var = k_xx - np.einsum("qi,ij,qj->q", K_q, A_inv, K_q)
    sigma = np.sqrt(np.clip(var, 0.0, None))
    return f, sigma


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

    print(f"\n{'='*78}")
    print("Gap-2 confidence bound (Massiani et al. Thm 4.3/4.4), vs plug-in point estimate")
    print(f"delta per side = {DELTA_PER_SIDE}, overall level = {2*DELTA_PER_SIDE:.2f}")
    print(f"{'='*78}")
    header = f"{'pair':<14}{'n1':>5}{'n2':>5}{'S1':>8}{'S2':>8}{'beta1':>9}{'beta2':>9}{'E[CMMD^2]':>12}{'bound':>12}{'ratio':>8}"
    print(header)

    for y1, y2 in zip(years[:-1], years[1:]):
        idx1, idx2 = np.where(masks[y1])[0], np.where(masks[y2])[0]
        X1, Y1 = cov_std[idx1], Y_all[idx1]
        X2, Y2 = cov_std[idx2], Y_all[idx2]
        n1, n2 = len(idx1), len(idx2)

        fit1 = fit_krr(X1, Y1, h2_cov, LAMBDA * n1)
        fit2 = fit_krr(X2, Y2, h2_cov, LAMBDA * n2)
        beta1 = beta_lambda(fit1, DELTA_PER_SIDE)
        beta2 = beta_lambda(fit2, DELTA_PER_SIDE)

        pool = np.vstack([X1, X2])
        query = pool[rng.choice(len(pool), size=min(N_QUERY, len(pool)), replace=False)]

        f1, sig1 = predict_and_sigma(fit1, query)
        f2, sig2 = predict_and_sigma(fit2, query)

        cmmd_sq = (f1 - f2) ** 2
        point_est = float(cmmd_sq.mean())

        B1 = beta1 * sig1
        B2 = beta2 * sig2
        bound_rhs = float(np.mean((B1 + B2) ** 2))

        ratio = bound_rhs / point_est if point_est > 0 else float("nan")
        print(f"{y1}v{y2:<10}{n1:>5}{n2:>5}{fit1['S']:>8.3f}{fit2['S']:>8.3f}"
              f"{beta1:>9.3f}{beta2:>9.3f}{point_est:>12.6f}{bound_rhs:>12.6f}{ratio:>8.1f}x")

    print(f"\n{'='*78}")
    print("Note: S_i, sigma_i^2 estimated in-sample from the fitted KRR model itself")
    print("(RKHS norm of the fit, residual variance) -- a plug-in approximation, not a")
    print("rigorous a priori bound. See bound_sketch.md Section 6, assumption-checking items.")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
