"""
Residual dependence Î(X; Y | Z) — the core quantity from Theorem 1.

THEORETICAL BACKGROUND
-----------------------
Theorem 1 (Mehta & Harchaoui 2025, eq. 15):
    The performance gap between the direct predictor η⋆(x) = E[r(Y)|X=x] and
    the indirect (zero-shot) predictor η_ρ(x) = E[g_ρ(Z)|X=x] satisfies

        ||η⋆ - η_ρ||²_{L2(P_X)} ≲ E_{P_Z}[I(X; Y|Z)] + prompt bias.

    The quantity

        I(X; Y|Z) := E_{P_Z}[I(X; Y|z)],   I(X;Y|z) = E_{P_{X|z}P_{Y|z}}[(S_z-1)²]

    is the **residual dependence** — the χ²-divergence between the joint
    P_{X,Y|z} and the product P_{X|z}P_{Y|z}, averaged over z.

    A small I(X;Y|Z) means Z is a good proxy for predicting Y from X: the
    information in X about Y is well-mediated by Z. In our financial context:
    - Small I(X;Y|Z): price history adds little predictive power for abnormal
      returns beyond what the earnings call transcript already captures.
    - Large I(X;Y|Z): price features carry incremental information about
      post-announcement returns that transcripts don't explain.

CONDITIONAL MEAN APPROACH (Theorem 2, eq. 16):
    The indirect predictor η_ρ(x) = [M_{Z|X} g_ρ](x) uses the operator from
    Proposition 1 to map g_ρ(Z) = E[Y|Z] back into X-space.

    Empirically we estimate:
        ĝ_ρ(z_i) = KRR prediction of Y from Z  (proxy for E[Y|Z=z])
        η̂_ρ(x_i) = KRR prediction of ĝ_ρ(Z) from X  (proxy for E[E[Y|Z]|X])
        η̂⋆(x_i)  = KRR prediction of Y from X         (proxy for E[Y|X])

    Then:
        Î(X; Y|Z) ≈ (1/N) Σ_i [η̂⋆(x_i) - η̂_ρ(x_i)]²               (CM estimate)

    KRR predictions use leave-one-out (LOO) cross-validation to avoid
    overfitting — in-sample predictions would be identically zero residuals.

BINNED APPROXIMATION (sanity check):
    Cluster Z into K groups via K-means (after PCA to 10 dims for stability).
    Within each cluster c_k, approximate:
        I(X; Y|Z ∈ c_k) ≈ HSIC(X_k, Y_k) / Var(Y_k) + ε
    Average over clusters:
        Î_bin(X; Y|Z) = (1/K) Σ_k I(X; Y|Z ∈ c_k)

    The two estimates should agree in order of magnitude; large discrepancies
    signal that the Z-binning is too coarse or N is too small.

DATA LEAKAGE GUARDS:
    1. Kernel bandwidths are selected on the training indices only.
    2. KRR predictions are made LOO: when predicting for sample i, K[i,i]=0
       (or equivalently the i-th column of the kernel solve is excluded).
    3. KRR regularization (λ_krr) is fixed, not cross-validated on the target Y.
       Cross-validating on Y would use Y to select λ, which is circular.
    4. PCA for binning is fit on Z (unsupervised) — does not use Y.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dependence.kernel_cca import median_heuristic, rbf_kernel, center_kernel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernel Ridge Regression helpers
# ---------------------------------------------------------------------------

def _krr_loo_predictions(K: np.ndarray, y: np.ndarray, reg: float) -> np.ndarray:
    """
    Kernel ridge regression leave-one-out predictions.

    For KRR with kernel matrix K and regularizer λ:
        α = (K + NλI)^{-1} y

    The LOO prediction at sample i is:
        ŷ_{-i}(x_i) = (y_i - K_{ii} / [(K+NλI)^{-1}]_{ii} · ... )

    We use the efficient LOO formula (Rifkin & Klautau 2003):
        ŷ_{LOO,i} = y_i - α_i / [(K + NλI)^{-1}]_{ii}

    which avoids N separate matrix solves.

    DATA LEAKAGE GUARD: LOO ensures that the prediction for point i
    does not use y_i. This is critical when plugging ŷ_ρ into the
    I(X;Y|Z) estimator.

    Parameters
    ----------
    K   : (N, N) centered kernel matrix.
    y   : (N,) response vector.
    reg : λ regularization parameter.

    Returns
    -------
    (N,) LOO predictions.
    """
    N = K.shape[0]
    # Solve (K + NλI) α = y
    A = K + N * reg * np.eye(N)          # Regularized kernel matrix (K + NλI)
    alpha = np.linalg.solve(A, y)        # α = (K + NλI)^{-1} y

    # Diagonal of A^{-1}: [(K + NλI)^{-1}]_{ii}
    # Efficient: solve A^T W = I column by column is expensive; instead solve AW=I once
    A_inv = np.linalg.inv(A)             # (K + NλI)^{-1}   [N×N, cheap for small N]
    diag_Ainv = np.diag(A_inv)          # [(K + NλI)^{-1}]_{ii}

    # LOO formula: ŷ_{LOO,i} = y_i - α_i / A^{-1}_{ii}
    loo_pred = y - alpha / diag_Ainv
    return loo_pred


def _krr_predict(K_train: np.ndarray, y_train: np.ndarray, K_cross: np.ndarray, reg: float) -> np.ndarray:
    """
    KRR prediction on new points.

    ŷ_new = K_cross (K_train + NλI)^{-1} y_train

    Parameters
    ----------
    K_train  : (N, N) training kernel matrix (centered).
    y_train  : (N,) training responses.
    K_cross  : (M, N) cross-kernel k(x_new_i, x_train_j).
    reg      : λ regularizer.

    Returns
    -------
    (M,) predictions at new points.
    """
    N = K_train.shape[0]
    alpha = np.linalg.solve(K_train + N * reg * np.eye(N), y_train)
    return K_cross @ alpha


# ---------------------------------------------------------------------------
# HSIC helper for binned approximation
# ---------------------------------------------------------------------------

def _hsic_unbiased(K_X: np.ndarray, K_Y: np.ndarray) -> float:
    """
    Unbiased HSIC estimator (Song et al. 2012):
        HSIC(X,Y) = (1/(N(N-3))) [Tr(K̃_X K̃_Y) + 1^T K̃_X 1 · 1^T K̃_Y 1 / ((N-1)(N-2))
                                   - 2/(N-2) · 1^T K̃_X K̃_Y 1]

    HSIC is a positive measure of dependence: HSIC=0 iff X⊥Y (under the
    characteristic kernel). We use it as a proxy for I(X;Y|z_bin) in the
    binned approximation.

    Parameters
    ----------
    K_X, K_Y : (N, N) kernel matrices with diagonal zeroed out.
    """
    N = K_X.shape[0]
    if N < 4:
        return 0.0

    # Zero the diagonal (required for unbiased estimator)
    H_X = K_X.copy(); np.fill_diagonal(H_X, 0.0)
    H_Y = K_Y.copy(); np.fill_diagonal(H_Y, 0.0)

    t1 = np.trace(H_X @ H_Y)                        # Tr(K̃_X K̃_Y)
    t2 = np.sum(H_X) * np.sum(H_Y) / ((N-1)*(N-2)) # 1^T K̃_X 1 · 1^T K̃_Y 1 / denom
    t3 = 2.0 / (N-2) * np.einsum('ij,jk,ki->', H_X, H_Y, np.ones((N,N)) - np.eye(N))

    return float((t1 + t2 - t3) / (N * (N-3)))


# ---------------------------------------------------------------------------
# Primary estimator: Conditional Mean approach
# ---------------------------------------------------------------------------

@dataclass
class ResidualDependenceResult:
    """
    Output of residual_dependence().

    Attributes
    ----------
    I_XYZ_cm     : Î(X;Y|Z) via Conditional Mean approach (primary, Thm 2).
    I_XYZ_bin    : Î(X;Y|Z) via binned approximation (sanity check).
    log_ratio    : log10(I_cm / I_bin) — should be < 1 in order of magnitude.
    eta_star_loo : η̂⋆(x_i) LOO predictions (E[Y|X] proxy).
    eta_rho_loo  : η̂_ρ(x_i) LOO predictions (E[E[Y|Z]|X] proxy).
    g_rho_loo    : ĝ_ρ(z_i) LOO predictions (E[Y|Z] proxy).
    """
    I_XYZ_cm: float
    I_XYZ_bin: float
    log_ratio: float
    eta_star_loo: np.ndarray
    eta_rho_loo: np.ndarray
    g_rho_loo: np.ndarray


def residual_dependence(
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    reg_krr: float = 1e-2,
    reg_cca: float = 1e-3,
    n_bins: int = 3,
    h2_X: Optional[float] = None,
    h2_Z: Optional[float] = None,
    pca_dims_for_binning: int = 5,
) -> ResidualDependenceResult:
    """
    Estimate I(X; Y | Z) — residual dependence from Theorem 1.

    TWO ESTIMATORS (should agree in order of magnitude):

    1. Conditional Mean (primary, Theorem 2):
       a. ĝ_ρ(z) = LOO-KRR of Y on Z  →  proxy for g_ρ(z) = E[Y|Z=z]
       b. η̂_ρ(x) = LOO-KRR of ĝ_ρ(Z) on X  →  proxy for η_ρ(x) = E[E[Y|Z]|X]
       c. η̂⋆(x) = LOO-KRR of Y on X  →  proxy for η⋆(x) = E[Y|X]
       d. Î(X;Y|Z) = (1/N) Σ_i [η̂⋆(x_i) - η̂_ρ(x_i)]²

    2. Binned (sanity check):
       a. PCA-reduce Z to low dims, K-means into n_bins clusters
       b. For each bin: compute HSIC(X_k, Y_k) normalized by Var(Y_k)
       c. Î_bin = (1/K) Σ_k HSIC(X_k, Y_k) / max(Var(Y_k), ε)

    DATA LEAKAGE GUARDS:
        - h2_X, h2_Z selected on X, Z (not Y).
        - All KRR predictions are LOO.
        - PCA/K-means for binning uses Z only (no Y or X).
        - reg_krr is not cross-validated on Y.

    Parameters
    ----------
    X      : (N, d_X) alpha(X) features (pre-announcement price features).
    Z      : (N, d_Z) beta(Z) features (transcript embeddings).
    Y      : (N,)     forward abnormal returns (the label r(Y)).
    reg_krr : λ for KRR used in both step-a and step-b predictions.
    reg_cca : λ for the NOCCO / kernel CCA (passed through, not used here).
    n_bins  : number of Z-clusters for the binned approximation.
    h2_X, h2_Z : squared RBF bandwidths. None → median heuristic on training data.
    pca_dims_for_binning : PCA rank for Z before K-means (avoids curse of dimensionality).

    Returns
    -------
    ResidualDependenceResult
    """
    N = X.shape[0]
    if N < 4:
        raise ValueError(f"Need at least 4 samples, got {N}.")
    Y = Y.ravel().astype(float)

    # --- Bandwidth selection (training data = all N samples here) -----------
    # h2 is chosen from X and Z alone — never from Y — to prevent leakage.
    if h2_X is None:
        h2_X = median_heuristic(X)
        logger.debug("h²_X = %.4f (median heuristic)", h2_X)
    if h2_Z is None:
        h2_Z = median_heuristic(Z)
        logger.debug("h²_Z = %.4f (median heuristic)", h2_Z)

    # --- Build and center kernel matrices ------------------------------------
    K_X_raw = rbf_kernel(X, h2=h2_X)   # k_X(x_i, x_j) — for alpha(X) space
    K_Z_raw = rbf_kernel(Z, h2=h2_Z)   # k_Z(z_i, z_j) — for beta(Z) space
    K_X = center_kernel(K_X_raw)        # K̃_X = H K_X H
    K_Z = center_kernel(K_Z_raw)        # K̃_Z = H K_Z H

    # =========================================================================
    # Approach 1: Conditional Mean (Theorem 2)
    # =========================================================================

    # --- Step a: ĝ_ρ(z_i) = LOO-KRR(K_Z, Y) ---------------------------------
    # Estimates g_ρ(z) = E_{ρ_{Y,Z}}[r(Y)|Z=z] (eq. 5).
    # LOO is essential: an in-sample KRR perfectly fits Y when λ→0, making
    # ĝ_ρ ≡ Y and the subsequent regression trivial.
    g_rho_loo = _krr_loo_predictions(K_Z, Y, reg=reg_krr)

    # --- Step b: η̂_ρ(x_i) = LOO-KRR(K_X, ĝ_ρ(Z)) --------------------------
    # Estimates η_ρ(x) = E_{Q_{X,Z}}[g_ρ(Z)|X=x]  (eq. 4 / eq. 7).
    # This is the operator M_{Z|X} applied to ĝ_ρ, evaluated at training X.
    eta_rho_loo = _krr_loo_predictions(K_X, g_rho_loo, reg=reg_krr)

    # --- Step c: η̂⋆(x_i) = LOO-KRR(K_X, Y) ---------------------------------
    # Estimates the direct predictor η⋆(x) = E[r(Y)|X=x]  (eq. 3).
    eta_star_loo = _krr_loo_predictions(K_X, Y, reg=reg_krr)

    # --- Step d: Î(X;Y|Z) = (1/N) Σ [η̂⋆(x_i) - η̂_ρ(x_i)]² ---------------
    # This approximates ||η⋆ - η_ρ||²_{L2(P_X)} from Theorem 1 eq. (15).
    # A larger value means X contains information about Y not mediated by Z.
    residuals_cm = eta_star_loo - eta_rho_loo   # (N,)
    I_XYZ_cm = float(np.mean(residuals_cm ** 2))

    # =========================================================================
    # Approach 2: Binned approximation
    # =========================================================================

    # --- PCA-reduce Z for stable clustering (no Y used — no leakage) --------
    # The 384-dim transcript embeddings live on a low-dim manifold in practice.
    # PCA is purely unsupervised, fit on Z only.
    Z_c = Z - Z.mean(axis=0)
    d_pca = min(pca_dims_for_binning, N - 1, Z_c.shape[1])
    if d_pca > 0:
        _, _, Vt = np.linalg.svd(Z_c, full_matrices=False)
        Z_pca = Z_c @ Vt[:d_pca].T     # (N, d_pca)
    else:
        Z_pca = Z_c

    # --- K-means cluster on Z_pca -------------------------------------------
    # K-means is fit on Z (no Y, no X), so it cannot leak the target.
    n_bins_eff = min(n_bins, N // 2)    # ensure at least 2 points per bin
    if n_bins_eff < 2:
        logger.warning("Too few samples (%d) for binning; skipping binned estimate.", N)
        I_XYZ_bin = float("nan")
    else:
        labels = _kmeans_simple(Z_pca, n_bins_eff)
        bin_stats: list[float] = []
        for k in range(n_bins_eff):
            mask = labels == k
            n_k = mask.sum()
            if n_k < 4:
                continue   # too small for HSIC
            X_k = X[mask]
            Y_k = Y[mask]
            h2_Xk = median_heuristic(X_k) if n_k >= 3 else h2_X
            K_Xk = rbf_kernel(X_k, h2=h2_Xk)
            # For Y (1-D), use a simple linear kernel or RBF with auto bw
            Y_k_col = Y_k.reshape(-1, 1)
            h2_Yk = median_heuristic(Y_k_col) if n_k >= 3 else 1.0
            K_Yk = rbf_kernel(Y_k_col, h2=h2_Yk)

            hsic_k = _hsic_unbiased(K_Xk, K_Yk)
            var_Yk = float(np.var(Y_k, ddof=1)) if n_k > 1 else 1.0
            # Normalize by Var(Y|bin) so the units match the CM estimate
            norm_k = hsic_k / max(var_Yk, 1e-10)
            bin_stats.append(norm_k)

        I_XYZ_bin = float(np.mean(bin_stats)) if bin_stats else float("nan")

    # --- Log-ratio sanity check ---------------------------------------------
    if I_XYZ_cm > 1e-10 and np.isfinite(I_XYZ_bin) and I_XYZ_bin > 1e-10:
        log_ratio = float(np.log10(I_XYZ_cm / I_XYZ_bin))
        if abs(log_ratio) > 2:
            logger.warning(
                "CM and binned I(X;Y|Z) estimates differ by >2 orders of magnitude "
                "(log10 ratio=%.2f). With N=%d samples this is expected — "
                "both estimates have high variance at small N.",
                log_ratio, N,
            )
    else:
        log_ratio = float("nan")

    logger.info(
        "Î(X;Y|Z): CM=%.4f, Binned=%.4f, log10-ratio=%.2f, N=%d",
        I_XYZ_cm, I_XYZ_bin, log_ratio if np.isfinite(log_ratio) else -999, N,
    )

    return ResidualDependenceResult(
        I_XYZ_cm=I_XYZ_cm,
        I_XYZ_bin=I_XYZ_bin,
        log_ratio=log_ratio,
        eta_star_loo=eta_star_loo,
        eta_rho_loo=eta_rho_loo,
        g_rho_loo=g_rho_loo,
    )


# ---------------------------------------------------------------------------
# Simple K-means (no sklearn dependency)
# ---------------------------------------------------------------------------

def _kmeans_simple(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 0) -> np.ndarray:
    """Lloyd's K-means with k-means++ initialization (no sklearn required)."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]

    # k-means++ initialization
    centers = [X[rng.integers(N)]]
    for _ in range(k - 1):
        dists = np.array([min(np.sum((x - c)**2) for c in centers) for x in X])
        probs = dists / dists.sum()
        centers.append(X[rng.choice(N, p=probs)])
    centers = np.array(centers)   # (k, d)

    labels = np.zeros(N, dtype=int)
    for _ in range(max_iter):
        # Assignment step
        dists = np.array([np.sum((X - c)**2, axis=1) for c in centers])  # (k, N)
        new_labels = np.argmin(dists, axis=0)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update step
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                centers[j] = X[mask].mean(axis=0)

    return labels


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Estimate I(X;Y|Z) for synthetic data")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--theta", type=float, default=0.5,
                        help="Residual dependence parameter θ ∈ [0,1] per Appx. F.4")
    parser.add_argument("--reg", type=float, default=5e-2)
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    N, theta = args.n, args.theta

    # Synthetic data following the simulation from Appendix F.4 (simplified):
    # Z = sqrt(theta) * X + noise,  Y = sign(Z + noise)
    # theta=0: X ⊥ Y|Z (residual dep = 0)
    # theta=1: X ≡ Z (dep fully mediated)
    d = 2
    X_syn = rng.standard_normal((N, d))
    Z_syn = theta * X_syn + np.sqrt(max(1 - theta**2, 1e-6)) * rng.standard_normal((N, d))
    # Y depends on Z but not on X beyond what Z captures
    noise_Y = rng.standard_normal(N) * 0.5
    Y_syn = Z_syn[:, 0] + noise_Y   # continuous regression target

    res = residual_dependence(X_syn, Z_syn, Y_syn, reg_krr=args.reg)
    print(f"\nθ = {theta:.2f},  N = {N}")
    print(f"  Î(X;Y|Z) CM     : {res.I_XYZ_cm:.4f}")
    print(f"  Î(X;Y|Z) Binned : {res.I_XYZ_bin:.4f}")
    print(f"  log10(CM/Bin)   : {res.log_ratio:.2f}")
    print()

    # Sanity: θ=1 should give near-0 residual dependence (X fully mediated by Z)
    X_full = rng.standard_normal((N, d))
    Z_full = X_full.copy() + 0.01 * rng.standard_normal((N, d))
    Y_full = Z_full[:, 0] + 0.5 * rng.standard_normal(N)
    res_full = residual_dependence(X_full, Z_full, Y_full, reg_krr=args.reg)
    print(f"  θ≈1 (X≡Z): Î(X;Y|Z) CM = {res_full.I_XYZ_cm:.4f}  (should be near 0)")

    # Sanity: θ=0 (X⊥Z) → large residual dependence
    X_indep = rng.standard_normal((N, d))
    Z_indep = rng.standard_normal((N, d))   # independent of X
    Y_indep = X_indep[:, 0] + 0.5 * rng.standard_normal(N)   # Y depends on X, not Z
    res_indep = residual_dependence(X_indep, Z_indep, Y_indep, reg_krr=args.reg)
    print(f"  θ=0 (X⊥Z, Y=f(X)): Î(X;Y|Z) CM = {res_indep.I_XYZ_cm:.4f}  (should be > 0)")
