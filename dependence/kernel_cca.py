"""
Kernel CCA estimator for the conditional mean operator M_{Z|X}.

THEORETICAL BACKGROUND
----------------------
Proposition 1 (Mehta & Harchaoui 2025, Appendix B.3, eq. 26-27):
    The conditional mean operator M_{Z|X}: L2(Q_Z) → L2(Q_X), defined by
        [M_{Z|X} g](x) = E_{Q_{X,Z}}[g(Z) | X = x],
    is compact and admits a singular value decomposition
        M_{Z|X}[g] = Σ_i σ_i ⟨g, β_i⟩_{L2(Q_Z)} α_i,
    with σ_1 = 1 ≥ σ_2 ≥ ... ≥ 0, left singular functions αi ∈ L2(Q_X),
    and right singular functions β_i ∈ L2(Q_Z). The pair (σ_1, α_1, β_1)
    is trivial: α_1 = 1_X, β_1 = 1_Z (constant functions).

NOCCO IDENTITY (Appendix E / Fukumizu et al. 2007b, Theorem 4):
    The Normalized Cross-Covariance Operator
        V_{X,Z} = C_{XX}^{-1/2} C_{XZ} C_{ZZ}^{-1/2}: L2(Q_Z) → L2(Q_X),   (eq. 93)
    satisfies
        ||V_{X,Z}||²_HS = I(X; Z) = Σ_{i≥2} σ²_i.                            (eq. 94)
    Its singular values are {σ_i}_{i≥2} — the non-trivial singular values of
    M_{Z|X}. The first pair (σ_1=1, const functions) is absorbed by centering.

EMPIRICAL ESTIMATOR (text after eq. 94):
    V̂_{X,Z} = (Ĉ_{XX} + λI)^{-1/2} Ĉ_{XZ} (Ĉ_{ZZ} + λI)^{-1/2},
    where Ĉ_{XX} = (1/N)Φ̃Φ̃^T, Ĉ_{XZ} = (1/N)Φ̃Ψ̃^T are empirical (centered)
    covariance operators in the RKHS, and λ > 0 is a Tikhonov regularizer.

KERNEL DUAL FORMULA (derived via push-through identity):
    In the N-dimensional sample space, with centered Gram matrices
        K̃_X = H K_X H,  K̃_Z = H K_Z H,  H = I - (1/N)11^T,
    and regularized matrices R_X = K̃_X + NλI, R_Z = K̃_Z + NλI:

        ||V̂_{X,Z}||²_HS = Tr[K̃_X R_X^{-1} K̃_Z R_Z^{-1}]           (NOCCO trace)

        σ̂²_i = eigenvalues of K̃_X R_X^{-1} K̃_Z R_Z^{-1}              (squared sv)

    Derivation sketch:
        ||V̂||²_HS = Tr[V̂^T V̂]
                  = (1/N²) Tr[Φ̃^T R̂_X^{-1} Φ̃ · Ψ̃^T R̂_Z^{-1} Ψ̃]
        push-through: Φ̃^T (Ĉ_{XX}+λI)^{-1} Φ̃ = N K̃_X R_X^{-1}
                  = Tr[K̃_X R_X^{-1} K̃_Z R_Z^{-1}]

BANDWIDTH SELECTION:
    RBF bandwidth h is set by the median heuristic (Gretton et al. 2012):
        h² = median({||x_i - x_j||² : i < j}) / (2 log(N+1)).
    Critically, h is computed on TRAINING data only. When a train/test split
    is provided, the test kernel uses the training bandwidth — never re-fits
    h on test data to avoid leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RBF kernel utilities
# ---------------------------------------------------------------------------

def median_heuristic(X: np.ndarray) -> float:
    """
    Estimate RBF bandwidth h² via the median heuristic (Gretton et al. 2012):
        h² = median(||x_i - x_j||²) / (2 log(N+1))

    Paper context: bandwidth selection for the RBF kernel used to build K̃_X
    or K̃_Z. Must be called on training data only to avoid data leakage.

    Parameters
    ----------
    X : (N, d) array.

    Returns
    -------
    h² > 0 (the squared bandwidth).
    """
    N = X.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 samples for median heuristic.")

    # Pairwise squared distances via ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2x_i·x_j
    sq_norms = np.sum(X ** 2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
    sq_dists = np.maximum(sq_dists, 0.0)  # numerical clamp

    # Off-diagonal only (i < j)
    upper = sq_dists[np.triu_indices(N, k=1)]
    median_sq_dist = np.median(upper)

    if median_sq_dist == 0.0:
        logger.warning("Median pairwise distance is 0 — all points identical? Defaulting h²=1.")
        return 1.0

    # Silverman-style denominator suppresses overfitting in small N regimes.
    h2 = median_sq_dist / (2.0 * np.log(N + 1))
    return float(h2)


def rbf_kernel(X: np.ndarray, h2: Optional[float] = None, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix k(x_i, x_j) = exp(-||x_i-x_j||²/(2h²)).

    K[i,j] lives in the RKHS H and represents ⟨φ(x_i), φ(x_j)⟩_H (eq. B.4 background).

    Parameters
    ----------
    X  : (N, d) training points.
    h2 : squared bandwidth. If None, computed via median_heuristic(X).
    Y  : (M, d) test points. If None, returns N×N training kernel.

    Returns
    -------
    (N, N) or (N, M) kernel matrix.
    """
    if h2 is None:
        h2 = median_heuristic(X)

    if Y is None:
        # Symmetric N×N kernel
        sq_norms = np.sum(X ** 2, axis=1)
        sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
    else:
        # Rectangular N×M cross-kernel
        sq_norms_X = np.sum(X ** 2, axis=1)
        sq_norms_Y = np.sum(Y ** 2, axis=1)
        sq_dists = sq_norms_X[:, None] + sq_norms_Y[None, :] - 2.0 * (X @ Y.T)

    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-sq_dists / (2.0 * h2))


def center_kernel(K: np.ndarray) -> np.ndarray:
    """
    Apply the centering matrix H = I - (1/N)11^T to produce K̃ = H K H.

    K̃ is the RKHS covariance without the mean:
        Ĉ_{XX} = (1/N) Φ̃Φ̃^T,  K̃ = Φ̃^T Φ̃  (centered Gram matrix).

    Parameters
    ----------
    K : (N, N) kernel matrix.

    Returns
    -------
    (N, N) centered kernel matrix K̃.
    """
    N = K.shape[0]
    row_means = K.mean(axis=1, keepdims=True)   # K 1 / N
    col_means = K.mean(axis=0, keepdims=True)   # 1^T K / N
    grand_mean = K.mean()                        # 1^T K 1 / N²
    return K - row_means - col_means + grand_mean


# ---------------------------------------------------------------------------
# Core kernel CCA estimator
# ---------------------------------------------------------------------------

@dataclass
class KernelCCAResult:
    """
    Output of kernel_cca().

    Attributes
    ----------
    singular_values : σ̂_i for i = 2, ..., rank  (non-trivial, descending).
        These approximate the singular values from Proposition 1, excluding
        σ_1 = 1 which is removed by centering.
    left_coefficients  : (N, r) array. Dual-space coefficients of α̂_i (left
        singular functions of V̂_{X,Z}, i.e., functions of X).
    right_coefficients : (N, r) array. Dual-space coefficients of β̂_i.
    h2_X : bandwidth used for K_X.
    h2_Z : bandwidth used for K_Z.
    reg  : regularization parameter λ.
    K_X_centered : K̃_X stored for downstream I(X;Z) computation.
    K_Z_centered : K̃_Z stored for downstream I(X;Z) computation.
    """
    singular_values: np.ndarray
    left_coefficients: np.ndarray
    right_coefficients: np.ndarray
    h2_X: float
    h2_Z: float
    reg: float
    K_X_centered: np.ndarray
    K_Z_centered: np.ndarray


def kernel_cca(
    X: np.ndarray,
    Z: np.ndarray,
    reg: float = 1e-3,
    n_components: Optional[int] = None,
    h2_X: Optional[float] = None,
    h2_Z: Optional[float] = None,
) -> KernelCCAResult:
    """
    Estimate the singular values σ_i of M_{Z|X} via kernel CCA (Prop. 1).

    The estimator is the regularized NOCCO (Normalized Cross-Covariance Operator):
        V̂_{X,Z} = (Ĉ_{XX} + λI)^{-1/2} Ĉ_{XZ} (Ĉ_{ZZ} + λI)^{-1/2}   (eq. 94)

    In the kernel dual, the squared singular values are:
        σ̂²_i = eigenvalues of D_X D_Z,                                      (dual form)
    where D_X = K̃_X (K̃_X + NλI)^{-1} and D_Z = K̃_Z (K̃_Z + NλI)^{-1}.

    Numerically we symmetrize D_X D_Z via eigendecompositions of K̃_X and K̃_Z:
        K̃_X = V_X Λ_X V_X^T,  γ_X = Λ_X / (Λ_X + NλI)  (filter gains)
        K̃_Z = V_Z Λ_Z V_Z^T,  γ_Z = Λ_Z / (Λ_Z + NλI)
        C = diag(γ_X^{1/2}) (V_X^T V_Z) diag(γ_Z) (V_Z^T V_X) diag(γ_X^{1/2})
        σ̂_i = sqrt(eigenvalues(C))                   [symmetric PSD, stable]

    DATA LEAKAGE GUARD:
        h2_X and h2_Z are computed on (X, Z) = training data only. If you
        have a test set, pass the training-derived h2_* explicitly when
        calling rbf_kernel() for test points.

    Parameters
    ----------
    X   : (N, d_X) array of alpha(X) features (e.g., price features ∈ R^3).
    Z   : (N, d_Z) array of beta(Z) features (e.g., transcript embeddings ∈ R^384).
    reg : λ > 0, Tikhonov regularization parameter.
    n_components : number of singular values to return (default: N-1).
    h2_X, h2_Z : squared bandwidths. None → computed via median heuristic on X, Z.

    Returns
    -------
    KernelCCAResult with σ̂_i in descending order.
    """
    N = X.shape[0]
    if N < 3:
        raise ValueError(f"Need at least 3 samples, got {N}.")
    if X.shape[0] != Z.shape[0]:
        raise ValueError("X and Z must have the same number of rows.")

    # --- Bandwidth selection (training data only, no leakage) ---------------
    # Median heuristic on the training X and Z separately.
    if h2_X is None:
        h2_X = median_heuristic(X)
        logger.debug("Median heuristic h²_X = %.4f", h2_X)
    if h2_Z is None:
        h2_Z = median_heuristic(Z)
        logger.debug("Median heuristic h²_Z = %.4f", h2_Z)

    # --- Build kernel matrices K_X, K_Z ∈ R^{N×N} --------------------------
    # K_X[i,j] = ⟨φ(x_i), φ(x_j)⟩_H = exp(-||x_i - x_j||² / (2h²_X))
    K_X = rbf_kernel(X, h2=h2_X)
    # K_Z[i,j] = ⟨ψ(z_i), ψ(z_j)⟩_G = exp(-||z_i - z_j||² / (2h²_Z))
    K_Z = rbf_kernel(Z, h2=h2_Z)

    # --- Center: K̃ = H K H,  H = I - (1/N)11^T ----------------------------
    # Centering removes the trivial singular pair (σ_1=1, constant functions),
    # so V̂_{X,Z} = (Ĉ_{XX}+λI)^{-1/2} Ĉ_{XZ} (Ĉ_{ZZ}+λI)^{-1/2} only
    # captures the non-trivial spectrum from Proposition 1.
    K_X_c = center_kernel(K_X)   # K̃_X = H K_X H
    K_Z_c = center_kernel(K_Z)   # K̃_Z = H K_Z H

    # --- Eigendecompose K̃_X and K̃_Z ----------------------------------------
    # K̃_X = V_X Λ_X V_X^T  (eigenvalues ≥ 0 since K̃_X is PSD after centering)
    lam_X, V_X = np.linalg.eigh(K_X_c)   # eigh: symmetric, returns ascending order
    lam_Z, V_Z = np.linalg.eigh(K_Z_c)

    # Clamp to remove numerical negatives (rounding error in a PSD matrix)
    lam_X = np.maximum(lam_X, 0.0)
    lam_Z = np.maximum(lam_Z, 0.0)

    # --- Filter gains γ_i = λ_i / (λ_i + Nλ) --------------------------------
    # D_X = K̃_X (K̃_X + NλI)^{-1} has the same eigenvectors as K̃_X
    # with eigenvalues γ_{X,i} = λ_{X,i} / (λ_{X,i} + Nλ).
    # These are the "filter gains" — large for informative directions,
    # small (≈0) for noise directions regularized away.
    Nreg = N * reg   # = N * λ (the scalar that enters (K̃ + NλI))
    gamma_X = lam_X / (lam_X + Nreg)   # ∈ [0, 1]  (D_X eigenvalues)
    gamma_Z = lam_Z / (lam_Z + Nreg)   # ∈ [0, 1]  (D_Z eigenvalues)

    # --- Symmetric form of D_X D_Z for stable eigenvalue computation --------
    # D_X D_Z = V_X diag(γ_X) V_X^T  V_Z diag(γ_Z) V_Z^T
    # To symmetrize without forming the full N×N product:
    #   C = diag(γ_X^{1/2}) (V_X^T V_Z) diag(γ_Z) (V_Z^T V_X) diag(γ_X^{1/2})
    # C is symmetric PSD; eigenvalues(C) = eigenvalues(D_X D_Z) = σ̂²_i.
    sqg_X = np.sqrt(gamma_X)             # γ_X^{1/2},  shape (N,)
    G = V_X.T @ V_Z                      # V_X^T V_Z,  shape (N, N)  [overlap matrix]
    # C = diag(sqg_X) @ G @ diag(gamma_Z) @ G.T @ diag(sqg_X)
    # Avoid explicit diag() products with broadcasting:
    C = (sqg_X[:, None] * G) * gamma_Z[None, :] @ (G.T * sqg_X[None, :])

    # Eigenvalues of C are σ̂²_i (real, nonneg by construction)
    sig2 = np.linalg.eigvalsh(C)         # ascending order
    sig2 = np.maximum(sig2, 0.0)         # clamp numerical negatives
    sig2 = sig2[::-1]                    # descending order (σ_2 ≥ σ_3 ≥ ...)

    sigma = np.sqrt(sig2)                # singular values σ̂_i

    if n_components is not None:
        sigma = sigma[:n_components]
        sig2 = sig2[:n_components]

    # --- Dual-space singular functions (left/right coefficients) -------------
    # The left singular function α̂_i corresponds to the eigenvector of D_X D_Z
    # in sample space. Via the symmetrized form C, if C v = μ v then:
    #   α̂_i ↔ V_X diag(sqg_X) v  (after normalization)
    # We compute this but expose as coefficient arrays — the user can evaluate
    # ĥ_i(x_new) = Σ_j [left_coefficients[:,i]]_j k_X(x_j, x_new) (with centering).
    _, evec_C = np.linalg.eigh(C)
    evec_C = evec_C[:, ::-1]   # descending order

    # Left coefficients α̂_i in sample space (N-vectors over training points)
    left_coef = V_X @ (sqg_X[:, None] * evec_C)
    # Right coefficients β̂_i
    right_coef = V_Z @ (gamma_Z[:, None] * (G.T @ (sqg_X[:, None] * evec_C)))
    # Normalize both (Gram-Schmidt already ensures unit kernel norm, but safeguard)
    norms_left  = np.sqrt(np.sum(left_coef ** 2, axis=0) + 1e-12)
    norms_right = np.sqrt(np.sum(right_coef ** 2, axis=0) + 1e-12)
    left_coef  = left_coef  / norms_left[None, :]
    right_coef = right_coef / norms_right[None, :]

    if n_components is not None:
        left_coef  = left_coef[:, :n_components]
        right_coef = right_coef[:, :n_components]

    logger.info(
        "Kernel CCA: N=%d, λ=%.2e, h²_X=%.4f, h²_Z=%.4f → "
        "top-5 σ̂_i: %s",
        N, reg, h2_X, h2_Z,
        np.round(sigma[:5], 4),
    )

    return KernelCCAResult(
        singular_values=sigma,
        left_coefficients=left_coef,
        right_coefficients=right_coef,
        h2_X=h2_X,
        h2_Z=h2_Z,
        reg=reg,
        K_X_centered=K_X_c,
        K_Z_centered=K_Z_c,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Synthetic sample size")
    parser.add_argument("--rho", type=float, default=0.7, help="True correlation")
    parser.add_argument("--reg", type=float, default=1e-2)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    N = args.n

    # Ground truth: X, Z jointly Gaussian with correlation rho
    # Theoretical σ_2 (first non-trivial canonical correlation) ≈ rho
    rho = args.rho
    X_syn = rng.standard_normal((N, 2))
    Z_syn = rho * X_syn[:, :1] + np.sqrt(1 - rho**2) * rng.standard_normal((N, 1))

    result = kernel_cca(X_syn, Z_syn, reg=args.reg)
    print(f"\nGround truth ρ = {rho:.2f}")
    print(f"Estimated σ̂_i (top 5): {np.round(result.singular_values[:5], 4)}")
    print(f"I(X;Z) = Σ σ̂²_i = {np.sum(result.singular_values**2):.4f}")
    print(f"Expected I(X;Z) (Gaussian) ≈ ρ² / (1-ρ²) = {rho**2/(1-rho**2):.4f}")
