"""
Mean square contingency I(X; Z) — the Hilbert-Schmidt norm of V̂_{X,Z}.

THEORETICAL BACKGROUND
-----------------------
Definition (eq. 10): The mean square contingency between X and Z under
the pre-training distribution Q_{X,Z} is

    I(X; Z) = E_{Q_X ⊗ Q_Z}[(R(X, Z) - 1)²]                       (eq. 10)

where R = dQ_{X,Z} / d(Q_X ⊗ Q_Z) is the information density.

Equivalently (eq. 11 + Proposition 2):

    I(X; Z) = ||M_{Z|X}||²_HS - 1 = Σ_{i≥2} σ²_i                  (eq. 11)

And via the NOCCO identity (eq. 94, Fukumizu et al. 2007b Theorem 4):

    I(X; Z) = ||V_{X,Z}||²_{HS}                                      (eq. 94)
             = Tr[(Ĉ_{XX} + λI)^{-1} Ĉ_{XZ} (Ĉ_{ZZ} + λI)^{-1} Ĉ_{ZX}]

KERNEL DUAL FORMULA (derived in kernel_cca.py):

    Î(X; Z) = Tr[K̃_X R_X^{-1} K̃_Z R_Z^{-1}]                        (*)

where K̃_X = H K_X H, K̃_Z = H K_Z H are centered Gram matrices,
and R_X = K̃_X + NλI, R_Z = K̃_Z + NλI.

This is the primary estimator implemented here.

FINITE-SAMPLE BIAS WARNING:
    The NOCCO trace estimator (*) has a positive bias at finite N: under
    independence (X ⊥ Z), Tr[K̃_X R_X^{-1} K̃_Z R_Z^{-1}] > 0 due to
    finite-sample correlations. With N≈200 and λ=0.01, the null-case
    bias is ~0.5 in our experiments. This means:
    1. Absolute values of Î(X;Z) should not be interpreted as population
       quantities at small N; use them for relative comparisons only.
    2. Significance tests should compute a null distribution (e.g., by
       permuting Z labels before computing the trace).
    An unbiased alternative is the HSIC estimator (Song et al. 2012),
    but it does not directly give the singular values σ̂_i.

CROSS-CHECK via SVD (eq. 11):
    Î(X; Z) = Σ_i σ̂²_i  (sum of squared singular values from kernel CCA)

Both computations should agree to floating-point precision since they
derive from the same kernel matrices (they differ only in computational path).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dependence.kernel_cca import (
    KernelCCAResult,
    kernel_cca,
    median_heuristic,
    rbf_kernel,
    center_kernel,
)

logger = logging.getLogger(__name__)


@dataclass
class ContingencyResult:
    """
    Output of mean_square_contingency().

    Attributes
    ----------
    I_XZ_nocco : Î(X;Z) via NOCCO trace formula (*) — primary estimate.
    I_XZ_svd   : Î(X;Z) = Σ σ̂²_i via SVD — should match I_XZ_nocco.
    singular_values : σ̂_i from kernel CCA (Prop. 1, non-trivial).
    rel_error   : |I_nocco - I_svd| / max(I_nocco, 1e-10) — sanity check.
    kcca_result : full KernelCCAResult for downstream use.
    """
    I_XZ_nocco: float
    I_XZ_svd: float
    singular_values: np.ndarray
    rel_error: float
    kcca_result: KernelCCAResult


def mean_square_contingency(
    X: np.ndarray,
    Z: np.ndarray,
    reg: float = 1e-3,
    h2_X: Optional[float] = None,
    h2_Z: Optional[float] = None,
) -> ContingencyResult:
    """
    Estimate I(X; Z) = ||V̂_{X,Z}||²_HS (equations 10-11 and 94).

    Uses two complementary computation paths:
    1. NOCCO trace:  Î = Tr[K̃_X R_X^{-1} K̃_Z R_Z^{-1}]                    (*)
    2. SVD sum:      Î = Σ_i σ̂²_i  (from kernel_cca())

    These must agree; a large relative error signals numerical problems.

    DATA LEAKAGE GUARD:
        h2_X and h2_Z are selected from (X, Z) alone. If you later evaluate
        I(X;Z) on a hold-out set, pass the same h2_X, h2_Z here.

    Parameters
    ----------
    X   : (N, d_X) alpha(X) feature matrix — the source modality features.
    Z   : (N, d_Z) beta(Z) embedding matrix — the auxiliary modality features.
    reg : λ, Tikhonov regularizer (same as in kernel_cca).
    h2_X, h2_Z : squared bandwidths; None → median heuristic on training data.

    Returns
    -------
    ContingencyResult with I(X;Z) estimates and a relative error check.
    """
    N = X.shape[0]

    # --- Step 1: Run kernel CCA to get σ̂_i and centered kernel matrices ----
    # kernel_cca() selects bandwidths, builds K̃_X, K̃_Z, eigendecomposes.
    kcca = kernel_cca(X, Z, reg=reg, h2_X=h2_X, h2_Z=h2_Z)
    K_X_c = kcca.K_X_centered   # K̃_X = H K_X H
    K_Z_c = kcca.K_Z_centered   # K̃_Z = H K_Z H

    # --- Step 2: NOCCO trace formula (*) ------------------------------------
    # Î(X;Z) = Tr[K̃_X R_X^{-1} K̃_Z R_Z^{-1}]
    #         = Tr[D_X D_Z]
    # where D_X = K̃_X (K̃_X + NλI)^{-1} and D_Z = K̃_Z (K̃_Z + NλI)^{-1}.
    #
    # Efficient computation: eigendecompose K̃ = V Λ V^T, then
    # D = V (Λ/(Λ+Nλ)) V^T, and Tr[D_X D_Z] = Tr[(V_X γ_X V_X^T)(V_Z γ_Z V_Z^T)]
    #                                           = ||diag(γ_X)^{1/2} V_X^T V_Z diag(γ_Z)^{1/2}||²_F
    Nreg = N * reg
    lam_X, V_X = np.linalg.eigh(K_X_c)
    lam_Z, V_Z = np.linalg.eigh(K_Z_c)
    lam_X = np.maximum(lam_X, 0.0)
    lam_Z = np.maximum(lam_Z, 0.0)

    # Filter gains γ_i = λ_i / (λ_i + Nλ)  [dimension-wise shrinkage]
    gamma_X = lam_X / (lam_X + Nreg)   # eigenvalues of D_X
    gamma_Z = lam_Z / (lam_Z + Nreg)   # eigenvalues of D_Z

    # Tr[D_X D_Z] = Tr[V_X Γ_X V_X^T V_Z Γ_Z V_Z^T]
    #             = Σ_{i,j} γ_{X,i} γ_{Z,j} (V_X^T V_Z)²_{ij}
    # Numerically: form B = diag(γ_X^{1/2}) (V_X^T V_Z) diag(γ_Z^{1/2})
    #             then Tr[D_X D_Z] = ||B||²_F  (Frobenius norm squared)
    overlap = V_X.T @ V_Z           # (N, N) overlap matrix V_X^T V_Z
    B = np.sqrt(gamma_X)[:, None] * overlap * np.sqrt(gamma_Z)[None, :]
    I_XZ_nocco = float(np.sum(B ** 2))   # = ||B||²_F = Tr[D_X D_Z]

    # --- Step 3: SVD cross-check Σ σ̂²_i ------------------------------------
    # From eq. (11): I(X;Z) = Σ_{i≥2} σ²_i.
    # kernel_cca() returns σ̂_i with the trivial σ_1=1 already removed
    # (because centering projects out constant functions).
    I_XZ_svd = float(np.sum(kcca.singular_values ** 2))

    rel_error = abs(I_XZ_nocco - I_XZ_svd) / max(I_XZ_nocco, 1e-10)
    if rel_error > 0.05:
        logger.warning(
            "I(X;Z) relative error between NOCCO and SVD is %.1f%% "
            "(NOCCO=%.4f, SVD=%.4f). Check for numerical issues.",
            rel_error * 100, I_XZ_nocco, I_XZ_svd,
        )
    else:
        logger.debug(
            "I(X;Z) NOCCO=%.4f, SVD=%.4f, rel_err=%.2e",
            I_XZ_nocco, I_XZ_svd, rel_error,
        )

    return ContingencyResult(
        I_XZ_nocco=I_XZ_nocco,
        I_XZ_svd=I_XZ_svd,
        singular_values=kcca.singular_values,
        rel_error=rel_error,
        kcca_result=kcca,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Estimate I(X;Z) for synthetic data")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--rho", type=float, default=0.6, help="True Pearson correlation")
    parser.add_argument("--reg", type=float, default=1e-2)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    N, rho = args.n, args.rho

    # Bivariate Gaussian: I(X;Z) should scale with ρ
    X_syn = rng.standard_normal((N, 1))
    Z_syn = rho * X_syn + np.sqrt(1 - rho**2) * rng.standard_normal((N, 1))

    res = mean_square_contingency(X_syn, Z_syn, reg=args.reg)
    print(f"\nρ = {rho:.2f},  N = {N},  λ = {args.reg:.2e}")
    print(f"  Î(X;Z) NOCCO : {res.I_XZ_nocco:.4f}")
    print(f"  Î(X;Z) SVD   : {res.I_XZ_svd:.4f}")
    print(f"  Relative error: {res.rel_error:.2e}")
    print(f"  Top σ̂_i: {np.round(res.singular_values[:6], 4)}")
    print()

    # Null check: independent X, Z → I(X;Z) ≈ 0
    X_null = rng.standard_normal((N, 1))
    Z_null = rng.standard_normal((N, 1))
    res_null = mean_square_contingency(X_null, Z_null, reg=args.reg)
    print(f"  Null (independent): Î(X;Z) NOCCO = {res_null.I_XZ_nocco:.4f}  (should be near 0)")
