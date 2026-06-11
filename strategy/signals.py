"""
Information-density signal for earnings events.

THEORETICAL BASIS (equation 8)
-------------------------------
The information density R(x, z) = dQ_{X,Z} / d(Q_X ⊗ Q_Z) measures how much
the joint (X=x, Z=z) deviates from independence. Via the spectral expansion of
M_{Z|X} (Proposition 1), it admits the kernel CCA decomposition:

    R(x, z) - 1 = Σ_{i≥2} σ_i α_i(x) β_i(z)                         (eq. 8)

where σ_i are the non-trivial singular values, α_i ∈ L2(Q_X) are left singular
functions (of X), and β_i ∈ L2(Q_Z) are right singular functions (of Z).

The per-event score is:
    c(x_i, z_i) = Σ_j σ̂_j f̂_j(x_i) ĝ_j(z_i)

where f̂_j, ĝ_j are the estimated canonical functions evaluated at (x_i, z_i).

INTERPRETATION
--------------
Large |c(x_i, z_i)| means the joint occurrence of X=x_i (price features) and
Z=z_i (transcript embedding) deviates strongly from what you'd expect if X and Z
were independent. In other words:
  - c > 0: the transcript and price behavior reinforce each other — both signal
    the same latent direction (e.g., strong momentum + bullish tone)
  - c < 0: the transcript and price are in tension (mean-reversion opportunity)

The long/short strategy in backtest.py ranks events by |c| to identify the
events where X and Z are most informative about each other.

IN-SAMPLE VS OUT-OF-SAMPLE
---------------------------
In-sample scoring uses:
    F = K̃_X @ α  (N × k canonical variables in X-space)
    G = K̃_Z @ β  (N × k canonical variables in Z-space)
    c_i = Σ_j σ_j F_{ij} G_{ij}

Out-of-sample scoring centers the test kernel against training statistics —
the standard KPCA centering formula — then evaluates the fitted coefficients α, β.
The training bandwidth h2_X, h2_Z is frozen at fit time: NEVER re-fit on test data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dependence.kernel_cca import (
    KernelCCAResult,
    kernel_cca,
    rbf_kernel,
    center_kernel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test kernel centering helper
# ---------------------------------------------------------------------------

def _center_kernel_test(K_train: np.ndarray, K_test: np.ndarray) -> np.ndarray:
    """
    Center a test kernel matrix against training statistics.

    If K_train (N×N) is the training Gram matrix and K_test (M×N) is the
    cross-kernel between M test points and N training points, then the
    centered test kernel is (standard KPCA centering formula):

        K̃_test[i,j] = K_test[i,j]
                       - (1/N) Σ_k K_test[i,k]        ← row mean of K_test
                       - (1/N) Σ_k K_train[k,j]        ← col mean of K_train
                       + (1/N²) Σ_{k,l} K_train[k,l]   ← grand mean of K_train

    This ensures that test canonical variables are in the same affine subspace
    as training canonical variables (i.e., the centering H is applied consistently).

    Parameters
    ----------
    K_train : (N, N) training Gram matrix (raw, NOT centered).
    K_test  : (M, N) cross-kernel k(x*_i, x_j) for M test and N training points.

    Returns
    -------
    (M, N) centered cross-kernel.
    """
    return (
        K_test
        - K_test.mean(axis=1, keepdims=True)      # (M,1) row means of K_test
        - K_train.mean(axis=0, keepdims=True)      # (1,N) column means of K_train
        + K_train.mean()                            # scalar grand mean
    )


# ---------------------------------------------------------------------------
# Signal estimator
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    """
    Output of SignalEstimator.score_insample() or .score().

    Attributes
    ----------
    scores      : (N,) per-event information-density signal c(x_i, z_i).
    F           : (N, k) left canonical variables f̂_j(x_i).
    G           : (N, k) right canonical variables ĝ_j(z_i).
    singular_values : σ̂_j used for weighting.
    """
    scores: np.ndarray
    F: np.ndarray
    G: np.ndarray
    singular_values: np.ndarray


class SignalEstimator:
    """
    Fit a kernel CCA model on training events and score new events.

    Usage
    -----
    >>> est = SignalEstimator(reg=1e-3, n_components=20)
    >>> est.fit(X_train, Z_train)
    >>> scores = est.score(X_test, Z_test)      # out-of-sample
    >>> scores_is = est.score_insample()        # in-sample (no double-fitting)

    The signal c(x, z) = Σ_j σ_j f_j(x) g_j(z) is from eq. (8).
    """

    def __init__(
        self,
        reg: float = 1e-3,
        n_components: int = 20,
    ) -> None:
        self.reg = reg
        self.n_components = n_components
        self._is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        Z_train: np.ndarray,
        h2_X: Optional[float] = None,
        h2_Z: Optional[float] = None,
    ) -> "SignalEstimator":
        """
        Fit kernel CCA on training events.

        DATA LEAKAGE GUARD: h2_X and h2_Z are selected from X_train, Z_train
        alone (median heuristic inside kernel_cca). Never pass test data here.

        Parameters
        ----------
        X_train : (N, d_X) training price features.
        Z_train : (N, d_Z) training transcript embeddings.
        h2_X, h2_Z : bandwidths (None → median heuristic on training data).
        """
        kcca = kernel_cca(
            X_train, Z_train,
            reg=self.reg,
            n_components=self.n_components,
            h2_X=h2_X,
            h2_Z=h2_Z,
        )
        self.kcca_: KernelCCAResult = kcca
        self.X_train_ = X_train
        self.Z_train_ = Z_train

        # Store the raw (un-centered) training kernels for test centering
        self.K_X_raw_ = rbf_kernel(X_train, h2=kcca.h2_X)
        self.K_Z_raw_ = rbf_kernel(Z_train, h2=kcca.h2_Z)

        # In-sample canonical variables
        self.F_train_: np.ndarray = kcca.K_X_centered @ kcca.left_coefficients   # (N, k)
        self.G_train_: np.ndarray = kcca.K_Z_centered @ kcca.right_coefficients  # (N, k)

        self._is_fitted = True
        logger.info(
            "SignalEstimator fitted: N=%d, k=%d, top σ̂: %s",
            X_train.shape[0], self.n_components,
            np.round(kcca.singular_values[:5], 4),
        )
        return self

    def score_insample(self) -> SignalResult:
        """
        Compute in-sample information-density scores c(x_i, z_i) = Σ_j σ_j F_{ij} G_{ij}.

        Uses already-computed training canonical variables — no re-fitting.
        This is correct for backtesting on the training set only; for forward
        evaluation always use score() with held-out data.

        Returns
        -------
        SignalResult with scores of shape (N_train,).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        sigma = self.kcca_.singular_values                    # (k,)
        scores = np.sum(self.F_train_ * self.G_train_ * sigma[None, :], axis=1)  # (N,)
        return SignalResult(
            scores=scores,
            F=self.F_train_,
            G=self.G_train_,
            singular_values=sigma,
        )

    def score(
        self,
        X_test: np.ndarray,
        Z_test: np.ndarray,
    ) -> SignalResult:
        """
        Score out-of-sample events using the fitted kernel CCA model.

        The test kernel is centered against training statistics — the standard
        KPCA centering formula — so canonical variables are comparable to training.

        DATA LEAKAGE GUARD: uses frozen h2_X, h2_Z from fit(); never re-fits
        bandwidth on test data.

        Parameters
        ----------
        X_test : (M, d_X) test price features.
        Z_test : (M, d_Z) test transcript embeddings.

        Returns
        -------
        SignalResult with scores of shape (M,).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        h2_X = self.kcca_.h2_X
        h2_Z = self.kcca_.h2_Z

        # Cross-kernels: k(x_test_i, x_train_j), shape (M, N)
        K_X_cross = rbf_kernel(self.X_train_, h2=h2_X, Y=X_test).T
        K_Z_cross = rbf_kernel(self.Z_train_, h2=h2_Z, Y=Z_test).T

        # Center against training statistics (KPCA centering formula)
        K_X_cross_c = _center_kernel_test(self.K_X_raw_, K_X_cross)
        K_Z_cross_c = _center_kernel_test(self.K_Z_raw_, K_Z_cross)

        # Evaluate canonical functions at test points
        F_test = K_X_cross_c @ self.kcca_.left_coefficients    # (M, k)
        G_test = K_Z_cross_c @ self.kcca_.right_coefficients   # (M, k)

        sigma = self.kcca_.singular_values                      # (k,)
        scores = np.sum(F_test * G_test * sigma[None, :], axis=1)   # (M,)

        return SignalResult(
            scores=scores,
            F=F_test,
            G=G_test,
            singular_values=sigma,
        )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Signal estimator smoke test")
    parser.add_argument("--n-train", type=int, default=60)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--rho", type=float, default=0.6, help="X-Z correlation")
    args = parser.parse_args()

    rng = np.random.default_rng(99)
    N_tr, N_te, rho = args.n_train, args.n_test, args.rho

    # Synthetic: X and Z correlated at rho, Y = Z[:,0] + noise
    X_tr = rng.standard_normal((N_tr, 3))
    Z_tr = rho * X_tr[:, :2] + np.sqrt(1 - rho**2) * rng.standard_normal((N_tr, 2))
    X_te = rng.standard_normal((N_te, 3))
    Z_te = rho * X_te[:, :2] + np.sqrt(1 - rho**2) * rng.standard_normal((N_te, 2))

    est = SignalEstimator(reg=1e-2, n_components=10)
    est.fit(X_tr, Z_tr)

    # In-sample
    is_res = est.score_insample()
    print(f"\nIn-sample (N={N_tr}):")
    print(f"  Score range: [{is_res.scores.min():.4f}, {is_res.scores.max():.4f}]")
    print(f"  Score std: {is_res.scores.std():.4f}")

    # Out-of-sample
    oos_res = est.score(X_te, Z_te)
    print(f"\nOut-of-sample (M={N_te}):")
    print(f"  Score range: [{oos_res.scores.min():.4f}, {oos_res.scores.max():.4f}]")
    print(f"  Score std: {oos_res.scores.std():.4f}")

    # Rank correlation: scores should correlate with X[:,0]*Z[:,0] (direct link)
    # Under H₀ (no dependence): scores ≈ 0 everywhere
    X_null = rng.standard_normal((N_te, 3))
    Z_null = rng.standard_normal((N_te, 2))   # independent of X_null
    null_est = SignalEstimator(reg=1e-2, n_components=10)
    null_est.fit(rng.standard_normal((N_tr, 3)), rng.standard_normal((N_tr, 2)))
    null_res = null_est.score(X_null, Z_null)
    print(f"\nNull (independent X,Z):")
    print(f"  Score std: {null_res.scores.std():.4f}  (should be near 0)")
