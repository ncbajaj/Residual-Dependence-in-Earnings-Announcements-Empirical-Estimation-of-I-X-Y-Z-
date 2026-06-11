"""
I(X;Y|Z) stability analysis across detected regimes.

CENTRAL EMPIRICAL CLAIM (from Theorem 1)
-----------------------------------------
Theorem 1 bounds the zero-shot prediction gap:
    ||η⋆ - η_ρ||²_{L2} ≲ I(X;Y|Z) + prompt bias.

Under a regime shift — when the Z-embedding distribution drifts away from the
training distribution Q_Z — the conditional structure Z→Y breaks down.
Concretely: the function g_ρ(z) = E_ρ[Y|Z=z] learned under the training
distribution Q_{Y,Z} becomes a poor predictor under the shifted P_{Y,Z}.

This means:
  - η_ρ(x) = E[g_ρ(Z)|X=x] is no longer a good proxy for E[Y|X=x]
  - The residual I(X;Y|Z) = (1/N)||η⋆ - η_ρ||² grows during regime shifts
  - X features contain incremental return-predictive information that Z fails
    to mediate — which is what the strategy in signals.py tries to exploit

This module tests that claim empirically:
    H₁: I(X;Y|Z)_regime  >  I(X;Y|Z)_stable

DESIGN NOTES
------------
- At small N (< ~50 per subsample), the LOO-KRR estimator has high variance.
  Effect sizes should be interpreted with the variance in mind.
- We report both point estimates and a sign-based sanity check (H₁ consistent?).
- The permutation p-value shuffles the regime label (not Y) to avoid Y-leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from dependence.mutual_info import residual_dependence, ResidualDependenceResult

logger = logging.getLogger(__name__)


@dataclass
class StabilityResult:
    """
    Output of regime_stability_analysis().

    Attributes
    ----------
    I_full      : I(X;Y|Z) on the full dataset.
    I_regime    : I(X;Y|Z) on regime-shift events (None if n < min_n).
    I_stable    : I(X;Y|Z) on stable events (None if n < min_n).
    ratio       : I_regime / I_stable — should be > 1 per H₁ (None if either is None).
    n_regime    : number of regime events.
    n_stable    : number of stable events.
    h1_consistent : True if I_regime > I_stable (supports the central claim).
    perm_p_value : permutation p-value for (I_regime - I_stable) > 0 under label shuffle.
    full_result  : ResidualDependenceResult on the full dataset.
    regime_result : ResidualDependenceResult on regime subset (None if too small).
    stable_result : ResidualDependenceResult on stable subset (None if too small).
    """
    I_full: float
    I_regime: Optional[float]
    I_stable: Optional[float]
    ratio: Optional[float]
    n_regime: int
    n_stable: int
    h1_consistent: Optional[bool]
    perm_p_value: Optional[float]
    full_result: ResidualDependenceResult
    regime_result: Optional[ResidualDependenceResult]
    stable_result: Optional[ResidualDependenceResult]


def _effect_size(a: float, b: float) -> float:
    """Normalized difference: (a - b) / (a + b). ∈ (-1, 1); positive means a > b."""
    denom = a + b
    return float((a - b) / max(denom, 1e-12))


def regime_stability_analysis(
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    regime_flags: np.ndarray | pd.Series,
    reg_krr: float = 1e-2,
    reg_cca: float = 1e-3,
    min_n_per_regime: int = 8,
    n_permutations: int = 200,
    seed: int = 0,
) -> StabilityResult:
    """
    Estimate I(X;Y|Z) separately for regime-shift and stable events.

    Tests the central claim: residual dependence increases during regime shifts.

    DATA LEAKAGE GUARDS
    -------------------
    - regime_flags is derived from Z only (unsupervised; no Y) — see RegimeDetector.
    - Bandwidths are selected within each subset using only that subset's X, Z.
    - KRR predictions are always LOO (no in-sample fit-eval).
    - The permutation test shuffles regime_flags, not Y.

    Parameters
    ----------
    X            : (N, d_X) alpha(X) price features.
    Z            : (N, d_Z) beta(Z) transcript embeddings.
    Y            : (N,) forward abnormal returns.
    regime_flags : (N,) boolean array — True for regime-shift events.
    reg_krr      : KRR regularization λ (same for all subsets for comparability).
    reg_cca      : CCA regularization λ (unused here, passed for consistency).
    min_n_per_regime : minimum events in a subset to compute I(X;Y|Z) on it.
    n_permutations   : permutations for p-value estimation (0 → skip).
    seed             : random seed for permutations.

    Returns
    -------
    StabilityResult with point estimates, ratio, and permutation p-value.
    """
    flags = np.asarray(regime_flags, dtype=bool)
    Y = Y.ravel().astype(float)
    N = len(flags)

    if len(X) != N or len(Z) != N or len(Y) != N:
        raise ValueError("X, Z, Y, regime_flags must all have the same length N.")

    n_regime = int(flags.sum())
    n_stable = int((~flags).sum())
    logger.info("Stability analysis: N=%d  regime=%d  stable=%d", N, n_regime, n_stable)

    # --- Full dataset ---------------------------------------------------------
    full_result = residual_dependence(X, Z, Y, reg_krr=reg_krr)
    I_full = full_result.I_XYZ_cm
    logger.info("I(X;Y|Z) full = %.4f", I_full)

    # --- Regime subset --------------------------------------------------------
    regime_result: Optional[ResidualDependenceResult] = None
    I_regime: Optional[float] = None
    if n_regime >= min_n_per_regime:
        try:
            regime_result = residual_dependence(
                X[flags], Z[flags], Y[flags], reg_krr=reg_krr
            )
            I_regime = regime_result.I_XYZ_cm
            logger.info("I(X;Y|Z) regime = %.4f  (n=%d)", I_regime, n_regime)
        except Exception as exc:
            logger.warning("I(X;Y|Z) regime failed: %s", exc)
    else:
        logger.warning(
            "Too few regime events (%d < %d) to estimate I(X;Y|Z)_regime.",
            n_regime, min_n_per_regime,
        )

    # --- Stable subset --------------------------------------------------------
    stable_result: Optional[ResidualDependenceResult] = None
    I_stable: Optional[float] = None
    if n_stable >= min_n_per_regime:
        try:
            stable_result = residual_dependence(
                X[~flags], Z[~flags], Y[~flags], reg_krr=reg_krr
            )
            I_stable = stable_result.I_XYZ_cm
            logger.info("I(X;Y|Z) stable = %.4f  (n=%d)", I_stable, n_stable)
        except Exception as exc:
            logger.warning("I(X;Y|Z) stable failed: %s", exc)
    else:
        logger.warning(
            "Too few stable events (%d < %d) to estimate I(X;Y|Z)_stable.",
            n_stable, min_n_per_regime,
        )

    # --- Ratio and directional check ------------------------------------------
    ratio: Optional[float] = None
    h1_consistent: Optional[bool] = None
    if I_regime is not None and I_stable is not None and I_stable > 1e-12:
        ratio = I_regime / I_stable
        h1_consistent = bool(I_regime > I_stable)
        eff = _effect_size(I_regime, I_stable)
        logger.info(
            "I_regime / I_stable = %.3f  (effect size = %.3f)  H₁ consistent: %s",
            ratio, eff, h1_consistent,
        )

    # --- Permutation test -----------------------------------------------------
    perm_p_value: Optional[float] = None
    if (
        n_permutations > 0
        and I_regime is not None
        and I_stable is not None
        and n_regime >= min_n_per_regime
        and n_stable >= min_n_per_regime
    ):
        observed_diff = I_regime - I_stable
        rng = np.random.default_rng(seed)
        perm_diffs: list[float] = []
        for _ in range(n_permutations):
            perm_flags = rng.permutation(flags)
            n_r = perm_flags.sum()
            n_s = (~perm_flags).sum()
            if n_r < min_n_per_regime or n_s < min_n_per_regime:
                continue
            try:
                r_r = residual_dependence(X[perm_flags], Z[perm_flags], Y[perm_flags], reg_krr=reg_krr)
                r_s = residual_dependence(X[~perm_flags], Z[~perm_flags], Y[~perm_flags], reg_krr=reg_krr)
                perm_diffs.append(r_r.I_XYZ_cm - r_s.I_XYZ_cm)
            except Exception:
                continue

        if perm_diffs:
            perm_arr = np.array(perm_diffs)
            perm_p_value = float(np.mean(perm_arr >= observed_diff))
            logger.info(
                "Permutation test: observed diff=%.4f, p-value=%.3f (n_perm=%d)",
                observed_diff, perm_p_value, len(perm_diffs),
            )
        else:
            logger.warning("No valid permutations completed.")

    return StabilityResult(
        I_full=I_full,
        I_regime=I_regime,
        I_stable=I_stable,
        ratio=ratio,
        n_regime=n_regime,
        n_stable=n_stable,
        h1_consistent=h1_consistent,
        perm_p_value=perm_p_value,
        full_result=full_result,
        regime_result=regime_result,
        stable_result=stable_result,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="I(X;Y|Z) stability smoke test")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--frac-regime", type=float, default=0.3,
                        help="Fraction of events that are 'regime shift'")
    parser.add_argument("--regime-strength", type=float, default=2.0,
                        help="How much larger I(X;Y|Z) is during regime (multiplier on noise)")
    args = parser.parse_args()

    rng = np.random.default_rng(11)
    N = args.n
    flags = rng.random(N) < args.frac_regime

    d = 3
    X = rng.standard_normal((N, d))
    Z = 0.6 * X + 0.8 * rng.standard_normal((N, d))  # correlated with X

    # During stable periods: Y mediated by Z (low residual dep)
    # During regime shifts: Y depends on X directly (high residual dep)
    noise_scale = np.where(flags, args.regime_strength, 0.3)
    Y = Z[:, 0] + noise_scale * X[:, 1] + 0.5 * rng.standard_normal(N)

    result = regime_stability_analysis(
        X, Z, Y, flags, n_permutations=50, min_n_per_regime=6
    )

    print(f"\nN={N}, regime fraction={flags.mean():.2f}")
    print(f"  I(X;Y|Z) full   : {result.I_full:.4f}")
    print(f"  I(X;Y|Z) regime : {result.I_regime}")
    print(f"  I(X;Y|Z) stable : {result.I_stable}")
    print(f"  Ratio (r/s)     : {result.ratio}")
    print(f"  H₁ consistent   : {result.h1_consistent}")
    print(f"  Permutation p   : {result.perm_p_value}")
