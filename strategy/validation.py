"""
Walk-forward validation with Newey-West t-statistics and Benjamini-Hochberg correction.

VALIDATION DESIGN
-----------------
Walk-forward splits:
    Train window : 2 years
    Test window  : 1 year
    Step         : 6 months

For each fold:
  1. Fit SignalEstimator on training events (kernel CCA with median heuristic bandwidth).
  2. Fit RegimeDetector on training embeddings.
  3. Score test events with the fitted estimator.
  4. Flag test events using the fitted detector (training threshold only).
  5. Run backtest on test events using scores and flags.
  6. Record per-event P&L and regime classification.

STATISTICAL TESTS
-----------------
Newey-West t-statistic (Newey & West 1987):
    Accounts for heteroskedasticity and autocorrelation in event P&L.
    With 5-day holding periods and an irregular event calendar, we use
    L = 5 lags in the Bartlett kernel. For each fold:
        Ω_NW = Ω_0 + 2 Σ_{l=1}^L (1 - l/(L+1)) Ω_l
        SE_NW = sqrt(Ω_NW / T)
        t_NW  = mean(r) / SE_NW

Benjamini-Hochberg (BH) correction (Benjamini & Hochberg 1995):
    Applied across the k canonical signal components to control FDR
    when testing whether each component individually predicts returns.
    Rejection criterion: p_j ≤ (rank_j / m) × α where α is the FDR level.

Signal degradation test:
    Tests H₀: mean_return_regime = mean_return_stable using a two-sample
    Newey-West t-test. A significant positive difference (returns higher in
    stable periods) supports Theorem 1's prediction that strategy performance
    degrades during regime shifts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from strategy.signals import SignalEstimator
from strategy.backtest import run_backtest, BacktestResult, summarize_backtest
from regime.regime_detection import RegimeDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def newey_west_se(returns: np.ndarray, n_lags: int = 5) -> float:
    """
    Newey-West HAC standard error for the sample mean.

    Corrects for heteroskedasticity and autocorrelation up to `n_lags` lags
    using the Bartlett kernel:
        w_l = 1 - l / (n_lags + 1)   (triangular / Bartlett weights)

    Ω_NW = (1/T) [Ω_0 + 2 Σ_{l=1}^L w_l Ω_l],  where Ω_l = Σ_t r_t r_{t-l}
    SE_NW = sqrt(Ω_NW / T)

    Parameters
    ----------
    returns : (T,) time series of P&L observations (event-ordered).
    n_lags  : number of autocorrelation lags (default 5, matching holding period).

    Returns
    -------
    Newey-West standard error of the mean.
    """
    T = len(returns)
    if T < n_lags + 2:
        return float(np.std(returns, ddof=1) / np.sqrt(max(T, 1)))

    r = returns - returns.mean()
    gamma_0 = float(np.dot(r, r) / T)
    omega = gamma_0
    for l in range(1, n_lags + 1):
        gamma_l = float(np.dot(r[l:], r[:-l]) / T)
        w_l = 1.0 - l / (n_lags + 1)
        omega += 2.0 * w_l * gamma_l

    omega = max(omega, 0.0)    # numerical clamp (omega should be non-negative)
    return float(np.sqrt(omega / T))


def newey_west_tstat(returns: np.ndarray, n_lags: int = 5) -> tuple[float, float]:
    """
    Newey-West t-statistic and two-sided p-value for H₀: E[r] = 0.

    Uses the normal approximation for the p-value.

    Returns
    -------
    (t_stat, p_value) tuple.
    """
    if len(returns) < 4:
        return float("nan"), float("nan")

    mu = float(returns.mean())
    se = newey_west_se(returns, n_lags=n_lags)
    if se < 1e-12:
        return float("nan"), float("nan")

    t = mu / se
    # Two-sided p-value from the standard normal (large-T approximation)
    from math import erfc, sqrt
    p = float(erfc(abs(t) / sqrt(2)))
    return t, p


def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg procedure for controlling False Discovery Rate at level α.

    For m hypotheses with sorted p-values p_{(1)} ≤ ... ≤ p_{(m)}, rejects H_{(j)}
    if p_{(j)} ≤ (j/m) × α.

    Parameters
    ----------
    p_values : (m,) array of p-values (one per hypothesis).
    alpha    : FDR level (default 0.05).

    Returns
    -------
    (rejected, adjusted_p) where:
        rejected    : (m,) boolean array (True = null rejected).
        adjusted_p  : (m,) BH-adjusted p-values (Benjamini-Yekutieli formula).
    """
    m = len(p_values)
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)

    rank = np.argsort(p_values)
    sorted_p = p_values[rank]
    thresholds = (np.arange(1, m + 1) / m) * alpha

    # Find the largest j such that p_{(j)} ≤ threshold_{(j)}
    below = sorted_p <= thresholds
    if below.any():
        k_max = below.nonzero()[0][-1]
        reject_sorted = np.zeros(m, dtype=bool)
        reject_sorted[:k_max + 1] = True   # reject all up to and including k_max
    else:
        reject_sorted = np.zeros(m, dtype=bool)

    # Un-sort
    rejected = np.zeros(m, dtype=bool)
    rejected[rank] = reject_sorted

    # BH adjusted p-values: min(p_{(j)} × m/j) evaluated from the right
    adj_sorted = np.minimum.accumulate((sorted_p * m / np.arange(1, m + 1))[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adjusted_p = np.zeros(m)
    adjusted_p[rank] = adj_sorted

    return rejected, adjusted_p


def _two_sample_nw_tstat(
    a: np.ndarray,
    b: np.ndarray,
    n_lags: int = 5,
) -> tuple[float, float]:
    """
    Two-sample Newey-West t-statistic for H₀: E[a] = E[b].

    Approximation: we concatenate the two samples and use the difference of
    means divided by sqrt(SE_a² + SE_b²) (heteroskedastic Welch-style).
    """
    if len(a) < 4 or len(b) < 4:
        return float("nan"), float("nan")
    se_a = newey_west_se(a, n_lags=n_lags)
    se_b = newey_west_se(b, n_lags=n_lags)
    se_diff = np.sqrt(se_a ** 2 + se_b ** 2)
    if se_diff < 1e-12:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / se_diff
    from math import erfc, sqrt
    p = float(erfc(abs(t) / sqrt(2)))
    return float(t), p


# ---------------------------------------------------------------------------
# Walk-forward fold container
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardFold:
    """Results for one walk-forward fold."""
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    backtest: BacktestResult
    nw_tstat: float
    nw_pvalue: float
    regime_nw_tstat: Optional[float]
    regime_nw_pvalue: Optional[float]
    stable_nw_tstat: Optional[float]
    stable_nw_pvalue: Optional[float]
    degradation_tstat: Optional[float]   # two-sample: stable - regime (H₁: stable > regime)
    degradation_pvalue: Optional[float]


@dataclass
class ValidationResult:
    """
    Output of walk_forward_validate().

    Attributes
    ----------
    folds               : Per-fold results.
    pooled_pnl          : Concatenated event P&L across all test folds.
    pooled_nw_tstat     : NW t-stat on pooled P&L (primary hypothesis test).
    pooled_nw_pvalue    : Two-sided p-value for pooled NW t-stat.
    pooled_regime_pnl   : Pooled P&L for regime-shift events.
    pooled_stable_pnl   : Pooled P&L for stable events.
    degradation_tstat   : NW t-stat for stable − regime return difference.
    degradation_pvalue  : p-value for degradation test.
    bh_component_rejected : (k,) bool — which canonical components reject H₀ after BH.
    bh_adjusted_p       : (k,) BH-adjusted p-values per component.
    mean_sharpe         : Mean Sharpe across folds (equal-weighted).
    sharpe_std          : Std of fold Sharpes (consistency measure).
    """
    folds: list[WalkForwardFold]
    pooled_pnl: np.ndarray
    pooled_nw_tstat: float
    pooled_nw_pvalue: float
    pooled_regime_pnl: np.ndarray
    pooled_stable_pnl: np.ndarray
    degradation_tstat: Optional[float]
    degradation_pvalue: Optional[float]
    bh_component_rejected: np.ndarray
    bh_adjusted_p: np.ndarray
    mean_sharpe: float
    sharpe_std: float


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validate(
    events: pd.DataFrame,
    train_years: float = 2.0,
    test_years: float = 1.0,
    step_months: int = 6,
    reg_cca: float = 1e-3,
    n_components: int = 20,
    n_quintiles: int = 5,
    tc_bps: float = 5.0,
    newey_west_lags: int = 5,
    bh_alpha: float = 0.05,
    regime_detector_kwargs: Optional[dict] = None,
    min_train_events: int = 20,
    min_test_events: int = 10,
) -> ValidationResult:
    """
    Walk-forward validation: 2-year train / 1-year test / 6-month step.

    Parameters
    ----------
    events : DataFrame with columns:
        date (Timestamp), ticker (str), X (np.ndarray or columns x_0..x_d),
        Z (np.ndarray), Y (float), abnormal_return (float).

        Expected column layout:
          - date, ticker, abnormal_return : metadata
          - columns starting with 'x_' : price features (alpha(X))
          - columns starting with 'z_' : embedding components (beta(Z))
        OR:
          - 'X' column containing np.ndarray per row
          - 'Z' column containing np.ndarray per row

    train_years : training window length in years.
    test_years  : test window length in years.
    step_months : step size between folds in months.
    reg_cca     : NOCCO regularization λ for kernel CCA.
    n_components : number of canonical components (signal dimensions).
    n_quintiles : quintile count for the backtest (top/bottom quintile traded).
    tc_bps      : one-way transaction cost in basis points.
    newey_west_lags : lag order L for the Bartlett kernel.
    bh_alpha    : FDR level for Benjamini-Hochberg correction.
    regime_detector_kwargs : kwargs forwarded to RegimeDetector (window_days, etc.).
    min_train_events : skip fold if training set has fewer than this many events.
    min_test_events  : skip fold if test set has fewer than this many events.

    Returns
    -------
    ValidationResult

    Notes on data leakage prevention
    ---------------------------------
    1. kernel CCA bandwidth (h2_X, h2_Z): median heuristic on train X, Z only.
    2. Regime threshold: calibrated on train embeddings; applied frozen to test.
    3. Signal scores for test events: out-of-sample via SignalEstimator.score().
    4. No Y-dependent hyperparameter selection anywhere.
    """
    df = events.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- Resolve X and Z column layout --------------------------------------
    x_cols = [c for c in df.columns if c.startswith("x_")]
    z_cols = [c for c in df.columns if c.startswith("z_")]
    has_array_cols = ("X" in df.columns and "Z" in df.columns)

    def _get_XZ(subset: pd.DataFrame):
        if has_array_cols:
            X = np.vstack(subset["X"].values)
            Z = np.vstack(subset["Z"].values)
        elif x_cols and z_cols:
            X = subset[x_cols].values.astype(float)
            Z = subset[z_cols].values.astype(float)
        else:
            raise ValueError(
                "events must have columns prefixed 'x_' and 'z_' for features, "
                "or array-valued 'X' and 'Z' columns."
            )
        return X, Z

    def _get_Y(subset: pd.DataFrame) -> np.ndarray:
        return subset["abnormal_return"].values.astype(float)

    # --- Build fold boundaries ----------------------------------------------
    t_min = df["date"].min()
    t_max = df["date"].max()

    train_td = timedelta(days=int(train_years * 365.25))
    test_td  = timedelta(days=int(test_years  * 365.25))
    step_td  = timedelta(days=int(step_months * 30.44))

    fold_starts: list[pd.Timestamp] = []
    t = t_min
    while t + train_td + test_td <= t_max + step_td:
        fold_starts.append(t)
        t += step_td

    if not fold_starts:
        raise ValueError(
            f"Not enough date range for even one fold: "
            f"span={t_max - t_min}, need train+test={train_td + test_td}."
        )

    logger.info(
        "Walk-forward: %d folds, train=%s, test=%s, step=%s",
        len(fold_starts), train_td, test_td, step_td,
    )

    # --- Per-component BH test: collect (t_stat, p_value) per component -----
    component_tstats: list[float] = []
    component_pvals: list[float] = []

    # --- Run folds ----------------------------------------------------------
    folds: list[WalkForwardFold] = []
    all_pnl: list[np.ndarray] = []
    all_pnl_regime: list[np.ndarray] = []
    all_pnl_stable: list[np.ndarray] = []

    rd_kwargs = regime_detector_kwargs or {}

    for fold_idx, t_start in enumerate(fold_starts):
        train_start = t_start
        train_end   = t_start + train_td
        test_start  = train_end
        test_end    = train_end + test_td

        train_mask = (df["date"] >= train_start) & (df["date"] < train_end)
        test_mask  = (df["date"] >= test_start)  & (df["date"] < test_end)

        df_train = df[train_mask].copy()
        df_test  = df[test_mask].copy()

        n_tr = len(df_train)
        n_te = len(df_test)

        if n_tr < min_train_events:
            logger.info(
                "Fold %d: skipping (n_train=%d < %d)", fold_idx, n_tr, min_train_events
            )
            continue
        if n_te < min_test_events:
            logger.info(
                "Fold %d: skipping (n_test=%d < %d)", fold_idx, n_te, min_test_events
            )
            continue

        logger.info(
            "Fold %d: train [%s, %s) n=%d, test [%s, %s) n=%d",
            fold_idx, train_start.date(), train_end.date(), n_tr,
            test_start.date(), test_end.date(), n_te,
        )

        try:
            X_tr, Z_tr = _get_XZ(df_train)
            X_te, Z_te = _get_XZ(df_test)
        except (ValueError, KeyError) as exc:
            logger.warning("Fold %d: feature extraction failed: %s", fold_idx, exc)
            continue

        # --- Fit signal estimator on training data (no Y) -------------------
        try:
            signal_est = SignalEstimator(reg=reg_cca, n_components=n_components)
            signal_est.fit(X_tr, Z_tr)
        except Exception as exc:
            logger.warning("Fold %d: SignalEstimator.fit() failed: %s", fold_idx, exc)
            continue

        # --- Fit regime detector on training embeddings ----------------------
        try:
            detector = RegimeDetector(**rd_kwargs)
            detector.fit(df_train["date"], Z_tr)
        except Exception as exc:
            logger.warning("Fold %d: RegimeDetector.fit() failed: %s", fold_idx, exc)
            detector = None

        # --- Score test events (out-of-sample) --------------------------------
        try:
            sig_result = signal_est.score(X_te, Z_te)
        except Exception as exc:
            logger.warning("Fold %d: score() failed: %s", fold_idx, exc)
            continue

        # --- Regime flags for test events ------------------------------------
        if detector is not None:
            try:
                test_flags = detector.transform(
                    df_test["date"], Z_te, event_dates=df_test["date"]
                )
                df_test = df_test.copy()
                df_test["is_regime"] = test_flags.values
            except Exception as exc:
                logger.warning("Fold %d: regime transform failed: %s", fold_idx, exc)
                df_test["is_regime"] = False
        else:
            df_test["is_regime"] = False

        df_test = df_test.copy()
        df_test["signal"] = sig_result.scores

        # --- Run backtest on test fold ----------------------------------------
        try:
            bt = run_backtest(df_test, tc_bps=tc_bps, n_quintiles=n_quintiles)
        except Exception as exc:
            logger.warning("Fold %d: backtest failed: %s", fold_idx, exc)
            continue

        if bt.n_events == 0:
            logger.info("Fold %d: no tradable events.", fold_idx)
            continue

        # --- NW t-statistics for this fold ------------------------------------
        fold_pnl = bt.event_returns["pnl_net"].values
        nw_t, nw_p = newey_west_tstat(fold_pnl, n_lags=newey_west_lags)

        regime_pnl_fold = fold_pnl[bt.event_returns["is_regime"].values]
        stable_pnl_fold = fold_pnl[~bt.event_returns["is_regime"].values]

        r_t, r_p = newey_west_tstat(regime_pnl_fold) if len(regime_pnl_fold) >= 4 else (None, None)
        s_t, s_p = newey_west_tstat(stable_pnl_fold) if len(stable_pnl_fold) >= 4 else (None, None)
        deg_t, deg_p = (None, None)
        if len(regime_pnl_fold) >= 4 and len(stable_pnl_fold) >= 4:
            # H₁: stable P&L > regime P&L (degradation during shifts)
            deg_t, deg_p = _two_sample_nw_tstat(stable_pnl_fold, regime_pnl_fold)

        # Per-component t-stats (one per canonical direction)
        sigma = signal_est.kcca_.singular_values
        F_te = sig_result.F   # (n_test, k)
        G_te = sig_result.G   # (n_test, k)
        Y_te = _get_Y(df_test)
        for j in range(min(n_components, len(sigma))):
            comp_scores = sigma[j] * F_te[:, j] * G_te[:, j]
            # Simplified: does this component correlate with Y at all?
            ct, cp = newey_west_tstat(comp_scores * Y_te, n_lags=newey_west_lags)
            component_tstats.append(ct)
            component_pvals.append(cp if np.isfinite(cp) else 1.0)

        folds.append(WalkForwardFold(
            fold_idx=fold_idx,
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            n_train=n_tr, n_test=n_te,
            backtest=bt,
            nw_tstat=nw_t, nw_pvalue=nw_p,
            regime_nw_tstat=r_t, regime_nw_pvalue=r_p,
            stable_nw_tstat=s_t, stable_nw_pvalue=s_p,
            degradation_tstat=deg_t, degradation_pvalue=deg_p,
        ))

        all_pnl.append(fold_pnl)
        if len(regime_pnl_fold) > 0:
            all_pnl_regime.append(regime_pnl_fold)
        if len(stable_pnl_fold) > 0:
            all_pnl_stable.append(stable_pnl_fold)

    if not folds:
        raise RuntimeError("No valid walk-forward folds completed.")

    # --- Pooled statistics --------------------------------------------------
    pooled_pnl    = np.concatenate(all_pnl)
    pooled_regime = np.concatenate(all_pnl_regime) if all_pnl_regime else np.array([])
    pooled_stable = np.concatenate(all_pnl_stable) if all_pnl_stable else np.array([])

    p_t, p_p = newey_west_tstat(pooled_pnl, n_lags=newey_west_lags)

    deg_t_pool, deg_p_pool = None, None
    if len(pooled_regime) >= 4 and len(pooled_stable) >= 4:
        deg_t_pool, deg_p_pool = _two_sample_nw_tstat(pooled_stable, pooled_regime)

    # --- BH correction across canonical components --------------------------
    bh_rejected, bh_adj = benjamini_hochberg(
        np.array(component_pvals) if component_pvals else np.array([]),
        alpha=bh_alpha,
    )

    fold_sharpes = np.array([
        f.backtest.sharpe for f in folds if np.isfinite(f.backtest.sharpe)
    ])
    mean_sharpe = float(fold_sharpes.mean()) if len(fold_sharpes) > 0 else float("nan")
    sharpe_std  = float(fold_sharpes.std())  if len(fold_sharpes) > 1 else float("nan")

    logger.info(
        "Validation complete: %d folds, pooled NW t=%.2f (p=%.3f), "
        "mean Sharpe=%.2f ± %.2f, BH rejected %d/%d components",
        len(folds), p_t, p_p if np.isfinite(p_p) else -1,
        mean_sharpe, sharpe_std if np.isfinite(sharpe_std) else 0,
        bh_rejected.sum(), len(bh_rejected),
    )

    return ValidationResult(
        folds=folds,
        pooled_pnl=pooled_pnl,
        pooled_nw_tstat=p_t,
        pooled_nw_pvalue=p_p,
        pooled_regime_pnl=pooled_regime,
        pooled_stable_pnl=pooled_stable,
        degradation_tstat=deg_t_pool,
        degradation_pvalue=deg_p_pool,
        bh_component_rejected=bh_rejected,
        bh_adjusted_p=bh_adj,
        mean_sharpe=mean_sharpe,
        sharpe_std=sharpe_std,
    )


def summarize_validation(result: ValidationResult) -> str:
    """Format walk-forward validation summary."""
    bh_n = int(result.bh_component_rejected.sum()) if len(result.bh_component_rejected) > 0 else 0
    bh_m = len(result.bh_component_rejected)

    p_nw = f"{result.pooled_nw_pvalue:.3f}" if np.isfinite(result.pooled_nw_pvalue) else "nan"
    deg_t = f"{result.degradation_tstat:.3f}" if result.degradation_tstat is not None else "n/a"
    deg_p = f"{result.degradation_pvalue:.3f}" if result.degradation_pvalue is not None else "n/a"

    lines = [
        "Walk-Forward Validation Summary",
        f"  Folds completed     : {len(result.folds)}",
        f"  Pooled NW t-stat    : {result.pooled_nw_tstat:.3f}  (p={p_nw})",
        f"  Mean fold Sharpe    : {result.mean_sharpe:.3f} ± {result.sharpe_std:.3f}",
        f"  BH components sig.  : {bh_n} / {bh_m} at FDR α=5%",
        f"  Degradation t-stat  : {deg_t}  (stable − regime; p={deg_p})",
        "",
        "  Fold-level summary:",
    ]
    for f in result.folds:
        sr = (
            f"{f.backtest.sharpe:.2f}" if np.isfinite(f.backtest.sharpe) else "nan"
        )
        lines.append(
            f"    [{f.test_start.date()} – {f.test_end.date()}]  "
            f"n={f.backtest.n_events}  Sharpe={sr}  NW t={f.nw_tstat:.2f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Walk-forward validation smoke test")
    parser.add_argument("--n", type=int, default=400, help="Total synthetic events")
    parser.add_argument("--skill", type=float, default=0.2,
                        help="Signal-to-return correlation")
    parser.add_argument("--d-x", type=int, default=3, help="X feature dim")
    parser.add_argument("--d-z", type=int, default=8, help="Z embedding dim")
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    N = args.n
    d_x, d_z = args.d_x, args.d_z

    # Generate synthetic events spanning ~4 years
    from datetime import timedelta
    base = pd.Timestamp("2020-01-01")
    event_dates = [base + timedelta(days=int(d)) for d in np.sort(rng.integers(0, 1460, N))]

    X_all = rng.standard_normal((N, d_x))
    X_pad = np.zeros((N, d_z))
    X_pad[:, :d_x] = X_all
    Z_all = 0.6 * X_pad + rng.standard_normal((N, d_z)) * 0.8
    Z_all /= np.linalg.norm(Z_all, axis=1, keepdims=True) + 1e-8

    Y_all = (Z_all[:, 0] + args.skill * X_all[:, 0] + 0.5 * rng.standard_normal(N)) * 0.02

    events_df = pd.DataFrame({
        "date": event_dates,
        "ticker": [f"T{i%30:02d}" for i in range(N)],
        "abnormal_return": Y_all,
        "is_regime": False,     # placeholder; overwritten by detector inside validation
    })
    for j in range(d_x):
        events_df[f"x_{j}"] = X_all[:, j]
    for j in range(d_z):
        events_df[f"z_{j}"] = Z_all[:, j]

    result = walk_forward_validate(
        events_df,
        train_years=2.0,
        test_years=1.0,
        step_months=6,
        reg_cca=1e-2,
        n_components=5,
        n_quintiles=5,
        tc_bps=5.0,
        newey_west_lags=5,
    )

    print(f"\n{summarize_validation(result)}")

    # NW test on pooled
    t, p = newey_west_tstat(result.pooled_pnl)
    print(f"\nPooled NW: t={t:.3f}, p={p:.3f}  (N={len(result.pooled_pnl)} events)")
