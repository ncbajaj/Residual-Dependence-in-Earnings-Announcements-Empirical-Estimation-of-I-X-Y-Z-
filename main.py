"""
Residual Dependence Pipeline — end-to-end CLI entry point.

Full pipeline:
  1. Fetch 8-K filings (EDGAR) and filter for earnings calls (Item 2.02)
  2. Embed transcripts via all-MiniLM-L6-v2          → Z matrix
  3. Compute pre-announcement price features          → X matrix
  4. Compute forward 5-day abnormal returns           → Y vector
  5. Align X, Z, Y into a single events table
  6. Regime detection on training embeddings
  7. Compute I(X;Z) and I(X;Y|Z) on training events
  8. Score all events with information-density signal
  9. Backtest: long/short quintile on test events
 10. Walk-forward validation (2yr/1yr/6mo)
 11. Stability analysis: I(X;Y|Z) by regime
 12. Generate four publication figures
 13. Generate structured markdown report
 14. Save all outputs to --output-dir

Usage
-----
python main.py \\
    --tickers AAPL MSFT JPM \\
    --start 2021-01-01 \\
    --end   2023-12-31 \\
    --train-end 2022-12-31 \\
    --output-dir output/

All random seeds are set at startup and logged. Every leakage-sensitive
decision (bandwidth selection, regime threshold, normalisation) uses only the
training window. See LEAKAGE_AUDIT below.

LEAKAGE_AUDIT
-------------
Point                     Guard
h²_X, h²_Z               median_heuristic() called on train X, Z only
Regime threshold          RegimeDetector.fit() on train; .transform() on test
Signal scores             SignalEstimator.fit() on train; .score() on test
KRR regularisation λ      Fixed constant; never cross-validated on Y
KRR predictions           LOO (Rifkin & Klautau 2003) — excludes y_i for point i
Price normalisation        Cross-sectional per filing_date; no cross-date information
Price window              [t-22, t-2] exclusive; no overlap with [t+1, t+5] return window
Walk-forward splits       Each fold fits exclusively on events before fold train_end
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Lazy imports (avoids 80 MB model import at startup before args are parsed)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    logger.info("Random seed set to %d", seed)


def _mkdir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _rolling_ixyz(
    events_df: pd.DataFrame,
    X_all: np.ndarray,
    Z_all: np.ndarray,
    Y_all: np.ndarray,
    window_days: int = 90,
    step_days: int = 30,
    reg_krr: float = 1e-2,
    min_window_size: int = 8,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Compute rolling I(X;Y|Z) estimates for Figure 2.

    For each window ending at t_end (stepped by step_days), compute I(X;Y|Z)
    on events in [t_end - window_days, t_end]. Returns (dates, ixyz_values,
    regime_flags) aligned to window center dates.

    DATA LEAKAGE GUARD: uses only events within each window — no future data.
    """
    from dependence.mutual_info import residual_dependence

    dates_all = pd.DatetimeIndex(events_df["date"])
    t_min = dates_all.min()
    t_max = dates_all.max()

    window_centers: list[pd.Timestamp] = []
    ixyz_vals: list[float] = []
    regime_flags_out: list[bool] = []

    t = t_min + pd.Timedelta(days=window_days)
    while t <= t_max:
        mask = (dates_all >= t - pd.Timedelta(days=window_days)) & (dates_all <= t)
        n_w = mask.sum()
        if n_w >= min_window_size:
            try:
                res = residual_dependence(
                    X_all[mask], Z_all[mask], Y_all[mask], reg_krr=reg_krr
                )
                window_centers.append(t)
                ixyz_vals.append(res.I_XYZ_cm)
                regime_flags_out.append(bool(events_df["is_regime"].values[mask].any()))
            except Exception as exc:
                logger.debug("Rolling I(X;Y|Z) window at %s failed: %s", t.date(), exc)
        t += pd.Timedelta(days=step_days)

    return (
        pd.DatetimeIndex(window_centers),
        np.array(ixyz_vals),
        np.array(regime_flags_out, dtype=bool),
    )


def _power_law_exponent(singular_values: np.ndarray, max_k: int = 30) -> float:
    """Fit σ_i ~ C * i^{-γ} and return γ (the decay exponent)."""
    sigma = singular_values[singular_values > 1e-8][:max_k]
    if len(sigma) < 3:
        return float("nan")
    i_idx = np.arange(1, len(sigma) + 1, dtype=float)
    b, _ = np.polyfit(np.log(i_idx), np.log(sigma), 1)
    return float(-b)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_fetch_and_embed(
    tickers: list[str],
    start: str,
    end: str,
    raw_dir: str,
    embed_cache_dir: str,
    seed: int,
) -> pd.DataFrame:
    """
    Stage 1-2: Fetch 8-K filings and embed transcripts.

    Returns a DataFrame with columns: ticker, filing_date, embed_path, text_len.
    Rows with failed fetches or embedding errors are logged and dropped.
    """
    from data.edgar import save_filings
    from features.embeddings import embed_transcripts_batch, load_embedding

    logger.info("Stage 1: Fetching 8-K filings for %s [%s → %s]", tickers, start, end)
    save_filings(tickers, start_date=start, end_date=end, out_dir=raw_dir)

    # Scan raw_dir for fetched filings, collect (ticker, date) pairs
    rows: list[dict] = []
    raw_path = Path(raw_dir)
    for ticker in tickers:
        ticker_dir = raw_path / ticker
        if not ticker_dir.exists():
            logger.warning("No filings directory for %s", ticker)
            continue
        for jpath in sorted(ticker_dir.glob("*.json")):
            try:
                with open(jpath) as fh:
                    rec = json.load(fh)
                filing_date = rec.get("filing_date", "")
                # Only keep Item 2.02 (earnings call transcripts)
                if not rec.get("item_202_present", False):
                    continue
                text = rec.get("raw_text", "") or rec.get("text", "") or ""
                if len(text) < 50:
                    logger.debug("Skip %s/%s: text too short (%d chars)", ticker, jpath.name, len(text))
                    continue
                rows.append({
                    "ticker": ticker,
                    "filing_date": filing_date,
                    "text": text,
                    "accession": rec.get("accession_number", ""),
                })
            except Exception as exc:
                logger.warning("Failed to read %s: %s", jpath, exc)

    if not rows:
        logger.error("No valid 8-K Item 2.02 filings found. Check EDGAR output in %s.", raw_dir)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df.dropna(subset=["filing_date"]).reset_index(drop=True)

    # Filter by date range
    df = df[
        (df["filing_date"] >= pd.Timestamp(start))
        & (df["filing_date"] <= pd.Timestamp(end))
    ].reset_index(drop=True)

    logger.info("Stage 1 complete: %d valid Item 2.02 filings", len(df))

    # Stage 2: embed transcripts
    logger.info("Stage 2: Embedding transcripts via all-MiniLM-L6-v2")
    records = [
        {"ticker": row["ticker"],
         "filing_date": str(row["filing_date"].date()),
         "raw_text": row["text"]}
        for _, row in df.iterrows()
    ]
    embeddings = embed_transcripts_batch(records, cache_dir=Path(embed_cache_dir))

    # Attach embeddings to DataFrame
    embed_list: list[Optional[np.ndarray]] = []
    for _, row in df.iterrows():
        key = (row["ticker"], str(row["filing_date"].date()))
        embed_list.append(embeddings.get(key))

    df["embedding"] = embed_list
    n_before = len(df)
    df = df[df["embedding"].apply(lambda e: e is not None)].reset_index(drop=True)
    logger.info(
        "Stage 2 complete: %d / %d events have embeddings",
        len(df), n_before,
    )
    return df


def stage_price_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Stage 3: Compute pre-announcement price features X = alpha(X).

    Adds column 'X' (np.ndarray of shape (3,)) to df.
    Rows with missing price data are dropped.
    """
    from features.price_features import compute_features_batch

    logger.info("Stage 3: Computing price features for %d events", len(df))
    filing_dates: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        t = row["ticker"]
        d = str(row["filing_date"].date())
        filing_dates.setdefault(t, []).append(d)

    features_df = compute_features_batch(filing_dates)
    # features_df has columns: ticker, filing_date, momentum_20d, realized_vol_20d, volume_ratio_20d, error

    features_df["filing_date"] = pd.to_datetime(features_df["filing_date"])
    features_df = features_df[features_df["error"].isna()].copy()

    feature_cols = ["momentum_20d", "realized_vol_20d", "volume_ratio_20d"]
    features_df["X"] = list(
        features_df[feature_cols].values.astype(float)
    )

    merged = df.merge(
        features_df[["ticker", "filing_date", "X"]],
        on=["ticker", "filing_date"],
        how="inner",
    )
    logger.info(
        "Stage 3 complete: %d / %d events have price features",
        len(merged), len(df),
    )
    return merged


def stage_abnormal_returns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Stage 4: Compute forward 5-day abnormal returns Y.

    Adds column 'abnormal_return' (float). Rows with missing data are dropped.
    """
    from data.prices import compute_abnormal_returns_batch

    logger.info("Stage 4: Computing forward abnormal returns for %d events", len(df))
    filing_dates: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        t = row["ticker"]
        d = str(row["filing_date"].date())
        filing_dates.setdefault(t, []).append(d)

    returns_df = compute_abnormal_returns_batch(filing_dates)
    returns_df["filing_date"] = pd.to_datetime(returns_df["filing_date"])
    returns_df = returns_df[returns_df["error"].isna()].copy()
    returns_df = returns_df.rename(columns={"filing_date": "filing_date"})

    merged = df.merge(
        returns_df[["ticker", "filing_date", "abnormal_return"]],
        on=["ticker", "filing_date"],
        how="inner",
    )
    merged = merged.dropna(subset=["abnormal_return"]).reset_index(drop=True)
    logger.info(
        "Stage 4 complete: %d / %d events have forward returns",
        len(merged), len(df),
    )
    return merged


def stage_align(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build aligned arrays X_all, Z_all, Y_all from the events DataFrame.

    Returns (events_df, X_all, Z_all, Y_all).
    events_df has columns: ticker, filing_date (as date str 'YYYY-MM-DD'), date (Timestamp),
    abnormal_return, is_regime (placeholder False).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["filing_date"])
    df = df.sort_values("date").reset_index(drop=True)

    Z_all = np.vstack(df["embedding"].values)   # (N, 384)
    X_all = np.vstack(df["X"].values)           # (N, 3)
    Y_all = df["abnormal_return"].values.astype(float)

    df["is_regime"] = False   # placeholder; set in stage_regime

    events_df = df[["ticker", "filing_date", "date", "abnormal_return", "is_regime"]].copy()
    return events_df, X_all, Z_all, Y_all


def stage_regime(
    events_df: pd.DataFrame,
    Z_all: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[pd.DataFrame, object]:
    """
    Stage 6: Fit RegimeDetector on training embeddings, flag all events.

    Returns updated events_df (with is_regime filled) and the fitted detector.
    LEAKAGE GUARD: RegimeDetector.fit() uses only training embeddings;
    threshold applied frozen to all events.
    """
    from regime.regime_detection import RegimeDetector

    logger.info(
        "Stage 6: Fitting RegimeDetector on %d training events", train_mask.sum()
    )
    detector = RegimeDetector(window_days=90, step_days=30, threshold_std=1.5)
    train_dates = events_df["date"][train_mask]
    train_Z = Z_all[train_mask]
    detector.fit(train_dates, train_Z)

    # Apply to full dataset (training threshold frozen)
    all_dates = events_df["date"]
    flags = detector.transform(all_dates, Z_all, event_dates=all_dates)
    events_df = events_df.copy()
    events_df["is_regime"] = flags.values

    n_flagged = events_df["is_regime"].sum()
    logger.info(
        "Stage 6 complete: %d / %d events flagged as regime-shift",
        n_flagged, len(events_df),
    )
    return events_df, detector


def stage_dependence(
    events_df: pd.DataFrame,
    X_all: np.ndarray,
    Z_all: np.ndarray,
    Y_all: np.ndarray,
    train_mask: np.ndarray,
    reg_cca: float,
    reg_krr: float,
    n_components: int,
) -> dict:
    """
    Stage 7: Compute I(X;Z) and I(X;Y|Z) on training events.

    LEAKAGE GUARD: all computations use only training events.
    """
    from dependence.contingency import mean_square_contingency
    from dependence.mutual_info import residual_dependence
    from regime.stability import regime_stability_analysis

    X_tr = X_all[train_mask]
    Z_tr = Z_all[train_mask]
    Y_tr = Y_all[train_mask]
    flags_tr = events_df["is_regime"].values[train_mask]

    logger.info("Stage 7a: Estimating I(X;Z) on %d training events", train_mask.sum())
    ixz_result = mean_square_contingency(X_tr, Z_tr, reg=reg_cca)
    logger.info("I(X;Z) = %.4f (NOCCO)", ixz_result.I_XZ_nocco)

    logger.info("Stage 7b: Estimating I(X;Y|Z) on %d training events", train_mask.sum())
    ixyz_result = residual_dependence(X_tr, Z_tr, Y_tr, reg_krr=reg_krr, reg_cca=reg_cca)
    logger.info("I(X;Y|Z) CM = %.4f", ixyz_result.I_XYZ_cm)

    logger.info("Stage 7c: Stability analysis by regime")
    stability = regime_stability_analysis(
        X_tr, Z_tr, Y_tr, flags_tr,
        reg_krr=reg_krr, n_permutations=100, min_n_per_regime=8,
    )

    return {
        "ixz_result": ixz_result,
        "ixyz_result": ixyz_result,
        "stability": stability,
    }


def stage_signal_and_backtest(
    events_df: pd.DataFrame,
    X_all: np.ndarray,
    Z_all: np.ndarray,
    Y_all: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    reg_cca: float,
    n_components: int,
    tc_bps: float,
) -> tuple[pd.DataFrame, object, object]:
    """
    Stage 8-9: Fit signal on training data, score test events, run backtest.

    LEAKAGE GUARD: SignalEstimator.fit() on training events;
    .score() on test events with frozen bandwidth.
    Returns (events_df_with_signal, signal_estimator, backtest_result).
    """
    from strategy.signals import SignalEstimator
    from strategy.backtest import run_backtest

    logger.info(
        "Stage 8: Fitting SignalEstimator on %d training events", train_mask.sum()
    )
    est = SignalEstimator(reg=reg_cca, n_components=n_components)
    est.fit(X_all[train_mask], Z_all[train_mask])

    # Score all events: train in-sample, test out-of-sample
    is_scores = est.score_insample()
    oos_scores = est.score(X_all[test_mask], Z_all[test_mask])

    signals_all = np.full(len(events_df), np.nan)
    signals_all[train_mask] = is_scores.scores
    signals_all[test_mask]  = oos_scores.scores
    events_df = events_df.copy()
    events_df["signal"] = signals_all

    logger.info("Stage 9: Running backtest on %d test events", test_mask.sum())
    test_df = events_df[test_mask].copy()
    test_df = test_df.rename(columns={"date": "date"})
    bt_result = run_backtest(test_df, tc_bps=tc_bps, n_quintiles=5)

    return events_df, est, bt_result


def stage_validation(
    events_df: pd.DataFrame,
    X_all: np.ndarray,
    Z_all: np.ndarray,
    reg_cca: float,
    n_components: int,
    tc_bps: float,
) -> object:
    """
    Stage 10: Walk-forward validation (2yr train / 1yr test / 6mo step).

    Attaches 'X' and 'Z' columns to events_df for the validation module,
    runs the walk-forward, returns ValidationResult.
    LEAKAGE GUARD: all splitting and fitting inside walk_forward_validate().
    """
    from strategy.validation import walk_forward_validate

    logger.info("Stage 10: Walk-forward validation")
    df = events_df.copy()
    df["X"] = list(X_all)
    df["Z"] = list(Z_all)

    try:
        val_result = walk_forward_validate(
            df,
            train_years=2.0,
            test_years=1.0,
            step_months=6,
            reg_cca=reg_cca,
            n_components=n_components,
            tc_bps=tc_bps,
            newey_west_lags=5,
        )
    except RuntimeError as exc:
        logger.warning("Walk-forward validation failed: %s", exc)
        val_result = None

    return val_result


def stage_plots(
    out_dir: Path,
    ixz_result,
    events_df: pd.DataFrame,
    X_all: np.ndarray,
    Z_all: np.ndarray,
    Y_all: np.ndarray,
    bt_result,
    reg_krr: float,
) -> None:
    """Stage 12: Generate four publication figures."""
    from analysis.plots import (
        build_publication_figure,
        plot_singular_value_decay,
        plot_residual_dependence_timeline,
        plot_cumulative_returns,
        plot_rolling_ic,
    )

    logger.info("Stage 12: Generating plots")

    # Rolling I(X;Y|Z) for Figure 2
    dates_win, ixyz_win, flags_win = _rolling_ixyz(
        events_df, X_all, Z_all, Y_all, reg_krr=reg_krr, min_window_size=8
    )

    # Signal IC for Figure 4 (only available when stage_signal_and_backtest ran)
    if "signal" in events_df.columns:
        has_signal = events_df["signal"].notna()
        dates_ic = pd.DatetimeIndex(events_df["date"][has_signal])
        signals_ic = events_df["signal"].values[has_signal]
        returns_ic = Y_all[has_signal]
    else:
        dates_ic = pd.DatetimeIndex([])
        signals_ic = np.array([])
        returns_ic = np.array([])

    try:
        fig = build_publication_figure(
            singular_values=ixz_result.singular_values,
            dates_xyz=dates_win,
            i_xyz_values=ixyz_win,
            regime_flags_xyz=flags_win,
            event_returns=bt_result.event_returns if bt_result.n_events > 0 else pd.DataFrame(
                {"date": [], "pnl_net": [], "pnl_gross": [], "is_regime": []}
            ),
            dates_ic=dates_ic,
            signals_ic=signals_ic,
            returns_ic=returns_ic,
        )
        fig_path = out_dir / "figures.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        logger.info("Figures saved to %s", fig_path)
    except Exception as exc:
        logger.warning("Figure generation failed: %s", exc)


def stage_report(
    out_dir: Path,
    tickers: list[str],
    date_start: str,
    date_end: str,
    events_df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    ixz_result,
    ixyz_result,
    stability,
    bt_result,
    val_result,
    reg_cca: float,
    reg_krr: float,
    n_components: int,
    tc_bps: float,
    seed: int,
    gamma_xz: float,
) -> None:
    """Stage 13: Generate markdown report."""
    from analysis.report import generate_report, write_report

    logger.info("Stage 13: Generating report")

    def _safe(obj, attr, default=None):
        return getattr(obj, attr, default)

    report = generate_report(
        tickers=tickers,
        date_start=date_start,
        date_end=date_end,
        n_events=len(events_df),
        n_events_train=int(train_mask.sum()),
        n_events_test=int(test_mask.sum()),
        ixz_nocco=ixz_result.I_XZ_nocco,
        ixz_svd=ixz_result.I_XZ_svd,
        top_singular_values=ixz_result.singular_values,
        gamma_xz=gamma_xz,
        ixyz_full=ixyz_result.I_XYZ_cm,
        ixyz_regime=_safe(stability, "I_regime"),
        ixyz_stable=_safe(stability, "I_stable"),
        ixyz_ratio=_safe(stability, "ratio"),
        n_regime_events=_safe(stability, "n_regime", 0),
        n_stable_events=_safe(stability, "n_stable", 0),
        h1_consistent=_safe(stability, "h1_consistent"),
        perm_p_value=_safe(stability, "perm_p_value"),
        sharpe_all=_safe(bt_result, "sharpe", float("nan")),
        sharpe_regime=_safe(bt_result, "sharpe_regime"),
        sharpe_stable=_safe(bt_result, "sharpe_stable"),
        max_drawdown=_safe(bt_result, "max_drawdown", 0.0),
        mean_pnl_bps=_safe(bt_result, "mean_return_bps", float("nan")),
        turnover_pa=_safe(bt_result, "turnover_pa", float("nan")),
        n_backtest_events=_safe(bt_result, "n_events", 0),
        n_folds=len(_safe(val_result, "folds", [])) if val_result else 0,
        pooled_nw_tstat=_safe(val_result, "pooled_nw_tstat", float("nan")),
        pooled_nw_pvalue=_safe(val_result, "pooled_nw_pvalue", float("nan")),
        mean_fold_sharpe=_safe(val_result, "mean_sharpe", float("nan")),
        sharpe_std=_safe(val_result, "sharpe_std", float("nan")),
        degradation_tstat=_safe(val_result, "degradation_tstat"),
        degradation_pvalue=_safe(val_result, "degradation_pvalue"),
        bh_rejected=int(_safe(val_result, "bh_component_rejected", np.zeros(0)).sum()),
        bh_total=len(_safe(val_result, "bh_component_rejected", [])),
        reg_cca=reg_cca,
        reg_krr=reg_krr,
        n_components=n_components,
        tc_bps=tc_bps,
        seed=seed,
    )

    report_path = out_dir / "report.md"
    write_report(report, str(report_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Residual Dependence Pipeline — I(X;Y|Z) in a financial context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers", nargs="+",
        default=["AAPL", "MSFT", "JPM"],
        help="S&P 500 ticker symbols to include (default: DEV_TICKERS)",
    )
    parser.add_argument(
        "--start", default="2021-01-01",
        help="Earliest 8-K filing date to fetch (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", default="2023-12-31",
        help="Latest 8-K filing date to fetch (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-end", default="2022-12-31",
        help="Train/test split date — events before this are training (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory to write figures, report, and cached data",
    )
    parser.add_argument(
        "--raw-dir", default="raw_filings",
        help="Directory where 8-K filing JSON is stored",
    )
    parser.add_argument(
        "--embed-cache-dir", default="embeddings_cache",
        help="Directory for transcript embedding cache (.npy files)",
    )
    parser.add_argument(
        "--n-components", type=int, default=20,
        help="Number of canonical components k for kernel CCA signal",
    )
    parser.add_argument(
        "--reg-cca", type=float, default=1e-3,
        help="NOCCO regularisation λ (for kernel CCA and I(X;Z))",
    )
    parser.add_argument(
        "--reg-krr", type=float, default=1e-2,
        help="KRR regularisation λ (for I(X;Y|Z) LOO predictions)",
    )
    parser.add_argument(
        "--tc-bps", type=float, default=5.0,
        help="One-way transaction cost in basis points",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="Skip EDGAR fetch and use existing raw_filings/ directory",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip walk-forward validation (saves time during development)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _set_seeds(args.seed)

    out_dir = _mkdir(args.output_dir)
    logger.info(
        "Pipeline start: tickers=%s, range=%s→%s, train_end=%s",
        args.tickers, args.start, args.end, args.train_end,
    )
    t0 = time.time()

    # ------------------------------------------------------------------
    # Stages 1-4: fetch, embed, price features, returns
    # ------------------------------------------------------------------
    if args.skip_fetch:
        logger.info("--skip-fetch: loading existing filings from %s", args.raw_dir)
        # Mimic the fetch stage but only scan existing JSON
        from features.embeddings import embed_transcripts_batch
        import json

        rows = []
        raw_path = Path(args.raw_dir)
        for ticker in args.tickers:
            for jpath in sorted((raw_path / ticker).glob("*.json")) if (raw_path / ticker).exists() else []:
                try:
                    with open(jpath) as fh:
                        rec = json.load(fh)
                    # edgar.py stores Item 2.02 flag in 'item_202_present'
                    if not rec.get("item_202_present", False):
                        continue
                    text = rec.get("raw_text", "") or rec.get("text", "") or ""
                    if len(text) < 50:
                        continue
                    rows.append({
                        "ticker": ticker,
                        "filing_date": pd.Timestamp(rec["filing_date"]),
                        "text": text,
                        "accession": rec.get("accession_number", ""),
                    })
                except Exception as exc:
                    logger.debug("Skip %s: %s", jpath, exc)

        if not rows:
            logger.error("No valid Item 2.02 filings in %s. Run without --skip-fetch.", args.raw_dir)
            return 1

        df = pd.DataFrame(rows)
        df = df[
            (df["filing_date"] >= pd.Timestamp(args.start))
            & (df["filing_date"] <= pd.Timestamp(args.end))
        ].reset_index(drop=True)

        records = [
            {"ticker": r["ticker"],
             "filing_date": str(r["filing_date"].date()),
             "raw_text": r["text"]}
            for _, r in df.iterrows()
        ]
        embeddings = embed_transcripts_batch(records, cache_dir=Path(args.embed_cache_dir))
        df["embedding"] = [
            embeddings.get((r["ticker"], str(r["filing_date"].date())))
            for _, r in df.iterrows()
        ]
        df = df[df["embedding"].apply(lambda e: e is not None)].reset_index(drop=True)
        logger.info("Loaded %d events from cache", len(df))
    else:
        df = stage_fetch_and_embed(
            args.tickers, args.start, args.end,
            args.raw_dir, args.embed_cache_dir, args.seed,
        )

    if df.empty:
        logger.error("No events after Stage 1-2. Aborting.")
        return 1

    df = stage_price_features(df)
    if df.empty:
        logger.error("No events after Stage 3. Aborting.")
        return 1

    df = stage_abnormal_returns(df)
    if df.empty:
        logger.error("No events after Stage 4. Aborting.")
        return 1

    # ------------------------------------------------------------------
    # Stage 5: align
    # ------------------------------------------------------------------
    events_df, X_all, Z_all, Y_all = stage_align(df)
    N = len(events_df)
    logger.info("Events aligned: N=%d  X=%s  Z=%s", N, X_all.shape, Z_all.shape)

    # Train/test split
    train_end_ts = pd.Timestamp(args.train_end)
    train_mask = events_df["date"].values <= train_end_ts
    test_mask  = events_df["date"].values >  train_end_ts
    n_train = train_mask.sum()
    n_test  = test_mask.sum()
    logger.info(
        "Train/test split at %s: %d train, %d test",
        args.train_end, n_train, n_test,
    )

    if n_train < 8:
        logger.error("Too few training events (%d). Expand --start or adjust --train-end.", n_train)
        return 1

    # ------------------------------------------------------------------
    # Stage 6: regime detection
    # ------------------------------------------------------------------
    events_df, detector = stage_regime(events_df, Z_all, train_mask)

    # ------------------------------------------------------------------
    # Stage 7: dependence estimation
    # ------------------------------------------------------------------
    dep_results = stage_dependence(
        events_df, X_all, Z_all, Y_all,
        train_mask, args.reg_cca, args.reg_krr, args.n_components,
    )
    ixz_result  = dep_results["ixz_result"]
    ixyz_result = dep_results["ixyz_result"]
    stability   = dep_results["stability"]

    # Power-law exponent for report
    gamma_xz = _power_law_exponent(ixz_result.singular_values)
    logger.info("Power-law exponent γ_XZ = %.4f", gamma_xz)

    # ------------------------------------------------------------------
    # Stages 8-9: signal and backtest
    # ------------------------------------------------------------------
    if n_test >= 4:
        events_df, signal_est, bt_result = stage_signal_and_backtest(
            events_df, X_all, Z_all, Y_all,
            train_mask, test_mask, args.reg_cca, args.n_components, args.tc_bps,
        )
        from strategy.backtest import summarize_backtest
        logger.info("\n%s", summarize_backtest(bt_result))
    else:
        logger.warning("Too few test events (%d) for backtest. Skipping.", n_test)
        from strategy.backtest import BacktestResult
        bt_result = BacktestResult(
            sharpe=float("nan"), sharpe_gross=float("nan"),
            mean_return_bps=float("nan"), std_return_bps=float("nan"),
            max_drawdown=0.0, turnover_pa=float("nan"),
            n_events=0, n_cohorts=0,
            sharpe_regime=None, sharpe_stable=None,
            n_regime_events=0, n_stable_events=0,
            event_returns=pd.DataFrame(), cohort_pnl=pd.Series(dtype=float),
        )

    # ------------------------------------------------------------------
    # Stage 10: walk-forward validation
    # ------------------------------------------------------------------
    val_result = None
    if not args.skip_validation and N >= 40:
        val_result = stage_validation(
            events_df, X_all, Z_all, args.reg_cca, args.n_components, args.tc_bps
        )
        if val_result:
            from strategy.validation import summarize_validation
            logger.info("\n%s", summarize_validation(val_result))
    else:
        logger.info("Skipping walk-forward validation (--skip-validation or N=%d too small).", N)

    # ------------------------------------------------------------------
    # Stage 12: plots
    # ------------------------------------------------------------------
    stage_plots(out_dir, ixz_result, events_df, X_all, Z_all, Y_all, bt_result, args.reg_krr)

    # ------------------------------------------------------------------
    # Stage 13: report
    # ------------------------------------------------------------------
    stage_report(
        out_dir, args.tickers, args.start, args.end,
        events_df, train_mask, test_mask,
        ixz_result, ixyz_result, stability, bt_result, val_result,
        args.reg_cca, args.reg_krr, args.n_components, args.tc_bps, args.seed, gamma_xz,
    )

    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.1fs. Outputs in %s/", elapsed, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
