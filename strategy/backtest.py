"""
Long/short quintile backtest on earnings events.

STRATEGY MECHANICS
------------------
For each "event cohort" (all events on the same filing date):
  1. Rank events by information-density signal c(x_i, z_i).
  2. Long top quintile (top 20% by signal), short bottom quintile (bottom 20%).
  3. Equal dollar weights within each leg: ±1/n_long per long, ∓1/n_short per short.
  4. Dollar neutral: same dollar exposure long and short.
  5. Hold for 5 trading days (matching the abnormal return window in prices.py).

P&L per cohort:
    gross = mean(r_long) - mean(r_short)
    cost  = 2 × tc_bps/10000      (one round trip per leg, normalized)
    net   = gross - cost

Transaction cost model:
    5 bps per side × 2 sides (entry + exit) = 10 bps round-trip per leg.
    For a dollar-neutral book investing $1 gross per leg:
        cost = 2 × round_trip = 2 × 10 bps = 20 bps per cohort.

Sharpe annualization:
    Events are irregularly spaced (earnings calendar), so we annualize by
    the observed event rate: Sharpe = (mean_pnl / std_pnl) × sqrt(events_per_year).
    events_per_year = N_events × (252 / observation_span_trading_days) — rough.
    (This treats events as IID; a more rigorous approach would account for
    clustering in earnings season.)

Regime split:
    Events with regime_flag=True are separated. The Sharpe computation is run
    independently on regime and stable subsets to test for differential performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROUND_TRIP_BPS = 10    # bps per leg (entry + exit)
_DEFAULT_TC_BPS = 5     # bps per side (one-way)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EventReturn:
    """Per-event backtest record."""
    date: pd.Timestamp
    ticker: str
    signal: float
    abnormal_return: float
    weight: float            # +1/n_long or -1/n_short (dollar-neutral)
    pnl_gross: float         # weight × abnormal_return
    pnl_net: float           # pnl_gross − per-event cost
    is_long: bool
    is_regime: bool          # True if event falls in a regime-shift window


@dataclass
class BacktestResult:
    """
    Full backtest output.

    Attributes
    ----------
    sharpe          : Annualized Sharpe ratio on net P&L (all events).
    sharpe_gross    : Annualized Sharpe ratio on gross P&L (pre-cost).
    mean_return_bps : Mean net P&L per event in basis points.
    std_return_bps  : Std of net P&L per event in basis points.
    max_drawdown    : Maximum peak-to-trough drawdown (as a fraction of portfolio value).
    turnover_pa     : Approximate one-way turnover per year (number of portfolio turns).
    n_events        : Total events traded.
    n_cohorts       : Number of event cohorts (distinct dates with ≥2 events).
    sharpe_regime   : Sharpe during regime-shift events (None if < 4 events).
    sharpe_stable   : Sharpe during stable events (None if < 4 events).
    n_regime_events : Events in regime-shift cohorts.
    n_stable_events : Events in stable cohorts.
    event_returns   : pd.DataFrame with one row per event (all backtest details).
    cohort_pnl      : pd.Series of per-cohort net P&L, indexed by date.
    """
    sharpe: float
    sharpe_gross: float
    mean_return_bps: float
    std_return_bps: float
    max_drawdown: float
    turnover_pa: float
    n_events: int
    n_cohorts: int
    sharpe_regime: Optional[float]
    sharpe_stable: Optional[float]
    n_regime_events: int
    n_stable_events: int
    event_returns: pd.DataFrame
    cohort_pnl: pd.Series


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _sharpe(pnl: np.ndarray, events_per_year: float) -> float:
    """Annualized Sharpe from an array of event-level P&L observations."""
    if len(pnl) < 4 or pnl.std() < 1e-12:
        return float("nan")
    return float(pnl.mean() / pnl.std() * np.sqrt(events_per_year))


def _max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown on a cumulative P&L series.

    Returns a non-negative fraction: 0 = no drawdown, 0.1 = 10% peak-to-trough.
    """
    if len(cumulative_pnl) == 0:
        return 0.0
    peak = np.maximum.accumulate(cumulative_pnl)
    drawdown = peak - cumulative_pnl
    return float(drawdown.max())


def _events_per_year(dates: pd.DatetimeIndex) -> float:
    """Estimate average events per year from observation dates."""
    if len(dates) < 2:
        return float("nan")
    span_days = (dates.max() - dates.min()).days
    if span_days == 0:
        return float("nan")
    return float(len(dates) / (span_days / 365.25))


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest(
    events: pd.DataFrame,
    tc_bps: float = _DEFAULT_TC_BPS,
    n_quintiles: int = 5,
    min_cohort_size: int = 2,
) -> BacktestResult:
    """
    Run the long/short quintile backtest on earnings events.

    Parameters
    ----------
    events : DataFrame with columns:
        - date          : pd.Timestamp (filing date / event date)
        - ticker        : str
        - signal        : float (information-density score from SignalEstimator)
        - abnormal_return : float (5-day abnormal return, from prices.py)
        - is_regime     : bool (True if event is in a regime-shift window)
      Additional columns are passed through to the output.
    tc_bps      : one-way transaction cost in basis points (default 5 bps).
    n_quintiles : number of signal quintiles; top and bottom traded (default 5).
    min_cohort_size : minimum events on a date to form long+short positions.

    Returns
    -------
    BacktestResult

    Notes
    -----
    Cohort = all events on the same filing date. Within a cohort:
      - n_long = n_short = floor(n_cohort / n_quintiles)
      - If n_cohort < 2*n_long, both legs have floor(n_cohort/2) events.
      - Dollar-neutral: weights sum to 0; each leg summed weight = 0.5.
    """
    required = {"date", "ticker", "signal", "abnormal_return", "is_regime"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events DataFrame missing columns: {missing}")

    df = events.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["signal", "abnormal_return"])
    df = df.sort_values("date").reset_index(drop=True)

    round_trip_cost = 2 * tc_bps / 10_000   # fraction (e.g., 0.001 for 10 bps)

    event_rows: list[EventReturn] = []
    cohort_dates: list[pd.Timestamp] = []
    cohort_pnls: list[float] = []

    for date, cohort in df.groupby("date", sort=True):
        n = len(cohort)
        if n < min_cohort_size:
            continue

        n_per_leg = max(1, n // n_quintiles)
        n_per_leg = min(n_per_leg, n // 2)   # never more than half the cohort

        cohort_sorted = cohort.sort_values("signal")
        short_idx = cohort_sorted.index[:n_per_leg]   # bottom quintile → short
        long_idx  = cohort_sorted.index[-n_per_leg:]  # top quintile → long

        w_long  = +1.0 / n_per_leg
        w_short = -1.0 / n_per_leg

        cohort_pnl_net = 0.0
        cohort_events: list[EventReturn] = []

        for idx in long_idx:
            row = cohort.loc[idx]
            ret = float(row["abnormal_return"])
            gross = w_long * ret
            net = gross - round_trip_cost / n_per_leg   # share cost equally across positions
            cohort_pnl_net += net
            cohort_events.append(EventReturn(
                date=date, ticker=str(row["ticker"]),
                signal=float(row["signal"]), abnormal_return=ret,
                weight=w_long, pnl_gross=gross, pnl_net=net,
                is_long=True, is_regime=bool(row["is_regime"]),
            ))

        for idx in short_idx:
            row = cohort.loc[idx]
            ret = float(row["abnormal_return"])
            gross = w_short * ret
            net = gross - round_trip_cost / n_per_leg
            cohort_pnl_net += net
            cohort_events.append(EventReturn(
                date=date, ticker=str(row["ticker"]),
                signal=float(row["signal"]), abnormal_return=ret,
                weight=w_short, pnl_gross=gross, pnl_net=net,
                is_long=False, is_regime=bool(row["is_regime"]),
            ))

        event_rows.extend(cohort_events)
        cohort_dates.append(date)
        cohort_pnls.append(cohort_pnl_net)

    if not event_rows:
        logger.warning("No tradable cohorts found.")
        empty = pd.DataFrame()
        empty_s = pd.Series(dtype=float)
        return BacktestResult(
            sharpe=float("nan"), sharpe_gross=float("nan"),
            mean_return_bps=float("nan"), std_return_bps=float("nan"),
            max_drawdown=0.0, turnover_pa=float("nan"),
            n_events=0, n_cohorts=0,
            sharpe_regime=None, sharpe_stable=None,
            n_regime_events=0, n_stable_events=0,
            event_returns=empty, cohort_pnl=empty_s,
        )

    # --- Aggregate results ---------------------------------------------------
    er_df = pd.DataFrame([
        {
            "date": e.date, "ticker": e.ticker, "signal": e.signal,
            "abnormal_return": e.abnormal_return, "weight": e.weight,
            "pnl_gross": e.pnl_gross, "pnl_net": e.pnl_net,
            "is_long": e.is_long, "is_regime": e.is_regime,
        }
        for e in event_rows
    ])
    cohort_pnl_s = pd.Series(cohort_pnls, index=pd.DatetimeIndex(cohort_dates), name="pnl_net")

    # Event-level P&L arrays (for Sharpe, drawdown)
    pnl_net   = er_df["pnl_net"].values
    pnl_gross = er_df["pnl_gross"].values
    event_dates = pd.DatetimeIndex(er_df["date"])
    epa = _events_per_year(event_dates)

    sharpe       = _sharpe(pnl_net, epa)
    sharpe_gross = _sharpe(pnl_gross, epa)

    # Max drawdown on cumulative cohort P&L (calendar-ordered)
    cum_pnl = np.cumsum(cohort_pnls)
    mdd = _max_drawdown(cum_pnl)

    # Rough one-way turnover per year: 2 trades per event (entry + exit)
    span_years = (event_dates.max() - event_dates.min()).days / 365.25
    turnover_pa = float(2 * len(er_df) / max(span_years, 1.0 / 12))

    # Regime vs stable split
    regime_mask = er_df["is_regime"].values.astype(bool)
    n_regime = int(regime_mask.sum())
    n_stable = int((~regime_mask).sum())

    sharpe_regime = _sharpe(pnl_net[regime_mask], epa) if n_regime >= 4 else None
    sharpe_stable = _sharpe(pnl_net[~regime_mask], epa) if n_stable >= 4 else None

    logger.info(
        "Backtest: %d events, %d cohorts, Sharpe=%.2f (gross=%.2f), "
        "MDD=%.1f%%, regime Sharpe=%.2f, stable Sharpe=%.2f",
        len(er_df), len(cohort_dates), sharpe, sharpe_gross,
        mdd * 100,
        sharpe_regime if sharpe_regime is not None else float("nan"),
        sharpe_stable if sharpe_stable is not None else float("nan"),
    )

    return BacktestResult(
        sharpe=sharpe,
        sharpe_gross=sharpe_gross,
        mean_return_bps=float(pnl_net.mean() * 10_000),
        std_return_bps=float(pnl_net.std() * 10_000),
        max_drawdown=mdd,
        turnover_pa=turnover_pa,
        n_events=len(er_df),
        n_cohorts=len(cohort_dates),
        sharpe_regime=sharpe_regime,
        sharpe_stable=sharpe_stable,
        n_regime_events=n_regime,
        n_stable_events=n_stable,
        event_returns=er_df,
        cohort_pnl=cohort_pnl_s,
    )


def summarize_backtest(result: BacktestResult) -> str:
    """Format key backtest statistics as a multi-line string."""
    lines = [
        f"Backtest Summary",
        f"  Events traded   : {result.n_events}  ({result.n_cohorts} cohorts)",
        f"  Sharpe (net)    : {result.sharpe:.3f}",
        f"  Sharpe (gross)  : {result.sharpe_gross:.3f}",
        f"  Mean P&L        : {result.mean_return_bps:.1f} bps/event",
        f"  Std  P&L        : {result.std_return_bps:.1f} bps/event",
        f"  Max drawdown    : {result.max_drawdown * 100:.2f}%",
        f"  Turnover/yr     : {result.turnover_pa:.1f}×",
        f"  Regime events   : {result.n_regime_events}  "
        f"Sharpe={result.sharpe_regime if result.sharpe_regime is not None else 'n/a':.3g}",
        f"  Stable events   : {result.n_stable_events}  "
        f"Sharpe={result.sharpe_stable if result.sharpe_stable is not None else 'n/a':.3g}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Backtest smoke test with synthetic events")
    parser.add_argument("--n", type=int, default=200, help="Total synthetic events")
    parser.add_argument("--skill", type=float, default=0.15,
                        help="Signal-to-return correlation (strategy 'edge')")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    N = args.n
    skill = args.skill

    # Synthetic events spread over 2 years, ~4 tickers per day
    from datetime import timedelta
    base = pd.Timestamp("2022-01-01")
    event_dates = [base + timedelta(days=int(d)) for d in np.sort(rng.integers(0, 500, N))]

    signals = rng.standard_normal(N)
    # With skill: returns partially correlated with signals
    returns = skill * signals + np.sqrt(1 - skill**2) * rng.standard_normal(N) * 0.02
    is_regime = rng.random(N) < 0.25

    events_df = pd.DataFrame({
        "date": event_dates,
        "ticker": [f"T{i%20:02d}" for i in range(N)],
        "signal": signals,
        "abnormal_return": returns,
        "is_regime": is_regime,
    })

    result = run_backtest(events_df, tc_bps=5.0, n_quintiles=5)
    print(f"\n{summarize_backtest(result)}")
