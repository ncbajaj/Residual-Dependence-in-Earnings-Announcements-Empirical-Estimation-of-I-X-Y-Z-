"""
Fetch adjusted OHLCV data via yfinance and compute forward 5-day abnormal
returns (stock minus SPY) aligned to 8-K filing dates.

ALIGNMENT / LOOKAHEAD BIAS DECISIONS
-------------------------------------
All decisions that could introduce lookahead bias are documented below.

1. Filing date as event date t=0:
   The SEC filing date is the day the 8-K appears on EDGAR. After-hours
   filings are NOT adjusted to t+1 here — we treat the filing date as the
   announcement date regardless of intraday timing. This is a known
   simplification; production code should check filing time vs. market
   close (4pm ET) and shift to t+1 for after-hours filings.

2. Price observation at t=0:
   We use the CLOSING price on the filing date as the base price (P_t).
   This assumes we could not trade on the filing-day announcement itself
   (conservative and standard in event studies). If the filing arrives
   pre-market, using P_{t-1} close would be more accurate.

3. Forward window [t+1, t+5]:
   Return = (P_{t+5} close / P_{t+1} open) - 1, where:
     - P_{t+1} open is the first tradeable price after the announcement
     - P_{t+5} close is 5 TRADING DAYS after t+1 (not calendar days)
   We never use any price on day t in the forward return calculation,
   which eliminates any overlap with the announcement-day move.

4. SPY as market benchmark:
   Abnormal return = stock return - SPY return over the identical window
   [t+1 open, t+5 close]. This matches trading days exactly, so market
   conditions are controlled for the same period.

5. Data download window:
   We download [event_date - 10 trading days, event_date + 10 trading days]
   to handle weekends/holidays without fetching the entire price history.
   The expansion is purely backward-looking relative to the return window,
   so no lookahead contamination.

6. Adjusted prices:
   yfinance returns split/dividend-adjusted prices by default (`auto_adjust=True`).
   This is correct for return computation but means prices are restated
   retroactively — using adjusted prices for a live/real-time system would
   introduce lookahead. Here (historical research) it is appropriate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Number of trading days in the forward return window
FORWARD_DAYS = 5

# Calendar-day buffer used when downloading price data around an event.
# 14 calendar days ≈ 10 trading days, giving enough room for holidays.
_PRICE_BUFFER_DAYS = 14


@dataclass
class AbnormalReturn:
    ticker: str
    filing_date: str          # YYYY-MM-DD (t=0, the event date)
    entry_date: str           # YYYY-MM-DD (t+1, first tradeable day)
    exit_date: str            # YYYY-MM-DD (t+5 in trading days from t+1)
    stock_return: Optional[float]    # (P_{t+5} close / P_{t+1} open) - 1
    spy_return: Optional[float]      # same window for SPY
    abnormal_return: Optional[float] # stock_return - spy_return
    error: Optional[str] = None      # non-None if computation failed


def _next_trading_day(ref_date: date, prices: pd.DataFrame) -> Optional[date]:
    """Return the first date in `prices` index that is strictly after `ref_date`."""
    future = prices.index[prices.index > pd.Timestamp(ref_date)]
    return future[0].date() if len(future) > 0 else None


def _nth_trading_day_after(start: date, n: int, prices: pd.DataFrame) -> Optional[date]:
    """Return the date of the n-th trading day on or after `start` in `prices`."""
    future = prices.index[prices.index >= pd.Timestamp(start)]
    if len(future) < n:
        return None
    return future[n - 1].date()


def compute_abnormal_return(
    ticker: str,
    filing_date: str,
) -> AbnormalReturn:
    """
    Download prices for `ticker` and SPY around `filing_date` and return
    the forward 5-day abnormal return. See module docstring for alignment
    decisions.

    Returns an AbnormalReturn with error set (and return fields None) on
    any data failure — never raises.
    """
    event = date.fromisoformat(filing_date)
    dl_start = (event - timedelta(days=_PRICE_BUFFER_DAYS)).isoformat()
    dl_end = (event + timedelta(days=_PRICE_BUFFER_DAYS)).isoformat()

    try:
        # Download stock and benchmark together to minimise API calls.
        # progress=False suppresses yfinance console output.
        raw = yf.download(
            [ticker, "SPY"],
            start=dl_start,
            end=dl_end,
            auto_adjust=True,
            progress=False,
            # group_by="ticker" gives a MultiIndex; we handle it below
        )
    except Exception as exc:
        logger.error("yfinance download failed for %s around %s: %s", ticker, filing_date, exc)
        return AbnormalReturn(
            ticker=ticker, filing_date=filing_date,
            entry_date="", exit_date="",
            stock_return=None, spy_return=None, abnormal_return=None,
            error=f"Download failed: {exc}",
        )

    if raw.empty:
        return AbnormalReturn(
            ticker=ticker, filing_date=filing_date,
            entry_date="", exit_date="",
            stock_return=None, spy_return=None, abnormal_return=None,
            error="Empty price data returned",
        )

    # yfinance 1.x always returns MultiIndex columns (Price, Ticker),
    # even for multi-ticker downloads. Access as (price_type, ticker_symbol).
    def _extract(df: pd.DataFrame, sym: str, col: str) -> pd.Series:
        if isinstance(df.columns, pd.MultiIndex):
            return df[(col, sym)].dropna()
        # Flat-column fallback for hypothetical future yfinance versions
        return df[col].squeeze().dropna()

    try:
        stock_close = _extract(raw, ticker, "Close")
        stock_open  = _extract(raw, ticker, "Open")
        spy_close   = _extract(raw, "SPY",   "Close")
        spy_open    = _extract(raw, "SPY",   "Open")
    except KeyError as exc:
        return AbnormalReturn(
            ticker=ticker, filing_date=filing_date,
            entry_date="", exit_date="",
            stock_return=None, spy_return=None, abnormal_return=None,
            error=f"Column extraction failed: {exc}",
        )

    # Determine t+1 (entry) and t+5 (exit) trading days.
    # We use stock_close index as the trading calendar (SPY should match).
    entry = _next_trading_day(event, stock_close.to_frame())
    if entry is None:
        return AbnormalReturn(
            ticker=ticker, filing_date=filing_date,
            entry_date="", exit_date="",
            stock_return=None, spy_return=None, abnormal_return=None,
            error=f"No trading day found after {filing_date} in download window",
        )

    exit_ = _nth_trading_day_after(entry, FORWARD_DAYS, stock_close.to_frame())
    if exit_ is None:
        return AbnormalReturn(
            ticker=ticker, filing_date=filing_date,
            entry_date=entry.isoformat(), exit_date="",
            stock_return=None, spy_return=None, abnormal_return=None,
            error=f"Fewer than {FORWARD_DAYS} trading days available after {entry}",
        )

    entry_ts = pd.Timestamp(entry)
    exit_ts  = pd.Timestamp(exit_)

    def _safe_get(series: pd.Series, ts: pd.Timestamp, label: str) -> float | None:
        if ts not in series.index:
            logger.warning("%s: %s not in price index for %s", ticker, ts.date(), label)
            return None
        return float(series[ts])

    p_entry_stock = _safe_get(stock_open,  entry_ts, "stock open t+1")
    p_exit_stock  = _safe_get(stock_close, exit_ts,  "stock close t+5")
    p_entry_spy   = _safe_get(spy_open,    entry_ts, "SPY open t+1")
    p_exit_spy    = _safe_get(spy_close,   exit_ts,  "SPY close t+5")

    if any(v is None for v in [p_entry_stock, p_exit_stock, p_entry_spy, p_exit_spy]):
        return AbnormalReturn(
            ticker=ticker, filing_date=filing_date,
            entry_date=entry.isoformat(), exit_date=exit_.isoformat(),
            stock_return=None, spy_return=None, abnormal_return=None,
            error="Missing price at entry or exit date",
        )

    stock_ret = p_exit_stock / p_entry_stock - 1.0   # type: ignore[operator]
    spy_ret   = p_exit_spy   / p_entry_spy   - 1.0   # type: ignore[operator]
    ab_ret    = stock_ret - spy_ret

    return AbnormalReturn(
        ticker=ticker,
        filing_date=filing_date,
        entry_date=entry.isoformat(),
        exit_date=exit_.isoformat(),
        stock_return=round(stock_ret, 6),
        spy_return=round(spy_ret, 6),
        abnormal_return=round(ab_ret, 6),
    )


def compute_abnormal_returns_batch(
    filing_dates: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Compute abnormal returns for a dict of {ticker: [filing_date, ...]}.

    Returns a DataFrame with columns:
        ticker, filing_date, entry_date, exit_date,
        stock_return, spy_return, abnormal_return, error

    Rows with errors have NaN returns but are retained (never dropped).
    """
    rows = []
    for ticker, dates in filing_dates.items():
        for fd in dates:
            ar = compute_abnormal_return(ticker, fd)
            rows.append({
                "ticker": ar.ticker,
                "filing_date": ar.filing_date,
                "entry_date": ar.entry_date,
                "exit_date": ar.exit_date,
                "stock_return": ar.stock_return,
                "spy_return": ar.spy_return,
                "abnormal_return": ar.abnormal_return,
                "error": ar.error,
            })
    df = pd.DataFrame(rows)
    n_err = df["error"].notna().sum()
    if n_err:
        logger.warning("%d / %d return computations had errors", n_err, len(df))
    return df


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compute abnormal returns around 8-K dates")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--date", default="2023-11-02", dest="filing_date",
                        help="Example AAPL earnings 8-K date (2023-11-02)")
    args = parser.parse_args()

    result = compute_abnormal_return(args.ticker, args.filing_date)
    print(result)
