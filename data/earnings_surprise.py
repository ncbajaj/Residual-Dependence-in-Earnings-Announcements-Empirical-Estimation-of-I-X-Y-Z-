"""
Fetch analyst EPS estimate vs. reported EPS ("earnings surprise") via
yfinance and align it to 8-K filing dates.

WHY THIS MODULE EXISTS
-----------------------
The rest of the pipeline uses two modalities at announcement time:
  X = pre-announcement price features ([t-22, t-2])
  Z = earnings call transcript embedding (8-K Item 2.02)
Neither one contains the single most well-documented predictor of
post-earnings-announcement drift direction in the finance literature:
whether the company beat or missed analyst consensus. This module adds
that as a third, same-timing feature: S = earnings surprise.

LOOKAHEAD BIAS
---------------
The EPS estimate and the reported EPS are BOTH public as of the earnings
release -- the surprise is known at the exact same moment as the
transcript content (t=0), same as Z. Using it to predict the forward
[t+1, t+5] return is not lookahead, for the same reason using transcript
content isn't: both are revealed at the announcement, before the forward
return window begins.

MATCHING TO FILING DATES
--------------------------
yfinance's earnings-date timestamps and EDGAR's 8-K filing dates are
usually the same calendar day but can differ by a day (e.g. after-hours
earnings release vs. next-day 8-K filing, or data-vendor timezone
rounding). We match each (ticker, filing_date) event to the closest
earnings-date record within a tolerance window and log/drop anything
that doesn't match -- no silent padding with an unrelated quarter's
surprise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# How many calendar days apart an earnings-date record and a filing date
# may be and still be considered the same event.
MATCH_TOLERANCE_DAYS = 3

# How many quarters of earnings-date history to request per ticker.
# yfinance's `limit` is a soft cap; in practice more rows are often
# returned, but we request generously to cover a multi-year backtest.
HISTORY_LIMIT = 40


@dataclass
class SurpriseRecord:
    ticker: str
    earnings_date: pd.Timestamp
    eps_estimate: Optional[float]
    eps_reported: Optional[float]
    surprise_pct: Optional[float]


def fetch_surprise_history(ticker: str, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Return a DataFrame of (earnings_date, eps_estimate, eps_reported,
    surprise_pct) for `ticker`, optionally cached to disk as CSV.

    Returns an empty DataFrame (never raises) on fetch failure, so a bad
    ticker never aborts a batch job -- callers should check len() == 0.
    """
    cache_path = (cache_dir / f"{ticker}.csv") if cache_dir else None
    if cache_path and cache_path.exists():
        try:
            df = pd.read_csv(cache_path, parse_dates=["earnings_date"])
            logger.debug("%s: loaded %d surprise records from cache", ticker, len(df))
            return df
        except Exception as exc:
            logger.warning("%s: failed to read surprise cache, refetching: %s", ticker, exc)

    try:
        raw = yf.Ticker(ticker).get_earnings_dates(limit=HISTORY_LIMIT)
    except Exception as exc:
        logger.error("%s: get_earnings_dates failed: %s", ticker, exc)
        return pd.DataFrame(columns=["earnings_date", "eps_estimate", "eps_reported", "surprise_pct"])

    if raw is None or raw.empty:
        logger.warning("%s: no earnings-date history returned", ticker)
        return pd.DataFrame(columns=["earnings_date", "eps_estimate", "eps_reported", "surprise_pct"])

    df = raw.reset_index().rename(columns={
        "Earnings Date": "earnings_date",
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "eps_reported",
        "Surprise(%)": "surprise_pct",
    })
    df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.tz_localize(None)
    df = df[["earnings_date", "eps_estimate", "eps_reported", "surprise_pct"]]

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    logger.info("%s: fetched %d earnings-date records (%d with a reported surprise)",
                ticker, len(df), df["surprise_pct"].notna().sum())
    return df


def match_surprise_to_events(
    events_df: pd.DataFrame,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    For each row of `events_df` (needs columns: ticker, filing_date or date),
    find the closest earnings-date surprise record within
    MATCH_TOLERANCE_DAYS and attach eps_estimate, eps_reported,
    surprise_pct columns (NaN if no match within tolerance).

    Does not mutate events_df; returns a new DataFrame with the three
    columns appended, same row order and length as the input.
    """
    df = events_df.copy()
    date_col = "date" if "date" in df.columns else "filing_date"
    df["_event_date"] = pd.to_datetime(df[date_col])

    out_rows = []
    surprise_cache: dict[str, pd.DataFrame] = {}
    n_matched = 0

    for ticker, group in df.groupby("ticker"):
        if ticker not in surprise_cache:
            surprise_cache[ticker] = fetch_surprise_history(ticker, cache_dir=cache_dir)
        hist = surprise_cache[ticker]

        for idx, row in group.iterrows():
            match = {"eps_estimate": None, "eps_reported": None, "surprise_pct": None}
            if not hist.empty:
                deltas = (hist["earnings_date"] - row["_event_date"]).abs()
                best = deltas.idxmin()
                if deltas.loc[best] <= timedelta(days=MATCH_TOLERANCE_DAYS):
                    match = {
                        "eps_estimate": hist.loc[best, "eps_estimate"],
                        "eps_reported": hist.loc[best, "eps_reported"],
                        "surprise_pct": hist.loc[best, "surprise_pct"],
                    }
                    n_matched += 1
            out_rows.append({"_idx": idx, **match})

    match_df = pd.DataFrame(out_rows).set_index("_idx")
    df = df.join(match_df)
    df = df.drop(columns=["_event_date"])

    logger.info("Matched %d / %d events to an earnings surprise record (tolerance=%d days)",
                n_matched, len(df), MATCH_TOLERANCE_DAYS)
    return df
