"""
Price-based features — alpha(X) in Mehta & Harchaoui (ICML 2025).

In the paper's zero-shot framework, alpha: X -> H_X is the encoder for the
source modality X. Here X is the pre-announcement price history, and alpha
maps it to a feature vector in R^3:

    alpha(X) = [momentum_20d, realized_vol_20d, volume_ratio_20d]

These features summarize the market's prior beliefs about the firm before
the earnings announcement. The residual dependence I(X;Y|Z) — Theorem 1 —
asks how much incremental predictive information alpha(X) contains about
Y (forward abnormal return) after controlling for beta(Z) (the transcript
embedding). A researcher's goal is to find Z rich enough that alpha(X)
becomes redundant.

LOOKAHEAD BIAS POLICY (applies to every function in this module)
----------------------------------------------------------------
All feature values are computed using only prices that are strictly BEFORE
the announcement date. Specifically:

  - The lookback window is [t-21, t-2] trading days relative to the
    announcement date t=0, giving 20 days of data.
  - We exclude t-1 to guard against any information leakage through
    same-day pre-market activity that might correlate with filing content.
  - Cross-sectional normalization (z-scoring per date) uses only the
    firms that have announced on THAT date. In practice a researcher
    normalizes over all firms with the same filing date. Using the
    full-sample distribution would be a subtle form of lookahead.

Any future change that widens the lookback window beyond t-2 or uses
post-announcement data must update this comment block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

LOOKBACK_DAYS = 20          # number of trading days in each feature window
EXCLUDE_PRE_ANNOUNCEMENT = 2  # exclude t-1 to guard against same-day leakage
# Window used: [t - LOOKBACK_DAYS - EXCLUDE_PRE_ANNOUNCEMENT + 1, t - EXCLUDE_PRE_ANNOUNCEMENT]
# i.e., 20 trading days ending two days before the filing date.

FEATURE_NAMES = ["momentum_20d", "realized_vol_20d", "volume_ratio_20d"]


# -------------------------------------------------------------------------
# Per-filing raw feature computation
# -------------------------------------------------------------------------

@dataclass
class RawFeatures:
    ticker: str
    filing_date: str

    # alpha(X) components — paper notation in docstrings below
    momentum_20d: Optional[float] = None
    realized_vol_20d: Optional[float] = None
    volume_ratio_20d: Optional[float] = None
    error: Optional[str] = None

    def to_array(self) -> Optional[np.ndarray]:
        """Return alpha(X) as a (3,) float32 array, or None if any feature is missing."""
        if any(v is None for v in [self.momentum_20d, self.realized_vol_20d, self.volume_ratio_20d]):
            return None
        return np.array(
            [self.momentum_20d, self.realized_vol_20d, self.volume_ratio_20d],
            dtype=np.float32,
        )


def compute_raw_features(ticker: str, filing_date: str) -> RawFeatures:
    """
    Compute the three components of alpha(X) for a single ticker/date.

    Paper reference: alpha: X -> H_X (Section 2). The output is the
    pre-announcement feature vector used in the source modality regression.

    Lookahead guard: only prices from [t-21, t-2] trading days are used.
    See module docstring for the full policy.

    Returns a RawFeatures with .error set (never raises) on any data failure.
    """
    rec = RawFeatures(ticker=ticker, filing_date=filing_date)

    # Download a generous window of price history.
    # 40 calendar days gives ~28 trading days — more than the 22 we need.
    # Using calendar days avoids off-by-one errors around holidays.
    import datetime
    event = datetime.date.fromisoformat(filing_date)
    dl_start = (event - datetime.timedelta(days=60)).isoformat()
    dl_end   = (event - datetime.timedelta(days=1)).isoformat()  # strictly before t=0

    try:
        raw = yf.download(
            ticker,
            start=dl_start,
            end=dl_end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        rec.error = f"Download failed: {exc}"
        logger.error("Price download failed for %s %s: %s", ticker, filing_date, exc)
        return rec

    if raw.empty:
        rec.error = "Empty price data"
        return rec

    # yfinance 1.x returns MultiIndex columns (Price, Ticker) even for single-
    # ticker downloads. Access via tuple to get a Series, not a DataFrame.
    try:
        close = raw[("Close", ticker)].dropna()
        volume = raw[("Volume", ticker)].dropna()
    except KeyError:
        # Fallback for hypothetical future yfinance versions that flatten columns
        close = raw["Close"].squeeze().dropna()
        volume = raw["Volume"].squeeze().dropna()

    # Align close and volume to a shared trading-day index
    shared_idx = close.index.intersection(volume.index)
    close = close.loc[shared_idx]
    volume = volume.loc[shared_idx]

    # The lookback window ends at t-EXCLUDE_PRE_ANNOUNCEMENT (i.e., t-2).
    # We exclude the last (EXCLUDE_PRE_ANNOUNCEMENT - 1) rows = 1 row.
    # After the download's end=t-1 cutoff, the last row IS t-1, which we exclude.
    # So the effective window is the 20 rows ending at index [-2] (= t-2).
    window_close  = close.iloc[-(LOOKBACK_DAYS + EXCLUDE_PRE_ANNOUNCEMENT - 1):-1]
    window_volume = volume.iloc[-(LOOKBACK_DAYS + EXCLUDE_PRE_ANNOUNCEMENT - 1):-1]

    if len(window_close) < LOOKBACK_DAYS:
        rec.error = f"Insufficient price history: need {LOOKBACK_DAYS} days, got {len(window_close)}"
        logger.warning("%s %s: %s", ticker, filing_date, rec.error)
        return rec

    rec.momentum_20d    = _momentum(window_close)
    rec.realized_vol_20d = _realized_vol(window_close)
    rec.volume_ratio_20d = _volume_ratio(window_volume)
    return rec


def _momentum(close: pd.Series) -> float:
    """
    20-day price momentum — the cumulative log-return over the lookback window.

    alpha(X) component: captures the trending direction of the stock in the
    20 trading days before the announcement. Positive momentum may reflect
    market anticipation of strong earnings.

    Computed as log(P_{t-2} / P_{t-22}) — uses only the start and end prices
    of the window, so intraday noise does not inflate the estimate.
    """
    return float(np.log(close.iloc[-1] / close.iloc[0]))


def _realized_vol(close: pd.Series) -> float:
    """
    20-day realized volatility — annualized standard deviation of log-returns.

    alpha(X) component: captures uncertainty in the stock price before the
    announcement. High pre-announcement volatility often indicates elevated
    uncertainty about earnings outcomes, which may inflate Y variance
    independently of Z.

    Annualization factor sqrt(252) matches the standard event-study convention.
    """
    log_returns = np.diff(np.log(close.values))
    return float(np.std(log_returns, ddof=1) * np.sqrt(252))


def _volume_ratio(volume: pd.Series) -> float:
    """
    20-day volume ratio — recent 5-day average volume divided by the prior
    15-day average volume.

    alpha(X) component: captures unusual trading activity in the days just
    before the announcement. An elevated ratio signals heightened investor
    attention or informed trading, which may be correlated with Y even
    after conditioning on Z.

    Split as [last 5 days] / [prior 15 days] rather than a single average
    because the ratio is more informative about directional change than
    absolute volume level, which varies widely across tickers.
    """
    if len(volume) < LOOKBACK_DAYS:
        return float("nan")
    recent = volume.iloc[-5:].mean()
    baseline = volume.iloc[:15].mean()
    if baseline <= 0:
        return float("nan")
    return float(recent / baseline)


# -------------------------------------------------------------------------
# Cross-sectional normalization
# -------------------------------------------------------------------------

def normalize_cross_sectionally(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each feature within each filing_date group.

    Paper reference: normalization ensures alpha(X) lives in a standardized
    feature space, which matters for kernel-based estimators of I(X;Y|Z)
    (Section 3 of the paper) that assume zero-mean, unit-variance inputs.

    LOOKAHEAD GUARD: normalization is done per-date, using only the set of
    firms that filed on that exact date. This is the only valid approach —
    normalizing over the full panel would use future firms' statistics to
    standardize past observations.

    Parameters
    ----------
    df : DataFrame with columns ['ticker', 'filing_date', 'momentum_20d',
         'realized_vol_20d', 'volume_ratio_20d']. May contain NaN for
         failed features.

    Returns
    -------
    DataFrame with the same shape; feature columns replaced by their
    per-date z-scores. Rows where normalization is undefined (single-firm
    date or all-NaN group) are set to NaN and logged as warnings.
    """
    out = df.copy()
    feature_cols = FEATURE_NAMES

    for date, group in df.groupby("filing_date"):
        idx = group.index
        for col in feature_cols:
            vals = group[col].values.astype(float)
            std = np.nanstd(vals, ddof=1)
            if std == 0 or np.isnan(std):
                # Can't normalize: single observation or all identical values.
                # Set to NaN so the downstream estimator can filter them.
                logger.warning(
                    "Cannot normalize %s on %s (std=%.4f, n=%d) — setting to NaN",
                    col, date, std if not np.isnan(std) else -1, len(vals),
                )
                out.loc[idx, col] = np.nan
            else:
                out.loc[idx, col] = (vals - np.nanmean(vals)) / std

    return out


# -------------------------------------------------------------------------
# Batch computation
# -------------------------------------------------------------------------

def compute_features_batch(
    filing_dates: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Compute alpha(X) for a dict of {ticker: [filing_date, ...]}.

    Returns a DataFrame with columns:
        ticker, filing_date, momentum_20d, realized_vol_20d, volume_ratio_20d, error

    Rows with errors have NaN features but are retained — never dropped.
    Normalization is NOT applied here; call normalize_cross_sectionally()
    separately after assembling the full panel for a given study period.
    """
    rows = []
    for ticker, dates in filing_dates.items():
        for fd in dates:
            rf = compute_raw_features(ticker, fd)
            rows.append({
                "ticker": ticker,
                "filing_date": fd,
                "momentum_20d": rf.momentum_20d,
                "realized_vol_20d": rf.realized_vol_20d,
                "volume_ratio_20d": rf.volume_ratio_20d,
                "error": rf.error,
            })

    df = pd.DataFrame(rows)
    n_err = df["error"].notna().sum()
    if n_err:
        logger.warning("%d / %d feature computations had errors", n_err, len(df))
    return df


# -------------------------------------------------------------------------
# CLI smoke test
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compute price features for 8-K dates")
    parser.add_argument("--filings-dir", default="raw_filings")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "JPM"])
    parser.add_argument("--normalize", action="store_true",
                        help="Apply cross-sectional normalization to the output")
    args = parser.parse_args()

    filings_root = Path(args.filings_dir)
    filing_dates: dict[str, list[str]] = {}

    for ticker in args.tickers:
        ticker_dir = filings_root / ticker
        if not ticker_dir.exists():
            continue
        dates = []
        for jf in sorted(ticker_dir.glob("*.json")):
            rec = json.loads(jf.read_text())
            if rec.get("item_202_present") and not rec.get("parse_error"):
                dates.append(rec["filing_date"])
        if dates:
            filing_dates[ticker] = dates

    df = compute_features_batch(filing_dates)

    if args.normalize:
        df_norm = normalize_cross_sectionally(df)
        print("\n--- Normalized features ---")
        print(df_norm[["ticker", "filing_date"] + FEATURE_NAMES + ["error"]].to_string(index=False))
    else:
        print(df[["ticker", "filing_date"] + FEATURE_NAMES + ["error"]].to_string(index=False))
