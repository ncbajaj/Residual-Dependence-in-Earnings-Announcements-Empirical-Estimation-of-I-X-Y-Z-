"""
Rolling sliced Wasserstein distance between consecutive 90-day windows of Z embeddings.

THEORETICAL CONNECTION
----------------------
Lemma 14 (Mehta & Harchaoui 2025): The generalization gap of the zero-shot predictor
η_ρ under a shifted test distribution P_X ≠ Q_X (the training distribution) is
bounded in part by TV(P_X, Q_X) — the total variation distance between distributions.

We proxy TV(P_X, Q_X) by the Wasserstein-2 distance between consecutive 90-day
windows of Z embeddings (the auxiliary modality available at prediction time).
A large W_2 between adjacent temporal windows indicates the embedding distribution
has shifted — a regime change that breaks the Z→Y mapping learned in training.

WINDOW DESIGN
-------------
- Window width: 90 calendar days
- Step size:    30 calendar days
- Comparison:   each window is compared to the next (consecutive pair)
- Assignment:   an event at date d is tagged as a "regime-shift event" if the
  nearest window-transition midpoint has W_2 above the calibrated threshold

SLICED WASSERSTEIN
------------------
W_2 in R^{384} (transcript embedding dimension) requires inverting a 384×384
covariance matrix, which is ill-conditioned when a window contains only 10-50
events. We use sliced W_2^2 instead:

    SW_2^2(P, Q) = E_{v ~ Unif(S^{d-1})}[W_2^2(v#P, v#Q)]

which is consistent (SW_2 → W_2 as n_proj → ∞) and computable from O(n log n)
sorted projections. With n_proj=200 random unit vectors, variance is low.

LEAKAGE GUARD
-------------
The threshold (mean + threshold_std × std) is computed on training windows only,
then applied unchanged to test windows. Call fit() on training events, then
transform() on test events passing the same fitted detector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_WINDOW_DAYS = 90     # calendar days per comparison window
_STEP_DAYS = 30       # calendar days between consecutive window centers
_MIN_WINDOW_SIZE = 5  # minimum events per window; smaller windows → NaN distance


# ---------------------------------------------------------------------------
# Sliced Wasserstein kernel
# ---------------------------------------------------------------------------

def _interp_quantiles(sorted_vals: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate a sorted array to the quantile positions in q.

    Parameters
    ----------
    sorted_vals : (n_samples, n_proj) array — one sorted projection per column.
    q           : (n_q,) quantile points in [0, 1].

    Returns
    -------
    (n_q, n_proj) interpolated quantile values.
    """
    n = sorted_vals.shape[0]
    xp = np.linspace(0.0, 1.0, n)
    idx = np.searchsorted(xp, q)
    idx = np.clip(idx, 1, n - 1)
    t = (q - xp[idx - 1]) / np.maximum(xp[idx] - xp[idx - 1], 1e-12)  # (n_q,)
    return (1.0 - t[:, None]) * sorted_vals[idx - 1] + t[:, None] * sorted_vals[idx]


def sliced_wasserstein_sq(
    A: np.ndarray,
    B: np.ndarray,
    n_proj: int = 200,
    n_quantiles: int = 100,
    seed: int = 0,
) -> float:
    """
    Sliced Wasserstein-2 squared distance between empirical distributions A and B.

    SW_2^2(A, B) = (1/n_proj) Σ_k W_2^2(proj_k # A, proj_k # B)

    where each proj_k is a unit vector drawn uniformly from S^{d-1}.
    For 1D: W_2^2 is approximated by mean squared difference of sorted quantiles.

    Parameters
    ----------
    A, B      : (n_A, d) and (n_B, d) sample arrays from the two distributions.
    n_proj    : number of random projections; higher → lower variance.
    n_quantiles : number of quantile points for 1D W_2 approximation.
    seed      : random seed for reproducibility.

    Returns
    -------
    Scalar SW_2^2 estimate (non-negative).
    """
    if A.shape[0] < 2 or B.shape[0] < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    d = A.shape[1]

    # Random unit projections: (n_proj, d)
    dirs = rng.standard_normal((n_proj, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # Projected samples: (n_A, n_proj) and (n_B, n_proj)
    proj_A = A @ dirs.T
    proj_B = B @ dirs.T

    proj_A.sort(axis=0)
    proj_B.sort(axis=0)

    q = np.linspace(0.0, 1.0, n_quantiles + 2)[1:-1]   # (n_quantiles,) in (0,1)
    qa = _interp_quantiles(proj_A, q)   # (n_q, n_proj)
    qb = _interp_quantiles(proj_B, q)   # (n_q, n_proj)

    # W_2^2 per projection, then average
    w2_per_proj = np.mean((qa - qb) ** 2, axis=0)    # (n_proj,)
    return float(np.mean(w2_per_proj))


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class WindowDistance:
    """Wasserstein distance for a consecutive window pair."""
    window_a_center: pd.Timestamp    # center date of window A
    window_b_center: pd.Timestamp    # center date of window B
    midpoint: pd.Timestamp           # midpoint between the two centers
    distance_sq: float               # SW_2^2 (NaN if either window is too small)
    n_a: int                         # events in window A
    n_b: int                         # events in window B


@dataclass
class RegimeDetectionResult:
    """Full output of RegimeDetector.fit_transform()."""
    window_distances: list[WindowDistance]
    threshold: float                  # fitted threshold (NaN if not enough windows)
    threshold_mean: float
    threshold_std: float
    event_flags: pd.Series            # bool Series indexed by date: True = regime shift
    distances_series: pd.Series       # SW_2^2 indexed by midpoint date (float, NaN-filled)


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Detect regime shifts by comparing consecutive 90-day windows of Z embeddings.

    Usage (train/test split)
    ------------------------
    >>> detector = RegimeDetector()
    >>> detector.fit(train_dates, train_embeddings)
    >>> test_flags = detector.transform(test_dates, test_embeddings, test_event_dates)

    Usage (single dataset)
    ----------------------
    >>> result = detector.fit_transform(dates, embeddings, event_dates)
    >>> regime_flags = result.event_flags
    """

    def __init__(
        self,
        window_days: int = _WINDOW_DAYS,
        step_days: int = _STEP_DAYS,
        threshold_std: float = 1.5,
        n_proj: int = 200,
        n_quantiles: int = 100,
        min_window_size: int = _MIN_WINDOW_SIZE,
    ) -> None:
        self.window_days = window_days
        self.step_days = step_days
        self.threshold_std = threshold_std
        self.n_proj = n_proj
        self.n_quantiles = n_quantiles
        self.min_window_size = min_window_size

        # Fitted attributes (set by fit())
        self.threshold_: float = float("nan")
        self.threshold_mean_: float = float("nan")
        self.threshold_std_: float = float("nan")
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_window_distances(
        self,
        dates: np.ndarray,            # (N,) array of pd.Timestamp
        embeddings: np.ndarray,       # (N, d) embeddings
    ) -> list[WindowDistance]:
        """Compute SW_2^2 for all consecutive 90-day window pairs."""
        if len(dates) == 0:
            return []

        dates = np.array(dates, dtype="datetime64[D]")
        t_min = pd.Timestamp(dates.min())
        t_max = pd.Timestamp(dates.max())

        # Window centers: t_min, t_min + step, t_min + 2*step, ...
        half = timedelta(days=self.window_days // 2)
        step = timedelta(days=self.step_days)

        centers: list[pd.Timestamp] = []
        t = t_min + half
        while t <= t_max - half + step:
            centers.append(t)
            t += step

        if len(centers) < 2:
            logger.warning(
                "Only %d window centers found (need ≥2). "
                "Increase date range or decrease window_days.",
                len(centers),
            )
            return []

        def events_in_window(center: pd.Timestamp) -> np.ndarray:
            lo = np.datetime64(center - half, "D")
            hi = np.datetime64(center + half, "D")
            mask = (dates >= lo) & (dates <= hi)
            return embeddings[mask]

        distances: list[WindowDistance] = []
        for i in range(len(centers) - 1):
            ca, cb = centers[i], centers[i + 1]
            A = events_in_window(ca)
            B = events_in_window(cb)
            midpoint = ca + (cb - ca) / 2

            if len(A) < self.min_window_size or len(B) < self.min_window_size:
                dist_sq = float("nan")
                logger.debug(
                    "Window pair (%s, %s): too few events (n_A=%d, n_B=%d), skipping.",
                    ca.date(), cb.date(), len(A), len(B),
                )
            else:
                dist_sq = sliced_wasserstein_sq(
                    A, B, n_proj=self.n_proj, n_quantiles=self.n_quantiles
                )
                logger.debug(
                    "Window (%s → %s): SW_2^2 = %.4f  (n_A=%d, n_B=%d)",
                    ca.date(), cb.date(), dist_sq, len(A), len(B),
                )

            distances.append(WindowDistance(
                window_a_center=ca,
                window_b_center=cb,
                midpoint=midpoint,
                distance_sq=dist_sq,
                n_a=len(A),
                n_b=len(B),
            ))

        return distances

    def _assign_flags(
        self,
        window_distances: list[WindowDistance],
        event_dates: pd.DatetimeIndex,
        threshold: float,
    ) -> pd.Series:
        """
        Assign regime-shift flag to each event.

        An event at date d is True if the nearest window-transition midpoint
        (within ±60 calendar days) has SW_2^2 above the threshold.
        """
        flags = pd.Series(False, index=event_dates, dtype=bool)
        if not window_distances or np.isnan(threshold):
            return flags

        midpoints = np.array([wd.midpoint for wd in window_distances], dtype="datetime64[D]")
        dists_sq = np.array([wd.distance_sq for wd in window_distances])
        max_gap = np.timedelta64(60, "D")

        for d in event_dates:
            d_np = np.datetime64(d, "D")
            gaps = np.abs(midpoints - d_np)
            nearest_idx = np.argmin(gaps)
            if gaps[nearest_idx] <= max_gap and not np.isnan(dists_sq[nearest_idx]):
                flags[d] = bool(dists_sq[nearest_idx] > threshold)

        return flags

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        dates: list | pd.DatetimeIndex,
        embeddings: np.ndarray,
    ) -> "RegimeDetector":
        """
        Fit the regime detector on training data.

        Computes rolling SW_2^2 distances and calibrates the threshold as
        mean + threshold_std × std, both computed on training windows only.

        Parameters
        ----------
        dates      : (N,) event dates (any date-like iterable).
        embeddings : (N, d) Z embeddings for those events.
        """
        dates_ts = pd.DatetimeIndex(dates)
        window_dists = self._compute_window_distances(dates_ts, embeddings)

        valid_dists = [wd.distance_sq for wd in window_dists if not np.isnan(wd.distance_sq)]
        if len(valid_dists) < 2:
            logger.warning(
                "Only %d valid window distances on training data — "
                "threshold cannot be calibrated. All events will be flagged False.",
                len(valid_dists),
            )
            self.threshold_mean_ = float("nan")
            self.threshold_std_ = float("nan")
            self.threshold_ = float("nan")
        else:
            arr = np.array(valid_dists)
            self.threshold_mean_ = float(arr.mean())
            self.threshold_std_ = float(arr.std(ddof=1))
            self.threshold_ = self.threshold_mean_ + self.threshold_std * self.threshold_std_
            logger.info(
                "Regime detector fitted: %d windows, SW_2^2 mean=%.4f std=%.4f "
                "→ threshold=%.4f (%.1f-sigma)",
                len(valid_dists),
                self.threshold_mean_, self.threshold_std_, self.threshold_, self.threshold_std,
            )

        self._train_window_distances = window_dists
        self._is_fitted = True
        return self

    def transform(
        self,
        dates: list | pd.DatetimeIndex,
        embeddings: np.ndarray,
        event_dates: Optional[list | pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """
        Assign regime-shift flags to events using the fitted threshold.

        Parameters
        ----------
        dates       : (N,) dates of all events used to compute windows.
        embeddings  : (N, d) Z embeddings for those events.
        event_dates : dates to flag (defaults to `dates` if None).

        Returns
        -------
        pd.Series of bool indexed by event_dates.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        dates_ts = pd.DatetimeIndex(dates)
        event_ts = pd.DatetimeIndex(event_dates if event_dates is not None else dates)
        window_dists = self._compute_window_distances(dates_ts, embeddings)
        return self._assign_flags(window_dists, event_ts, self.threshold_)

    def fit_transform(
        self,
        dates: list | pd.DatetimeIndex,
        embeddings: np.ndarray,
        event_dates: Optional[list | pd.DatetimeIndex] = None,
    ) -> RegimeDetectionResult:
        """
        Fit on and transform the same dataset. Returns the full result object.

        Parameters
        ----------
        dates       : (N,) dates of all events used to compute windows.
        embeddings  : (N, d) Z embeddings for those events.
        event_dates : dates to flag (defaults to `dates` if None).

        Returns
        -------
        RegimeDetectionResult with event_flags, window_distances, threshold.
        """
        self.fit(dates, embeddings)

        dates_ts = pd.DatetimeIndex(dates)
        event_ts = pd.DatetimeIndex(event_dates if event_dates is not None else dates)

        flags = self._assign_flags(self._train_window_distances, event_ts, self.threshold_)

        dist_series = pd.Series(
            {wd.midpoint: wd.distance_sq for wd in self._train_window_distances},
            name="SW2_sq",
        )

        n_flagged = flags.sum()
        logger.info(
            "Regime flagging: %d / %d events flagged as regime-shift (threshold SW_2^2=%.4f)",
            n_flagged, len(event_ts), self.threshold_,
        )

        return RegimeDetectionResult(
            window_distances=self._train_window_distances,
            threshold=self.threshold_,
            threshold_mean=self.threshold_mean_,
            threshold_std=self.threshold_std_,
            event_flags=flags,
            distances_series=dist_series,
        )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Regime detection smoke test")
    parser.add_argument("--n", type=int, default=120, help="Total synthetic events")
    parser.add_argument("--d", type=int, default=16, help="Embedding dim (use small d for speed)")
    parser.add_argument("--shift-at", type=int, default=60, help="Event index where shift occurs")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    N, d, shift_at = args.n, args.d, args.shift_at

    # Synthetic: first half N(0,I), second half N(μ, I) with large μ → regime shift
    Z1 = rng.standard_normal((shift_at, d))
    Z2 = rng.standard_normal((N - shift_at, d)) + 3.0   # shifted mean
    Z_all = np.vstack([Z1, Z2])
    Z_all /= np.linalg.norm(Z_all, axis=1, keepdims=True)   # L2 normalize like all-MiniLM

    base = pd.Timestamp("2022-01-01")
    dates = pd.DatetimeIndex([base + timedelta(days=3 * i) for i in range(N)])

    detector = RegimeDetector(window_days=90, step_days=30, threshold_std=1.5, n_proj=50)
    result = detector.fit_transform(dates, Z_all)

    print(f"\nN={N}, d={d}, shift at event {shift_at}")
    print(f"  Threshold (SW_2^2 > {result.threshold:.4f}) → regime shift")
    print(f"  Events flagged: {result.event_flags.sum()} / {len(result.event_flags)}")
    print(f"\nWindow distances (midpoint → SW_2^2):")
    for wd in result.window_distances:
        flag = "*** SHIFT ***" if (not np.isnan(wd.distance_sq) and wd.distance_sq > result.threshold) else ""
        print(f"  {wd.midpoint.date()}  SW_2^2={wd.distance_sq:.4f}  n=({wd.n_a},{wd.n_b})  {flag}")
