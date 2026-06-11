"""
Publication-quality matplotlib figures for the residual dependence pipeline.

Four figures:
  1. Singular value decay of M_{Z|X} — log-log plot with power-law fit
  2. I(X;Y|Z) over time with regime-shift shading
  3. Cumulative strategy returns split by regime vs stable periods
  4. Rolling 90-day signal IC (rank correlation of score with forward return)

All functions return (fig, axes) tuples. Call fig.savefig(...) to persist.
No scipy dependency — power-law fit and Spearman correlation implemented here.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import LogLocator, LogFormatter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_PALETTE = {
    "regime":  "#d62728",   # red — regime-shift events
    "stable":  "#1f77b4",   # blue — stable events
    "signal":  "#2ca02c",   # green — signal-based series
    "neutral": "#7f7f7f",   # grey — reference lines, shading
    "accent":  "#ff7f0e",   # orange — highlights
}
_ALPHA_SHADE = 0.18
_FIG_SIZE_SINGLE = (7, 4)
_FIG_SIZE_GRID = (13, 9)


def _style_ax(ax: Axes, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient (no scipy)."""
    def _ranks(v):
        order = np.argsort(v)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        return r

    rx = _ranks(x); ry = _ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    return float(np.dot(rx, ry) / denom) if denom > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Figure 1 — Singular value decay
# ---------------------------------------------------------------------------

def plot_singular_value_decay(
    singular_values: np.ndarray,
    ax: Optional[Axes] = None,
    title: str = "Singular Value Decay of  M̂_{Z|X}",
    max_components: int = 40,
) -> tuple[Figure, Axes]:
    """
    Log-log plot of σ̂_i vs component index i with a power-law fit.

    The power law σ_i ~ C × i^{-γ} characterizes the compressibility of the
    dependence between X and Z in the RKHS basis. A steeper decay (larger γ)
    means the dependence is more concentrated in the first few canonical
    directions — easier to estimate and exploit.

    Parameters
    ----------
    singular_values : σ̂_i from kernel_cca() (descending order, non-trivial).
    ax              : Matplotlib axes (created if None).
    max_components  : maximum i to plot (avoids noise tail at high indices).
    """
    sigma = singular_values[singular_values > 1e-8][:max_components]
    k = len(sigma)
    i_idx = np.arange(1, k + 1, dtype=float)

    # Power-law fit on log-log (OLS on log(sigma) ~ a + b*log(i))
    log_i = np.log(i_idx)
    log_s = np.log(sigma)
    b, a = np.polyfit(log_i, log_s, 1)  # b = slope, a = intercept
    gamma = float(-b)
    C = float(np.exp(a))
    fit_vals = C * i_idx ** (-gamma)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIG_SIZE_SINGLE)
    else:
        fig = ax.figure

    ax.scatter(i_idx, sigma, s=18, color=_PALETTE["stable"], zorder=3, label="σ̂_i")
    ax.plot(i_idx, fit_vals, "--", color=_PALETTE["accent"], linewidth=1.5,
            label=f"Power law fit:  γ̂ = {gamma:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=9, framealpha=0.7)
    _style_ax(ax, xlabel="Component index  i", ylabel="σ̂_i", title=title)
    ax.annotate(
        f"γ̂_XZ = {gamma:.3f}\n(σ_i ~ i^{{−γ}})",
        xy=(i_idx[min(5, k-1)], sigma[min(5, k-1)]),
        xytext=(i_idx[min(5, k-1)] * 1.5, sigma[min(5, k-1)] * 1.8),
        fontsize=8, color=_PALETTE["accent"],
        arrowprops=dict(arrowstyle="->", color=_PALETTE["accent"], lw=0.8),
    )

    if fig is not None:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 2 — I(X;Y|Z) timeline with regime shading
# ---------------------------------------------------------------------------

def plot_residual_dependence_timeline(
    dates: pd.DatetimeIndex,
    i_xyz_values: np.ndarray,
    regime_flags: np.ndarray,
    ax: Optional[Axes] = None,
    title: str = "Residual Dependence  Î(X;Y|Z)  over Time",
    window_label: str = "rolling 90-day window",
) -> tuple[Figure, Axes]:
    """
    Line/scatter plot of I(X;Y|Z) over time with regime-shift shading.

    Parameters
    ----------
    dates        : (N,) event dates corresponding to each I estimate.
    i_xyz_values : (N,) I(X;Y|Z) estimate for each date.
    regime_flags : (N,) bool — True = event is in a regime-shift window.
    window_label : label for the x-axis note.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIG_SIZE_SINGLE)
    else:
        fig = ax.figure

    dates_arr = pd.DatetimeIndex(dates)
    vals = np.asarray(i_xyz_values, dtype=float)
    flags = np.asarray(regime_flags, dtype=bool)

    # Shade regime-shift periods
    # Find contiguous regime blocks and shade the full date span
    in_block = False
    block_start = None
    for i, (d, f) in enumerate(zip(dates_arr, flags)):
        if f and not in_block:
            block_start = d
            in_block = True
        elif not f and in_block:
            ax.axvspan(block_start, d, color=_PALETTE["regime"], alpha=_ALPHA_SHADE)
            in_block = False
    if in_block:
        ax.axvspan(block_start, dates_arr[-1], color=_PALETTE["regime"], alpha=_ALPHA_SHADE)

    # Plot stable events
    stable_mask = ~flags
    if stable_mask.any():
        ax.scatter(dates_arr[stable_mask], vals[stable_mask],
                   s=20, color=_PALETTE["stable"], alpha=0.75, zorder=3, label="Stable")

    # Plot regime events
    if flags.any():
        ax.scatter(dates_arr[flags], vals[flags],
                   s=25, color=_PALETTE["regime"], alpha=0.85, zorder=4, label="Regime shift")

    # Smoothed trend (simple moving average over sorted events)
    sort_idx = np.argsort(dates_arr)
    smooth_window = max(3, len(vals) // 10)
    if len(vals) >= smooth_window:
        smooth = np.convolve(vals[sort_idx], np.ones(smooth_window) / smooth_window, mode="valid")
        smooth_dates = dates_arr[sort_idx][smooth_window - 1:]
        ax.plot(smooth_dates, smooth, color=_PALETTE["neutral"], linewidth=1.5,
                alpha=0.6, label=f"Trend ({smooth_window}-event MA)")

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.legend(fontsize=8, framealpha=0.7)

    regime_patch = mpatches.Patch(color=_PALETTE["regime"], alpha=_ALPHA_SHADE,
                                  label="Regime-shift window")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [regime_patch], labels + ["Regime-shift window"], fontsize=8, framealpha=0.7)

    _style_ax(ax, xlabel=f"Date  ({window_label})", ylabel="Î(X;Y|Z)", title=title)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30, ha="right")

    if fig is not None:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 3 — Cumulative returns by regime
# ---------------------------------------------------------------------------

def plot_cumulative_returns(
    event_returns: pd.DataFrame,
    ax: Optional[Axes] = None,
    title: str = "Cumulative Strategy Returns by Regime",
) -> tuple[Figure, Axes]:
    """
    Cumulative P&L (net of costs) split into regime-shift and stable event series.

    Both series start at 0 and accumulate in event-date order.
    The gap between the two illuminates performance differential.

    Parameters
    ----------
    event_returns : DataFrame from BacktestResult.event_returns with columns:
        date, pnl_net, pnl_gross, is_regime.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIG_SIZE_SINGLE)
    else:
        fig = ax.figure

    if event_returns.empty or "date" not in event_returns.columns:
        if fig is None:
            fig, ax = plt.subplots(figsize=_FIG_SIZE_SINGLE)
        ax.text(0.5, 0.5, "No backtest events", ha="center", va="center",
                transform=ax.transAxes, color=_PALETTE["neutral"])
        _style_ax(ax, title=title)
        if fig is not None:
            fig.tight_layout()
        return fig, ax
    df = event_returns.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    def _cumret(mask):
        sub = df[mask].sort_values("date")
        if sub.empty:
            return pd.Series(dtype=float), pd.DatetimeIndex([])
        cum = sub["pnl_net"].cumsum().values * 100   # → percentage points
        return cum, pd.DatetimeIndex(sub["date"])

    cum_stable, dates_stable = _cumret(~df["is_regime"])
    cum_regime, dates_regime = _cumret(df["is_regime"])
    cum_all, dates_all = _cumret(pd.Series(True, index=df.index))

    if len(dates_all) > 1:
        ax.plot(dates_all, cum_all, color=_PALETTE["neutral"], linewidth=1.0,
                linestyle="--", alpha=0.7, label="All events")
    if len(dates_stable) > 0:
        ax.step(dates_stable, cum_stable, color=_PALETTE["stable"], linewidth=1.8,
                where="post", label=f"Stable  (n={len(cum_stable)})")
    if len(dates_regime) > 0:
        ax.step(dates_regime, cum_regime, color=_PALETTE["regime"], linewidth=1.8,
                where="post", label=f"Regime shift  (n={len(cum_regime)})")

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.legend(fontsize=9, framealpha=0.7)
    _style_ax(ax, xlabel="Date", ylabel="Cumulative P&L (pp)", title=title)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30, ha="right")

    if fig is not None:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 4 — Rolling signal IC
# ---------------------------------------------------------------------------

def plot_rolling_ic(
    dates: pd.DatetimeIndex,
    signals: np.ndarray,
    returns: np.ndarray,
    window_days: int = 90,
    ax: Optional[Axes] = None,
    title: str = "Rolling 90-Day Signal IC  (Spearman Rank Correlation)",
) -> tuple[Figure, Axes]:
    """
    Rolling Spearman IC between signal scores and forward abnormal returns.

    For each event date d, the IC is computed over all events in [d−90d, d].
    A persistently positive IC confirms the signal has directional predictive
    power. Dips during regime-shift periods test whether the signal degrades.

    Parameters
    ----------
    dates    : (N,) event dates.
    signals  : (N,) information-density scores from SignalEstimator.
    returns  : (N,) forward abnormal returns (Y).
    window_days : rolling window in calendar days.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=_FIG_SIZE_SINGLE)
    else:
        fig = ax.figure

    dates_arr = pd.DatetimeIndex(dates)
    signals = np.asarray(signals, dtype=float)
    returns = np.asarray(returns, dtype=float)

    n = len(dates_arr)
    ics = np.full(n, np.nan)
    for i in range(n):
        d = dates_arr[i]
        mask = (dates_arr >= d - pd.Timedelta(days=window_days)) & (dates_arr <= d)
        if mask.sum() >= 5:
            ics[i] = _spearman(signals[mask], returns[mask])

    valid = ~np.isnan(ics)
    if valid.any():
        ax.bar(
            dates_arr[valid],
            ics[valid],
            color=np.where(ics[valid] >= 0, _PALETTE["signal"], _PALETTE["regime"]),
            width=max(2, window_days // 10),
            alpha=0.75,
        )
        # Smooth trend
        smooth_window = max(3, valid.sum() // 8)
        if valid.sum() >= smooth_window:
            ic_valid = ics[valid]
            smooth = np.convolve(ic_valid, np.ones(smooth_window) / smooth_window, mode="valid")
            smooth_dates = dates_arr[valid][smooth_window - 1:]
            ax.plot(smooth_dates, smooth, color="black", linewidth=1.5,
                    label=f"{smooth_window}-event MA")

    mean_ic = float(np.nanmean(ics)) if valid.any() else 0.0
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.axhline(mean_ic, color=_PALETTE["accent"], linewidth=1.2, linestyle="--",
               label=f"Mean IC = {mean_ic:.3f}")
    ax.legend(fontsize=8, framealpha=0.7)
    _style_ax(ax,
              xlabel="Date",
              ylabel="Spearman IC",
              title=title)
    ax.set_ylim(-1, 1)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30, ha="right")

    if fig is not None:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Combined publication figure (2×2)
# ---------------------------------------------------------------------------

def build_publication_figure(
    singular_values: np.ndarray,
    dates_xyz: pd.DatetimeIndex,
    i_xyz_values: np.ndarray,
    regime_flags_xyz: np.ndarray,
    event_returns: pd.DataFrame,
    dates_ic: pd.DatetimeIndex,
    signals_ic: np.ndarray,
    returns_ic: np.ndarray,
    suptitle: str = "Residual Dependence — Empirical Results",
) -> Figure:
    """
    Assemble all four figures into a 2×2 grid suitable for publication.

    Parameters
    ----------
    singular_values   : σ̂_i from KernelCCAResult (Fig 1).
    dates_xyz         : event dates for I(X;Y|Z) timeline (Fig 2).
    i_xyz_values      : I(X;Y|Z) estimates aligned to dates_xyz (Fig 2).
    regime_flags_xyz  : bool array aligned to dates_xyz (Fig 2).
    event_returns     : BacktestResult.event_returns DataFrame (Fig 3).
    dates_ic          : event dates for IC computation (Fig 4).
    signals_ic        : signal scores aligned to dates_ic (Fig 4).
    returns_ic        : abnormal returns aligned to dates_ic (Fig 4).
    suptitle          : overall figure title.

    Returns
    -------
    matplotlib Figure (call fig.savefig() to persist).
    """
    fig, axes = plt.subplots(2, 2, figsize=_FIG_SIZE_GRID)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.98)

    plot_singular_value_decay(singular_values, ax=axes[0, 0])
    plot_residual_dependence_timeline(
        dates_xyz, i_xyz_values, regime_flags_xyz, ax=axes[0, 1]
    )
    plot_cumulative_returns(event_returns, ax=axes[1, 0])
    plot_rolling_ic(dates_ic, signals_ic, returns_ic, ax=axes[1, 1])

    plt.subplots_adjust(hspace=0.38, wspace=0.32)
    return fig
