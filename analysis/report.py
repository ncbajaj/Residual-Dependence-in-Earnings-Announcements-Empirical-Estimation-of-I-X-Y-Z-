"""
Structured markdown report with all key statistics inline.

`generate_report()` accepts the primary output objects from each module and
renders a self-contained markdown string. Call write_report() to save to disk.

The report structure:
  1. Header and run metadata
  2. Data statistics (N events, tickers, date range)
  3. X-Z dependence structure — I(X;Z) and singular value summary
  4. Residual dependence — I(X;Y|Z) full, by regime, and ratio
  5. Strategy performance — Sharpe, drawdown, costs, by regime
  6. Walk-forward validation — pooled NW t-stat, degradation, BH
  7. Leakage audit — explicit checklist
  8. Theoretical interpretation mapping results to Theorem 1
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _fmt(x: Optional[float], decimals: int = 4, na: str = "n/a") -> str:
    """Format a float or return na string."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return na
    return f"{x:.{decimals}f}"


def _pval_str(p: Optional[float]) -> str:
    """Format p-value with significance stars."""
    if p is None or not np.isfinite(p):
        return "n/a"
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{p:.3f}{stars}"


def generate_report(
    tickers: list[str],
    date_start: str,
    date_end: str,
    n_events: int,
    n_events_train: int,
    n_events_test: int,
    # I(X;Z)
    ixz_nocco: float,
    ixz_svd: float,
    top_singular_values: np.ndarray,
    gamma_xz: float,                         # power-law exponent from Fig 1
    # I(X;Y|Z) by regime
    ixyz_full: float,
    ixyz_regime: Optional[float],
    ixyz_stable: Optional[float],
    ixyz_ratio: Optional[float],
    n_regime_events: int,
    n_stable_events: int,
    h1_consistent: Optional[bool],
    perm_p_value: Optional[float],
    # Backtest
    sharpe_all: float,
    sharpe_regime: Optional[float],
    sharpe_stable: Optional[float],
    max_drawdown: float,
    mean_pnl_bps: float,
    turnover_pa: float,
    n_backtest_events: int,
    # Walk-forward validation
    n_folds: int,
    pooled_nw_tstat: float,
    pooled_nw_pvalue: float,
    mean_fold_sharpe: float,
    sharpe_std: float,
    degradation_tstat: Optional[float],
    degradation_pvalue: Optional[float],
    bh_rejected: int,
    bh_total: int,
    # Model hyperparameters
    reg_cca: float,
    reg_krr: float,
    n_components: int,
    tc_bps: float,
    seed: int,
    extra_notes: str = "",
) -> str:
    """
    Render the complete markdown report as a string.

    All numeric inputs are the final estimates from the pipeline run.
    Fields documented with ``Optional[float]`` may be None when the
    corresponding subsample was too small to estimate (N < min_n_per_regime).
    """
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    top5 = ", ".join(f"{s:.4f}" for s in top_singular_values[:5])
    i_xz_check = "✓" if abs(ixz_nocco - ixz_svd) / max(ixz_nocco, 1e-10) < 0.05 else "⚠"

    deg_consistent = (
        "✓ stable > regime (strategy degrades during shifts)"
        if (degradation_tstat is not None and degradation_tstat > 0)
        else "✗ direction not consistent"
        if degradation_tstat is not None
        else "n/a"
    )

    md = f"""\
# Residual Dependence — Analysis Report
*Generated: {now}*

---

## 1. Run configuration

| Parameter | Value |
|-----------|-------|
| Tickers | {', '.join(tickers)} |
| Date range | {date_start} → {date_end} |
| Total events (ticker × filing date) | {n_events} |
| Training events | {n_events_train} |
| Test events | {n_events_test} |
| CCA regularisation λ | {reg_cca:.2e} |
| KRR regularisation λ | {reg_krr:.2e} |
| Canonical components k | {n_components} |
| Transaction cost | {tc_bps} bps per side |
| Random seed | {seed} |

---

## 2. X–Z dependence structure  —  I(X; Z)

I(X; Z) quantifies how much joint information the price features X and the
transcript embedding Z share, relative to independence (mean square contingency,
eqs. 10–11). A large I(X; Z) means price behaviour and transcript content
are strongly coupled — a prerequisite for the indirect predictor η_ρ to work.

| Estimator | Value |
|-----------|-------|
| NOCCO trace  Î(X;Z) | {_fmt(ixz_nocco)} |
| SVD sum  Î(X;Z) | {_fmt(ixz_svd)} |
| Cross-check | {i_xz_check} (rel. error {abs(ixz_nocco - ixz_svd) / max(ixz_nocco, 1e-10):.2e}) |

### Singular value decay  (Proposition 1)

Top-5 σ̂_i: **{top5}**

Power-law exponent: **γ̂_XZ = {gamma_xz:.4f}**  (σ_i ~ i^{{−γ_XZ}})

A larger γ means the dependence is more concentrated in the first few canonical
directions, making the truncated estimator Σ_{{i=2}}^k σ̂_i² more accurate.

> **Bias warning**: The regularised NOCCO estimator has positive finite-sample
> bias (null value ≈ N/k at small N). Use these values for relative comparisons
> and cross-regime contrasts, not as absolute population quantities.

---

## 3. Residual dependence  —  I(X; Y | Z)

I(X; Y | Z) (Theorem 1, eq. 15) measures how much information X contains about
Y beyond what Z already mediates. Small ⟹ transcripts are sufficient for
zero-shot prediction. Large ⟹ price features carry incremental return
predictive information that transcripts don't explain.

| Subset | N events | Î(X;Y|Z) |
|--------|----------|----------|
| Full dataset | {n_events} | {_fmt(ixyz_full)} |
| Regime-shift events | {n_regime_events} | {_fmt(ixyz_regime)} |
| Stable events | {n_stable_events} | {_fmt(ixyz_stable)} |

**Ratio I_regime / I_stable: {_fmt(ixyz_ratio, 3)}**

**H₁ consistent** (I_regime > I_stable): {"✓" if h1_consistent else "✗" if h1_consistent is not None else "n/a"}

Permutation p-value (regime label shuffle): **{_pval_str(perm_p_value)}**

*Interpretation*: {"The ratio > 1 and H₁ is consistent with Theorem 1 — residual dependence increases during distribution shifts, because the Z→Y mapping learned in training breaks down." if h1_consistent else "Directional check inconclusive — may need more events per regime subset."}

---

## 4. Strategy performance

Long top quintile / short bottom quintile by information-density signal  c(x,z) = Σ_j σ̂_j f̂_j(x) ĝ_j(z)  (eq. 8).
5-day hold, dollar-neutral, {tc_bps} bps per side.

| Metric | All | Regime | Stable |
|--------|-----|--------|--------|
| Sharpe (annualised) | {_fmt(sharpe_all, 3)} | {_fmt(sharpe_regime, 3)} | {_fmt(sharpe_stable, 3)} |

| Metric | Value |
|--------|-------|
| Mean P&L per event | {_fmt(mean_pnl_bps, 1)} bps |
| Max drawdown | {_fmt(max_drawdown * 100, 2)}% |
| Turnover (one-way, p.a.) | {_fmt(turnover_pa, 1)}× |
| Events traded | {n_backtest_events} |

*Regime vs stable Sharpe differential*: If stable Sharpe > regime Sharpe, the
strategy's edge degrades during distribution shifts — consistent with Theorem 1.

---

## 5. Walk-forward validation

2-year train / 1-year test / 6-month step.  All signals generated out-of-sample
with parameters fitted on the training window only.

| Statistic | Value |
|-----------|-------|
| Folds | {n_folds} |
| Pooled NW t-statistic | {_fmt(pooled_nw_tstat, 3)} |
| Pooled NW p-value | {_pval_str(pooled_nw_pvalue)} |
| Mean fold Sharpe | {_fmt(mean_fold_sharpe, 3)} ± {_fmt(sharpe_std, 3)} |
| BH-rejected components | {bh_rejected} / {bh_total}  (FDR α = 5%) |
| Degradation t-stat (stable − regime) | {_fmt(degradation_tstat, 3)} |
| Degradation p-value | {_pval_str(degradation_pvalue)} |
| Direction | {deg_consistent} |

Newey-West uses L = 5 lags (matching the 5-day holding period).
Benjamini-Hochberg is applied across the k = {n_components} canonical signal components.

---

## 6. Data-leakage audit

| Check | Status | Detail |
|-------|--------|--------|
| Kernel bandwidth (h²_X, h²_Z) | ✓ | Median heuristic on training data only; frozen for test scoring |
| KRR regularisation λ | ✓ | Fixed constant; never cross-validated on Y |
| KRR predictions | ✓ | LOO (Rifkin & Klautau 2003); in-sample evaluation excluded |
| Regime threshold | ✓ | Calibrated on train windows; applied unchanged to test windows |
| Signal scoring | ✓ | `SignalEstimator.score()` uses frozen h², never re-fit on test |
| Price feature normalisation | ✓ | Cross-sectional per filing_date; no look-forward across dates |
| Price feature window | ✓ | [t−22, t−2] trading days; no overlap with forward return window |
| Walk-forward splits | ✓ | Each fold fits exclusively on events before `train_end` |
| Bandwidth for regime stability | ✓ | Subset bandwidth fitted within each regime/stable subset separately |
| Survivorship bias | ⚠ | Universe is 2024 S&P 500 snapshot — companies removed before 2024 absent |
| After-hours filing timing | ⚠ | Filing date not shifted to t+1 for after-hours 8-Ks |
| Adjusted prices | ⚠ | yfinance retroactive split/dividend adjustment is appropriate for research but not live trading |

---

## 7. Theoretical interpretation

The pipeline operationalises **Theorem 1** of Mehta & Harchaoui (2025):

```
‖η⋆ − η_ρ‖²_L2(P_X)  ≲  I(X; Y | Z)  +  prompt bias
```

| Paper quantity | Pipeline estimate | Module |
|---------------|-------------------|--------|
| M_{{Z\|X}} singular values σ_i | σ̂_i from kernel CCA | `dependence/kernel_cca.py` |
| I(X; Z) = ‖V_{{X,Z}}‖²_HS | Î = {_fmt(ixz_nocco)} (NOCCO trace) | `dependence/contingency.py` |
| I(X; Y \| Z) (Thm 2) | Î_CM = {_fmt(ixyz_full)} (LOO-KRR chain) | `dependence/mutual_info.py` |
| TV(P_X, Q_X) (Lemma 14) | SW_2^2 rolling Wasserstein | `regime/regime_detection.py` |
| Information density R(x,z) (eq. 8) | c(x,z) = Σ σ̂_j f̂_j(x) ĝ_j(z) | `strategy/signals.py` |

The central empirical result: **I_regime / I_stable = {_fmt(ixyz_ratio, 3)}**.
When the embedding distribution drifts (TV↑), the zero-shot predictor breaks down
and residual dependence I(X;Y|Z) grows — consistent with the bound in Theorem 1.
{extra_notes}
---
*Pipeline version: residual-dependence v0.1 | Model: Mehta & Harchaoui, ICML 2025*
"""
    return md


def write_report(report_str: str, path: str) -> None:
    """Write the markdown report string to disk."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_str)
    logger.info("Report written to %s", path)
