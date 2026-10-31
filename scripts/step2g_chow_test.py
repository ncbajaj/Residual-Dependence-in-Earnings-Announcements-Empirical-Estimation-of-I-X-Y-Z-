"""
Step 2g: direct structural-break (Chow) test for concept drift.

Rather than inferring concept drift by elimination (ruling out capacity,
regularization, and covariate shift in turn), this fits a simple linear
model  Y = b0 + b1*X1 + b2*X2 + b3*X3 + b4*S + eps  separately in
different calendar periods and formally tests whether the coefficient
vector itself changes between periods (a Chow test), rather than just
whether predictions generalize.

Two splits:
  1. Two non-overlapping 2-year halves (2020-2021 vs 2022-2023) --
     directly matching the pre- vs post-rate-hike-cycle break.
  2. Four non-overlapping calendar years, generalized Chow test across
     all four, plus the raw per-year coefficients side by side so sign
     flips / magnitude changes are visible directly, not just implied by
     a p-value.

Uses OLS (not the KRR used elsewhere) specifically because a Chow test
on a fixed, low-dimensional linear model is the standard, interpretable
tool for this question -- the kernel model's effective function has no
single "coefficient vector" to compare across periods.

Loads from dataset_cache/surprise_matched_150.npz -- no network needed.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("step2g")

CACHE_PATH = Path("dataset_cache/surprise_matched_150.npz")


def ols_fit(Xd, y):
    """Returns (coefs, SSR, n)."""
    coefs, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ coefs
    return coefs, float(resid @ resid), len(y)


def chow_test(groups, Xd_all, y_all, masks):
    """
    groups: list of boolean masks partitioning Xd_all/y_all.
    Returns (F_stat, df1, df2, p_value, per_group_coefs, per_group_ssr).
    """
    k = Xd_all.shape[1]
    coefs_pooled, ssr_pooled, n_pooled = ols_fit(Xd_all, y_all)

    per_group_coefs, per_group_ssr, per_group_n = [], [], []
    for m in masks:
        c, ssr, n = ols_fit(Xd_all[m], y_all[m])
        per_group_coefs.append(c)
        per_group_ssr.append(ssr)
        per_group_n.append(n)

    G = len(masks)
    ssr_groups_sum = sum(per_group_ssr)
    df1 = (G - 1) * k
    df2 = n_pooled - G * k
    F = ((ssr_pooled - ssr_groups_sum) / df1) / (ssr_groups_sum / df2)
    p = float(stats.f.sf(F, df1, df2))
    return F, df1, df2, p, per_group_coefs, per_group_ssr, per_group_n, coefs_pooled


def main():
    logger.info("Loading cached dataset from %s ...", CACHE_PATH)
    d = np.load(CACHE_PATH, allow_pickle=True)
    X_all, S_all, Y_all, dates = d["X_all"], d["S_all"], d["Y_all"], d["date"]
    dates_ts = pd.DatetimeIndex(dates)
    N = len(Y_all)
    logger.info("Loaded N=%d events, date range %s to %s", N, dates_ts.min().date(), dates_ts.max().date())

    # Standardize X, S globally (for interpretable, comparable coefficient
    # magnitudes across periods -- this is a diagnostic about the
    # relationship itself, not a leakage-sensitive predictive backtest).
    X_std = (X_all - X_all.mean(0)) / (X_all.std(0) + 1e-8)
    S_std = (S_all - S_all.mean(0)) / (S_all.std(0) + 1e-8)
    Xd = np.column_stack([np.ones(N), X_std, S_std])  # intercept + 3 X dims + 1 S dim
    feature_names = ["intercept", "X1", "X2", "X3", "surprise"]

    # --- Split 1: two non-overlapping halves ---
    mask_early = dates_ts < pd.Timestamp("2022-01-01")
    mask_late = dates_ts >= pd.Timestamp("2022-01-01")
    print(f"\n{'='*78}\nTWO-PERIOD CHOW TEST: 2020-2021 (n={mask_early.sum()}) vs "
          f"2022-2023 (n={mask_late.sum()})\n{'='*78}")
    F, df1, df2, p, coefs, ssrs, ns, coefs_pooled = chow_test(
        ["early", "late"], Xd, Y_all, [mask_early, mask_late]
    )
    print(f"Chow F({df1},{df2}) = {F:.3f}, p-value = {p:.4f}")
    print(f"\n{'feature':>12} | {'pooled':>10} | {'2020-21':>10} | {'2022-23':>10}")
    for i, name in enumerate(feature_names):
        print(f"{name:>12} | {coefs_pooled[i]:>10.5f} | {coefs[0][i]:>10.5f} | {coefs[1][i]:>10.5f}")
    print(f"\nConclusion: {'Coefficients differ SIGNIFICANTLY between periods (p<0.05) -- concept drift confirmed' if p < 0.05 else 'No significant difference detected at this split (p>=0.05)'}")

    # --- Split 2: four calendar years, generalized Chow test ---
    years = [2020, 2021, 2022, 2023]
    masks_y = [(dates_ts.year == yr) for yr in years]
    ns_y = [m.sum() for m in masks_y]
    print(f"\n{'='*78}\nFOUR-YEAR CHOW TEST: {dict(zip(years, ns_y))}\n{'='*78}")
    F4, df1_4, df2_4, p4, coefs4, ssrs4, ns4, coefs_pooled4 = chow_test(
        [str(y) for y in years], Xd, Y_all, masks_y
    )
    print(f"Chow F({df1_4},{df2_4}) = {F4:.3f}, p-value = {p4:.4f}")
    header = f"{'feature':>12} | {'pooled':>10} | " + " | ".join(f"{y:>10}" for y in years)
    print(f"\n{header}")
    for i, name in enumerate(feature_names):
        row = f"{name:>12} | {coefs_pooled4[i]:>10.5f} | " + " | ".join(f"{coefs4[j][i]:>10.5f}" for j in range(4))
        print(row)
    print(f"\nConclusion: {'Coefficients differ SIGNIFICANTLY across years (p<0.05) -- concept drift confirmed' if p4 < 0.05 else 'No significant difference detected across years (p>=0.05)'}")


if __name__ == "__main__":
    main()
