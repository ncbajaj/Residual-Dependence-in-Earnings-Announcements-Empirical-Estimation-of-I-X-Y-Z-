"""
Step 2f: step2e showed the Z-embedding-based regime detector finds almost
no shifts in the two worst-performing folds, even though the model fails
there just as badly as anywhere else. This checks whether a regime
detector built on X (price features) instead of Z catches something the
embedding-based one misses -- i.e. whether the non-stationarity lives in
the X-Y relationship specifically, not the Z marginal distribution.

Same sliced-Wasserstein / rolling-window machinery (regime/regime_detection.py),
just fed X instead of Z. Loads from the cached dataset, no network needed.
"""
import logging
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("step2f")

from dependence.kernel_cca import median_heuristic, rbf_kernel
from regime.regime_detection import RegimeDetector

CACHE_PATH = Path("dataset_cache/surprise_matched_150.npz")
TRAIN_YEARS, TEST_YEARS, STEP_MONTHS = 2.0, 1.0, 6
MIN_TRAIN, MIN_TEST = 50, 15
KRR_REG = 1e-2


def krr_fit_predict(K_train, y_train, K_cross, reg):
    N = K_train.shape[0]
    alpha = np.linalg.solve(K_train + N * reg * np.eye(N), y_train)
    return K_cross @ alpha


def corr(a, b):
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    logger.info("Loading cached dataset from %s ...", CACHE_PATH)
    d = np.load(CACHE_PATH, allow_pickle=True)
    X_all, S_all, Y_all, dates = d["X_all"], d["S_all"], d["Y_all"], d["date"]
    dates_ts = pd.DatetimeIndex(dates)
    logger.info("Loaded N=%d events", len(Y_all))

    # Standardize X globally for the regime detector (SW distance is
    # scale-sensitive; the 3 price-feature dims may have very different
    # scales). This uses the FULL sample's mean/std only to put features
    # on comparable footing for the detector itself, not for prediction --
    # the KRR model below still standardizes using TRAIN-only statistics.
    X_for_regime = (X_all - X_all.mean(0)) / (X_all.std(0) + 1e-8)

    t_min, t_max = dates_ts.min(), dates_ts.max()
    train_td = timedelta(days=int(TRAIN_YEARS * 365.25))
    test_td = timedelta(days=int(TEST_YEARS * 365.25))
    step_td = timedelta(days=int(STEP_MONTHS * 30.44))

    fold_starts = []
    t = t_min
    while t + train_td + test_td <= t_max + step_td:
        fold_starts.append(t)
        t += step_td

    all_stable_corr, all_shift_corr = [], []

    for fold_idx, t_start in enumerate(fold_starts):
        train_start, train_end = t_start, t_start + train_td
        test_start, test_end = train_end, train_end + test_td
        train_mask = (dates_ts >= train_start) & (dates_ts < train_end)
        test_mask = (dates_ts >= test_start) & (dates_ts < test_end)
        n_tr, n_te = train_mask.sum(), test_mask.sum()
        if n_tr < MIN_TRAIN or n_te < MIN_TEST:
            continue

        # --- Regime detection on X instead of Z ---
        detector = RegimeDetector(window_days=90, step_days=30, threshold_std=1.5)
        detector.fit(dates_ts[train_mask], X_for_regime[train_mask])

        combined_mask = train_mask | test_mask
        flags = detector.transform(
            dates_ts[combined_mask], X_for_regime[combined_mask],
            event_dates=dates_ts[test_mask],
        )
        test_is_shift = flags.values.astype(bool)
        n_shift, n_stable = test_is_shift.sum(), (~test_is_shift).sum()

        # --- Same low-capacity direction model as step2c/2d/2e ---
        X_tr, S_tr, Y_tr = X_all[train_mask], S_all[train_mask], Y_all[train_mask]
        X_te, S_te, Y_te = X_all[test_mask], S_all[test_mask], Y_all[test_mask]

        x_mean, x_std = X_tr.mean(0), X_tr.std(0) + 1e-8
        s_mean, s_std = S_tr.mean(0), S_tr.std(0) + 1e-8
        X_tr_s, X_te_s = (X_tr - x_mean) / x_std, (X_te - x_mean) / x_std
        S_tr_s, S_te_s = (S_tr - s_mean) / s_std, (S_te - s_mean) / s_std

        h2_X, h2_S = median_heuristic(X_tr_s), median_heuristic(S_tr_s)
        K_tt = (rbf_kernel(X_tr_s, h2=h2_X) + rbf_kernel(S_tr_s, h2=h2_S)) / 2.0
        K_tetr = (rbf_kernel(X_te_s, h2=h2_X, Y=X_tr_s) + rbf_kernel(S_te_s, h2=h2_S, Y=S_tr_s)) / 2.0

        dir_pred = krr_fit_predict(K_tt, Y_tr, K_tetr, KRR_REG)

        stable_corr = corr(dir_pred[~test_is_shift], Y_te[~test_is_shift])
        shift_corr = corr(dir_pred[test_is_shift], Y_te[test_is_shift]) if n_shift >= 3 else float("nan")

        all_stable_corr.append((dir_pred[~test_is_shift], Y_te[~test_is_shift]))
        if n_shift >= 3:
            all_shift_corr.append((dir_pred[test_is_shift], Y_te[test_is_shift]))

        print(f"\n{'='*78}\nFOLD {fold_idx}: test [{test_start.date()}-{test_end.date()}), "
              f"n_test={n_te}, X-regime: n_stable={n_stable}, n_shift={n_shift}\n{'='*78}")
        print(f"  Detector threshold: SW_2^2 = {detector.threshold_:.5f} "
              f"(mean={detector.threshold_mean_:.5f}, std={detector.threshold_std_:.5f})")
        print(f"  Out-of-sample corr, STABLE test events (n={n_stable}): {stable_corr:.3f}")
        print(f"  Out-of-sample corr, SHIFT  test events (n={n_shift}):  {shift_corr:.3f}")

    if all_stable_corr:
        pred_s = np.concatenate([p for p, _ in all_stable_corr])
        true_s = np.concatenate([y for _, y in all_stable_corr])
        print(f"\n=== POOLED across all folds (X-based regime split) ===")
        print(f"Stable events (N={len(pred_s)}): pooled corr = {corr(pred_s, true_s):.3f}")
    if all_shift_corr:
        pred_h = np.concatenate([p for p, _ in all_shift_corr])
        true_h = np.concatenate([y for _, y in all_shift_corr])
        print(f"Shift events  (N={len(pred_h)}): pooled corr = {corr(pred_h, true_h):.3f}")
    else:
        print("No fold had enough X-based regime-shift test events for a pooled estimate.")


if __name__ == "__main__":
    main()
