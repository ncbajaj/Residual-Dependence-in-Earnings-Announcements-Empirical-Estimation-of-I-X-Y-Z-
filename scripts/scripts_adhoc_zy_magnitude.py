"""
Does transcript content (Z) predict the MAGNITUDE of the forward return,
the SIGN, or both? This decomposes the I(Z;Y) finding from
scripts_adhoc_zy_link.py / scripts_adhoc_zy_demeaned.py to find out whether
it's a volatility signal (magnitude) or a directional signal (sign) --
the two have completely different trading implications.

Uses the same fixed random subsample (N=400) and permutation-null approach
as scripts_adhoc_zy_demeaned.py, for speed and comparability.
"""
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("zy_magnitude")

from dependence.contingency import mean_square_contingency

data = np.load("/tmp/zy_aligned.npz", allow_pickle=True)
Z_all, Y_all, tickers = data["Z_all"], data["Y_all"], data["ticker"]
N_full = len(Y_all)

df = pd.DataFrame({"ticker": tickers, "y": Y_all})
ticker_means = df.groupby("ticker")["y"].transform("mean").values
Y_demeaned = Y_all - ticker_means  # same within-company return used before

N_SUB = 400
rng = np.random.default_rng(7)  # same seed -> same subsample as the demeaned test
sub_idx = rng.choice(N_full, size=N_SUB, replace=False)
Z_sub = Z_all[sub_idx]
Y_sub = Y_demeaned[sub_idx]

Y_mag = np.abs(Y_sub).reshape(-1, 1)          # magnitude only
Y_sign = np.sign(Y_sub).reshape(-1, 1)        # direction only (-1/0/+1)

logger.info("Subsample N=%d. |Y| mean=%.5f std=%.5f. Sign balance: +%.0f%% / -%.0f%%",
            N_SUB, Y_mag.mean(), Y_mag.std(),
            100 * (Y_sign > 0).mean(), 100 * (Y_sign < 0).mean())

n_perm = 100


def run_test(label, Y_col):
    res_obs = mean_square_contingency(Z_sub, Y_col, reg=1e-3)
    logger.info("[%s] Observed Î(Z;%s) = %.6f", label, label, res_obs.I_XZ_nocco)
    null_vals = np.empty(n_perm)
    rng2 = np.random.default_rng(11)
    for i in range(n_perm):
        perm = rng2.permutation(N_SUB)
        res_null = mean_square_contingency(Z_sub, Y_col[perm], reg=1e-3)
        null_vals[i] = res_null.I_XZ_nocco
        if (i + 1) % 25 == 0:
            logger.info("  [%s] permutation %d/%d done", label, i + 1, n_perm)
    p_value = float(np.mean(null_vals >= res_obs.I_XZ_nocco))
    z_score = (res_obs.I_XZ_nocco - null_vals.mean()) / null_vals.std()
    print(f"\n=== Z -> {label} (N={N_SUB}) ===")
    print(f"Observed  = {res_obs.I_XZ_nocco:.6f}")
    print(f"Null      = {null_vals.mean():.6f} +/- {null_vals.std():.6f}")
    print(f"z-score   = {z_score:.2f}")
    print(f"p-value   = {p_value:.4f}")
    return p_value, z_score


p_mag, z_mag = run_test("|Y| (magnitude)", Y_mag)
p_sign, z_sign = run_test("sign(Y) (direction)", Y_sign)

print("\n=== SUMMARY ===")
print(f"Magnitude: z={z_mag:.2f}, p={p_mag:.4f}  -> {'SIGNIFICANT' if p_mag < 0.05 else 'not significant'}")
print(f"Direction: z={z_sign:.2f}, p={p_sign:.4f}  -> {'SIGNIFICANT' if p_sign < 0.05 else 'not significant'}")
if p_mag < 0.05 and p_sign >= 0.05:
    print("CONCLUSION: Transcripts predict HOW BIG the move will be, not WHICH WAY -- a volatility signal, not a directional one.")
elif p_sign < 0.05 and p_mag >= 0.05:
    print("CONCLUSION: Transcripts predict direction, not magnitude -- the opposite of the volatility hypothesis.")
elif p_mag < 0.05 and p_sign < 0.05:
    print("CONCLUSION: Transcripts predict both magnitude AND direction.")
else:
    print("CONCLUSION: Neither magnitude nor direction alone is significant at this subsample size -- the original I(Z;Y) signal may be more diffuse/nonlinear than a simple magnitude/direction split captures.")
