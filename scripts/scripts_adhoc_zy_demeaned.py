"""
Robustness check on the I(Z;Y) finding from scripts_adhoc_zy_link.py.

With 384-dim embeddings and ~16 filings per company on average, the kernel
dependence test could be picking up "which company is this" (each firm has
a stylistic/template fingerprint in its transcripts) rather than genuine
quarter-to-quarter sentiment -- IF some companies simply have higher or
lower average returns than others over 2020-2023. Removing each ticker's
own mean return before re-testing isolates whatever is left: does WITHIN-
company variation in transcript content track WITHIN-company variation in
returns? That is the economically meaningful "sentiment" question.

Uses a fixed random subsample (not the full N) for speed under heavy host
load -- the eigendecomposition cost scales as O(N^3), so a subsample of
400 is ~11x cheaper per permutation than the full N=1407.
"""
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("zy_demeaned")

from dependence.contingency import mean_square_contingency

data = np.load("/tmp/zy_aligned.npz", allow_pickle=True)
Z_all, Y_all, tickers = data["Z_all"], data["Y_all"], data["ticker"]
N_full = len(Y_all)
logger.info("Loaded cached arrays: N=%d, %d unique tickers", N_full, len(set(tickers)))

df = pd.DataFrame({"ticker": tickers, "y": Y_all})
ticker_means = df.groupby("ticker")["y"].transform("mean").values
Y_demeaned = Y_all - ticker_means

N_SUB = 400
rng = np.random.default_rng(7)
sub_idx = rng.choice(N_full, size=N_SUB, replace=False)
Z_sub = Z_all[sub_idx]
Y_sub = Y_demeaned[sub_idx]
Y_col = Y_sub.reshape(-1, 1)
logger.info("Fixed random subsample: N=%d (from %d)", N_SUB, N_full)

res_obs = mean_square_contingency(Z_sub, Y_col, reg=1e-3)
logger.info("Observed Î(Z; Y_demeaned) NOCCO [subsample] = %.6f", res_obs.I_XZ_nocco)

n_perm = 100
null_vals = np.empty(n_perm)
for i in range(n_perm):
    perm = rng.permutation(N_SUB)
    res_null = mean_square_contingency(Z_sub, Y_col[perm], reg=1e-3)
    null_vals[i] = res_null.I_XZ_nocco
    if (i + 1) % 20 == 0:
        logger.info("  permutation %d/%d done", i + 1, n_perm)

p_value = float(np.mean(null_vals >= res_obs.I_XZ_nocco))
z_score = (res_obs.I_XZ_nocco - null_vals.mean()) / null_vals.std()

print("\n=== Z (transcript) -> Y_demeaned (within-ticker abnormal return), subsample N=400 ===")
print(f"Î(Z; Y_demeaned) observed = {res_obs.I_XZ_nocco:.6f}")
print(f"Permutation null mean     = {null_vals.mean():.6f} +/- {null_vals.std():.6f}")
print(f"z-score vs null           = {z_score:.2f}")
print(f"p-value                   = {p_value:.4f}")
print(f"Conclusion: {'Within-company sentiment link SURVIVES demeaning' if p_value < 0.05 else 'Effect VANISHES after removing ticker fixed effects -- original link was company identity, not sentiment'}")
