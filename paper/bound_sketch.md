# A concept-drift term for the zero-shot regime-shift bound: sketch

**Status: working sketch, not a finished theorem.** This lays out a decomposition that is provable now with standard technique, an exact bridging identity connecting it to the CMMD statistic validated empirically (`scripts/step3b_cmmd_permnull.py`), a covariate-shift bridge now matched word-for-word against the source paper's Lemma 14, and one remaining open gap (finite-sample confidence bound for the CMMD term, Gap 2) plus one smaller residual item (Gap 1b — verifying an assumption of the now-exact Lemma 14 bridge holds in this case study) before this becomes a real theorem. Honesty about which parts are solid and which aren't is the point of writing this down now rather than later.

**Scope note, important:** Mehta & Harchaoui's Theorem 1 ($\|\eta^\star-\eta_\rho\|^2_{L_2(P_X)} \lesssim I(X;Y\mid Z) + \text{prompt bias}$) is a *single-distribution* statement — no train/deploy shift appears in it at all. The shift-dependent machinery this sketch reuses is their Lemma 14 (arXiv:2507.09128v1, Appendix D.3, p.49), which in their paper bridges a *pretraining-vs-evaluation* covariate mismatch, not a temporal one. Section 3 below re-applies their exact inequality to a different pair of distributions (train-time $P$ → deploy-time $Q$) — a legitimate reuse of the same proof technique, but explicitly not the claim they make. The exact statement has now been pulled from the source and checked (previously this was a reconstructed guess; see Section 3 for what the guess got wrong).

---

## 1. Setup

Let $P$ be the training-time joint distribution over $(X,Y)$ and $Q$ the deployment-time joint distribution. Write $\eta_P(x) = \mathbb E_P[Y\mid X=x]$ and $\eta_Q(x) = \mathbb E_Q[Y\mid X=x]$ for the two regression functions, and let $\eta_\rho$ be a predictor trained on $P$-distributed data (in our case study: the KRR fit of Section 4.3 of the draft). The quantity of interest is the deployment-time error
$$
\mathcal E \;:=\; \|\eta_\rho - \eta_Q\|^2_{L_2(Q_X)}.
$$

## 2. Decomposition (Lemma A — provable now)

Write $\eta_\rho - \eta_Q = (\eta_\rho - \eta_P) + (\eta_P - \eta_Q)$ and apply $(a+b)^2 \le 2a^2+2b^2$ pointwise, then integrate against $Q_X$:
$$
\mathcal E \;\le\; 2\,\|\eta_\rho-\eta_P\|^2_{L_2(Q_X)} \;+\; 2\,\|\eta_P-\eta_Q\|^2_{L_2(Q_X)}.
\tag{1}
$$
This is elementary — no assumptions beyond both norms being finite. Call the first term $\mathcal A$ (estimation error, evaluated under the *deployment* covariate measure) and the second $\mathcal B$ (concept drift). The whole point of this sketch is that $\mathcal B = 0$ exactly under pure covariate shift ($P(Y\mid X) = Q(Y\mid X)$ everywhere, by definition), and the existing framework has no term that can become nonzero in that case while $\mathrm{TV}(P_X,Q_X) = 0$ — i.e., the existing bound is structurally silent on exactly the failure mode Section 5.7–5.8 of the draft shows is the operative one in our case study.

## 3. Converting $\mathcal A$ back to the existing machinery (Lemma B — now checked against the source)

$\mathcal A = \|\eta_\rho-\eta_P\|^2_{L_2(Q_X)}$ is an in-distribution-style estimation error (how far the trained predictor is from the *true training-time* regression function) but weighted by the *deployment* measure $Q_X$ instead of $P_X$.

**Gap 1 — resolved, with a correction.** Mehta & Harchaoui's actual Lemma 14 (arXiv:2507.09128v1, Appendix D.3, p.49) reads: for distributions $P_X,Q_X$ with densities w.r.t. a common dominating measure $\nu_X$, and any $\eta:\mathcal X\to\mathbb R$ with $\eta|_X\in L_2(P_X)\cap L_2(Q_X)$,

- **Additive form**, if $\|\eta\|_\infty<\infty$:
$$\|\eta\|^2_{L_2(P_X)} \;\le\; \|\eta\|^2_{L_2(Q_X)} + \|\eta\|_\infty^2\,\mathrm{TV}(P_X,Q_X).$$
- **Multiplicative form**, if $Q_X\ll P_X$ and $\frac{dQ_X}{dP_X}(x)\le B_{P,Q}$ for $P_X$-a.e. $x$:
$$\|\eta\|^2_{L_2(P_X)} \;\le\; B_{P,Q}\,\|\eta\|^2_{L_2(Q_X)}.$$

In their paper this bridges an estimation-error bound naturally stated under the *pretraining* marginal $Q_X$ (their Theorem 11) back to the *evaluation* marginal $P_X$ that their end-to-end guarantee (their eq. 87) needs — a pretrain/downstream mismatch, not a temporal one. I'm reusing the identical inequality for a different mismatch (train-time $P\to$ deploy-time $Q$ in this sketch's notation), which is a legitimate re-application of the same proof technique (Hölder / change-of-measure) but is **not the same claim they make** — relabeling their target role ($P_X$, the norm being bounded) as my deploy-time $Q$, and their source role ($Q_X$, the norm doing the bounding) as my train-time $P$:

$$
\mathcal A \;=\; \|\eta_\rho-\eta_P\|^2_{L_2(Q_X)} \;\le\; \|\eta_\rho-\eta_P\|^2_{L_2(P_X)} \;+\; \|\eta_\rho-\eta_P\|_\infty^2\,\mathrm{TV}(P_X,Q_X)
\tag{2}
$$

(additive case; requires $\|\eta_\rho-\eta_P\|_\infty<\infty$), or, under a bounded-density-ratio assumption in the other direction ($P_X\ll Q_X$, $\frac{dP_X}{dQ_X}(x)\le B$ for $Q_X$-a.e. $x$),

$$
\mathcal A \;\le\; B\cdot\|\eta_\rho-\eta_P\|^2_{L_2(P_X)}.
\tag{2'}
$$

$\|\eta_\rho-\eta_P\|^2_{L_2(P_X)}$ is then exactly what Theorem 1's machinery bounds via $I(X;Y\mid Z)$ + prompt bias.

**What my original guess got wrong.** The informal $(1+c\cdot d(P_X,Q_X))$ shape I'd written before reading the source was wrong on two counts, not just imprecise: (a) it conflated the additive and multiplicative cases into one multiplicative-looking expression, and (b) the real additive bound (2) scales the TV term by $\|\eta_\rho-\eta_P\|_\infty^2$ — a sup-norm bound on the *predictor-minus-truth gap itself*, not a generic constant times the whole squared norm. That's a materially different, and in our case non-trivial, assumption: KRR predictions aren't bounded a priori without an explicit clipping step or an RKHS-norm argument, so this would need to be checked (or enforced) before being assumed. The multiplicative alternative (2') needs the training covariate distribution $P_X$ absolutely continuous w.r.t. the deployment distribution $Q_X$ with a bounded ratio — i.e. training data can't put mass where deployment covariates essentially never occur — which should be checked empirically (e.g. via the regime detector's embedding space, Section 5.7 of the draft) rather than assumed.

**Residual open item (call it Gap 1b, smaller than the original Gap 1):** verify that $\|\eta_\rho-\eta_P\|_\infty<\infty$ is plausible for the fitted KRR predictor in the case study (or substitute the multiplicative form (2') and check the density-ratio assumption instead) before (4) below is stated as a clean theorem rather than a sketch.

## 4. The concept-drift term $\mathcal B$, and its exact connection to CMMD (Lemma C — provable now)

This is the part that's new, and it's the one with a clean, exact identity rather than an inequality. The conditional mean embedding of $Y\mid X=x$ under a kernel $\kappa$ on the output space is $\mu^P_{Y|X=x} = \mathbb E_P[\kappa(Y,\cdot)\mid X=x]$. **If $\kappa$ is the linear kernel** $\kappa(y,y')=yy'$, this embedding collapses to the scalar $\eta_P(x)$ itself (a 1-dimensional RKHS), and the CMMD statistic defined in Section 4 of the draft becomes, exactly:
$$
\mathrm{CMMD}_{\text{linear}}(x) \;=\; \big|\eta_P(x) - \eta_Q(x)\big|.
$$
Therefore
$$
\mathcal B \;=\; \|\eta_P-\eta_Q\|^2_{L_2(Q_X)} \;=\; \mathbb E_{x\sim Q_X}\big[\mathrm{CMMD}_{\text{linear}}(x)^2\big],
\tag{3}
$$
**exactly**, with no approximation — this is essentially a restatement of the definition of a regression function, not a deep result, but it is the bridge that lets a *finite-sample estimator with known statistical behavior* stand in for the unobservable population quantity $\mathcal B$.

**Important, and worth flagging plainly:** the CMMD runs validated empirically in `step3b_cmmd_permnull.py` used an **RBF** output kernel, not the linear kernel identity (3) needs. RBF-CMMD is a reasonable, more general empirical drift diagnostic (sensitive to differences anywhere in the conditional distribution, not just its mean). But it is *not* the exact plug-in estimator for $\mathcal B$ as defined for this specific bound.

**Done:** `step3c_cmmd_linear.py` reruns the same permutation-calibrated test with the linear output kernel, which collapses exactly to $\mathrm{CMMD}_{\text{linear}}(x) = |f_1(x)-f_2(x)|$ for two KRR fits — a direct, bound-matching point estimate of $\mathcal B$. Result, and it is more textured than a clean repeat of the RBF finding:

| Comparison | RBF CMMD (general shape) | Linear CMMD (mean only — matches $\mathcal B$ exactly) |
|---|---|---|
| 2020 vs 2021 | z=5.23, p≈0 | z=1.29, p=0.083 (not significant) |
| 2021 vs 2022 | z=7.86, p≈0 | z=3.78, p≈0 |
| 2022 vs 2023 | z=7.77, p≈0 | z=3.49, p≈0 |
| same-year nulls (all 4 years) | correctly non-significant | correctly non-significant |

The null-case calibration still passes cleanly under the linear kernel. But the 2020→2021 transition, strongly significant under the general (RBF) test, is only marginal under the mean-only (linear, bound-matching) test. Read together with the Chow test's own pattern — the coarse two-period split (2020–21 vs 2022–23) was only marginal at $p=0.067$ while the full four-year split was clearly significant at $p=0.0032$ — three independent signals now agree that the *mean-level* drift $\mathcal B$ specifically concentrates in the 2021→2022 and 2022→2023 transitions (coinciding with the rate-hiking cycle intensifying), while the 2020→2021 change looks more like a shift in the conditional distribution's shape (variance/tails) than in $\eta(x)$ itself. That distinction — not all drift is mean-drift, and the bound's $\mathcal B$ term specifically isolates the part that is — is a genuinely novel empirical point worth keeping in the eventual paper, not just a repeat of the Chow-test finding under a fancier estimator.

## 5. The combined bound (informal statement, pending Gaps 1b–2)

Putting (1), (2), (3) together (additive form of Lemma B; the multiplicative form (2') substitutes cleanly in place of the last two terms of the first bracket if its assumption is preferred):
$$
\|\eta_\rho - \eta_Q\|^2_{L_2(Q_X)} \;\lesssim\; \underbrace{\Big(I(X;Y\mid Z) + \text{prompt bias}\Big) \;+\; \|\eta_\rho-\eta_P\|_\infty^2\,\mathrm{TV}(P_X,Q_X)}_{\text{covariate-shift term (existing machinery + exact Lemma 14 bridge)}} \;+\; \underbrace{\mathbb E_{Q_X}\big[\mathrm{CMMD}_{\text{linear}}(x)^2\big]}_{\text{concept drift (new)}}.
\tag{4}
$$
This is a *combined* bound: it reduces to (a close cousin of) the existing regime-shift result when $\mathcal B=0$, and it remains informative — rather than vacuous — exactly in the regime our case study lives in: $\mathrm{TV}(P_X,Q_X)\approx 0$ (Section 5.7: neither modality's regime detector flags anything) while $\mathcal B$ is large (Section 5.8: the Chow test and the calibrated CMMD both reject at $p<0.01$). Note this is now the exact structural form of Mehta & Harchaoui's own eq. (87) (Appendix D.3) with the extra $\mathcal B$ term appended — the covariate-shift piece is no longer a reconstructed guess, it's their lemma verbatim, relabeled.

## 6. Gap 2 (open): turning the point estimate into a confidence bound

Section 5 of this sketch states (4) with the *population* quantity $\mathcal B$. To make it usable from data, the KRR-based CMMD estimator needs a finite-sample confidence bound, not just a permutation-test p-value — permutation testing tells you "$\mathcal B$ is probably not zero," which is what we used in Section 5.8 of the draft, but a *bound* needs an upper confidence bound on $\mathcal B$'s actual magnitude to be substituted into (4) and still hold with high probability. The kernel conditional two-sample test paper we pulled the CMMD construction from states exactly this kind of result (their Theorem 4.4, giving finite-sample type-I error control via explicit per-dataset confidence terms $B_i(D_i,x)$) — the concrete next step is to pull that theorem's exact statement and check whether it transfers directly to give a high-probability upper bound on $\mathcal B$ itself (aggregated over $x\sim Q_X$, not just a pointwise test), or whether it needs adaptation.

## 7. What's actually solid right now, versus what's still open

| Piece | Status |
|---|---|
| Decomposition (1) | Solid — elementary, no assumptions beyond finiteness. |
| Covariate-shift bridge (2)/(2') | **Resolved.** Exact statement of Mehta & Harchaoui's Lemma 14 (arXiv:2507.09128v1, Appx. D.3, p.49), re-applied to train→deploy shift instead of their pretrain→eval shift. Original guessed form was wrong in structure, not just constants (see Section 3). Residual item: verify $\|\eta_\rho-\eta_P\|_\infty<\infty$ or the multiplicative form's density-ratio bound holds for the case study (Gap 1b, smaller than original Gap 1). |
| CMMD-as-exact-identity for $\mathcal B$ under a linear output kernel (3) | Solid, essentially definitional. |
| Empirical validation that the underlying drift is real and detectable | Done, both with the general (RBF) and the bound-matching (linear) output kernel (`step2g_chow_test.py`, `step3b_cmmd_permnull.py`, `step3c_cmmd_linear.py`). The two kernels disagree on the 2020→2021 transition specifically, which is itself informative (mean-drift vs shape-drift, see Section 4). |
| Finite-sample confidence bound on $\mathcal B$, not just a p-value (Gap 2) | Open — needs the source CMMD paper's Theorem 4.4 read in full and checked for transfer. |
| Combined bound (4) as a fully rigorous theorem | Not yet — contingent on closing Gap 1b and Gap 2. |

## 8. Concrete next steps, in order

1. ~~Rerun with a linear output kernel to get the bound-matching point estimate of $\mathcal B$.~~ **Done** — see Section 4; result is more textured than expected (mean-drift concentrates in 2 of 3 year-pairs, not all 3).
2. ~~Pull the exact statement of Mehta & Harchaoui's Lemma 14 and check it against the reconstructed form in (2).~~ **Done** — see Section 3; the guessed form was structurally wrong (conflated additive/multiplicative cases), now replaced with the exact source statement. Resolves Gap 1, leaves a smaller residual item (Gap 1b): check the sup-norm or density-ratio assumption actually holds for the KRR fit used in the case study.
3. Pull the exact statement of the kernel conditional two-sample test paper's Theorem 4.4 and check whether it transfers to a high-probability bound on the *aggregated* quantity $\mathcal B = \mathbb E_{Q_X}[\mathrm{CMMD}^2]$, as opposed to a pointwise test at a single $x$ — resolves Gap 2.
4. Check Gap 1b empirically: either bound $\|\eta_\rho-\eta_P\|_\infty$ for the fitted KRR predictor (e.g. via clipping or an RKHS-norm argument) or verify the bounded-density-ratio assumption for (2') using the regime detector's embedding space.
5. Only after 3–4: write the combined bound as a real, numbered theorem with a complete proof, and re-verify the empirical numbers in the draft against the linear-kernel estimator from step 1.
