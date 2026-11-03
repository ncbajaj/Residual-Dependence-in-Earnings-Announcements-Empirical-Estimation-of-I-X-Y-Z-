# A concept-drift term for the zero-shot regime-shift bound: sketch

**Status: working sketch, not a finished theorem.** This lays out a decomposition that is provable now with standard technique, an exact bridging identity connecting it to the CMMD statistic validated empirically (`scripts/step3b_cmmd_permnull.py`), a covariate-shift bridge now matched word-for-word against the source paper's Lemma 14, and a finite-sample confidence bound for the CMMD term now matched against its source paper's Theorems 4.2–4.4. Both originally-flagged gaps are resolved *in structure* — no open logical gaps remain in the argument shape. What's left is confirming each source paper's regularity assumptions actually hold for this case study's fitted models (Gap 1b, and the analogous items in Section 6), and then instantiating the resulting bounds numerically. Honesty about which parts are solid and which aren't is the point of writing this down now rather than later.

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

## 6. Gap 2 — resolved: turning the point estimate into a confidence bound

**Source, exact.** The CMMD construction used in `step3b`/`step3c` comes from Massiani, Fiedler, Haverbeck, Solowjow & Trimpe, *"Kernel conditional tests from learning-theoretic bounds"* (arXiv:2506.03898v2; accepted NeurIPS 2025 — note the arXiv v1 title was "A kernel conditional two-sample test," which is what earlier notes in this repo call it). Their relevant results:

- **Theorem 4.2** (abstract confidence-bound-to-test result): given a learning method $\mathcal L_i: D\mapsto f_D^{(i)}$ and a function $B_i$ such that
$$\mathbb P\Big[\forall (x,n)\in S\times\mathcal N_i,\ \big\|f^{(i)}_{D_{:n}}(x)-\mathbb E[p](x)\big\|\le B_i\big(D_{:n},x\big)\Big]\ \ge\ 1-\delta_i,$$
the test $\mathcal T:\ \mathbb 1\{\|f^{(1)}_{D_1}(x)-f^{(2)}_{D_2}(x)\| > B_1(D_1,x)+B_2(D_2,x)\}$ has $(S,\mathcal N_1,\mathcal N_2)$-level $\delta_1+\delta_2$.
- **The critical detail, easy to miss:** the coverage statement is $\forall x\in S$ *simultaneously* — $B_i$ is a **uniform-in-$x$** confidence band over the whole region $S$, not a bound computed fresh at one query point. This is exactly the property Gap 2 needed and I wasn't sure would be there before reading the source.
- **Theorem 4.3** instantiates $B_i$ explicitly for (vector-valued) KRR, under: (a) the conditional law $p$ (of $Y\mid X$) is $R$-subgaussian, and (b) an RKHS-membership / source-condition assumption $\mathbb E[p]\in\mathcal H_K$ with $\|\mathbb E[p]\|_K\le S$ — giving $B(D,x)=\beta_\lambda(D,\delta)\cdot\sigma_{D,\lambda}(x)$, with $\beta_\lambda$ an explicit, computable formula (their eq. 10 or 11 depending on the data-collection case) and $\sigma_{D,\lambda}(x)$ the usual KRR posterior-variance-shaped quantity.
- **Theorem 4.4** packages Theorem 4.2 + Theorem 4.3 into the two-KRR-fit test directly — this is the "Theorem 4.4" referenced loosely in earlier notes in this repo, now pinned to its exact statement.

**The transfer argument (why Gap 2 closes).** Because the Theorem 4.3/4.4 bound holds uniformly for all $x\in S$ at once (not pointwise-only), the event
$$\forall x\in S:\ \ \mathrm{CMMD}_{\text{linear}}(x) = |f_1(x)-f_2(x)| \;\le\; B_1(D_1,x)+B_2(D_2,x)$$
holds with probability $\ge 1-(\delta_1+\delta_2)$. Squaring (monotone, preserves the event) and integrating against $Q_X$ — a deterministic operation on a function that is already uniformly bounded on an event not involving $Q_X$ at all, so no extra union bound is needed, provided $\mathrm{supp}(Q_X)\subseteq S$ — gives, on the *same* event:
$$
\mathcal B \;=\; \mathbb E_{Q_X}\big[\mathrm{CMMD}_{\text{linear}}(x)^2\big] \;\le\; \mathbb E_{Q_X}\Big[\big(B_1(D_1,x)+B_2(D_2,x)\big)^2\Big] \qquad\text{w.p.}\ge 1-(\delta_1+\delta_2).
\tag{3'}
$$
The right-hand side is a concrete, computable quantity (a $Q_X$-weighted average of a known function of the fitted KRR models), not just a p-value — this is a genuine high-probability **upper bound** on the population concept-drift term $\mathcal B$, which is what the combined bound (4) needs to be stated with confidence rather than just a rejection decision.

**What's still open before this is numerically usable (not a new gap, but real remaining work):**
1. Assumption-checking: $R$-subgaussianity of the conditional return distribution and the RKHS-membership/source-condition bound $\|\mathbb E[p]\|_K\le S$ for our covariate kernel — structurally the same kind of residual item as Gap 1b, i.e. a regularity assumption on the fitted functions rather than a logical gap in the argument.
2. ~~Actually instantiating $\beta_\lambda(D,\delta)$ (their eq. 10/11) numerically for the `step3c_cmmd_linear.py` KRR fits, and computing (3') on the cached dataset.~~ **Done** — see below.
3. Handling that $S$ must contain $\mathrm{supp}(Q_X)$ while $D_1,D_2$ (the two years' data) may have different covariate supports — plausible in our setting (same standardized feature space across years) but worth a direct check via the regime detector's embedding coverage, same spirit as Gap 1b's density-ratio check.

**Numeric instantiation (`step3d_cmmd_confidence_bound.py`) — the bound is correct but, as expected for this class of result, very loose.** For scalar-output KRR, Theorem 4.3's operator-valued $\beta_\lambda,\sigma_{D,\lambda}$ reduce to a closed form in the eigenvalues of the training Gram matrix (derivation in the script's docstring): with $\{\mu_i\}$ the eigenvalues of $K$, $\lambda$ the ridge constant actually used in the fit, $r_i=\mu_i/(\mu_i+\lambda)$, $\sigma^2$ an estimated residual variance, and $S$ an estimated RKHS-norm bound,
$$\beta_\lambda(D,\delta) = S + \lambda^{-1/2}\sqrt{\sigma^2\textstyle\sum_i r_i + 2\sqrt{\ln(1/\delta)}\,\sigma^2\sqrt{\textstyle\sum_i r_i^2} + 2\ln(1/\delta)\,\sigma^2\max_i r_i}.$$
$S$ and $\sigma^2$ were estimated *in-sample* from the fitted KRR model (the fit's own RKHS norm, $\sqrt{\alpha^\top K\alpha}$, and its residual variance) — a plug-in approximation standing in for the assumption-checking in item 1 above, not a rigorous a priori bound. Running this at $\delta_1=\delta_2=0.05$ (overall level 0.10) on the cached dataset:

| Pair | $n_1,n_2$ | plug-in $\widehat{\mathcal B}$ (point estimate) | upper bound (3') | ratio |
|---|---|---|---|---|
| 2020 vs 2021 | 306, 317 | 0.000182 | 0.1709 | ~940× |
| 2021 vs 2022 | 317, 348 | 0.000246 | 0.1562 | ~640× |
| 2022 vs 2023 | 348, 325 | 0.000243 | 0.1404 | ~580× |

The bound holds (it's not violated, and it's a legitimate high-probability statement), but it is roughly 500–900× looser than the permutation-test point estimate at this sample size (n≈300–350 per year) — a standard feature of these learning-theoretic kernel confidence bounds, which are built from worst-case concentration arguments (matrix Bernstein-type terms) rather than tuned for tightness. **This is a real, honest limitation to report, not a bug to paper over:** the structural result (Gap 2) is resolved — a finite-sample confidence bound on $\mathcal B$ genuinely exists and is computable — but it is not yet informative enough at this $n$ to replace the permutation test as the paper's primary evidence. The permutation-test result (Section 4) remains the practically load-bearing evidence for drift; the confidence bound is a theoretical completion of the argument, not (yet) a tighter empirical tool. Loosening factors like this are a known property of this bound family and would need either much larger $n$, a tighter bootstrapped variant (the source paper's own Appendix B discusses bootstrapping specifically to address this), or a different proof technique to become practically competitive.

## 7. What's actually solid right now, versus what's still open

| Piece | Status |
|---|---|
| Decomposition (1) | Solid — elementary, no assumptions beyond finiteness. |
| Covariate-shift bridge (2)/(2') | **Resolved.** Exact statement of Mehta & Harchaoui's Lemma 14 (arXiv:2507.09128v1, Appx. D.3, p.49), re-applied to train→deploy shift instead of their pretrain→eval shift. Original guessed form was wrong in structure, not just constants (see Section 3). Residual item: verify $\|\eta_\rho-\eta_P\|_\infty<\infty$ or the multiplicative form's density-ratio bound holds for the case study (Gap 1b, smaller than original Gap 1). |
| CMMD-as-exact-identity for $\mathcal B$ under a linear output kernel (3) | Solid, essentially definitional. |
| Empirical validation that the underlying drift is real and detectable | Done, both with the general (RBF) and the bound-matching (linear) output kernel (`step2g_chow_test.py`, `step3b_cmmd_permnull.py`, `step3c_cmmd_linear.py`). The two kernels disagree on the 2020→2021 transition specifically, which is itself informative (mean-drift vs shape-drift, see Section 4). |
| Finite-sample confidence bound on $\mathcal B$, not just a p-value (Gap 2) | **Resolved, structurally, and now instantiated numerically.** Massiani et al. (arXiv:2506.03898v2) Theorems 4.2–4.4 give a confidence bound $B_i(D,x)$ that is uniform in $x$ over a region $S$, not pointwise-only — squaring and integrating against $Q_X$ transfers it directly to a high-probability bound on $\mathcal B$ itself (eq. 3'), with no extra union bound needed. `step3d_cmmd_confidence_bound.py` computes it on the cached dataset: the bound is valid but ~500–900× looser than the permutation-test point estimate at $n\approx 300$ per year — a real, expected limitation of this bound family, not a bug. See Section 6. |
| Combined bound (4) as a fully rigorous theorem | Not yet — both remaining gaps are now "resolved in structure, unverified in our specific instance": Gap 1b (sup-norm/density-ratio assumption) and the Section 6 items (subgaussianity/RKHS-membership assumption, numeric $\beta_\lambda$ instantiation, support-containment check). No open *logical* gaps remain in the argument shape; what's left is confirming the paper's assumptions hold for this case study's fitted models. |

## 8. Concrete next steps, in order

1. ~~Rerun with a linear output kernel to get the bound-matching point estimate of $\mathcal B$.~~ **Done** — see Section 4; result is more textured than expected (mean-drift concentrates in 2 of 3 year-pairs, not all 3).
2. ~~Pull the exact statement of Mehta & Harchaoui's Lemma 14 and check it against the reconstructed form in (2).~~ **Done** — see Section 3; the guessed form was structurally wrong (conflated additive/multiplicative cases), now replaced with the exact source statement. Resolves Gap 1, leaves a smaller residual item (Gap 1b): check the sup-norm or density-ratio assumption actually holds for the KRR fit used in the case study.
3. ~~Pull the exact statement of the kernel conditional two-sample test paper's Theorem 4.4 and check whether it transfers to a high-probability bound on the aggregated quantity $\mathcal B$.~~ **Done** — see Section 6; it transfers cleanly because the source bound is already uniform in $x$, not pointwise. Resolves Gap 2 structurally; leaves numeric instantiation as follow-on work.
4. Check Gap 1b empirically: either bound $\|\eta_\rho-\eta_P\|_\infty$ for the fitted KRR predictor (e.g. via clipping or an RKHS-norm argument) or verify the bounded-density-ratio assumption for (2') using the regime detector's embedding space.
5. Check the Section 6 assumptions empirically: subgaussianity of the conditional-return distribution, an RKHS-norm bound on $\eta_P,\eta_Q$, and that the two years' covariate supports both sit inside a region $S\supseteq\mathrm{supp}(Q_X)$.
6. ~~Write a `step3d_cmmd_confidence_bound.py` that instantiates $\beta_\lambda(D,\delta)$ (eq. 10/11 of Massiani et al.) for the existing KRR fits and computes the right-hand side of (3') numerically, alongside the already-computed permutation-test point estimates.~~ **Done** — see Section 6; the bound is valid but ~500–900× looser than the point estimate at current sample sizes, a real limitation worth stating plainly rather than a reason to hide the result.
7. Only after 4–5: write the combined bound as a real, numbered theorem with a complete proof, and re-verify the empirical numbers in the draft against the linear-kernel estimator from step 1. Decide, as part of that writeup, whether the loose numeric bound belongs in the main text (as an honest "theory vs. practice" point) or an appendix.
