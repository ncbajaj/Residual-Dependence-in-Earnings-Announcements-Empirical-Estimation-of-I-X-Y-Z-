# Diagnosing Generalization Failure Beyond Covariate Shift: A Concept-Drift Case Study in Zero-Shot Financial Prediction

**Draft v0.1 — arXiv-bound. Not yet peer reviewed.**

## Abstract

Generalization bounds for zero-shot prediction typically decompose the error of a predictor trained on one distribution and deployed on another into an in-distribution term plus a *covariate-shift* penalty — a divergence between the training and deployment input distributions, usually total variation (TV) or a related integral probability metric. This framing implicitly assumes that when such a predictor fails to generalize, a detectable shift in the input distribution is the mechanism responsible, or at least a necessary accompanying symptom. We test this assumption empirically using earnings-call transcripts, pre-announcement price features, and analyst earnings-surprise data as a richly instrumented case study, chosen because it admits exactly the kind of controlled, multi-angle diagnosis the question requires: a real, independently-verifiable population-level dependence signal; a natural walk-forward temporal validation structure; and cheap, principled ablations on model capacity, regularization, and distributional shift. We find a statistically robust population-level dependence between transcript content and forward returns (permutation z≈4.6–7.7, surviving a company-fixed-effects control), but a model trained to exploit it fails walk-forward out-of-sample validation (net Sharpe −0.89, Newey-West p=0.08, reversing the apparently profitable sign of a naive single-split evaluation). We then run a four-step elimination protocol — capacity ablation, a six-order-of-magnitude regularization sweep, and covariate-shift detection on both input modalities via a rolling distributional-distance detector — and find that none of them explain the failure: a 4-dimensional model fails identically to a 388-dimensional one, no regularization setting recovers out-of-sample correlation, and neither the transcript-embedding nor the price-feature marginal distribution shows a detectable shift during the failure windows. A direct structural-break (Chow) test on a reduced linear model confirms the remaining candidate explanation directly rather than by elimination: the conditional relationship between features and outcomes changes significantly across the training/deployment boundary (F(15,1276)=2.30, p=0.003), with the instability concentrated in specific feature coordinates. This is *concept drift* — a change in P(Y|X), not P(X) — which a covariate-shift bound is structurally unable to detect or budget for. We argue this gap is not specific to our case study: any zero-shot bound stated purely in terms of input-distribution divergence will silently mis-attribute, or fail to flag, generalization failures whose actual source is concept drift, and we propose the four-step protocol used here as a general, reusable diagnostic for telling the two failure modes apart.

---

## 1. Introduction

A standard move in the zero-shot and transfer-learning generalization literature is to bound the deployment-time error of a predictor trained on a source distribution by its source error plus a term measuring how far the deployment (target) input distribution has moved from the source input distribution — typically total variation, Wasserstein distance, or another integral probability metric between $P_X$ and $Q_X$. This is a *covariate-shift* bound: it attributes generalization failure to a detectable change in the marginal distribution of the inputs, and, conversely, it implies that in the absence of such a detectable shift, the predictor's in-distribution performance should approximately carry over.

This is a clean and useful theoretical framing, and it is correct as a bound — a bound is always correct, in the sense that it holds; the open empirical question is whether the covariate-shift term is the one doing the explanatory work when a *real* predictor fails to generalize, or whether it is frequently vacuous or irrelevant while the actual failure comes from a different source entirely: *concept drift*, a change in the conditional distribution $P(Y\mid X)$ itself, which no bound stated purely in terms of $P_X$ versus $Q_X$ can see.

Distinguishing these two failure modes matters practically. If a practitioner observes a generalization gap and checks only for covariate shift — the natural first instinct, and the one a covariate-shift bound invites — a null result ("the input distribution looks the same") will be read as reassurance that the model should still be trusted, or as a puzzle with no available explanation. If the true cause is concept drift, that reassurance is false, and no amount of input-distribution monitoring will ever catch it.

We investigate this question using a case study chosen specifically because it lets us check every link in the causal chain independently, rather than relying on a single global generalization-gap number: earnings-call-transcript-based prediction of short-horizon stock returns. This domain has three properties that make it an unusually well-instrumented testbed for the covariate-shift-versus-concept-drift question, independent of any interest in trading:

1. **A verifiable population-level signal exists**, so a null walk-forward result cannot be dismissed as "there was never anything to generalize in the first place" — we can show the signal is real, directly, before asking whether a trained model recovers it out-of-sample.
2. **A natural, non-adversarial temporal train/test structure** (announcements occur continuously over calendar time) gives walk-forward validation real content, rather than an arbitrary train/test split.
3. **Multiple independent input modalities** (transcript text, pre-announcement price behavior, analyst earnings surprise) let us run the covariate-shift check on more than one representation of $X$, which matters: a shift undetectable in one modality's marginal distribution could in principle be visible in another's, and finding neither raises our confidence that the absence of detectable covariate shift is not simply an artifact of which representation we happened to inspect.

**Contributions.**

- We give a concrete empirical case in which a real, statistically verified population-level dependence between inputs and a target fails to survive a trained, walk-forward-validated predictor, and show the apparent single-split "success" of a naive strategy *reverses sign* under honest out-of-sample validation — a cautionary result about the trap of evaluating zero-shot or transfer claims on a single split.
- We give a four-step elimination protocol — capacity ablation, regularization sweep, and covariate-shift detection on each input modality — that rules out the three most common "fixable" explanations for a generalization gap (overfitting, under-regularization, detectable input-distribution shift) before concluding concept drift, and we confirm the conclusion directly with a structural-break test rather than resting on elimination alone.
- We argue, and give a worked example of, the general point: covariate-shift bounds are silent on concept drift by construction, and a predictor can fail to generalize with *no detectable change in $P_X$ at all* — which means empirical covariate-shift monitoring is not a sufficient diagnostic for trusting a zero-shot predictor's deployment-time behavior, however reassuring a null covariate-shift test looks.

---

## 2. Background

### 2.1 Zero-shot generalization bounds and the covariate-shift framing

[Mehta & Harchaoui, ICML 2025] bound the $L_2(P_X)$ error of a zero-shot predictor $\eta_\rho$ relative to the Bayes-optimal in-distribution predictor $\eta^\star$:
$$\|\eta^\star - \eta_\rho\|^2_{L_2(P_X)} \;\lesssim\; I(X; Y \mid Z) \;+\; \text{prompt bias},$$
where $Z$ is an auxiliary (zero-shot) representation and $I(X;Y\mid Z)$ is the residual conditional dependence between the *direct* input $X$ and the target $Y$ left unexplained by $Z$. Within their framework, a companion result (their Lemma 14, in the paper's numbering) further relates generalization degradation under *distribution shift* to the total-variation distance $\mathrm{TV}(P_X, Q_X)$ between the training and deployment input distributions — a covariate-shift bound in the classical sense of [Quionero-Candela et al., 2009].

This is the theoretical object we set out to stress-test empirically: does a detectable TV-style shift in the relevant input distribution actually accompany observed generalization failures, or can a zero-shot-style predictor fail to generalize with the input distribution looking entirely ordinary?

### 2.2 Covariate shift versus concept drift

The distinction we lean on throughout is standard in the dataset-shift literature [Quionero-Candela et al., 2009]: **covariate shift** is a change in $P(X)$ (or $P(Z)$) between training and deployment with $P(Y\mid X)$ held fixed; **concept drift** is a change in $P(Y\mid X)$ itself, which can occur with $P(X)$ completely unchanged. A bound stated as a function of $\mathrm{TV}(P_X, Q_X)$, or any other divergence between the two *marginal* input distributions, is by construction a covariate-shift bound: it is mathematically blind to concept drift, because concept drift can occur at TV distance zero.

### 2.3 Post-earnings-announcement drift, briefly

Our case-study domain has one well-established empirical regularity worth noting up front, because it gives us an internal check on whether our pipeline is finding something real: analyst earnings surprises predict same-direction subsequent returns, a finding with a decades-long replication record beginning with [Bernard & Thomas, 1989]. We use this as a sanity check in Section 6: if our structural-break analysis recovers the surprise-return relationship as the *most* stable coefficient across years, relative to price-based features with no comparable theoretical grounding, that is independent corroboration that our pipeline is measuring something real rather than an artifact of our own construction.

---

## 3. Case study: data and construction

### 3.1 Sources and universe

We construct a dataset from three sources: SEC EDGAR 8-K Item 2.02 filings (earnings-call transcripts), yfinance adjusted daily OHLCV (price data), and yfinance's analyst earnings-date history (estimated vs. reported EPS, i.e. the earnings surprise). The universe is 150 large- and mid-cap US equities, systematically sampled from the 2024 S&P 500 constituent list (every $n$-th ticker by alphabetical order, for even coverage across the list rather than a contiguous block), over January 2020–December 2023. An earlier, smaller run (66 tickers, same period) is reported alongside the 150-ticker results in Section 5.1 specifically because comparing the two is itself informative about estimator reliability at small $N$ (Section 5.1).

### 3.2 Variables

- $X$: 3-dimensional pre-announcement price features computed over the $[t-22, t-2]$ trading-day window relative to the announcement date $t$.
- $Z$: a 384-dimensional sentence embedding (all-MiniLM-L6-v2) of the transcript text.
- $Y$: the forward 5-trading-day abnormal return, $(P_{t+5}^{\text{close}} / P_{t+1}^{\text{open}} - 1) - $ the identical-window SPY return, computed with no overlap with the $[t-22,t-2]$ window used to build $X$.
- $S$: the earnings surprise (analyst EPS estimate vs. reported EPS, percent), matched to each transcript event within a 3-day tolerance.

$S$ and $Z$ are revealed at the same moment — the earnings release — so using $S$ to predict the forward return window is not lookahead, for the same reason using $Z$ is not.

### 3.3 Two data-quality findings worth flagging independently

Two issues surfaced during construction that are not specific to our analysis and are likely to recur in any EDGAR-based earnings-event pipeline:

**Ticker-to-CIK drift.** EDGAR's `company_tickers.json` maps a ticker to its *current* filer only. Two tickers in our universe (XOM, BLK) currently resolve to a post-reorganization CIK with no pre-reorganization filing history, silently zeroing the ticker out of any backtest spanning the reorganization date unless corrected with a historical-CIK override.

**Non-earnings Item 2.02 filings.** A subset of filings flagged "Item 2.02 present" are not the quarterly earnings release at all. One airline in our universe files a mid-quarter investor-guidance update under the same item code, consistently 9 days before the real earnings date; another company files a second, unrelated Item 2.02 filing roughly a month after every real earnings release. Both are detectable and excludable by requiring a successful match to an independently-sourced earnings-surprise record (Section 3.2), which we do throughout.

### 3.4 Leakage controls

All kernel bandwidths (median heuristic) and detector thresholds are computed on training data only and frozen for evaluation on held-out data. All conditional-mean predictions reported as "in-sample" in Section 5 are still ridge-regularized fits, not raw interpolation; "out-of-sample" predictions always use a cross-kernel against training data only, with the training kernel's own bandwidth. Train/test splits are strictly chronological (walk-forward, Section 4.3).

---

## 4. Methods

### 4.1 Population-level dependence estimation

We estimate $I(X;Z)$ via a regularized kernel-CCA / NOCCO estimator (Hilbert–Schmidt norm of the normalized cross-covariance operator), cross-checked against the direct sum of squared canonical correlations (relative error $\sim 10^{-9}$ across all runs). We estimate $I(X;Y\mid Z)$ via a leave-one-out kernel-ridge-regression chain [Rifkin & Klautau, 2004]. Both are pre-existing, validated estimators; we use them only in Section 5.1 as a calibration check before moving to the main diagnostic question.

### 4.2 Direct dependence testing with a permutation null

$I(X;Y\mid Z)\approx 0$ is consistent with two different stories — $Z$ fully mediates whatever $X$ would tell us about $Y$, or neither $X$ nor $Z$ tells us anything about $Y$ to begin with — so we additionally test $I(Z;Y)$ directly, using the same kernel dependence statistic with a label-permutation null (200 permutations unless noted), both on the full sample and after removing each ticker's own mean return (to rule out a company-identity confound: $Z$ could encode "which company is this" via a stylistic fingerprint, which would be spuriously informative about $Y$ if different companies simply have different average returns over the sample period).

### 4.3 Predictive model and walk-forward validation

We fit kernel ridge regression (KRR) on a combined kernel $K = \tfrac13(K_Z + K_X + K_S)$ (equal-weighted sum of per-modality RBF kernels, bandwidths by median heuristic on training data) to predict $|Y|$ (a magnitude model) and $Y$ directly (a direction model). A magnitude-filtered strategy trades only the top 40% of test events by predicted $|Y|$, using the sign of the direction model's prediction, at 5bps one-way transaction cost. We validate with a single chronological train/test split and, separately, with walk-forward folds (2-year train / 1-year test / 6-month step) using Newey-West standard errors (5 lags, matching the holding period) for significance.

### 4.4 Elimination protocol

When the walk-forward result fails, we run four diagnostic checks, in order, each designed to rule out one specific candidate explanation before concluding the next:

1. **Capacity ablation.** Compare the full model ($Z+X+S$, 388 dimensions) against a low-capacity model ($X+S$ only, 4 dimensions, no transcript) on in-sample vs. out-of-sample correlation, per fold. If the low-capacity model fails identically, the failure is not explained by having "too many" embedding dimensions to fit noise with.
2. **Regularization sweep.** Sweep the KRR ridge parameter $\lambda$ across six orders of magnitude on the low-capacity model. If some $\lambda$ recovers positive, stable out-of-sample correlation while in-sample correlation falls (the textbook bias-variance curve), the failure was fixable under-regularization. If out-of-sample correlation never leaves the noise band around zero at any $\lambda$, it was not.
3. **Covariate-shift detection.** Using the same rolling sliced-Wasserstein regime detector the covariate-shift bound in Section 2.1 motivates (90-day windows, threshold = train-window mean + 1.5 train-window std), flag test events as regime-shift or stable, separately using $Z$'s and $X$'s own distributions. Split the out-of-sample correlation by this flag. If the failure concentrates in flagged-shift events, covariate shift is implicated; if it is uniform across stable and shift events — or if no shift events are flagged at all during a demonstrably-failing test window — covariate shift (at least of the kind this detector is built to catch) is ruled out.
4. **Structural-break test.** Fit a simple linear model $Y = b_0 + b_1 X_1 + b_2 X_2 + b_3 X_3 + b_4 S + \varepsilon$ separately in each of several non-overlapping calendar periods and run a Chow test [Chow, 1960] comparing the pooled-fit residual sum of squares against the sum of per-period residual sums of squares. A significant Chow statistic is direct evidence that the conditional relationship $P(Y\mid X,S)$ itself differs across periods — concept drift, confirmed rather than inferred by the elimination in steps 1–3.

---

## 5. Results

### 5.1 Population-level dependence is real, and finite-sample estimates at small $N$ should not be over-trusted

At 66 tickers (905 events, 11 detected regime-shift events), the ratio $\hat{I}(X;Y\mid Z)_{\text{regime}} / \hat{I}(X;Y\mid Z)_{\text{stable}} = 2815$, with permutation $p=0.110$. At 150 tickers (1354 events, 18 detected regime-shift events), the same ratio is $37$, with $p=0.750$. The effect *weakens by two orders of magnitude* as the sample grows — the signature of a small-$N$ estimator artifact rather than a strengthening real effect, and a direct illustration of why a single small-sample NOCCO-style estimate should not be reported as a discovered effect without a replication check at a second scale. We flag this explicitly because it is exactly the kind of result that looks publishable at $N=11$ regime events and is not.

### 5.2 Direct test: $I(Z;Y)$ is real and survives a company-identity confound

On the 150-ticker sample ($N=1405$), $\hat I(Z;Y)=7.576$ against a permutation null of $6.385\pm0.154$ ($z\approx7.7$, $p<0.005$ at this permutation count). After removing each ticker's own mean return ($N=1407$, 84 tickers, between-ticker variance $\approx 8\%$ of total), $\hat I(Z;Y_{\text{demeaned}})=7.650$, essentially unchanged — ruling out the company-identity confound directly rather than assuming it away.

### 5.3 The dependence is mostly about magnitude, with a much weaker direction component

Decomposing $Y_{\text{demeaned}}$ into $|Y|$ and $\mathrm{sign}(Y)$ on a fixed $N=400$ subsample (for permutation-test tractability): $\hat I(Z;|Y|)=12.19$ against null $10.98\pm0.26$ ($z=4.61$); $\hat I(Z;\mathrm{sign}(Y))=1.16$ against null $0.97\pm0.12$ ($z=1.53$). Both exceed every one of 100 permutation draws ($p<0.01$ at this resolution), but the two effect sizes differ by roughly a factor of three. This matters for Section 5.4: a magnitude-dominated signal and a direction-dominated signal have very different implications for a directional trading strategy, independent of any generalization question.

### 5.4 A naive strategy "succeeds" on a single split and fails walk-forward, with the sign reversed

On a single chronological split (train through 2022, test 2023), the magnitude-filtered subset shows net Sharpe $+0.08$ (mean P&L $+3.15$bps/event, hit rate 51.9%) against a baseline (trade every test event) net Sharpe of $-0.19$ ($-8.33$bps/event, hit rate 51.5%) — the magnitude filter appears to turn a losing strategy into a marginally profitable one, by concentrating a fixed transaction cost against larger expected moves rather than by improving the directional hit rate at all.

Under proper walk-forward validation (3 folds, 2yr/1yr/6mo), this reverses: pooled baseline net Sharpe $-0.05$ (NW $t=-0.18$, $p=0.86$); pooled magnitude-filtered net Sharpe $-0.89$ (NW $t=-1.75$, $p=0.08$), the wrong sign from the single-split result, and consistent across all three folds individually (fold P&Ls: baseline $+18.65/-17.10/-10.10$bps vs. filtered $+6.76/-57.71/-71.03$bps). A single train/test split is not a safe way to validate a zero-shot or out-of-sample claim; this is a concrete illustration of exactly how it can mislead, in both direction and magnitude.

### 5.5 Elimination step 1: capacity is not the explanation

Comparing the full model (388-dim) to the low-capacity model (4-dim, no transcript) across the 3 walk-forward folds:

| Fold | Full: in-sample / out-of-sample (direction) | Low-capacity: in-sample / out-of-sample (direction) |
|---|---|---|
| 0 | 0.584 / 0.023 | 0.475 / 0.012 |
| 1 | 0.586 / 0.006 | 0.490 / −0.030 |
| 2 | 0.575 / −0.047 | 0.480 / −0.036 |

The 4-dimensional model fails identically to the 388-dimensional one in every fold. If "too many embedding dimensions" were the explanation, the low-capacity model should have generalized; it did not.

### 5.6 Elimination step 2: no regularization setting recovers out-of-sample correlation

Sweeping $\lambda\in\{10^{-3},\dots,10^{3}\}$ on the low-capacity direction model, per fold (full table in the appendix; summary below is the range across the sweep):

| Fold | In-sample corr range | Out-of-sample corr range |
|---|---|---|
| 0 | 0.669 → 0.308 | +0.056 → −0.032 |
| 1 | 0.684 → 0.357 | −0.030 → −0.044 |
| 2 | 0.683 → 0.267 | +0.005 → −0.040 |

In-sample correlation falls smoothly and monotonically with $\lambda$, exactly as the standard bias-variance tradeoff predicts. Out-of-sample correlation never leaves a band of roughly $[-0.05, +0.06]$ at *any* setting, in *any* fold — there is no intermediate $\lambda$ at which an under-regularization story would predict a peak, and none is observed.

### 5.7 Elimination step 3: no detectable covariate shift, in either modality

Using the Z-embedding-based regime detector (same criterion as the covariate-shift bound motivating this entire check): fold 0 flags 19 of 350 test events as regime-shift; folds 1 and 2 — the two *worst*-performing folds in Section 5.4 — flag **zero**. Pooled out-of-sample correlation on stable-flagged events is $-0.038$ ($N=979$); on the (small, fold-0-only) shift-flagged subset it is $+0.109$ ($N=19$), the opposite of what a covariate-shift story predicts, on too few points to trust.

Repeating with a regime detector built on $X$ (price features) instead of $Z$: **zero** shift events are flagged in *all three* folds. Neither modality's marginal distribution shows anything a sliced-Wasserstein distributional-distance detector — the same family of statistic the covariate-shift bound is built around — would flag as unusual, during windows where the model demonstrably fails.

### 5.8 Elimination step 4 → confirmation: a direct structural-break test finds concept drift

A Chow test on $Y \sim 1 + X_1 + X_2 + X_3 + S$, split into two non-overlapping halves (2020–2021 vs. 2022–2023): $F(5,1286)=2.07$, $p=0.067$ — suggestive but not significant at this coarse a split. Splitting into four non-overlapping calendar years: $F(15,1276)=2.30$, $p=0.0032$ — significant. The coefficients make the mechanism visible directly, not just statistically:

| Feature | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|
| $X_1$ | $+0.0034$ | $-0.0080$ | $+0.0042$ | $-0.0039$ |
| $X_2$ | $-0.0028$ | $-0.0013$ | $-0.0012$ | $-0.0041$ |
| $X_3$ | $-0.0002$ | $-0.0031$ | $-0.0044$ | $-0.0013$ |
| $S$ (surprise) | $+0.0014$ | $+0.0003$ | $+0.0018$ | $+0.0003$ |

$X_1$'s coefficient changes sign in every single year. $S$'s coefficient is the only one that keeps the same sign across all four years — consistent, independently, with the established post-earnings-announcement-drift literature (Section 2.3), and a useful internal check that this is a real property of the data rather than an artifact of how we built the pipeline: the one feature with a theoretical reason to be stable is the one that is stable, and the features without such grounding (simple pre-announcement price statistics) are the ones that drift.

This is concept drift, confirmed directly by a test of $P(Y\mid X,S)$ itself, not inferred from the absence of other explanations. It is also, by construction (Section 2.2), invisible to a covariate-shift bound: $X$ and $Z$'s marginal distributions passed every check in Section 5.7 precisely while the conditional relationship was moving underneath them.

---

## 6. Discussion

### 6.1 What this says about covariate-shift bounds

The zero-shot bound in Section 2.1 is correct as a bound — nothing here contradicts it. What this case study shows is that the *covariate-shift term specifically* can be empirically uninformative about why a real generalization gap exists, even when that gap is large and the underlying population-level dependence motivating the predictor in the first place is genuinely real and independently verified (Section 5.2–5.3). A practitioner who checked only for covariate shift here — which is the natural, and in our case the theoretically-motivated, first check — would have found nothing, and would have had no remaining tool, within that framework, to explain the failure. The Chow test in Section 5.8 is not part of the zero-shot generalization framework at all; it is imported from classical econometrics specifically because the covariate-shift framework has no mechanism for detecting the failure mode that turned out to be the actual one.

We think this generalizes beyond our case study: any bound stated purely as a function of $\mathrm{TV}(P_X,Q_X)$ or a similar input-distribution divergence is, by construction, silent on concept drift, and concept drift is not a rare or exotic failure mode — it is the default way a *relationship*, as opposed to a *distribution*, changes over time. A complete empirical diagnostic for "will my zero-shot predictor still work," should include a concept-drift check as a first-class step, not an afterthought reached only after a covariate-shift check comes back negative and the practitioner is left without an explanation.

### 6.2 The four-step protocol as a reusable tool

We present the elimination protocol in Section 4.4 as a general-purpose diagnostic, not specific to finance or to transcripts. Any setting with (a) a trained model that fails to generalize out-of-sample and (b) the ability to construct a lower-capacity ablation, sweep regularization, and fit a regime detector on each input modality, can run the same four steps. The value of running them *in this specific order* is that each step closes off one alternative explanation cheaply before a more expensive or more specific test (the structural-break test, which requires committing to a reduced functional form) is needed to confirm the remaining one.

### 6.3 Limitations

Our universe (150 of ~500 S&P constituents, systematically but not randomly sampled, 2020–2023, US equities only) is modest by empirical-asset-pricing standards; we make no claim that these specific numerical results (Sharpe ratios, Chow $F$-statistics) generalize beyond this sample, only that the *pattern* — real dependence, walk-forward failure, elimination of capacity/regularization/covariate-shift, confirmed concept drift — is a reproducible methodological finding on the data we have. The text representation (generic sentence embeddings, not finance-tuned) is a further-reaching choice than strictly necessary; a finance-tuned embedding could in principle show different, possibly more stable, dependence structure, which we have not tested. The regime detector in Section 5.7 is one specific instantiation of a covariate-shift test (sliced-Wasserstein, 90-day rolling windows, a 1.5-sigma threshold); we cannot rule out that a different covariate-shift detector, with different windowing or a different divergence measure, would have flagged something ours did not — though the fact that *two independent modalities*, with the *same* detector family, both came back clean is reassuring on this point.

---

## 7. Conclusion

We used earnings-announcement prediction as a case study to test whether covariate-shift bounds, the standard theoretical tool for reasoning about zero-shot generalization failure, actually explain real generalization failures when one is observed. They did not, in our case: a verified population-level signal, exploited by a trained model, failed walk-forward validation for a reason invisible to every covariate-shift check we ran, and visible only to a direct test of the conditional relationship itself. We offer the elimination protocol used to reach this conclusion — capacity ablation, regularization sweep, multi-modality covariate-shift detection, then a direct structural-break test — as a general diagnostic, and the result itself as a concrete argument that concept drift deserves equal billing with covariate shift in how zero-shot generalization failure gets reasoned about and tested for.

---

## References

- Mehta, H. & Harchaoui, Z. *A Generalization Theory for Zero-Shot Prediction.* ICML 2025.
- Quionero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (eds.) *Dataset Shift in Machine Learning.* MIT Press, 2009.
- Bernard, V. L. & Thomas, J. K. *Post-Earnings-Announcement Drift: Delayed Price Response or Risk Premium?* Journal of Accounting Research, 1989.
- Rifkin, R. & Klautau, A. *In Defense of One-Vs-All Classification.* Journal of Machine Learning Research, 5:101–141, 2004.
- Chow, G. C. *Tests of Equality Between Sets of Coefficients in Two Linear Regressions.* Econometrica, 1960.

## Appendix A: full regularization sweep

*(insert full per-lambda table from `.runlogs/step2d_v2.log`, all three folds, before submission)*

## Appendix B: full leakage audit

*(reuse the 12-point audit table from the main repo README)*

## Appendix C: reproducibility

All code, cached datasets, and the exact script producing each result table above are in the project repository: `scripts/step2_magnitude_direction.py` through `scripts/step2g_chow_test.py`, `dependence/`, `regime/`, `data/earnings_surprise.py`. Random seed 42 throughout. Package versions: see `requirements.txt`.
