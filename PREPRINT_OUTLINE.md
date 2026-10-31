# Preprint outline

**Working title options:**
1. *Do Earnings Call Transcripts Predict Returns? Evidence for Concept Drift, Not Alpha*
2. *Residual Dependence and Concept Drift: An Empirical Stress Test of Zero-Shot Generalization Theory in Earnings Announcements*
3. *Information Without Edge: Earnings Transcripts Carry Real Signal That Cannot Be Traded*

(1) is probably the right one for a q-fin/empirical-finance audience — it states the finding, not the method, which is what gets a reader past the title. (2) is better if targeting an ML venue that cares about the zero-shot-generalization framing. Pick based on where this is headed.

---

## Abstract (draft)

We test a widely assumed but rarely validated heuristic: that the content of earnings call transcripts predicts subsequent stock returns. Using 8-K Item 2.02 filings, pre-announcement price features, analyst earnings-surprise data, and a sentence-embedding representation of transcript text across up to 150 large- and mid-cap US equities (2020–2023), we find that transcript content carries real, statistically robust information about the *magnitude* of the forward 5-day abnormal return (permutation z≈4.6–7.7, surviving a company-fixed-effects control) and a weaker but still significant signal about its *direction* (z≈1.5). However, a trading strategy built to exploit this — a magnitude filter combined with a kernel-ridge direction model, selected to concentrate trades where transaction costs matter least — fails walk-forward validation, reversing from an apparently profitable single-split result (net Sharpe −0.53 → +0.08) to a loss once tested on rolling out-of-sample folds (net Sharpe −0.89, Newey-West p=0.08, wrong sign). We trace this failure through a complete elimination chain: it is not explained by model capacity (a 4-dimensional model fails identically to a 388-dimensional one), not fixable by regularization (no setting across six orders of magnitude recovers out-of-sample correlation), and not explained by covariate shift (neither the transcript-embedding nor the price-feature marginal distribution shows a detectable regime shift during the failure periods, by the same sliced-Wasserstein criterion that motivates the zero-shot generalization bound we set out to test). A direct structural-break test confirms the remaining explanation: the feature-to-return relationship itself changes significantly year to year (Chow F(15,1276)=2.30, p=0.003), with the instability concentrated in price-momentum-style features specifically, while the earnings-surprise coefficient remains directionally stable across all four years — consistent with the established post-earnings-announcement-drift literature. The result is a specific, falsifiable account of *why* a plausible-sounding text-based trading signal fails: not lack of information, but concept drift in the mapping from announcement-time features to outcomes.

---

## 1. Introduction

- Open with the heuristic as stated by practitioners/fintwit/finance blogs: "earnings call tone/sentiment predicts the stock reaction." Cite that this is widely assumed, rarely tested end-to-end with leakage controls and genuine out-of-sample validation.
- State the paper's actual contribution up front: not "we found alpha" and not "there's no signal" — a specific diagnosis of *why* a real, detectable signal fails to become a tradeable strategy.
- Connect to Mehta & Harchaoui (ICML 2025)'s zero-shot generalization bound: ‖η⋆−η_ρ‖² ≲ I(X;Y|Z) + prompt bias. This paper empirically operationalizes that framework in a financial context and, in doing so, surfaces a gap in it: the bound's regime-shift term is a covariate-shift (TV-distance) story, and our evidence says the actual failure mode here is concept drift, which that term does not capture.
- Roadmap paragraph.

## 2. Background

- **2.1 Post-earnings-announcement drift (PEAD)**: the established baseline — earnings surprises predict same-direction drift. Cite Bernard & Thomas 1989 and the large replication literature. This is the benchmark our "surprise" feature should recover, and it does (Section 6.3).
- **2.2 Text-based signals in finance**: brief survey of transcript/10-K sentiment literature (Loughran-McDonald dictionary studies, embedding-based studies). Position this paper as methodologically stricter than typical studies in this space: full walk-forward validation, leakage audit, and — distinctively — a diagnostic chain when the strategy fails, rather than stopping at "no significant alpha."
- **2.3 Zero-shot generalization theory**: summarize Theorem 1 from Mehta & Harchaoui, the I(X;Z), I(X;Y|Z) quantities, and the covariate-shift (TV-distance) framing of its regime-shift term. This is the theoretical scaffold the empirical pipeline was built to test.
- **2.4 Covariate shift vs. concept drift**: standard ML distribution-shift taxonomy (cite Quionero-Candela et al. or similar). Flag explicitly that Theorem 1's bound is a covariate-shift bound, and that this distinction is what Section 6 formally resolves.

## 3. Data

- **3.1 Sources**: SEC EDGAR 8-K Item 2.02 (transcripts), yfinance (adjusted OHLCV, forward returns), yfinance earnings-dates (analyst EPS estimate vs. reported, i.e. surprise).
- **3.2 Universe and period**: state both scales honestly — initial 66-ticker / 2020–2023 run, scaled to 150 tickers via systematic sampling of the S&P 500 list, same period. Report the abandoned 476-ticker/2017–2023 run and *why* it was abandoned (operational, not substantive — repeated host sleep/wake cycles corrupting fetch reliability over a 10+ hour run), since a careful reviewer will ask why it isn't in the paper.
- **3.3 Data-quality findings worth reporting as their own small contribution**:
  - Ticker→CIK drift: EDGAR's current ticker mapping can point at a post-reorganization entity with no historical filings (XOM, BLK), silently zeroing a ticker out of a historical backtest unless corrected.
  - Non-earnings Item 2.02 filings: some flagged "earnings" 8-Ks are actually interim guidance updates (airline-style, ~9 days pre-announcement) or unrelated recurring filings (DTE's second monthly 8-K) — requiring a surprise-match filter to exclude silently.
  - Both are concrete, citable data-hygiene gotchas for anyone building an EDGAR-based earnings dataset; worth a short subsection or appendix.
- **3.4 Variable construction**: X (3-dim pre-announcement price features, [t−22,t−2]), Z (384-dim MiniLM sentence embedding of transcript text), Y (forward 5-day abnormal return vs. SPY, [t+1 open, t+5 close]), S (earnings surprise %, matched within a 3-day tolerance). State the lookahead-bias argument for S explicitly: public at the same moment as Z.
- **3.5 Leakage audit**: reuse the existing 12-point audit table from the codebase (bandwidth/threshold freezing, LOO evaluation, no look-ahead in feature windows, survivorship caveat, etc.).

## 4. Methods

- **4.1 Dependence estimators**: kernel CCA, NOCCO (I(X;Z)), LOO-KRR conditional MI (I(X;Y|Z)) — brief, these are existing validated machinery, cite the cross-check (rel. error ~1e-9).
- **4.2 Direct Z→Y dependence test**: the permutation-test methodology used to test I(Z;Y), I(Z;Y_demeaned), I(Z;|Y|), I(Z;sign(Y)) directly — explain why this is a different, more direct question than the conditional-MI quantities in 4.1, and why it was necessary (I(X;Y|Z)≈0 is consistent with two very different stories; this test tells them apart).
- **4.3 Trading strategy construction**: magnitude filter (KRR on |Y|) + direction model (KRR on Y), both using a 3-block sum-of-RBF-kernels on (Z, X, S), median-heuristic bandwidths frozen from training.
- **4.4 Validation**: single train/test split vs. walk-forward (2yr train / 1yr test / 6mo step), Newey-West (L=5) significance testing — same convention as the existing pipeline's `strategy/validation.py`.
- **4.5 Diagnostic protocol** (this is the paper's methodological core, make it a clearly numbered sequence):
  1. Compare a high-capacity (384+4 dim) and low-capacity (4 dim, no transcript) model's in-sample vs. out-of-sample correlation, per fold — tests whether capacity/overfitting explains the failure.
  2. Sweep KRR regularization across six orders of magnitude on the low-capacity model — tests whether under-regularization explains the failure.
  3. Split out-of-sample correlation by a regime-shift flag from a Z-embedding-based detector (rolling sliced-Wasserstein distance, same criterion behind Theorem 1) — tests covariate shift in Z.
  4. Repeat with a regime detector built on X instead of Z — tests covariate shift in X.
  5. Chow structural-break test on a simple linear Y~X+S model, both as a 2-period and 4-period (annual) split — direct test of concept drift.

## 5. Results

- **5.1 Population-level dependence is real and survives a confound check.** Table: I(Z;Y) observed vs. permutation null, full sample and ticker-demeaned (z≈7.7, z≈5.0).
- **5.2 The signal is mostly about magnitude, not direction.** Table: I(Z;|Y|) z=4.61 vs I(Z;sign(Y)) z=1.53, both nominally significant but at very different strength — state the "p=0.0000 means <1/101, not literally zero" caveat here, reviewers will ask.
- **5.3 The naive strategy "works" on a single split and fails on honest validation.** Table: single-split net Sharpe −0.53→+0.08; walk-forward pooled net Sharpe −0.89, wrong sign, p=0.08 (not significant, and the wrong direction from what the single split implied). This is the paper's central cautionary result — emphasize it structurally with its own subsection, not buried in a table.
- **5.4 Overfitting ruled out.** Table: in-sample/out-of-sample correlation, full vs. low-capacity model, 3 folds × 2 targets. Note the identical failure pattern regardless of dimensionality.
- **5.5 Under-regularization ruled out.** Table/figure: regularization sweep, in-sample correlation falling smoothly while out-of-sample stays flat at ~0 across all λ, all folds. A line plot (λ on log x-axis, correlation on y-axis, two lines per fold) would be the single most useful figure in the paper — it visually makes the "textbook bias-variance curve on one axis, flat noise on the other" point immediately.
- **5.6 Covariate shift ruled out.** Table: regime-shift event counts (Z-based and X-based) per fold, out-of-sample correlation split by stable/shift. Emphasize: the two worst-performing folds have *zero* flagged shift events by either criterion.
- **5.7 Concept drift confirmed directly.** Table: per-year OLS coefficients (the 4-column table already built), Chow F-statistics and p-values for both splits. This is the paper's payoff table — the sign flips in the price-feature coefficients should be visually obvious, and the surprise coefficient's directional stability is the nice contrasting detail to end on.

## 6. Discussion

- **6.1 What this says about the heuristic**: transcripts are not informationally empty — the "no link" null hypothesis is actually rejected — but the gap between "detectable" and "tradeable" is concept drift, not insufficient data or a bad model.
- **6.2 What this says about Theorem 1 / the zero-shot bound empirically**: the bound's regime-shift term is TV-distance / covariate-shift flavored; our evidence is that the actual failure mode in this domain is concept drift, which that term doesn't address. Frame this constructively — not "the theorem is wrong" (it's a bound, not a claim about which shift type dominates empirically in finance) but "when applying this framework to earnings-announcement prediction, practitioners should budget for concept drift as a separate, likely dominant failure mode the covariate-shift bound won't flag."
- **6.3 The surprise-feature finding**: tie explicitly to PEAD literature — the one feature with theoretical grounding is the one that's stable; the price-momentum features, which lack comparable theoretical grounding, are the unstable ones. Suggests a practical prescription: weight theoretically-grounded features more, momentum-style features less, in any earnings-event model spanning multiple years/regimes.
- **6.4 Limitations**: universe size (150 of ~500 S&P names, systematically but not randomly sampled), survivorship (2024 constituent snapshot), single embedding model (generic MiniLM, not finance-tuned), 5-day horizon only, the abandoned full-universe run, single-country/single-exchange scope.
- **6.5 Future work**: a regime-aware or piecewise model that explicitly re-estimates per period rather than assuming stationarity; a finance-tuned embedding; a direct test of whether concept drift is detectable *ex ante* (e.g., via a changepoint test on rolling coefficients) rather than only visible in hindsight.

## 7. Conclusion

- Restate the three-part finding crisply: signal exists → naive exploitation fails → the failure is concept drift, not overfitting or covariate shift. State the actionable takeaway for practitioners and researchers evaluating similar text-based signals: validate walk-forward, and when it fails, diagnose *why* before concluding "no alpha" — the information may well be there.

## Appendix

- Full leakage-audit table (existing).
- Data-quality notes (CIK drift, non-earnings Item 2.02 filings) in full detail with examples.
- Reproducibility: repo link, exact script list mapping each results table to the script that produced it (`scripts/step2*.py`), package versions, random seeds.

---

## Figures/tables to produce before drafting prose

1. **Table**: dataset summary (tickers, events, date range, train/test split) at both the 66- and 150-ticker scales.
2. **Table**: I(Z;Y) / I(Z;|Y|) / I(Z;sign(Y)) permutation results (already have the numbers).
3. **Table**: single-split vs. walk-forward strategy performance (already have the numbers).
4. **Figure**: regularization sweep, in-sample vs out-of-sample correlation vs. λ, one small-multiple panel per fold. (Not yet plotted — data exists in `.runlogs/step2d_v2.log`.)
5. **Table**: regime-shift event counts and stable/shift out-of-sample correlation, Z-based and X-based detectors.
6. **Table**: per-year OLS coefficients + Chow test statistics (already have the numbers, from `step2g_chow_test.py` output).
7. **Figure (optional, nice-to-have)**: the per-year coefficient table as a small line plot (year on x-axis, one line per coefficient) — makes the sign-flip visually immediate without reading a table.

## Open decision before full drafting

Venue/format affects tone and length a lot:
- **arXiv q-fin.ST preprint**: can be long-form, methods-heavy, informal about not being peer-reviewed yet.
- **SSRN working paper**: similar, finance-audience framing, more PEAD-literature engagement expected.
- **ML workshop paper (e.g. NeurIPS/ICML workshop on time series, finance, or distribution shift)**: shorter, more emphasis on Section 6.2's framework critique, less finance-literature throat-clearing.

Pick one before I draft full prose — it changes how much of Sections 2.1/2.2 to keep.
