# Residual Dependence — I(X;Y|Z) Estimation in Finance

Research pipeline operationalizing the zero-shot generalization framework from
**Mehta & Harchaoui, "A Generalization Theory for Zero-Shot Prediction," ICML 2025**.

The core quantity is **residual dependence** I(X;Y|Z) from Theorem 1:

> ||η⋆ − η_ρ||²_{L2(P_X)} ≲ I(X; Y | Z) + prompt bias

When I(X;Y|Z) is small, earnings transcripts Z are sufficient to predict returns Y —
price features X add nothing. When I(X;Y|Z) is large (especially during distribution
shifts), the indirect predictor breaks down and price features carry incremental
predictive information.

| Symbol | Meaning | Source |
|--------|---------|--------|
| Z | Earnings call transcript (8-K Item 2.02) | SEC EDGAR |
| X | Pre-announcement price features | yfinance adjusted OHLCV |
| Y | Forward 5-day abnormal return vs SPY | yfinance |

---

## Theoretical background

### Theorem 1 (Mehta & Harchaoui 2025, eq. 15)

The L2 error of the zero-shot predictor η_ρ is bounded by residual dependence:

```
||η⋆ − η_ρ||²_{L2}  ≲  I(X; Y | Z)  +  prompt bias
```

### Proposition 1 — singular value structure of M_{Z|X}

The operator M_{Z|X}: L2(P_X) → L2(P_Z) has SVD σ_1 = 1 (trivial) and non-trivial
σ_i for i ≥ 2. We estimate these from kernel CCA of (X, Z) pairs (eq. 26–27).

### NOCCO estimator (eq. 94)

The Hilbert-Schmidt squared norm of the normalized cross-covariance operator:

```
V̂_{X,Z} = (Ĉ_{XX} + λI)^{-1/2} Ĉ_{XZ} (Ĉ_{ZZ} + λI)^{-1/2}
Î(X; Z) = ||V̂_{X,Z}||²_HS
```

### Information density (eq. 8)

Per-event signal score measuring joint (X, Z) deviation from independence:

```
c(x, z) = Σ_{j≥2} σ̂_j f̂_j(x) ĝ_j(z)
```

### Lemma 14 — regime shifts proxy TV(P_X, Q_X)

The generalization gap is bounded in part by the total variation distance between
training and test distributions. We proxy this with the sliced Wasserstein distance
SW_2² between consecutive 90-day windows of Z embeddings.

---

## Project layout

```
residual_dependence/
├── data/
│   ├── edgar.py            # SEC EDGAR 8-K fetcher (Item 2.02)
│   ├── prices.py           # Price features + abnormal return computation
│   └── universe.py         # S&P 500 universe (2024 snapshot)
├── dependence/
│   ├── kernel_cca.py       # Regularized kernel CCA (σ_i, f_i, g_i)
│   ├── contingency.py      # NOCCO estimator → I(X;Z)
│   └── mutual_info.py      # LOO-KRR chain → I(X;Y|Z)
├── regime/
│   ├── regime_detection.py # Rolling sliced Wasserstein regime detector
│   └── stability.py        # I(X;Y|Z) by regime vs stable periods
├── strategy/
│   ├── signals.py          # Per-event information density signal c(x,z)
│   ├── backtest.py         # Long/short quintile backtest
│   └── validation.py       # Walk-forward + Newey-West + BH correction
├── analysis/
│   ├── plots.py            # Publication-quality figures (4 panels)
│   └── report.py           # Structured markdown report generator
├── main.py                 # CLI entry point — full pipeline
├── requirements.txt
└── README.md
```

---

## Module-to-paper mapping

| Module | Paper quantity | Equation |
|--------|---------------|----------|
| `dependence/kernel_cca.py` | σ_i of M_{Z\|X} | Proposition 1 (eq. 26–27) |
| `dependence/contingency.py` | I(X;Z) = \|\|V_{X,Z}\|\|²_HS | NOCCO (eq. 94) |
| `dependence/mutual_info.py` | I(X;Y\|Z) via KRR chain | Theorem 2 |
| `regime/regime_detection.py` | TV(P_X, Q_X) proxy | Lemma 14 |
| `regime/stability.py` | I(X;Y\|Z) ↑ during drift | Theorem 1 consequence |
| `strategy/signals.py` | c(x,z) = R(x,z) − 1 | eq. 8 |
| `strategy/backtest.py` | Empirical strategy P&L | — |
| `strategy/validation.py` | Out-of-sample significance | — |
| `analysis/plots.py` | Figures 1–4 | — |

---

## Setup

```bash
conda create -n residual_dep python=3.11
conda activate residual_dep
pip install -r requirements.txt
```

Requirements:
```
requests==2.32.3       # SEC EDGAR API calls
yfinance==1.4.1        # Price data
pandas==2.2.3          # DataFrame operations
numpy==1.26.4          # Numerical kernels
sentence-transformers==3.3.1  # all-MiniLM-L6-v2 transcript embeddings
matplotlib==3.10.1     # Publication figures
```

All sentence-transformer model weights are downloaded automatically on first run and
cached in `~/.cache/huggingface/`. Subsequent runs are fully offline.

---

## Quick start

### Fetch real data and run the full pipeline

```bash
python main.py \
    --tickers AAPL MSFT JPM GS BAC \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --train-end 2022-12-31 \
    --output-dir output \
    --n-components 10 \
    --log-level INFO
```

Output files written to `output/`:
- `figures.png` — 2×2 publication figure grid
- `report.md` — structured markdown report with all key statistics

### Skip data fetching (use cached filings)

```bash
python main.py \
    --tickers AAPL MSFT JPM \
    --start 2023-01-01 --end 2023-12-31 \
    --train-end 2023-06-30 \
    --skip-fetch \
    --output-dir output
```

### Skip walk-forward validation (faster dev loop)

```bash
python main.py \
    --tickers AAPL MSFT JPM \
    --start 2023-01-01 --end 2023-12-31 \
    --train-end 2023-06-30 \
    --skip-fetch --skip-validation \
    --n-components 5 --reg-cca 1e-2 \
    --output-dir output
```

---

## CLI reference

```
usage: main.py [-h] [--tickers TICKERS [TICKERS ...]]
               [--start START] [--end END] [--train-end TRAIN_END]
               [--output-dir OUTPUT_DIR]
               [--raw-dir RAW_DIR] [--embed-cache-dir EMBED_CACHE_DIR]
               [--n-components N_COMPONENTS]
               [--reg-cca REG_CCA] [--reg-krr REG_KRR]
               [--tc-bps TC_BPS] [--seed SEED]
               [--skip-fetch] [--skip-validation]
               [--log-level {DEBUG,INFO,WARNING,ERROR}]

arguments:
  --tickers            Ticker symbols (default: AAPL MSFT JPM GS BAC C)
  --start              History start date YYYY-MM-DD (default: 2018-01-01)
  --end                History end date YYYY-MM-DD (default: 2023-12-31)
  --train-end          Walk-forward train cutoff YYYY-MM-DD (default: 2021-12-31)
  --output-dir         Directory for figures.png and report.md (default: output)
  --raw-dir            Directory for raw EDGAR JSON (default: raw_filings)
  --embed-cache-dir    Directory for embedding cache (default: embed_cache)
  --n-components       Canonical components k (default: 10)
  --reg-cca            CCA regularization λ (default: 1e-3)
  --reg-krr            KRR regularization λ (default: 1e-2)
  --tc-bps             Transaction cost per side in bps (default: 5.0)
  --seed               Global random seed (default: 42)
  --skip-fetch         Use cached EDGAR filings; skip HTTP requests
  --skip-validation    Skip walk-forward validation (faster)
  --log-level          Logging verbosity (default: INFO)
```

---

## Pipeline stages

The `main.py` pipeline runs these stages in sequence:

### Stage 1 — Fetch and embed
- Calls `data/edgar.py` to fetch 8-K Item 2.02 filings from SEC EDGAR (or loads cache)
- Embeds transcript text using `all-MiniLM-L6-v2` (384-dim) via `dependence/contingency.py`
- Caches embeddings keyed by `(ticker, filing_date)` to avoid re-computation

### Stage 2 — Price features
- Calls `data/prices.py` to extract 11 pre-announcement features over [t−22, t−2]:
  volatility, momentum, volume ratio, price-to-MA ratios, and return skew/kurtosis
- Features are cross-sectional z-scored per filing date to remove market-wide effects

### Stage 3 — Abnormal returns
- Forward 5-day return minus SPY benchmark over the identical window
- Entry at t+1 open, exit at t+5 close (t = filing date)

### Stage 4 — Align
- Merge filings, price features, and abnormal returns on `(ticker, filing_date)`
- Drop events with missing prices or embeddings

### Stage 5 — Regime detection
- `RegimeDetector`: 90-day embedding windows, 30-day step, sliced Wasserstein (200 projections)
- Threshold = mean + 1.5σ of SW_2² values, calibrated on training windows only
- Events flagged as regime-shift if the nearest window midpoint (within ±60 days) exceeds threshold

### Stage 6 — Dependence estimation
- Kernel CCA of (X, Z) → σ̂_i, f̂_i(x), ĝ_i(z) (Proposition 1)
- NOCCO trace → Î(X;Z) (eq. 94)
- LOO-KRR chain → Î(X;Y|Z) full, regime, stable (Theorem 2)

### Stage 7 — Signal and backtest
- `SignalEstimator.fit()` on training events; `score()` on test events
- Long top quintile / short bottom quintile; 5-day hold; dollar-neutral
- Sharpe, max drawdown, turnover reported overall and by regime

### Stage 8 — Walk-forward validation
- 2-year train / 1-year test / 6-month step
- Pooled Newey-West t-test (L=5 lags) across all test folds
- Benjamini-Hochberg correction across k canonical signal directions
- Degradation test: two-sample NW t-test of (stable P&L) − (regime P&L)

### Stage 9 — Figures
Four panels saved to `output/figures.png`:
1. **Singular value decay** — log-log σ̂_i vs i with power-law fit γ̂_XZ (Proposition 1)
2. **I(X;Y|Z) timeline** — rolling estimate with regime shading (Theorem 1)
3. **Cumulative returns** — strategy P&L split by regime vs stable (Lemma 14)
4. **Rolling IC** — 90-day rank-correlation of signal with forward return

### Stage 10 — Report
Markdown report with all key statistics saved to `output/report.md`.

---

## Reproducing the paper's central claim

The central empirical claim maps directly to Theorem 1:

> During distribution shifts (regime periods), I(X;Y|Z) increases and strategy
> P&L degrades — consistent with the theoretical bound.

Check these three quantities in `output/report.md`:
1. **I_regime / I_stable ratio** (Section 3): should be > 1
2. **Degradation t-stat** (Section 5): positive means stable > regime P&L
3. **Regime vs stable Sharpe** (Section 4): stable Sharpe should exceed regime Sharpe

All three are necessary to claim empirical support for Theorem 1.

---

## Data-leakage audit

The pipeline enforces a strict no-leakage policy:

| Check | Enforcement |
|-------|-------------|
| Bandwidth h²_X, h²_Z | Median heuristic on train data; frozen for test scoring |
| Regime threshold | Calibrated on train windows; applied unchanged to test |
| KRR regularization λ | Fixed constant; never cross-validated on Y |
| KRR in-sample evaluation | LOO formula (Rifkin & Klautau 2003); no test data seen |
| Signal scoring | `SignalEstimator.score()` uses frozen bandwidths and coefficients |
| Price feature normalization | Cross-sectional per date; no look-forward across dates |
| Price feature window | [t−22, t−2] trading days — no overlap with [t+1, t+5] return window |
| Walk-forward splits | Each fold fits exclusively on events before `train_end` |

Known limitations (not leakage, but caveats):
- **Survivorship bias**: Universe is 2024 S&P 500 snapshot
- **After-hours timing**: Filing date not shifted to t+1 for after-hours 8-Ks
- **Adjusted prices**: yfinance retroactive split adjustment is fine for research

---

## Module 1 — Data pipeline

### data/edgar.py

Fetches 8-K Item 2.02 filings from the SEC EDGAR submissions API.

- Rate-limited to 10 requests/second via a token-bucket (EDGAR fair-access policy)
- All parse failures are flagged in the JSON output (`parse_error` field); no records silently dropped
- Output: `raw_filings/<TICKER>/<filing_date>_<accession>.json`
- JSON fields: `ticker`, `filing_date`, `accession_number`, `item_202_present` (bool), `raw_text`, `parse_error`

```bash
python -m data.edgar --tickers AAPL MSFT JPM --start 2023-01-01 --end 2023-12-31 --out raw_filings
```

### data/prices.py

Fetches adjusted OHLCV via yfinance and computes forward 5-day abnormal returns.

- Entry: open price on t+1; exit: close on t+5 trading days after t+1
- Benchmark: SPY over the identical window
- Price features: 11-dimensional, covering [t−22, t−2] pre-announcement window

```bash
python -m data.prices --ticker AAPL --date 2023-11-02
```

### data/universe.py

Static S&P 500 constituent list (2024 snapshot). Pass `--tickers all` to `main.py` to use it.

---

## Module 2 — Dependence estimation

### dependence/kernel_cca.py

Regularized kernel CCA estimating the SVD of M_{Z|X}.

- RBF kernels with bandwidth = median heuristic on training data
- Regularization λ prevents ill-conditioning at small N
- Outputs: singular values σ̂_i, left/right canonical functions f̂_i(x), ĝ_i(z)

### dependence/contingency.py

NOCCO estimator for I(X;Z) (eq. 94). Two cross-checks:
- `ixz_nocco`: trace of V̂_{X,Z}ᵀ V̂_{X,Z}
- `ixz_svd`: sum of squared singular values (should agree to numerical precision)

Also calls `embed_transcripts_batch()` to produce Z embeddings via all-MiniLM-L6-v2.

### dependence/mutual_info.py

LOO-KRR chain estimator for I(X;Y|Z):

```
η̂_Z(z)  = KRR(Z → Y)        (ŷ from transcript embedding only)
ε̂_i     = y_i − η̂_Z(z_i)    (residual not explained by Z)
Î(X;Y|Z) = I(X; ε̂ | Z)      (how much X explains the residual)
```

LOO predictions use the Rifkin & Klautau (2003) closed-form formula to avoid refitting.

---

## Module 3 — Regime detection and stability

### regime/regime_detection.py

Rolling sliced Wasserstein distance between consecutive 90-day windows of Z embeddings.

- Window: 90-day width, 30-day step
- Sliced Wasserstein SW_2² with 200 random projections and 100 quantile points
- Threshold = mean + 1.5σ of SW_2² values (calibrated on training windows only)
- Events are flagged as regime-shift if the nearest window midpoint is within ±60 days

```python
from regime.regime_detection import RegimeDetector
detector = RegimeDetector(window_days=90, step_days=30, threshold_std=1.5)
result = detector.fit_transform(dates, Z_embeddings, train_cutoff_date)
print(result.regime_flags)   # bool array, length = n_events
```

### regime/stability.py

Tracks I(X;Y|Z) separately in regime-shift vs stable periods.

Core empirical claim: **I_regime > I_stable** — residual dependence grows when the
embedding distribution drifts, consistent with Theorem 1.

Permutation test shuffles regime_flags (not Y) to test whether the elevation
is explained by chance.

---

## Module 4 — Strategy

### strategy/signals.py

Per-event signal from information density (eq. 8):

```python
c(x, z) = Σ_j σ̂_j f̂_j(x) ĝ_j(z)
```

`SignalEstimator.fit(X_train, Z_train)` computes bandwidths and CCA coefficients.
`SignalEstimator.score(X_test, Z_test)` scores new events using frozen parameters —
no refitting on test data.

### strategy/backtest.py

Long top quintile / short bottom quintile by signal score:
- 5-day hold, dollar-neutral (+1/n_long per long, −1/n_short per short)
- 5 bps transaction cost per side (round-trip = 2 × 5bps × 2 legs)
- Reports Sharpe, max drawdown, turnover; split by regime vs stable

### strategy/validation.py

Walk-forward out-of-sample validation:
- 2-year train / 1-year test / 6-month step
- Newey-West HAC standard errors (L=5 lags, matching holding period)
- Benjamini-Hochberg FDR correction across k canonical directions
- Degradation test: two-sample NW t-test of (stable P&L) − (regime P&L)

---

## Module 5 — Analysis

### analysis/plots.py

Four publication-quality figures (2×2 grid, `output/figures.png`):

1. **Singular value decay**: log-log scatter of σ̂_i with power-law fit,
   annotated with γ̂_XZ (measures concentration of dependence)
2. **I(X;Y|Z) timeline**: rolling estimate with red-shaded regime periods
3. **Cumulative returns**: step plots for regime vs stable P&L separately
4. **Rolling IC**: 90-day Spearman rank correlation of c(x,z) with forward return

### analysis/report.py

`generate_report(...)` builds a self-contained markdown string with 7 sections:
configuration, I(X;Z) structure, I(X;Y|Z) by regime, strategy performance,
walk-forward validation, leakage audit, and theoretical interpretation.

---

## Known limitations

1. **Survivorship bias** — universe is a 2024 S&P 500 snapshot; companies removed earlier are absent
2. **After-hours filing timing** — filing date not shifted to t+1 for after-hours 8-Ks
3. **EDGAR pagination** — most recent ~1000 filings per CIK searched (see `edgar.py`)
4. **SPY as benchmark** — single-factor adjustment; no sector/style/size controls
5. **Small-N bias in NOCCO** — finite-sample positive bias; use for relative comparisons, not absolute quantities
6. **Proxy for TV distance** — sliced Wasserstein is not TV; the Lemma 14 connection is qualitative

---

## Citation

```bibtex
@inproceedings{mehta2025generalization,
  title     = {A Generalization Theory for Zero-Shot Prediction},
  author    = {Mehta, Harsh and Harchaoui, Zaid},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```
