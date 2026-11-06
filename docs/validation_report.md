# Independent Validation Report: UK credit-loss satellite models

| | |
|---|---|
| **Models validated** | Champions for credit cards, other consumer credit, mortgages and corporate lending ([model development document](model_development_document.md)) |
| **Evidence** | [Notebook 04: diagnostics](../notebooks/04_diagnostics.ipynb), [notebook 05: alternatives](../notebooks/05_regularisation_panel.ipynb), [notebook 06: validation](../notebooks/06_validation.ipynb), [notebook 07: benchmark](../notebooks/07_scenario_projection.ipynb) |
| **Standard** | PRA SS1/23, principle 4 (independent model validation): conceptual soundness, outcome analysis, ongoing monitoring |

## Overall opinion

**The models are fit for stress-testing use with the conditions below.** They are economically signed, statistically supported, robust to estimation choices for three of four portfolios, and benchmark sensibly against the Bank of England's own results. Their weaknesses are known, quantified and in the conservative direction for cards. None is rated red.

| Portfolio | Rating | Conditions |
|---|---|---|
| Corporate | 🟢 Green | Note possible under-statement of crisis peaks |
| Mortgages | 🟡 Amber | Report champion–challenger range; expert judgement for rate paths above 8% |
| Credit cards | 🟡 Amber, conservative | Document the credit-quality overlay; monitor the drifting sensitivity |
| Other consumer | 🟡 Amber, weak | Report alongside the pooled consumer panel challenger; redevelop when longer credit-growth history is available |

## 1. Conceptual soundness

| Test | Result |
|---|---|
| Economic signs | Enforced at selection; confirmed by block bootstrap (share of 1,000 resamples with the correct sign ≥ 98% for every macro term except the other-consumer unemployment term, 95%) |
| Spurious regression | All static candidates fail cointegration; champion residuals are stationary (KPSS; ADF on non-overlapping quarters p < 0.005). Standard ADF fails for two portfolios because of the MA(3) overlap, not a unit root |
| Residual autocorrelation | Present by construction (overlapping target); handled by Newey–West errors. Conclusions hold from 2 to 12 Newey–West lags, except other consumer credit (t = 1.95 at 12 lags) |
| Heteroskedasticity | Rejected for other consumer and mortgages; WLS and Huber regression move card and corporate coefficients by < 10%, other consumer by up to 40% |
| Functional form | RESET rejects mortgages only. Hinge, quadratic and interaction alternatives are not significant |
| Influential quarters | Dropping all quarters with Cook's D > 4/n changes macro coefficients by up to 24%; all stay significant |
| Parameter stability | CUSUM stable for cards; rejected at 5% for the others. Chow tests reject at 2008 (overstated under autocorrelation) |
| Alternative methods | The sign-constrained Lasso, choosing freely among 49 macro features, selects the champion's card drivers. Unconstrained penalised models use mostly wrong-signed features and understate stress |

## 2. Outcome analysis

### 2.1 Out-of-time backtests (re-estimated to the cutoff, projected on the actual economy)

| Portfolio | Window | Actual peak | Model peak | RMSE | Share of quarters in 90% band |
|---|---|---|---|---|---|
| Cards | GFC (fit to 2007 Q4) | 10.8% | 18.1% | 6.9 pp | 30% |
| Cards | COVID / rates (fit to 2019 Q4) | 3.1% | 4.4% | 0.9 pp | 85% |
| Other consumer | GFC | 5.5% | 6.2% | 1.0 pp | 80% |
| Other consumer | COVID / rates | 2.1% | 2.8% | 0.9 pp | 80% |
| Mortgages | GFC | 0.12% | 0.13% | 0.03 pp | 65% |
| Corporate | GFC | 1.79% | 1.34% | 0.30 pp | 100% |
| Corporate | COVID / rates | 0.28% | 0.94% | 0.21 pp | 45% |

**Selection-honest test:** re-running the whole selection procedure on pre-2008 data picks Δ unemployment (t−4) and Δ Bank Rate (t−8); its GFC peak error is +4.9 pp, smaller than with the hindsight drivers (+7.3 pp). A bank following this process in 2007 would have **over**-projected card losses in the crisis, the safer error for capital.

### 2.2 Rolling-origin forecasts (42 origins 2003–2024, eight quarters ahead)

| | Cards | Other consumer | Mortgages | Corporate |
|---|---|---|---|---|
| RMSE, model | 2.29 pp | 0.96 pp | 0.020 pp | 0.25 pp |
| RMSE, no-change | 1.96 pp | 1.09 pp | 0.029 pp | 0.41 pp |
| Skill | −0.17 | +0.12 | +0.30 | +0.40 |

The card model is not a good unconditional forecaster: it over-predicts on average by 0.85 pp. That is acceptable for a stress model, where the use is conditional projection under severe paths, but it rules out use for budgeting.

### 2.3 Benchmark: Bank of England 2025 results

| | This project (five-year write-off rate) | Bank of England (five-year impairment rate) | Ratio |
|---|---|---|---|
| UK consumer credit | 14.2% | 23.3% | 0.61 |
| UK mortgages | 0.53% | 1.0% | 0.53 |
| Total | 2.1% | 3.8% | 0.55 |

Source: Bank of England, *Financial Stability Report*, December 2025, Table 5.C. A ratio below one is expected (impairments front-load lifetime losses under IFRS 9; different population). The **consistency of the ratio across portfolios** is the evidence of sound relative calibration.

## 3. Ongoing monitoring (since the 2024 Q4 jump-off)

| Portfolio | Actual 2026 Q2 | Model 2026 Q2 | 90% band | Status |
|---|---|---|---|---|
| Cards | 2.39% | 3.78% | 2.56–5.08% | ❌ below band |
| Other consumer | 0.81% | 1.91% | 1.11–3.26% | ❌ below band |
| Mortgages | 0.01% | 0.02% | 0.01–0.05% | ✅ |
| Corporate | 0.15% | 0.24% | 0.16–0.38% | ❌ just below band |

The consumer models over-predict as unemployment rose without a rise in losses. The Bank of England made the same judgement in its 2025 test (Box E: removing past judgements lowered the five-year unsecured impairment rate by an estimated 4.6 percentage points). **Finding:** document the credit-quality overlay (cards 0.72, other consumer 0.62) and apply the monitoring trigger.

## 4. Sensitivity and model risk

Effect on the five-year cumulative loss rate of shifting one driver for the whole scenario:

| Shock | Cards | Other consumer | Mortgages | Corporate |
|---|---|---|---|---|
| Unemployment +1 pp | +2.73 pp | +0.59 pp | +0.28 pp | +1.91 pp |
| Bank Rate +1 pp | +0.87 pp | 0 | +0.06 pp | 0 |
| House prices −5 pp a year | 0 | 0 | +0.20 pp | 0 |
| GDP growth −1 pp | 0 | 0 | 0 | +0.14 pp |

Across modelling choices (jump-off adjustment, challenger, pooled panel, raw-rate transform, credit-quality overlay), total five-year losses range from **£35.2bn to £48.6bn** around the champion's £44.3bn. For consumer credit, this model-choice range is almost as wide as the statistical 90% interval.

Quantile regression: the card unemployment effect at the 90th percentile (0.20) is about a third larger than at the mean (0.15), so tail quarters are more sensitive.

## 5. Findings to remediate

| # | Finding | Severity | Owner action |
|---|---|---|---|
| F1 | Other consumer model fragile (single driver; bootstrap interval includes zero) | Medium | Report with pooled-panel challenger; redevelop with credit growth |
| F2 | Consumer models outside the 90% band since 2025 | Medium | Apply documented overlay; re-estimate if the breach continues for three quarters |
| F3 | Interest-rate effects extrapolated beyond observed data | Medium | Expert-judgement review of rate scenarios; affordability analysis |
| F4 | Mortgage functional form rejected (RESET) | Low (small £) | Explore loan-to-value or affordability drivers when available |
| F5 | Card sensitivity drifting (0.33 → 0.15 over time) | Low | Monitor annually via recursive estimates |
