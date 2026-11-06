# Model Development Document: UK credit-loss satellite models

| | |
|---|---|
| **Models** | Four satellite models: credit cards, other consumer credit, mortgages, corporate (PNFC) lending |
| **Purpose** | Project portfolio loss rates under macroeconomic stress scenarios (stress testing; IFRS 9 forward-looking scenarios) |
| **Owner / developer** | Tilak Heble |
| **Version** | 1.0, November 2026 |
| **Structure** | Follows the PRA's model risk management principles for banks (SS1/23): purpose and use, data, methodology, performance, limitations, use constraints, monitoring |
| **Independent review** | [Validation report](validation_report.md) |

## 1. Purpose and intended use

**Intended use:** projecting annual write-off rates (net of recoveries) for UK lending portfolios over a five-year horizon, conditional on paths for UK unemployment, GDP, house prices, Bank Rate and inflation, as specified in Bank of England stress scenarios.

**Not intended for:** point forecasts of next quarter's losses (a no-change forecast is competitive in benign periods); individual-bank or loan-level losses; impairment (IFRS 9 provision) amounts, which differ from write-offs in timing; scenarios far outside the 1993–2026 range of the drivers without expert overlay.

## 2. Data

| Data | Source | Coverage | Notes |
|---|---|---|---|
| Write-offs by portfolio (RPQTFHE, RPQTFHF, RPQTFHD, RPQTFHB) | Bank of England Bankstats C2.1 / database | 1993 Q1 – 2026 Q2 | Sterling, net of recoveries, not seasonally adjusted |
| Amounts outstanding for comparison (RPQB74D, RPQB75D, RPQB73D, RPQB69D) | Bank of England | 1993 Q1 – 2026 Q3 | All UK banks and building societies |
| Unemployment (MGSX), real GDP (ABMI), CPI inflation (D7G7) | ONS | 1971+ | Quarterly |
| Bank Rate (IUQABEDR) | Bank of England | 1990+ | Quarterly average |
| UK House Price Index | HM Land Registry | 1990+ | Monthly, averaged to quarters |
| 2025 Bank Capital Stress Test variable paths | Bank of England | 2000–2029 | History and scenario |

**Target:** the trailing four-quarter loss rate = write-offs over the last four quarters ÷ average lagged balance, in %. It absorbs lumpy block write-offs and matches how banks report annual loss rates. Modelled on the logit scale (bounded in 0–100%).

**Data controls** ([notebook 01](../notebooks/01_eda.ipynb), [`contracts.py`](../src/ukstress/ingest/contracts.py)):
- A Pandera data contract validates every build (ranges, uniqueness, ordering); CI rebuilds the panel from the committed raw files.
- Two write-off series downloaded twice reconcile exactly with the published Bankstats table.
- My macro history matches the Bank of England's scenario history (correlation ≥ 0.995), so scenario paths join without rebasing.
- 12 balance breaks (reporting-population changes) were found; at most of them the loss rate is unaffected, and step dummies at the card breaks are not significant (p = 0.08 and 0.96). No adjustment is made.
- 26 write-off spikes (robust z > 4) are kept: they are real losses, absorbed by the trailing target.

## 3. Methodology

### 3.1 Model form

$$\text{logit}(LR_t) = \alpha + \sum_j \beta_j X_{j,t-\ell_j} + \gamma\,\text{COVID}_t + \rho\,\text{logit}(LR_{t-4}) + \varepsilon_t$$

- $X_{j,t-\ell_j}$: macro driver *j* at lag $\ell_j \in \{0,1,2,3,4,6,8\}$ quarters
- COVID: 2020 Q2 – 2021 Q4 policy regime (furlough, payment holidays), when losses fell despite a 22% fall in GDP
- $\rho$: persistence via the loss rate four quarters earlier (no overlap with the trailing window)
- Estimation: OLS with Newey–West (HAC, 4 lags) standard errors, because the overlapping target makes residuals MA(3)
- Sample: 1995 Q2 – 2026 Q2 (125 quarters)

### 3.2 Candidate drivers and sign rules

| Driver | Required sign |
|---|---|
| Unemployment rate; change in unemployment | + |
| Real GDP growth | − |
| House price growth | − |
| Bank Rate; change in Bank Rate | + |
| CPI inflation | + |

Only drivers specified in the Bank of England scenario are eligible. Level and change of the same variable may not appear together.

### 3.3 Selection procedure ([notebook 03](../notebooks/03_model_development.ipynb))

All combinations of up to three drivers × seven lags × {static, dynamic}: **19,110 candidates per portfolio**. A candidate passes if:
1. every macro coefficient has the required sign (and 0 < ρ < 1);
2. every macro coefficient is significant at 5% (Newey–West);
3. all VIF < 5;
4. static models are cointegrated (Engle–Granger p < 0.10); dynamic models have stationary residuals.

The passing model with the lowest BIC is the **champion**; the best passing model on a different driver set is the **challenger**. Every sign-correct, significant static model failed the cointegration test for cards, other consumer and corporate (spurious regressions), so all champions are dynamic.

### 3.4 Champion models

Newey–West standard errors in brackets.

| | Credit cards | Other consumer | Mortgages | Corporate |
|---|---|---|---|---|
| Δ unemployment, t−2 | 0.148 (0.039) | 0.111 (0.052) | | |
| Unemployment, t | | | | 0.209 (0.038) |
| Unemployment, t−2 | | | 0.229 (0.089) | |
| GDP growth, t−6 | | | | −0.023 (0.005) |
| House price growth, t−1 | | | −0.035 (0.012) | |
| Bank Rate, t−8 | 0.030 (0.010) | | 0.097 (0.030) | |
| COVID regime | −0.057 (0.061) | −0.092 (0.092) | −0.248 (0.154) | −0.286 (0.073) |
| Persistence ρ | 0.874 (0.053) | 0.858 (0.107) | 0.522 (0.118) | 0.615 (0.066) |
| Adjusted R² | 0.914 | 0.720 | 0.848 | 0.911 |
| Candidates passing all rules | 32 | 4 | 334 | 49 |
| BIC gap to next model | 3.2 | 0.19 | 0.32 | 1.4 |

Interpretation at the end-2024 starting point: a 1 pp rise in unemployment over a year raises the card loss rate by about 0.33 pp within six months, and by about 2.6 pp once persistence plays out.

### 3.5 Alternatives considered

| Alternative | Result | Decision |
|---|---|---|
| Static levels models | Fail cointegration (spurious) | Rejected |
| Raw-rate or log target | Raw rate raises the card scenario peak by about 1 pp; Box-Cox λ = −0.31 (close to log) | Logit kept; raw rate reported as sensitivity |
| Asymmetric unemployment effect, quadratic unemployment, mortgage "double trigger" interaction | Not significant (p = 0.66, 0.47, 0.10 with the wrong sign) | Linear form kept |
| Ridge / Lasso / Elastic Net on 51 features | Lasso and Elastic Net use 17 features, 13 with wrong signs; card scenario peak 3.3–3.9% | Rejected for stress use |
| Sign-constrained Lasso | Independently selects the champion's card drivers; best cross-validated error | Validated challenger |
| Pooled panel of all portfolios | Slope homogeneity rejected (p ≈ 10⁻²²) | Separate models |
| Pooled consumer panel (cards + other) | Homogeneity not rejected (p = 0.39) | Challenger for other consumer credit |
| Consumer credit growth (t−12) | Significant (p = 0.005) but only from 2000 | Recommended future enhancement |

### 3.6 Projection

Recursive: the persistence term uses actual loss rates while available, then the model's own projection. Uncertainty from 2,000 simulations combining parameter draws from N(β̂, V_HAC) with residuals resampled in four-quarter blocks. Optional jump-off adjustment carries the end-2024 model gap into the projection, fading over eight quarters. Five-year cumulative rate = sum of the five calendar-year (Q4 trailing) rates; £ losses use the end-2024 balance (static balance sheet).

Implementation: [`satellite.py`](../src/ukstress/models/satellite.py), [`engine.py`](../src/ukstress/scenarios/engine.py). The OLS engine is also implemented from first principles ([`ols_scratch.py`](../src/ukstress/models/ols_scratch.py), [notebook 02](../notebooks/02_ols_from_scratch.ipynb)) and unit-tested against statsmodels to machine precision.

## 4. Performance

Summary only; full results in the [validation report](validation_report.md).

| | Cards | Other consumer | Mortgages | Corporate |
|---|---|---|---|---|
| 8-quarter skill vs no-change (2003–24) | −0.17 | +0.12 | +0.30 | +0.40 |
| GFC peak error, re-estimated to 2007 Q4 | +7.3 pp | +0.7 pp | +0.0 pp | −0.45 pp |
| Bootstrap share with correct sign (weakest macro term) | 99.6% | 95.1% | 98.1% | 99.7% |

## 5. Limitations

1. **One severe cycle.** Stress sensitivities are learned mainly from 2008–10. Recursive estimates show the card sensitivity fell from about 0.33 to 0.22 when that cycle entered the sample and has drifted to about 0.15 since 2024.
2. **Interest-rate extrapolation.** The 2025 scenario's 8% Bank Rate exceeds the 1993–2026 maximum (7.5%). Mortgage projections above about 1.5x the scenario severity are not credible.
3. **Unobserved credit cycles.** The 2004–07 UK personal-insolvency boom raised card losses with unemployment flat; the models under-fit it.
4. **Book quality has changed.** Consumer models over-predict 2025–26 by 30–40%, consistent with the Bank of England's 2025 judgement on improved consumer credit quality.
5. **Write-offs, not impairments.** Write-offs lag IFRS 9 impairments; the models are not a substitute for provisioning models.
6. **Aggregate data.** All UK banks and building societies; individual books differ in mix and risk appetite.

## 6. Use constraints and overlays

- Consumer projections may be scaled by the credit-quality overlay (cards 0.72, other consumer 0.62), recalibrated quarterly.
- Report other consumer credit and mortgages as champion–challenger ranges.
- Expert judgement is required for interest-rate scenarios above 8% or unemployment above 11%.

## 7. Monitoring

Quarterly, once the Bank of England publishes write-offs:
1. Re-run [notebook 06](../notebooks/06_validation.ipynb) section 2 (live monitoring since jump-off).
2. **Trigger:** actual loss rate outside the 90% simulation band for three consecutive quarters → re-estimate and re-validate.
3. Annual: re-run the full selection procedure; review the overlay.
