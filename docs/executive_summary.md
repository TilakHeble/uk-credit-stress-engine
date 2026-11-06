# UK credit losses under the Bank of England 2025 stress scenario

*Executive summary for a risk committee. Figures are from [notebook 07](../notebooks/07_scenario_projection.ipynb); methods and limitations are in the [model development document](model_development_document.md) and [validation report](validation_report.md).*

## The question

If the UK enters the Bank of England's 2025 Bank Capital Stress Test scenario, how much would UK lenders lose on credit cards, other consumer credit, mortgages and corporate lending, and how confident can we be? The scenario runs for five years from end-2024: UK GDP falls 5%, unemployment peaks at 8.5%, house prices fall 28% and Bank Rate peaks at 8%.

## The answer

**Five-year write-offs of about £44 billion across the four portfolios, about £32 billion more than end-2024 loss rates would produce. The plausible range across modelling choices is £35–49 billion.**

| Portfolio | Balance, end-2024 | Loss rate end-2024 | Peak loss rate (2028 Q1) | Five-year cumulative | 90% interval | Five-year losses |
|---|---|---|---|---|---|---|
| Credit cards | £53bn | 2.3% | 6.4% | 24.4% | 17.5–34.8% | **£12.9bn** |
| Other consumer credit | £75bn | 0.8% | 2.4% | 7.0% | 3.0–13.7% | **£5.3bn** |
| Mortgages | £1,474bn | 0.01% | 0.22% | 0.53% | 0.24–1.22% | **£7.8bn** |
| Corporate (PNFC) | £500bn | 0.13% | 1.08% | 3.6% | 2.3–5.6% | **£18.2bn** |
| **Total** | **£2,102bn** | | | **2.1%** | | **£44.3bn** |

Write-offs net of recoveries, all UK banks and building societies, static end-2024 balance sheet (as in the Bank of England test).

## What drives it

- **The labour market dominates.** A one-point higher unemployment path adds 2.7 percentage points to five-year card losses and 1.9 points to corporate losses. Card losses respond to the *change* in unemployment within about six months; corporate losses to its *level*.
- **Interest rates are the new risk, and the least certain one.** The scenario's 8% Bank Rate is above anything in the 1993–2026 data. Through a two-year lag, higher rates add to card and mortgage losses, and they are why **mortgage losses exceed their 2008–12 peak even at 80% of the scenario's severity**: in 2009 Bank Rate was cut to 0.5%; in this scenario it rises.
- **Losses peak late.** Every portfolio peaks in 2028 Q1, about a year after unemployment peaks, because write-offs follow arrears and persist.

## How confident are we?

| Portfolio | Validation rating | Main reason |
|---|---|---|
| Corporate | 🟢 Green | Best forecasting record; may understate crisis peaks (under-projected 2009–12 by 25%) |
| Mortgages | 🟡 Amber | Accurate backtests, but a tiny loss rate and a rejected functional-form test; challenger 30% lower |
| Credit cards | 🟡 Amber, conservative | Robust signs, but over-projects: models estimated before 2008 over-shot the crisis, and the model over-predicts 2025–26 |
| Other consumer | 🟡 Amber, weak | Single-driver model; the unemployment effect is not robust in bootstrap; pooled challenger projects 30% higher |

**Benchmark against the Bank of England's own results** (December 2025 *Financial Stability Report*, Table 5.C): the Bank's five-year *impairment* rates are 23.3% for UK consumer credit and 1.0% for mortgages; this project's five-year *write-off* rates are 14.2% and 0.53%. Impairments should be higher: IFRS 9 books lifetime losses before debts are written off, and the Bank's figures cover the seven largest banks only. The ratio (about 0.5–0.6) is consistent across portfolios, and the ranking of risks is the same.

**Current consumer books look safer than history.** Since end-2024 unemployment has risen to about 5%, yet card and personal-loan losses stayed flat, so the consumer models over-predicted 2025–26 by about 30–40%. The Bank of England reached the same view independently and cut its consumer loss assumptions in the 2025 test. With an overlay for this, card five-year losses fall from 24.4% to 17.6%.

## Reverse stress test

How severe would a recession need to be for each portfolio to exceed its 2008–12 peak loss rate?

| Portfolio | Severity needed (1 = 2025 scenario) | Peak unemployment at that severity |
|---|---|---|
| Mortgages | 0.8 | 7.7% (with Bank Rate 7.4%) |
| Corporate | 1.4 | 10.1% |
| Credit cards | 2.0 | 12.6% |
| Other consumer | not reached at 2.5 | n/a |

## Recommendations

1. **Use the corporate and card models for stress projections, the card model with a documented credit-quality overlay.** Report other consumer credit and mortgages as ranges (champion to challenger).
2. **Treat interest-rate effects as an expert-judgement area.** No UK data combine an 8% Bank Rate with rising unemployment; supplement with affordability analysis of mortgage and card books.
3. **Add credit growth as a driver when longer history is available.** Consumer credit growth three years earlier predicts card losses (p = 0.005) but its series starts only in 2000.
4. **Monitor quarterly.** Re-estimate if actual losses fall outside the 90% band for three consecutive quarters (currently the case for the consumer models).

*Data: Bank of England (write-offs, balances, Bank Rate, 2025 scenario), ONS (unemployment, GDP, CPI), HM Land Registry (UK HPI). Open Government Licence v3.0.*
