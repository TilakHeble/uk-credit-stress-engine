# 10 · Regression in industry: from a fitted model to a decision

🔴 Expert · Previous: [09](09_uncertainty_validation.md) · [Course index](README.md) · Extras: [cheat sheet](cheat_sheet.md) · [interview questions](interview_questions.md)

Lessons 1–9 were about getting the regression right. In a bank, the NHS, a tech company or government, that is only half the job. This lesson covers the other half: **using a model responsibly to support a decision**, with the stress-testing project as the worked example.

## 1. Where regression is used for real

| Sector | Typical regression task | What the model feeds |
|---|---|---|
| Banking (credit risk) | Satellite models: loss/default rates on macro drivers | Bank of England / ECB stress tests, IFRS 9 provisions, capital planning |
| Banking (market risk) | Factor models: returns on market, size, value factors | Hedging, risk attribution, performance evaluation |
| Insurance | Claims frequency/severity (GLMs) | Pricing, reserving |
| Tech / marketing | Marketing mix models, A/B test regression adjustment (CUPED) | Budget allocation, experiment readouts |
| NHS / public health | Demand and waiting-time drivers, risk adjustment | Capacity planning, fair performance comparison |
| Government / economics | Policy evaluation (difference-in-differences), forecasting | Spending decisions, impact assessments |

The techniques are the same; the bar for **trust** is different. A model that sets capital or allocates NHS beds must be explainable, validated and monitored.

## 2. Stress testing in one page

A stress test asks: if a severe-but-plausible recession happened, would a bank still have enough capital?

1. **Scenario:** the regulator publishes paths for unemployment, GDP, house prices, rates (e.g. the Bank of England's 2025 Bank Capital Stress Test).
2. **Satellite models:** regressions translate the scenario into loss rates per portfolio (this project, notebooks 03–05).
3. **Projection:** losses, income and balance-sheet effects over five years (notebook 07).
4. **Capital impact:** how far the capital ratio falls (the Bank of England reported an aggregate fall of 3.5 percentage points in 2025).

Regression is the engine in step 2, which is why everything in lessons 1–9 matters. A wrong-signed coefficient becomes a wrong capital number.

## 3. Model risk management

"Model risk" is the risk of losses or bad decisions from using a model that is wrong or misused. UK banks follow the PRA's **SS1/23** principles. In practice that means:

| Principle | What it means for a regression model | In this project |
|---|---|---|
| Model identification and tiering | Know which models exist and how material each is | Four satellite models, one per portfolio |
| Governance | Owners, approvals, policies | Model development document names purpose and use limits |
| **Development** | Sound data, methodology, documented choices and limitations | [Model development document](../docs/model_development_document.md) |
| **Independent validation** | Someone other than the developer tests it: conceptual soundness, outcome analysis, benchmarks | [Validation report](../docs/validation_report.md), notebook 06 |
| **Model risk mitigants** | Overlays, conservative adjustments, usage restrictions when weaknesses are known | Credit-quality overlay; ranges for weak models; expert judgement above 8% Bank Rate |
| **Monitoring** | Ongoing performance checks with triggers | Re-estimate if actual losses leave the 90% band for three quarters |

## 4. Writing a model up

A model nobody can review is a model nobody should use. A good development document answers, in order:
1. **Purpose and intended use**, and what the model must *not* be used for.
2. **Data:** sources, definitions, quality checks, known breaks.
3. **Methodology:** the equation, the candidates considered, the selection rules, and why alternatives were rejected.
4. **Performance:** fit, backtests, benchmarks, in numbers.
5. **Limitations:** stated plainly, ranked by importance.
6. **Use constraints and overlays**, and **monitoring** with triggers.

The validation report then gives an independent opinion with a rating (e.g. green/amber/red) and a list of findings to remediate.

## 5. Communicating to non-specialists

Risk committees, executives and clinicians do not want t-statistics. Lead with the decision-relevant answer, then the uncertainty, then the caveat that could change the decision.

> **Weak:** "The champion model's Newey–West t-statistic on Δu(t−2) is 3.8 and adj. R² is 0.91."
>
> **Strong:** "Under the 2025 scenario, card losses rise from 2.3% to a peak of about 6.4% a year in 2028, roughly 2004–07 levels; the plausible range for five-year losses is £9–16bn. The biggest uncertainty is the 8% Bank Rate, which is outside anything in our data."

See the [executive summary](../docs/executive_summary.md) for the full version.

## 6. Ethics and fairness

Regression can encode unfairness: a credit-scoring regression using postcode can act as a proxy for ethnicity. In lending, insurance, hiring and public services:
- Know which variables are **protected characteristics** (UK Equality Act 2010) or proxies for them.
- Test outcomes for disparate impact across groups, even if the protected variable is not in the model.
- Prefer explainable models where decisions affect individuals; people have a right to understand significant automated decisions (UK GDPR Article 22).

Aggregate macro models like this project do not score individuals, but the same discipline applies when you build a credit-scoring model (logistic regression, the next project in this portfolio).

## Capstone exercise

Using the repository's code (`ukstress.models.satellite` and `ukstress.scenarios.engine`), build your own scenario: a "stagflation" path where unemployment rises to 7% while Bank Rate rises to 9% and house prices fall 15%. Project card and mortgage losses, and write a five-sentence summary for a risk committee that states the result, the range, and the main limitation.

<details><summary>Hint</summary>

Start from `scenario_macro(macro)`, edit the projection quarters (index > 2024Q4) of `unemployment`, `bank_rate` and `house_price_growth_yoy`, call `add_changes()` to recompute the change variables, then pass the path to `run(models, panel, macro, path, n_sims=1000)` and `summary_table()`. The dashboard (`make dashboard`) does this interactively.
</details>

## Practice

**Q1.** A validator finds that your model's residuals have been outside the 90% band for four quarters running. Under the monitoring policy, what happens next, and what might you do in the meantime?

<details><summary>Answer</summary>

The trigger (three consecutive quarters) is breached, so the model is escalated for re-estimation and re-validation. In the meantime, apply a documented, conservative overlay or management adjustment, and report results with the model's known weakness.
</details>

**Q2.** Why should model development and validation be done by different people?

<details><summary>Answer</summary>

Developers are invested in their choices and test what they expect to work. An independent validator challenges assumptions, tries to break the model, and has no incentive to approve it, which reduces the risk of errors and optimistic bias reaching decisions.
</details>

**Q3.** Rewrite this for an executive: "The logit-scale coefficient on lagged house price growth is −0.035 (HAC s.e. 0.012) in the mortgage satellite model."

<details><summary>Answer</summary>

"Falling house prices raise mortgage losses: in the stress scenario, the 28% fall in prices is one of three drivers that take mortgage losses from almost zero to about 0.2% a year, still small compared with credit cards."
</details>
