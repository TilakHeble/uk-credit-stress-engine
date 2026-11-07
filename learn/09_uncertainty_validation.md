# 09 · Uncertainty and validation: bootstrap, quantile regression, simulation, backtesting

🔴 Expert · Previous: [08](08_panel_regression.md) · [Course index](README.md) · Next: [10 · Regression in industry](10_regression_in_industry.md)

A stress-test projection is not one number. This lesson turns a fitted regression into a **distribution of outcomes**, and then asks the validator's question: would this model have worked?

```python
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from ukstress.config import MODELS_DIR
from ukstress.ingest.build_panel import build_macro, build_panel
from ukstress.features.build import scenario_macro, trailing_loss_rate
from ukstress.models.satellite import load_models, project, sample, target
from ukstress.validation.backtest import backtest, summary

warnings.filterwarnings("ignore")
panel, macro = build_panel(), build_macro()
champ = load_models(MODELS_DIR / "champions.json")["credit_card"]     # the project's card model
y, X = sample(champ.spec, target(panel, "credit_card"), macro)
print(champ.spec.describe())
```

## 1. The bootstrap

Standard errors from formulas rely on assumptions. The bootstrap estimates uncertainty by **resampling the data**: draw a new sample with replacement, refit, repeat 1,000 times, and look at the spread of the estimates.

For time series, resampling single quarters destroys the autocorrelation. The **moving-block bootstrap** resamples blocks of consecutive quarters, keeping short-range dependence intact:

```python
rng = np.random.default_rng(0)
n, block = len(y), 8
draws = []
for _ in range(1000):
    starts = rng.integers(0, n - block + 1, n // block + 1)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    draws.append(np.linalg.lstsq(X.values[idx], y.values[idx], rcond=None)[0])
draws = pd.DataFrame(draws, columns=X.columns)
c = "unemployment_chg_yoy_L2"
se = np.sqrt(champ.cov.loc[c, c])
print(f"Newey-West 95% CI: [{champ.params[c] - 1.96 * se:.3f}, {champ.params[c] + 1.96 * se:.3f}]")
print(f"Block-bootstrap 95% CI: [{draws[c].quantile(.025):.3f}, {draws[c].quantile(.975):.3f}]")
print(f"share of bootstrap draws with the correct (positive) sign: {(draws[c] > 0).mean():.1%}")
```

If the two intervals roughly agree, the analytic standard errors are believable. The share with the right sign is a direct measure of how robust the economic conclusion is.

## 2. Quantile regression: modelling the tail

OLS models the **mean** of *y* given *x*. Quantile regression models a chosen **percentile**: it minimises asymmetrically weighted absolute errors (the "check" loss), so the 90th-percentile line has 10% of points above it.

$$\min_\beta \sum_i \rho_\tau(y_i - x_i'\beta), \qquad \rho_\tau(u) = u(\tau - \mathbb{1}[u < 0])$$

Risk managers care about bad outcomes. If drivers matter *more* in the upper tail, a mean model understates stress.

```python
rows = {f"q={q}": sm.QuantReg(y, X).fit(q=q, max_iter=5000).params for q in [0.1, 0.5, 0.9]}
rows["OLS mean"] = champ.params
print(pd.DataFrame(rows).T[["unemployment_chg_yoy_L2", "bank_rate_L8"]].round(3))
```

Here the unemployment effect is similar across the 10th, 50th and 90th percentiles (about 0.16–0.20) and somewhat above the OLS mean (0.15). There is no strong evidence that bad quarters are *more* sensitive than typical ones; rather, OLS is pulled down slightly by a few quarters with large negative residuals. That is a useful finding in itself: the tail behaves like the middle, so the mean model's simulation bands are a reasonable guide to tail risk.

## 3. Projecting with uncertainty: simulation

A projection has two sources of uncertainty:
1. **Parameter uncertainty:** we do not know β exactly. Draw β from $N(\hat\beta, \hat V_{NW})$.
2. **Shock uncertainty:** future quarters bring their own errors. Resample actual residuals in blocks (no normality assumption).

Each simulation draws both, runs the dynamic model forward through the scenario, and records the path. Percentiles across 1,000 paths give the fan chart.

```python
path = scenario_macro(macro)        # actual history to 2024 Q4, then the Bank of England 2025 stress scenario
pr = project(champ, target(panel, "credit_card"), path, n_sims=1000, seed=1)
print(pr[["central", "p05", "p50", "p95"]].iloc[::4].round(2))
```

**Read the bands correctly:** the 90% band at a quarter is pointwise. The 90% interval for a total over five years must be computed from the simulated totals, not by adding the pointwise bands (notebook 07 does this).

## 4. Out-of-time backtesting

The most convincing validation: pretend it is the past. Re-estimate the model using **only data up to a cutoff**, then project forward using the economy that actually happened. Comparing with actual losses isolates model error from scenario error.

```python
bt = backtest(champ.spec, panel, macro, cutoff="2007Q4", horizon=20, n_sims=300)
s = summary(bt)
print({k: round(float(v), 2) for k, v in s.items()})
```

Re-estimated before the crisis, the card model would have **over**-projected 2008–12 losses. For capital that is the safer error, but it shows how much the estimated sensitivity depends on what is in the sample (lesson 7, structural breaks).

## 5. Beating a naive benchmark

A model should beat the simplest alternative. For a loss rate, that is **no change**: next year's rate equals today's. Skill = 1 − RMSE(model) / RMSE(naive); positive means the model adds value.

```python
from ukstress.validation.backtest import rolling_origin

ro = rolling_origin(champ.spec, panel, macro, [str(q) for q in pd.period_range("2005Q4", "2022Q4", freq="Q")[::4]], horizon=8)
rmse = lambda e: float(np.sqrt((e ** 2).mean()))   # noqa: E731
print(f"RMSE model {rmse(ro['model error']):.2f} pp, naive {rmse(ro['naive error']):.2f} pp, "
      f"skill {1 - rmse(ro['model error']) / rmse(ro['naive error']):+.2f}")
```

A stress model can lose to "no change" in normal times and still be useful: its job is to answer "what if the economy turns?", which "no change" cannot. But the result must be reported, and it rules out using the model for routine forecasting.

## 6. Benchmarking against an external result

Compare with an independent estimate of the same quantity. The Bank of England published five-year impairment rates for its 2025 test: 23.3% for UK consumer credit, 1.0% for mortgages. The project's write-off rates are lower by a consistent ratio of about 0.5–0.6, which is expected (impairments book lifetime losses earlier than write-offs) and suggests the *relative* calibration is sound. See [notebook 07](../notebooks/07_scenario_projection.ipynb).

## Key takeaways
- Block bootstrap: resample blocks to respect autocorrelation; compare with analytic intervals.
- Quantile regression shows whether drivers bite harder in the tail.
- Simulate parameters and shocks together for honest projection bands; compute totals from simulated totals.
- Backtest out of time, compare with a naive benchmark, and benchmark against external estimates.

## Practice

**Q1.** Why does an ordinary (single-observation) bootstrap understate uncertainty for quarterly time-series regressions?

<details><summary>Answer</summary>

It treats quarters as independent, breaking the autocorrelation. Each bootstrap sample then looks like it contains more independent information than the real data do, so the spread of estimates (and hence the uncertainty) is too small. Block bootstrap fixes this.
</details>

**Q2.** The 95th-percentile band for each year's loss rate is 6%, 8%, 9%, 8%, 7%. Is the 95th percentile of the five-year total 38%?

<details><summary>Answer</summary>

No. Adding pointwise percentiles assumes every year is at its 95th percentile in the same simulation, which is rarer than 5%. The correct number is the 95th percentile of the simulated five-year totals, which is usually lower (unless years are perfectly correlated).
</details>

**Q3.** A model's out-of-time backtest through 2008 over-predicts losses by 50%. Give one argument for keeping it as a stress model, and one change you would make.

<details><summary>Answer</summary>

For: in capital stress testing, over-prediction is the conservative error, and the model still ranks scenarios correctly. Change: investigate why (here, a pre-2008 sample over-attributed the 2004–07 insolvency boom to unemployment), document it, and consider a calibration overlay or an extra driver for the credit cycle.
</details>
