# 07 · Regression with time series: lags, stationarity, spurious regression, Newey–West

🟠 Advanced · Previous: [06](06_model_selection.md) · [Course index](README.md) · Next: [08 · Panel data](08_panel_regression.md)

Most textbook regression assumes independent observations, like people in a survey. Quarterly economic data are nothing like that: this quarter looks like last quarter, series trend, and effects arrive with delays. This lesson covers what changes.

```python
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint, kpss
from ukstress.ingest.build_panel import build_macro, build_panel
from ukstress.features.build import covid_dummy, logit_pct, trailing_loss_rate

warnings.filterwarnings("ignore")
panel, macro = build_panel(), build_macro()
y = logit_pct(trailing_loss_rate(panel, "credit_card")).rename("y")
```

## 1. Lags: effects take time

Losses today reflect the economy months or years ago (job loss → arrears → default → write-off). A **distributed lag** model includes past values of the driver: $y_t = \alpha + \beta_0 x_t + \beta_1 x_{t-1} + \dots$. The simplest way to find the lag is the cross-correlation at each lead:

```python
du = macro["unemployment_chg_yoy"]
print({k: round(y.corr(du.shift(k)), 2) for k in range(0, 9)})
```

The correlation peaks a few quarters out: changes in unemployment show up in card losses with a lag of up to about a year.

## 2. Stationarity

A series is **stationary** if its mean, variance and autocorrelation do not change over time. Many economic series are not: they trend or wander (a "unit root", like a random walk).

Two complementary tests:
- **ADF (augmented Dickey–Fuller):** null hypothesis = unit root (non-stationary). Small p → stationary.
- **KPSS:** null hypothesis = stationary. Small p → non-stationary.

```python
def test(s):
    s = s.dropna()
    return {"ADF p": round(adfuller(s)[1], 3), "KPSS p": round(kpss(s, nlags="auto")[1], 3)}

print("card loss rate (logit):", test(y))
print("unemployment level:     ", test(macro["unemployment"]))
print("change in unemployment: ", test(macro["unemployment_chg_yoy"]))
```

The card loss rate cannot reject a unit root (ADF p ≈ 0.23) and KPSS is borderline: it is persistent enough to treat with care. For the unemployment level the two tests *disagree* (ADF says stationary, KPSS says not), which is common with about 130 quarters: the honest conclusion is "borderline". When tests conflict, design the model so it is valid either way.

## 3. Spurious regression: the classic trap

Regress one random walk on another *independent* random walk and you will usually get a "significant" slope and a high R². The trends line up by coincidence. Granger and Newbold showed this in 1974; it is why time-series regressions need extra care.

```python
rng = np.random.default_rng(0)
hits = 0
for _ in range(500):
    a, b = np.cumsum(rng.normal(size=125)), np.cumsum(rng.normal(size=125))   # two unrelated random walks
    p = sm.OLS(a, sm.add_constant(b)).fit().pvalues[1]
    hits += p < 0.05
print(f"'significant' at 5% in {hits / 5:.0f}% of regressions of unrelated random walks (should be 5%)")
```

**Remedies:**
1. Model *changes* instead of levels (both become stationary), or
2. Show the levels are **cointegrated**: a combination of them is stationary, so they share a genuine long-run relationship. The Engle–Granger test checks whether the regression residuals are stationary, or
3. Include the dependent variable's own past (a dynamic model), which soaks up the persistence.

```python
X_levels = macro[["unemployment", "bank_rate"]].shift(4)
d = pd.concat([y, X_levels], axis=1).dropna()
print(f"Engle-Granger cointegration p = {coint(d['y'], d[['unemployment', 'bank_rate']])[1]:.3f}  (small p = cointegrated)")
```

Here p ≈ 0.6: the card loss rate and these levels are **not** cointegrated, so a static levels regression would be spurious.

In this project, **every** economically signed, significant *static* model for cards failed the cointegration test: they were spurious. All champion models are dynamic.

## 4. Lagged dependent variables (dynamic models)

Adding the target's own past makes the model dynamic:

$$y_t = \alpha + \beta x_{t-\ell} + \rho\, y_{t-4} + \varepsilon_t$$

- $\rho$ measures **persistence**: how much of a loss shock is still there a year later.
- **Short-run effect** of $x$: $\beta$. **Long-run effect** of a permanent change in $x$: $\beta / (1 - \rho)$.
- Projection is **recursive**: once the forecast passes the last actual data, $y_{t-4}$ is the model's own earlier prediction.

Why $y_{t-4}$ and not $y_{t-1}$? The target is a trailing four-quarter rate, so $y_t$ and $y_{t-1}$ share three of the same quarters. Using $y_{t-4}$ (no shared quarters) avoids a mechanical, almost-perfect fit.

```python
df = pd.DataFrame({"y": y, "d_unemp_L2": macro["unemployment_chg_yoy"].shift(2),
                   "bank_rate_L8": macro["bank_rate"].shift(8), "covid": covid_dummy(macro.index),
                   "y_L4": y.shift(4)}).dropna()
X = sm.add_constant(df.drop(columns="y"))
res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
rho, b = res.params["y_L4"], res.params["d_unemp_L2"]
print(res.params.round(3).to_dict())
print(f"short-run effect {b:.3f}; long-run multiplier 1/(1-ρ) = {1 / (1 - rho):.1f}; long-run effect {b / (1 - rho):.3f}")
```

**Caution:** with a lagged dependent variable, autocorrelated errors make OLS *biased* (the lagged y is correlated with the error). Here the residual autocorrelation is from the overlap and dies out by lag 4–8, which is why $y_{t-4}$ rather than $y_{t-1}$ matters.

## 5. Newey–West (HAC) standard errors

When errors are autocorrelated, classical standard errors are too small (the sample has less independent information than *n* suggests). Newey–West estimates the variance allowing for autocorrelation up to *L* lags, using Bartlett weights $w_l = 1 - \frac{l}{L+1}$:

$$\hat V_{NW} = (X'X)^{-1}\Big[\sum_t e_t^2 x_t x_t' + \sum_{l=1}^{L} w_l \sum_t e_t e_{t-l}(x_t x_{t-l}' + x_{t-l} x_t')\Big] \, (X'X)^{-1}$$

A common bandwidth rule is $L \approx 4(n/100)^{2/9}$, about 4 here, which also matches the 3-quarter overlap. Check conclusions are not sensitive to $L$:

```python
for L in [0, 2, 4, 8, 12]:
    r = sm.OLS(df["y"], X).fit() if L == 0 else sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    print(f"L = {L:>2}: t(unemployment) = {r.tvalues['d_unemp_L2']:.2f}")
```

## 6. Structural breaks

Relationships can change: new regulation, a new type of lending, a pandemic. Tools:
- **Chow test:** fit before and after a known date and F-test whether the coefficients differ.
- **CUSUM:** cumulative sum of recursive residuals; drifting outside bounds signals instability at an unknown date.
- **Recursive estimates:** re-estimate on expanding samples and plot the coefficient over time.
- **Dummies** for known regimes (the COVID dummy here).

```python
for end in ["2007Q4", "2012Q4", "2019Q4", "2026Q2"]:
    r = sm.OLS(df["y"].loc[:end], X.loc[:end].drop(columns="covid") if end < "2020" else X.loc[:end]).fit()
    print(f"sample to {end}: unemployment effect {r.params['d_unemp_L2']:.3f}")
```

The card unemployment effect estimated before 2008 is about twice today's, so a model built in 2007 would have over-projected the crisis (lesson 9 tests this).

## Key takeaways
- Effects arrive with lags; find them with cross-correlations and confirm in the regression.
- Non-stationary levels can produce spurious regressions; difference, prove cointegration, or model the dynamics.
- Dynamic models: ρ is persistence, β/(1−ρ) the long-run effect; project recursively.
- Overlapping targets create autocorrelation by construction: use Newey–West errors and avoid lags that overlap.
- Check stability: relationships estimated before a crisis may not hold in it.

## Practice

**Q1.** A regression of house prices on the number of UK mobile phones (1995–2025) has R² = 0.93 and t = 25. What should you suspect, and how would you check?

<details><summary>Answer</summary>

Spurious regression: both series trend upwards. Check stationarity of each (ADF/KPSS), regress changes on changes, and test whether the residuals of the levels regression are stationary (Engle–Granger). Expect the relationship to vanish.
</details>

**Q2.** A dynamic model has β = 0.15 and ρ = 0.875. What is the long-run effect of a permanent 1-unit rise in x? Roughly how many years until half of a shock to y has faded, if the persistence term is $y_{t-4}$?

<details><summary>Answer</summary>

Long-run effect = 0.15 / (1 − 0.875) = 1.2, eight times the short-run effect. With y_{t-4}, ρ applies once per year, so half-life ≈ ln 0.5 / ln 0.875 ≈ 5.2 years.
</details>

**Q3.** Why is ADF sometimes unable to reject a unit root in residuals that are actually stationary?

<details><summary>Answer</summary>

ADF has low power in short samples and when errors have a moving-average structure (like the MA(3) created by an overlapping four-quarter target). KPSS (with stationarity as the null) and ADF on non-overlapping observations are useful cross-checks; notebook 04 uses both.
</details>

**Q4.** Why is it wrong to use classical standard errors when the dependent variable is a 12-month rolling average observed monthly?

<details><summary>Answer</summary>

Consecutive observations share 11 of 12 months, so residuals are strongly autocorrelated by construction. The sample contains far less independent information than its length suggests; classical standard errors are much too small. Use Newey–West with at least 11 lags, or use non-overlapping observations.
</details>
