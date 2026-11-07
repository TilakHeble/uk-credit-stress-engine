# 05 · Fixing problems: transforms, robust errors, WLS, robust regression, non-linearity

🟡 Intermediate · Previous: [04](04_diagnostics.md) · [Course index](README.md) · Next: [06 · Choosing a model](06_model_selection.md)

Lesson 4 found problems. This lesson is the toolbox for fixing them. A rule of thumb: **match the fix to the problem**. Robust standard errors fix inference, not bias; a transform fixes shape, not autocorrelation.

```python
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from ukstress.ingest.build_panel import build_macro, build_panel
from ukstress.features.build import covid_dummy, logit_pct, trailing_loss_rate

warnings.filterwarnings("ignore")
panel, macro = build_panel(), build_macro()
rate = trailing_loss_rate(panel, "credit_card")
base = pd.DataFrame({
    "rate": rate,
    "d_unemp_L2": macro["unemployment_chg_yoy"].shift(2),
    "bank_rate_L8": macro["bank_rate"].shift(8),
    "covid": covid_dummy(macro.index),
}).dropna()
```

## 1. Transforming the target

A loss rate is bounded below by 0 and right-skewed (a few crisis years are far above the typical 2–3%). A linear model on the raw rate can predict negative losses in a good scenario and treats a 1-point rise from 2% the same as from 10%.

| Transform | Formula | Coefficient means | Use when |
|---|---|---|---|
| None | $y$ | units of y per unit of x | y unbounded, roughly symmetric |
| Log | $\ln y$ | ≈ % change in y per unit of x | y positive, multiplicative effects |
| Logit | $\ln\frac{p}{1-p}$ | change in log-odds | y is a rate between 0 and 1 |
| Box–Cox | $\frac{y^\lambda - 1}{\lambda}$ | depends on λ | let the data choose; λ = 0 is log |

```python
lam = stats.boxcox(base["rate"])[1]
print(f"Box-Cox λ = {lam:.2f}  (0 = log, 1 = no transform)")
for name, yt in [("raw", base["rate"]), ("log", np.log(base["rate"])), ("logit", logit_pct(base["rate"]))]:
    r = sm.OLS(yt, sm.add_constant(base[["d_unemp_L2", "bank_rate_L8", "covid"]])).fit()
    print(f"{name:>5}: residual skew {stats.skew(r.resid):+.2f}")
```

Box–Cox picks λ ≈ −0.3, close to the log. On the raw rate the residuals are right-skewed (about +0.65); the log and logit both bring skew down to about 0.1. The project uses the logit because it also keeps projections below 100%.

**Interpreting a logit coefficient:** the effect on the rate depends on where you start. At loss rate $p$ (as a fraction), $\partial p / \partial x \approx \beta\, p (1-p)$. So the same coefficient implies a bigger absolute rise in a bad year than in a good one, which is what we see in data on default rates.

**Back-transform with care.** $E[\ln y] \ne \ln E[y]$: predicting on the log scale and exponentiating gives roughly the *median*, not the mean. For stress testing this is usually handled by simulating and transforming each draw (lesson 9).

## 2. Robust standard errors

If the errors are heteroskedastic (lesson 4), keep the OLS coefficients but replace the variance formula with White's "sandwich":

$$\widehat{\text{Var}}(\hat\beta) = (X'X)^{-1}\Big(\sum_i e_i^2 x_i x_i'\Big)(X'X)^{-1}$$

HC0 is the formula above; HC1 multiplies by $n/(n-k)$; HC3 adjusts for leverage and behaves best in small samples. If errors are also autocorrelated, use Newey–West (lesson 7).

```python
y = logit_pct(base["rate"])
X = sm.add_constant(base[["d_unemp_L2", "bank_rate_L8", "covid"]])
ses = pd.DataFrame({c: sm.OLS(y, X).fit(cov_type=c).bse for c in ["nonrobust", "HC1", "HC3"]})
ses["Newey-West"] = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4}).bse
print(ses.round(4))
```

The robust (HC) errors are close to the classical ones here, so heteroskedasticity is minor. Newey–West errors are about 1.6–1.9 times larger: autocorrelation, not heteroskedasticity, is the real problem in this data.

## 3. Weighted least squares (WLS)

If you know (or can estimate) which observations are noisier, give them less weight. Minimise $\sum w_i e_i^2$ with $w_i = 1/\sigma_i^2$. **Feasible GLS** estimates $\sigma_i$ from the OLS residuals first:

```python
ols = sm.OLS(y, X).fit()
aux = sm.OLS(np.abs(ols.resid), sm.add_constant(ols.fittedvalues)).fit()   # model the size of the errors
sd_hat = np.clip(aux.fittedvalues, 1e-3, None)
wls = sm.WLS(y, X, weights=1 / sd_hat**2).fit()
print(pd.DataFrame({"OLS": ols.params, "WLS": wls.params}).round(3))
```

If WLS and OLS give similar coefficients, heteroskedasticity is not distorting the estimates. Large differences mean the conclusions depend on how noisy quarters are treated, which is itself worth reporting.

Here WLS and OLS agree closely, so the noisier quarters are not distorting the estimates.

## 4. Robust regression (Huber)

OLS squares residuals, so one huge outlier can dominate. **Huber M-estimation** uses squared loss for small residuals and absolute loss for large ones: big outliers get down-weighted, not deleted.

$$\rho(e) = \begin{cases} \tfrac12 e^2 & |e| \le c \\ c|e| - \tfrac12 c^2 & |e| > c \end{cases}$$

```python
huber = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
weights = pd.Series(huber.weights, index=y.index)
print(pd.DataFrame({"OLS": ols.params, "Huber": huber.params}).round(3))
print("most down-weighted quarters:", weights.nsmallest(3).round(2).to_dict())
```

Huber barely moves the coefficients. The most down-weighted quarters (around 2008–10 and 2023) are the ones the model fits worst, but they are not driving the result.

## 5. Non-linear relationships within a linear model

"Linear regression" means linear in the *coefficients*, not in the variables. You can include transformed regressors freely.

**Polynomials:** add $x^2$ to allow curvature. **Hinge (piecewise linear) terms:** allow a different slope above and below a threshold. Do rising and falling unemployment matter equally?

```python
du = base["d_unemp_L2"]
Xh = sm.add_constant(pd.DataFrame({"du_up": du.clip(lower=0), "du_down": du.clip(upper=0),
                                   "bank_rate_L8": base["bank_rate_L8"], "covid": base["covid"]}))
hinge = sm.OLS(y, Xh).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(hinge.params[["du_up", "du_down"]].round(3).to_dict())
print(f"p-value for equal slopes: {float(hinge.wald_test('du_up = du_down', scalar=True).pvalue):.3f}")
```

A small p-value would mean the effect is asymmetric. Always test non-linear terms rather than add them by default: each one costs a degree of freedom and risks overfitting (lesson 6).

Here the two slopes are similar and the test does not reject equality (p ≈ 0.62): rising and falling unemployment move card losses symmetrically, so the simpler linear term is kept.

**Splines** generalise hinges: piecewise polynomials joined smoothly at several knots (`patsy`'s `bs()` or `cr()` in a formula). They are flexible but can behave wildly outside the data range, which matters for stress scenarios.

## Which fix for which problem?

| Problem (lesson 4) | Biases coefficients? | Fix |
|---|---|---|
| Heteroskedasticity | No | Robust (HC) errors; WLS for efficiency |
| Autocorrelation | No (unless a lagged y is in the model *and* errors are autocorrelated) | Newey–West errors; model the dynamics (lesson 7) |
| Non-normal errors | No | Transform y; bootstrap intervals |
| Outliers / influence | Can | Investigate; Huber regression; report with and without |
| Wrong functional form | **Yes** | Transforms, polynomials, hinges, interactions |
| Omitted variable | **Yes** | Add the variable; or argue its correlation with X is small |
| Multicollinearity | No (imprecise, not biased) | Drop or combine; Ridge |

## Practice

**Q1.** A colleague fixes autocorrelated residuals by switching to HC3 standard errors. What is wrong?

<details><summary>Answer</summary>

HC3 corrects for heteroskedasticity only; it assumes errors are uncorrelated across observations. Autocorrelation needs HAC (Newey–West) standard errors or a model of the dynamics.
</details>

**Q2.** In a log-linear model, $\ln y = 0.5 + 0.08x$. Approximately what does a one-unit increase in x do to y? Exactly?

<details><summary>Answer</summary>

Approximately +8%. Exactly, y is multiplied by e^0.08 ≈ 1.083, so +8.3%. The approximation is good for small coefficients.
</details>

**Q3.** The logit card model has a coefficient of 0.15 on the unemployment change. What is the approximate effect on the loss rate of a 1-point rise when the loss rate is 2%? When it is 8%?

<details><summary>Answer</summary>

∂p/∂x ≈ β p(1−p). At p = 0.02: 0.15 × 0.02 × 0.98 ≈ 0.0029 → +0.29 pp. At p = 0.08: 0.15 × 0.08 × 0.92 ≈ 0.011 → +1.1 pp. The same coefficient has a much bigger absolute effect when losses are already high.
</details>

**Q4.** Why might splines be dangerous in a stress-testing model, even if they fit history well?

<details><summary>Answer</summary>

Stress scenarios deliberately go to the edge of, or beyond, historical data. Polynomial pieces can curve sharply outside the data range, so projections may explode or turn the wrong way exactly where the model matters most. Simpler, economically constrained forms extrapolate more predictably.
</details>
