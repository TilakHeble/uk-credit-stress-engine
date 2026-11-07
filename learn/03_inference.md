# 03 · How sure are we? Inference

🟡 Intermediate · Previous: [02](02_multiple_regression.md) · [Course index](README.md) · Next: [04 · Diagnostics](04_diagnostics.md)

## Estimates are random

Run the same regression on a different 30 years of history and you would get different coefficients. Inference asks: **how far might the estimate be from the truth?** The answer is a standard error, and from it confidence intervals, t-tests and p-values.

## The assumptions (Gauss–Markov and friends)

| # | Assumption | If it fails |
|---|---|---|
| A1 | Linear in parameters: $y = X\beta + \varepsilon$ | Wrong functional form; biased predictions (lesson 5) |
| A2 | No perfect collinearity | $X'X$ cannot be inverted |
| A3 | Exogeneity: $E[\varepsilon \mid X] = 0$ | **Biased coefficients** (omitted variables, reverse causality) |
| A4 | Homoskedasticity: constant error variance | Standard errors wrong (lesson 4–5) |
| A5 | No autocorrelation of errors | Standard errors wrong, usually too small (lesson 7) |
| A6 | Normal errors (small samples only) | t and F distributions approximate |

Under A1–A5, OLS is **BLUE**: the Best (lowest-variance) Linear Unbiased Estimator. A3 is the one that matters most for *meaning*; A4 and A5 matter for *uncertainty*.

## Standard errors from the formula

Under A1–A5:

$$\text{Var}(\hat\beta) = \sigma^2 (X'X)^{-1}, \qquad \hat\sigma^2 = \frac{e'e}{n - k}$$

where $k$ counts parameters including the intercept. The standard error of coefficient $j$ is the square root of the $j$-th diagonal element.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from ukstress.ingest.build_panel import build_macro, build_panel
from ukstress.features.build import covid_dummy, logit_pct, trailing_loss_rate

panel, macro = build_panel(), build_macro()
df = pd.DataFrame({
    "y": logit_pct(trailing_loss_rate(panel, "credit_card")),     # logit scale (lesson 5)
    "d_unemp_L4": macro["unemployment_chg_yoy"].shift(4),
    "bank_rate_L8": macro["bank_rate"].shift(8),
    "covid": covid_dummy(macro.index),
}).dropna()
X = sm.add_constant(df[["d_unemp_L4", "bank_rate_L8", "covid"]])
y = df["y"]

n, k = X.shape
beta = np.linalg.lstsq(X, y, rcond=None)[0]
e = y - X @ beta
sigma2 = e @ e / (n - k)
se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
res = sm.OLS(y, X).fit()
print(pd.DataFrame({"coef": beta, "se (formula)": se, "se (statsmodels)": res.bse}, index=X.columns).round(4))
```

## t-tests and p-values

To test $H_0: \beta_j = 0$:

$$t = \frac{\hat\beta_j}{\text{se}(\hat\beta_j)} \sim t_{n-k} \text{ under } H_0$$

The p-value is the probability of a |t| at least this large if the true coefficient were zero. Small p (conventionally < 0.05) → reject $H_0$.

```python
t = beta / se
p = 2 * stats.t.sf(np.abs(t), n - k)
print(pd.DataFrame({"t": t, "p": p}, index=X.columns).round(4))
```

**What a p-value is not:** it is not the probability that the hypothesis is true, and a tiny p-value does not mean a large effect. With enough data, trivially small effects become "significant". Always read the coefficient's *size* too.

## Confidence intervals

$$\hat\beta_j \pm t_{0.975,\,n-k} \cdot \text{se}(\hat\beta_j)$$

Interpretation: if we repeated the sampling many times, 95% of intervals built this way would contain the true value.

```python
q = stats.t.ppf(0.975, n - k)
print(pd.DataFrame({"low": beta - q * se, "high": beta + q * se}, index=X.columns).round(3))
```

## F-tests: several coefficients at once

**Overall F-test:** are all slopes zero? **Nested F-test:** does adding a set of $q$ variables improve the model?

$$F = \frac{(\text{RSS}_\text{restricted} - \text{RSS}_\text{full}) / q}{\text{RSS}_\text{full} / (n - k)} \sim F_{q,\,n-k}$$

```python
restricted = sm.OLS(y, sm.add_constant(df[["d_unemp_L4"]])).fit()
full = res
F = ((restricted.ssr - full.ssr) / 2) / (full.ssr / (n - k))
print(f"F = {F:.2f}, p = {stats.f.sf(F, 2, n - k):.4f}")
print("statsmodels:", [round(float(v), 4) for v in full.compare_f_test(restricted)[:2]])
```

With a single restriction, F equals the square of the t-statistic.

## Confidence interval vs prediction interval

For a new quarter with regressors $x_0$:
- **Confidence interval for the mean:** uncertainty about the *line*: $\text{se}^2 = x_0' V x_0$
- **Prediction interval for one outcome:** adds the quarter's own error: $\text{se}^2 = x_0' V x_0 + \hat\sigma^2$

```python
x0 = pd.DataFrame({"const": [1.0], "d_unemp_L4": [2.0], "bank_rate_L8": [5.0], "covid": [0.0]})
pred = res.get_prediction(x0).summary_frame(alpha=0.05)
to_rate = lambda z: 100 / (1 + np.exp(-z))   # back from logit to %
print({"central %": round(float(to_rate(pred["mean"].iloc[0])), 2),
       "95% CI for the mean %": [round(float(to_rate(pred[c].iloc[0])), 2) for c in ["mean_ci_lower", "mean_ci_upper"]],
       "95% prediction interval %": [round(float(to_rate(pred[c].iloc[0])), 2) for c in ["obs_ci_lower", "obs_ci_upper"]]})
```

**Stress testing needs prediction intervals.** A bank lives through one actual outcome, not the average of many.

## The catch: these standard errors assume A4 and A5

Our residuals are strongly autocorrelated (the trailing loss rate overlaps by three quarters each quarter), which violates A5. Classical standard errors are then **too small**, so t-statistics are too big:

```python
hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(pd.DataFrame({"classical se": res.bse, "Newey-West se": hac.bse, "ratio": hac.bse / res.bse}).round(3))
```

Lessons 4 and 7 show how to detect this and why Newey–West standard errors fix it.

## Key takeaways
- Under the Gauss–Markov assumptions, OLS is BLUE, with Var(β̂) = σ²(X'X)⁻¹.
- t-tests test one coefficient, F-tests test several; with one restriction F = t².
- p-values measure evidence against zero, not effect size or truth.
- Prediction intervals (one outcome) are much wider than confidence intervals (the mean).
- Classical standard errors are only right if errors are homoskedastic and uncorrelated.

## Practice

**Q1.** A coefficient is 0.15 with standard error 0.05 and 120 residual degrees of freedom. Is it significant at 5%? What is the 95% CI?

<details><summary>Answer</summary>

t = 3.0 > 1.98, so yes (p ≈ 0.003). CI ≈ 0.15 ± 1.98 × 0.05 = [0.051, 0.249].
</details>

**Q2.** Your colleague doubles the sample by duplicating every row. What happens to the coefficients and standard errors, and why is this cheating?

<details><summary>Answer</summary>

Coefficients are unchanged; standard errors shrink by about √2 because n doubles. It is cheating because no new information was added: the observations are not independent. Overlapping time-series windows create a milder version of the same problem, which is why HAC errors are needed.
</details>

**Q3.** Which assumption failure biases the coefficients themselves, rather than only the standard errors?

<details><summary>Answer</summary>

Exogeneity (A3): omitted variables correlated with X, measurement error in X, or reverse causality. Heteroskedasticity and autocorrelation leave OLS unbiased but make the classical standard errors wrong.
</details>

**Q4.** Why is the prediction interval for a single quarter always wider than the confidence interval for the mean, however much data you have?

<details><summary>Answer</summary>

The confidence interval shrinks towards zero width as n grows (parameter uncertainty disappears), but the prediction interval also includes σ², the irreducible noise of a single outcome, which does not shrink with more data.
</details>
