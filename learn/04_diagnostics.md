# 04 · Checking the model: diagnostics

🟡 Intermediate · Previous: [03](03_inference.md) · [Course index](README.md) · Next: [05 · Fixing problems](05_remedies.md)

## The point of diagnostics

Every result in lesson 3 rested on assumptions. Diagnostics test them using the **residuals**, the model's mistakes. A good model's residuals look like noise: no pattern over time, no pattern against the fitted values, no single observation dominating.

We use a better card model from here on: the project's champion, which adds the loss rate four quarters earlier (lesson 7 explains why).

```python
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from ukstress.ingest.build_panel import build_macro, build_panel
from ukstress.features.build import covid_dummy, logit_pct, trailing_loss_rate

panel, macro = build_panel(), build_macro()
y_all = logit_pct(trailing_loss_rate(panel, "credit_card"))
df = pd.DataFrame({
    "y": y_all,
    "d_unemp_L2": macro["unemployment_chg_yoy"].shift(2),
    "bank_rate_L8": macro["bank_rate"].shift(8),
    "covid": covid_dummy(macro.index),
    "y_L4": y_all.shift(4),
}).dropna()
X = sm.add_constant(df.drop(columns="y"))
y = df["y"]
res = sm.OLS(y, X).fit()
warnings.filterwarnings("ignore")  # silence library deprecation notices in this lesson
e = res.resid
print(f"n = {len(y)}, adj R² = {res.rsquared_adj:.3f}")
```

## 1. Heteroskedasticity: is the error variance constant?

**Picture:** residuals plotted against fitted values fan out (bigger misses at higher values).

**Tests:**
- **Breusch–Pagan:** regress $e^2$ on $X$; if $X$ explains the squared residuals, variance depends on the regressors.
- **White:** the same with squares and cross-products of $X$, so it also catches non-linear patterns.

```python
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

print(f"Breusch-Pagan p = {het_breuschpagan(e, X)[1]:.3f}   White p = {het_white(e, X)[1]:.3f}")
```

**Consequence:** coefficients stay unbiased, but classical standard errors are wrong. **Fix:** robust (White/HC) standard errors, or weighted least squares (lesson 5).

## 2. Autocorrelation: are this quarter's errors related to last quarter's?

**Picture:** residuals plotted over time run in long streaks.

**Tests:**
- **Durbin–Watson** (≈ 2 means no first-order autocorrelation; near 0 means strong positive). **Invalid when the model includes a lagged dependent variable**, as ours does.
- **Breusch–Godfrey:** regress residuals on $X$ and lagged residuals; valid with lagged dependent variables and tests several lags.
- **Ljung–Box:** tests whether the first *m* residual autocorrelations are jointly zero.

```python
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

print(f"Durbin-Watson = {durbin_watson(e):.2f} (not valid here)")
print(f"Breusch-Godfrey p (4 lags) = {acorr_breusch_godfrey(res, nlags=4)[1]:.2e}")
print(f"autocorrelation at lags 1, 4, 8: {e.autocorr(1):.2f}, {e.autocorr(4):.2f}, {e.autocorr(8):.2f}")
```

Strong autocorrelation at lag 1 that dies away by lag 4–8 is the fingerprint of the **overlapping target**: each trailing four-quarter loss rate shares three quarters with its neighbour. This does not bias the coefficients but makes classical standard errors far too small. **Fix:** Newey–West (HAC) standard errors (lesson 7).

## 3. Normality of the errors

Matters for exact t and F tests in small samples and for prediction intervals that assume normal errors.

```python
from statsmodels.stats.stattools import jarque_bera

jb_p = jarque_bera(e)[1]
print(f"Jarque-Bera p = {jb_p:.4f}, skew = {stats.skew(e):.2f}, excess kurtosis = {stats.kurtosis(e):.2f}")
```

**Fix when it fails:** a transform of the target (lesson 5), or build prediction intervals by resampling the actual residuals instead of assuming normality (lesson 9). With 125 observations the central limit theorem protects the coefficient tests reasonably well.

## 4. Functional form: Ramsey's RESET

Adds powers of the fitted values to the model. If $\hat y^2$ helps explain $y$, the linear form is missing curvature or an interaction.

```python
from statsmodels.stats.diagnostic import linear_reset

print(f"RESET p = {linear_reset(res, power=2, use_f=True).pvalue:.3f}")
```

## 5. Multicollinearity: are the regressors too similar?

When regressors are highly correlated, the model cannot tell their effects apart: standard errors balloon and coefficients swing with small data changes. The **variance inflation factor** for regressor *j* is

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ comes from regressing $x_j$ on the other regressors. VIF = 1 means no overlap; above 5–10 is a warning.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vifs = {c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns) if c != "const"}
print({k: round(v, 2) for k, v in vifs.items()})
```

Multicollinearity is not a violation of an assumption (unless it is perfect); it is a lack of information. Fixes: drop one of the overlapping variables, combine them, get more data, or use Ridge regression (lesson 6).

## 6. Influential observations

Some quarters pull the line more than others.
- **Leverage** $h_{ii}$ (diagonal of the hat matrix $H = X(X'X)^{-1}X'$): how unusual a quarter's *regressors* are. Average leverage is $k/n$.
- **Cook's distance:** combines leverage and residual size; how much *all* fitted values move if the quarter is dropped. Rule of thumb: $D_i > 4/n$.
- **DFBETAS:** how much *each coefficient* moves (in standard errors) if the quarter is dropped. Rule of thumb: $|DFBETAS| > 2/\sqrt{n}$.

```python
infl = res.get_influence()
table = pd.DataFrame({"leverage": infl.hat_matrix_diag, "cooks_d": infl.cooks_distance[0],
                      "dfbeta_unemp": infl.dfbetas[:, list(X.columns).index("d_unemp_L2")]}, index=y.index)
print(table.sort_values("cooks_d", ascending=False).head(5).round(3))
print(f"quarters with Cook's D > 4/n: {(table.cooks_d > 4 / len(y)).sum()}")
```

**What to do with influential points:** never delete them silently. Ask whether they are errors (fix them), a different regime (model it, e.g. the COVID dummy), or genuine extremes (keep them, and check the conclusions survive without them):

```python
keep = table.cooks_d <= 4 / len(y)
res_drop = sm.OLS(y[keep], X[keep]).fit()
print(pd.DataFrame({"all quarters": res.params, "without influential": res_drop.params}).round(3))
```

## 7. Residuals over time

For time series, always plot the residuals in date order. Streaks, trends or a shift in variance signal a missing driver or a structural break (lesson 7).

```python
print(e.groupby(e.index.year).mean().round(2).loc[2005:2012])
```

## Diagnostic checklist

| Check | Tool | Our card model |
|---|---|---|
| Heteroskedasticity | Breusch–Pagan, White | BP p ≈ 0.51 (fine); White p ≈ 0.02 (mild, non-linear) → robust standard errors |
| Autocorrelation | Breusch–Godfrey, Ljung–Box (not DW with lagged y) | Strong (0.82 at lag 1, 0.03 by lag 8), from the overlapping target → Newey–West |
| Normality | Jarque–Bera, Q-Q plot | Rejected: skew ≈ −0.8 → resample actual residuals for intervals |
| Functional form | RESET | p ≈ 0.07: no strong evidence of missing curvature |
| Multicollinearity | VIF | All below 1.7 |
| Influence | Leverage, Cook's D, DFBETAS | 6 flagged quarters (mostly 1995); without them the unemployment effect falls from 0.148 to 0.113 but keeps its sign |

The full battery for all four portfolios, with a decision for each finding, is in [notebook 04](../notebooks/04_diagnostics.ipynb).

## Practice

**Q1.** A model's Durbin–Watson is 1.95, but it includes $y_{t-1}$ as a regressor. Can you conclude there is no autocorrelation?

<details><summary>Answer</summary>

No. With a lagged dependent variable, DW is biased towards 2 and can hide autocorrelation. Use Breusch–Godfrey (or Durbin's h).
</details>

**Q2.** Two regressors have VIFs of 25. Both coefficients are insignificant, but an F-test that both are zero rejects strongly. What is happening?

<details><summary>Answer</summary>

Classic multicollinearity: together they clearly matter, but the data cannot separate their individual effects, so each standard error is huge. Keep one, combine them, or use Ridge.
</details>

**Q3.** One quarter has a Cook's distance of 0.8 (threshold 4/n ≈ 0.03). List three legitimate responses, and one illegitimate one.

<details><summary>Answer</summary>

Legitimate: check it for a data error and correct it; model the cause (a dummy for a known event or regime); report the results with and without it, or use robust regression (Huber). Illegitimate: delete it because it makes the results look worse.
</details>

**Q4.** Heteroskedasticity makes classical standard errors wrong. Does it also bias the coefficients?

<details><summary>Answer</summary>

No: OLS stays unbiased (exogeneity does not depend on constant variance). It is no longer the most efficient estimator (WLS can do better), and the classical standard errors are wrong.
</details>
