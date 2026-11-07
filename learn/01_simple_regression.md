# 01 · The straight line: simple linear regression

🟢 Beginner · [Course index](README.md) · Next: [02 · Many variables](02_multiple_regression.md)

## The question

When unemployment rises, do credit-card losses rise too, and **by how much**? A chart can show that two things move together. Regression puts a number on it: "each 1-point rise in unemployment goes with an X-point rise in card losses".

## The model

Simple linear regression assumes

$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$$

| Symbol | Meaning here |
|---|---|
| $y_i$ | card loss rate in quarter *i* (%) |
| $x_i$ | change in unemployment over the previous year, measured a year earlier (pp) |
| $\beta_0$ | intercept: the expected loss rate when $x = 0$ |
| $\beta_1$ | slope: how much $y$ changes, on average, for a one-unit increase in $x$ |
| $\varepsilon_i$ | error: everything else that moves losses |

## Load the data

```python
import numpy as np
import pandas as pd
from ukstress.ingest.build_panel import build_macro, build_panel
from ukstress.features.build import trailing_loss_rate

panel, macro = build_panel(), build_macro()
df = pd.DataFrame({
    "loss_rate": trailing_loss_rate(panel, "credit_card"),       # % of balances written off over 4 quarters
    "d_unemp": macro["unemployment_chg_yoy"].shift(4),            # change in unemployment, one year earlier
}).dropna()
print(df.describe().round(2))
```

## Least squares: the "best" line

Many lines could go through the cloud of points. Ordinary least squares (OLS) picks the line that makes the **sum of squared vertical distances** (residuals) as small as possible:

$$\min_{\beta_0, \beta_1} \sum_i (y_i - \beta_0 - \beta_1 x_i)^2$$

Setting the derivatives to zero gives a closed form:

$$\hat\beta_1 = \frac{\text{Cov}(x, y)}{\text{Var}(x)}, \qquad \hat\beta_0 = \bar y - \hat\beta_1 \bar x$$

**Why squares?** Squaring makes every error positive, punishes big misses more than small ones, and gives a formula you can solve exactly. (Absolute errors give *median* regression, covered in lesson 9.)

```python
x, y = df["d_unemp"].values, df["loss_rate"].values
b1 = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
b0 = y.mean() - b1 * x.mean()
print(f"intercept = {b0:.3f}, slope = {b1:.3f}")
```

Check against a library:

```python
import statsmodels.api as sm

res = sm.OLS(y, sm.add_constant(x)).fit()
print(res.params.round(3))   # [intercept, slope]: identical to the formula
```

## Reading the result

The slope says: **when unemployment rose by 1 percentage point over a year, card losses a year later were about `b1` points higher on average.** The intercept is the average loss rate in a year when unemployment did not change.

Three cautions that apply to every regression:
1. **Units matter.** The slope is "percentage points of loss rate per percentage point of unemployment". Rescale $x$ and the slope rescales.
2. **"On average", not "always".** Individual quarters scatter around the line; that scatter is $\varepsilon$.
3. **Association, not causation.** The slope describes how the two moved together in this data. Causation needs an argument (here: job losses → missed payments → write-offs) and ideally ruling out other explanations. Lesson 2 starts on that.

## How good is the line? R²

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2}$$

The share of the variation in $y$ that the line explains. In simple regression it equals the squared correlation.

```python
y_hat = b0 + b1 * x
r2 = 1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"R² = {r2:.3f}, correlation² = {np.corrcoef(x, y)[0, 1] ** 2:.3f}")
```

Here R² is about 0.41: the unemployment change alone explains about 40% of the variation in card losses. That does not mean the slope is wrong. It means much of what drives losses is elsewhere (lending standards, the 2004–07 insolvency boom, the loss rate's own persistence), which later lessons add.

## Residuals: the first diagnostic

The residuals $e_i = y_i - \hat y_i$ are what the model misses. Always look at them. A pattern in the residuals is a pattern the model failed to capture.

```python
df["residual"] = y - y_hat
print(df["residual"].groupby(df.index.year).mean().loc[2003:2012].round(2))
```

From 2004 to 2011 the residuals are positive every year and grow to about +3.5 points in 2009: losses were higher than unemployment alone explains, first in the 2004–07 personal-insolvency boom, then in the crisis. Residuals that run in long same-sign streaks rather than bouncing randomly mean the model is missing something persistent. Lesson 7 explains why that matters so much for time series.

## Key takeaways
- OLS chooses the line that minimises squared residuals; the slope is Cov(x, y) / Var(x).
- The slope is an average association in the units of the data, not proof of causation.
- R² measures fit, not correctness. Always inspect residuals.

## Practice

**Q1.** The slope is about 2.0. If unemployment rises from 4% to 6% over a year, what does the model predict for the card loss rate a year later, compared with a year when unemployment is flat?

<details><summary>Answer</summary>

The change is 2 pp, so the predicted loss rate is about 2 × 2.0 = 4 percentage points higher than in a flat year (your exact slope may differ slightly as the data updates).
</details>

**Q2.** If you measured unemployment change in *basis points* (1 pp = 100 bp), what happens to the slope and to R²?

<details><summary>Answer</summary>

The slope is divided by 100 (each basis point is a hundredth of the effect). R² is unchanged: rescaling x does not change how well a line fits.
</details>

**Q3.** Why might the loss rate go with unemployment *a year earlier* rather than the same quarter?

<details><summary>Answer</summary>

Write-offs are the end of a process: job loss → missed payments → arrears → default → charge-off, typically 6–12 months or more. Notebook 01 measures this lead directly with cross-correlations (it peaks at about 4 quarters).
</details>

**Q4.** A colleague says "R² is only about 0.4, so unemployment doesn't matter for card losses." What is wrong with that argument?

<details><summary>Answer</summary>

R² measures how much of the total variation one variable explains, not whether its effect is real or large. A driver can have a large, highly significant effect and still leave most variation unexplained because other factors also matter. Significance (lesson 3) and economic size of the slope are the relevant questions.
</details>
