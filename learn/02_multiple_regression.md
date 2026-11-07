# 02 · Many variables at once: multiple regression

🟢 Beginner · Previous: [01](01_simple_regression.md) · [Course index](README.md) · Next: [03 · Inference](03_inference.md)

## Why one variable is not enough

Lesson 1 found card losses rise with unemployment. But Bank Rate also changed over the same years, and so did house prices. If two drivers move together, a one-variable regression gives the first driver credit for the second one's effect: **omitted-variable bias**. Multiple regression estimates each effect *holding the others constant*.

$$y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \dots + \beta_k x_{ki} + \varepsilon_i$$

## Matrix form

Stack the $n$ observations: $y$ is an $n \times 1$ vector, $X$ is an $n \times (k+1)$ matrix whose first column is all ones (the intercept), and

$$y = X\beta + \varepsilon$$

Minimising $\lVert y - X\beta \rVert^2$ gives the **normal equation**:

$$X'X\hat\beta = X'y \quad\Rightarrow\quad \hat\beta = (X'X)^{-1}X'y$$

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ukstress.ingest.build_panel import build_macro, build_panel
from ukstress.features.build import covid_dummy, trailing_loss_rate

panel, macro = build_panel(), build_macro()
df = pd.DataFrame({
    "loss_rate": trailing_loss_rate(panel, "credit_card"),
    "d_unemp_L4": macro["unemployment_chg_yoy"].shift(4),
    "bank_rate_L8": macro["bank_rate"].shift(8),
    "hpi_growth_L4": macro["house_price_growth_yoy"].shift(4),
}).dropna()

X = np.column_stack([np.ones(len(df)), df[["d_unemp_L4", "bank_rate_L8", "hpi_growth_L4"]].values])
y = df["loss_rate"].values
beta = np.linalg.solve(X.T @ X, X.T @ y)        # solve X'X b = X'y (never invert explicitly)
print(dict(zip(["const", "d_unemp_L4", "bank_rate_L8", "hpi_growth_L4"], beta.round(3))))
```

In practice libraries solve this with a QR decomposition, which is more accurate than forming $X'X$ (lesson 2 of [notebook 02](../notebooks/02_ols_from_scratch.ipynb) shows why: forming $X'X$ squares the condition number).

## Omitted-variable bias, seen directly

Compare the unemployment slope alone with the slope once the other drivers are included:

```python
alone = sm.OLS(y, sm.add_constant(df["d_unemp_L4"])).fit().params["d_unemp_L4"]
full = sm.OLS(y, sm.add_constant(df[["d_unemp_L4", "bank_rate_L8", "hpi_growth_L4"]])).fit()
print(f"unemployment slope alone: {alone:.3f}   with controls: {full.params['d_unemp_L4']:.3f}")
print(full.params.round(3))
```

Here the unemployment slope barely moves (about 2.02 alone, 2.05 with controls): in this sample, Bank Rate and house price growth are only weakly correlated with the *change* in unemployment, so leaving them out does little damage *to the unemployment coefficient*. Notice the Bank Rate coefficient is negative, implying higher rates *reduce* card losses. That is not economics, it is history (rates were cut to 0.5% just as losses peaked in 2009). Lesson 6 shows how banks stop such a coefficient reaching a stress test.

The bias formula: if the true model has $x_1$ and $x_2$ but you leave out $x_2$,

$$E[\hat\beta_1^{\text{short}}] = \beta_1 + \beta_2 \cdot \delta, \qquad \delta = \text{slope of } x_2 \text{ on } x_1$$

So the bias is large when the omitted variable matters ($\beta_2$ big) **and** is correlated with the included one ($\delta$ big).

## What "holding others constant" means: Frisch–Waugh–Lovell

The coefficient on $x_1$ in a multiple regression equals the slope from this two-step recipe:
1. Regress $y$ on the other regressors; keep the residuals (the part of $y$ the others cannot explain).
2. Regress $x_1$ on the other regressors; keep the residuals (the part of $x_1$ that is "new information").
3. Regress the first residuals on the second.

```python
others = sm.add_constant(df[["bank_rate_L8", "hpi_growth_L4"]])
ry = sm.OLS(y, others).fit().resid
rx = sm.OLS(df["d_unemp_L4"], others).fit().resid
print(f"FWL slope: {(rx @ ry) / (rx @ rx):.4f}   multiple-regression coefficient: {full.params['d_unemp_L4']:.4f}")
```

They match exactly. This is the precise meaning of "the effect of unemployment, holding Bank Rate and house prices fixed".

## Dummy (indicator) variables

A dummy is 1 in some quarters and 0 otherwise. Its coefficient shifts the intercept for those quarters. In 2020–21, furlough and payment holidays kept losses low even as GDP collapsed; a COVID dummy lets the model treat that period separately instead of distorting every slope.

```python
df["covid"] = covid_dummy(macro.index).reindex(df.index)
with_covid = sm.OLS(y, sm.add_constant(df[["d_unemp_L4", "bank_rate_L8", "hpi_growth_L4", "covid"]])).fit()
print(with_covid.params.round(3))
```

The COVID dummy is about −2.4: in 2020–21 the loss rate was 2.4 points lower than the drivers alone imply, and including it changes the other slopes.

Categorical variables with *m* categories need *m − 1* dummies (one category is the baseline absorbed by the intercept). Including all *m* plus an intercept makes $X'X$ singular: the **dummy-variable trap**.

## Interactions

An interaction lets one variable's effect depend on another. With $x_1 \times x_2$ in the model:

$$\frac{\partial y}{\partial x_1} = \beta_1 + \beta_3 x_2$$

Example question: does unemployment hurt more when house prices are falling? (For mortgages this is the "double trigger" theory: default needs both a payment shock and no equity.)

```python
df["u_x_hpi"] = (df["d_unemp_L4"] - df["d_unemp_L4"].mean()) * (df["hpi_growth_L4"] - df["hpi_growth_L4"].mean())
inter = sm.OLS(y, sm.add_constant(df[["d_unemp_L4", "hpi_growth_L4", "u_x_hpi"]])).fit()
print(inter.params.round(4), inter.pvalues["u_x_hpi"].round(3), sep="\n")
```

The interaction coefficient is negative and significant (p ≈ 0.002). Because both variables are centred, it reads: the unemployment effect is about 1.72 at average house price growth, and **larger when house prices are falling** (each point of weaker house price growth adds about 0.085 to the unemployment effect). Households with falling home equity have less room to refinance or borrow their way out of a job loss.

**Centre variables before interacting them** (subtract the mean). Otherwise the main-effect coefficients describe the effect when the other variable is zero, which may be an impossible value, and they become highly collinear with the interaction.

## Key takeaways
- Multiple regression estimates each effect holding the others fixed; leaving out a correlated driver biases the rest.
- $\hat\beta = (X'X)^{-1}X'y$, but compute it with a solver or QR, never an explicit inverse.
- FWL: a coefficient is the slope of "unexplained y" on "unexplained x".
- Dummies shift intercepts; interactions let slopes vary; centre before interacting.

## Practice

**Q1.** The unemployment slope changes when Bank Rate is added. Using the omitted-variable-bias formula, what must be true of Bank Rate for the short-regression slope to be biased *upwards*?

<details><summary>Answer</summary>

Bias = β₂ × δ. Upward bias needs β₂ and δ to have the same sign: for example, Bank Rate having a positive effect on losses *and* being positively correlated with the unemployment change in the sample (or both negative).
</details>

**Q2.** A model has an intercept and dummies for all four quarters of the year. What goes wrong, and what are two fixes?

<details><summary>Answer</summary>

The four dummies sum to the intercept column, so X is perfectly collinear (dummy-variable trap) and X'X cannot be inverted. Fix: drop one quarter's dummy (it becomes the baseline), or drop the intercept.
</details>

**Q3.** In `y = b0 + b1·u + b2·h + b3·u·h`, with u and h centred, how do you interpret b1?

<details><summary>Answer</summary>

The effect of a one-unit increase in u when h is at its average (h = 0 after centring). The effect at other values of h is b1 + b3·h.
</details>

**Q4.** Why does FWL matter in practice, beyond being a neat result?

<details><summary>Answer</summary>

It gives the partial regression (added-variable) plot used in diagnostics (lesson 4): plotting the two residual series shows exactly which observations drive a coefficient. It also underlies fixed-effects estimation in panels (lesson 8): demeaning by group is FWL with group dummies.
</details>
