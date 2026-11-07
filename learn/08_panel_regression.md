# 08 · Panel data: pooling, fixed effects and the right standard errors

🟠 Advanced · Previous: [07](07_time_series_regression.md) · [Course index](README.md) · Next: [09 · Uncertainty and validation](09_uncertainty_validation.md)

## What is a panel?

Several units (people, firms, countries, or here **loan portfolios**) observed over the same periods. We have four portfolios × about 125 quarters. Panels offer more data per coefficient, but only if the units genuinely share behaviour.

```python
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from ukstress.config import PORTFOLIOS
from ukstress.ingest.build_panel import build_macro, build_panel
from ukstress.features.build import covid_dummy, logit_pct, trailing_loss_rate

warnings.filterwarnings("ignore")
panel, macro = build_panel(), build_macro()
frames = []
for p in PORTFOLIOS:
    y = logit_pct(trailing_loss_rate(panel, p))
    frames.append(pd.DataFrame({
        "portfolio": p, "y": y, "y_L4": y.shift(4),
        "d_unemp_L2": macro["unemployment_chg_yoy"].shift(2),
        "bank_rate_L8": macro["bank_rate"].shift(8),
        "covid": covid_dummy(macro.index),
    }).loc["1995Q2":].dropna())
long = pd.concat(frames).reset_index()
print(long.groupby("portfolio")["y"].agg(["count", "mean"]).round(2))
```

## 1. Pooled OLS: one intercept for everyone

Stacking everything with a single intercept assumes all portfolios have the same average loss rate. They obviously do not (cards ≈ 3%, mortgages ≈ 0.05%), so pooled OLS mixes up "which portfolio" with "what the economy did".

## 2. Fixed effects: one intercept per unit

$$y_{it} = \alpha_i + \beta' x_{it} + \varepsilon_{it}$$

$\alpha_i$ absorbs everything constant about portfolio *i* (its typical riskiness). Equivalent ways to estimate it:
- **Dummy variables:** one dummy per portfolio, no common intercept.
- **Within transformation:** subtract each portfolio's mean from its *y* and *x* (this is the FWL theorem from lesson 2 applied to the dummies).

```python
FE = pd.get_dummies(long["portfolio"], dtype=float)
X_pooled = sm.add_constant(long[["d_unemp_L2", "bank_rate_L8", "covid", "y_L4"]])
X_fe = pd.concat([FE, long[["d_unemp_L2", "bank_rate_L8", "covid", "y_L4"]]], axis=1)
pooled = sm.OLS(long["y"], X_pooled).fit()
fe = sm.OLS(long["y"], X_fe).fit()
print(pd.DataFrame({"pooled": pooled.params, "fixed effects": fe.params}).round(3))

# F-test: do the portfolio intercepts differ?
q = FE.shape[1] - 1
F = ((pooled.ssr - fe.ssr) / q) / (fe.ssr / fe.df_resid)
print(f"F-test for fixed effects: F = {F:.1f}, p = {stats.f.sf(F, q, fe.df_resid):.1e}")
```

Look at the persistence coefficient: pooled OLS estimates ρ ≈ 1.0, fixed effects ≈ 0.85. Without portfolio intercepts, the lagged loss rate is the only thing that can tell a mortgage from a card, so it soaks up the level differences and looks like a unit root. Omitting the fixed effects biases the dynamics, not just the intercept. (The means in the first table are on the logit scale.)

**Random effects** treats $\alpha_i$ as random draws uncorrelated with $x$; it is efficient when that holds (Hausman test). With four portfolios that are clearly different by design, fixed effects is the natural choice.

## 3. Do the units share the same slopes?

Fixed effects let intercepts differ but force the **same slope** on every portfolio. Test it: allow every slope to differ by portfolio (interact each regressor with each portfolio dummy) and F-test the restriction. This is a Chow test across units.

```python
cols = ["d_unemp_L2", "bank_rate_L8", "covid", "y_L4"]
X_het = pd.concat([FE] + [long[cols].mul(FE[p], axis=0).add_suffix(f"_{p}") for p in FE], axis=1)
het = sm.OLS(long["y"], X_het).fit()
q = X_het.shape[1] - X_fe.shape[1]
F = ((fe.ssr - het.ssr) / q) / (het.ssr / het.df_resid)
print(f"slope homogeneity, all four portfolios: F = {F:.1f}, p = {stats.f.sf(F, q, het.df_resid):.1e}")
```

Rejected decisively for all four portfolios, so separate models per portfolio are justified. Repeat for the two unsecured consumer portfolios only:

```python
sub = long[long.portfolio.isin(["credit_card", "other_consumer"])].reset_index(drop=True)
fe2 = pd.get_dummies(sub["portfolio"], dtype=float)
base = pd.concat([fe2] + [sub[[c]].mul(fe2[p], axis=0).add_suffix(f"_{p}") for p in fe2 for c in ["covid", "y_L4"]], axis=1)
common = pd.concat([base, sub[["d_unemp_L2", "bank_rate_L8"]]], axis=1)
separate = pd.concat([base] + [sub[["d_unemp_L2", "bank_rate_L8"]].mul(fe2[p], axis=0).add_suffix(f"_{p}") for p in fe2], axis=1)
rc, rs = sm.OLS(sub["y"], common).fit(), sm.OLS(sub["y"], separate).fit()
q = separate.shape[1] - common.shape[1]
F = ((rc.ssr - rs.ssr) / q) / (rs.ssr / rs.df_resid)
print(f"slope homogeneity, cards vs other consumer: F = {F:.2f}, p = {stats.f.sf(F, q, rs.df_resid):.3f}")
```

Cards and personal loans *can* share their macro sensitivities, which gives the weaker personal-loan model twice the data. The project saves this pooled model as a challenger.

## 4. Standard errors in panels

Panel residuals are correlated in two ways:
1. **Across units in the same period:** every portfolio sees the same economy each quarter, so their shocks move together.
2. **Within a unit over time:** autocorrelation (lesson 7).

| Standard errors | Allows for |
|---|---|
| Classical | Neither |
| Clustered by unit | Arbitrary correlation over time within a unit (needs many units; we have 2–4, too few) |
| Clustered by period | Correlation across units within a period |
| **Driscoll–Kraay** | Both: cross-sectional correlation *and* autocorrelation (Newey–West applied to period-summed scores) |

```python
t_index = pd.factorize(sub["quarter"])[0]
se = pd.DataFrame({
    "classical": rc.bse,
    "cluster by quarter": sm.OLS(sub["y"], common).fit(cov_type="cluster", cov_kwds={"groups": sub["quarter"].astype(str)}).bse,
    "Driscoll-Kraay": sm.OLS(sub["y"], common).fit(cov_type="nw-groupsum", cov_kwds={"time": t_index, "maxlags": 4}).bse,
}).loc[["d_unemp_L2", "bank_rate_L8"]]
print(se.round(4))
```

A rule worth remembering: **cluster at the level where the shocks are shared.** Clustering by portfolio with two portfolios would be meaningless (cluster-robust errors need many clusters).

## 5. Why not time fixed effects too?

Two-way fixed effects (unit *and* period dummies) are popular, but here a quarter dummy would absorb every macro variable, because all portfolios see the same unemployment and Bank Rate in a given quarter. The coefficients we care about would not be identified. Time effects suit panels where the regressor of interest varies across units within a period (for example regional unemployment).

## Key takeaways
- Fixed effects remove each unit's constant differences; pooled OLS ignores them.
- Test slope homogeneity before pooling; pool only units that pass.
- Panel errors are correlated across units and over time: Driscoll–Kraay handles both.
- Time fixed effects absorb any regressor common to all units in a period.

## Practice

**Q1.** You pool 30 UK regions' unemployment and default rates with region fixed effects. Which standard errors would you use, and why?

<details><summary>Answer</summary>

With 30 regions, clustering by region (allowing within-region autocorrelation) is feasible. Regions also share national shocks each quarter, so two-way clustering (region and quarter) or Driscoll–Kraay is safer.
</details>

**Q2.** Why does the within transformation give the same slope as including unit dummies?

<details><summary>Answer</summary>

By the FWL theorem, the slope on x in a regression with unit dummies equals the slope from regressing y-residuals on x-residuals after removing the dummies' effect. Regressing on unit dummies just subtracts unit means, which is exactly the within transformation.
</details>

**Q3.** The slope-homogeneity F-test rejects strongly for four portfolios but not for two of them. What modelling strategy follows?

<details><summary>Answer</summary>

Do not pool all four. Estimate separate models, or a partial pool: pool the two homogeneous portfolios (more data, more precise shared slopes) and keep the others separate. That is what the project does for cards and other consumer credit.
</details>
