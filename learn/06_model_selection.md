# 06 · Choosing a model: overfitting, information criteria, cross-validation, regularisation

🟠 Advanced · Previous: [05](05_remedies.md) · [Course index](README.md) · Next: [07 · Time series](07_time_series_regression.md)

## The problem

There are 7 candidate drivers and 7 possible lags each: 49 features, for about 125 quarters of data. Put them all in and the model fits history beautifully and forecasts terribly. This is **overfitting**: the model learns the noise in this particular sample.

```python
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ukstress.ingest.build_panel import build_macro, build_panel
from ukstress.features.build import DRIVERS, all_lags, covid_dummy, logit_pct, trailing_loss_rate

warnings.filterwarnings("ignore")
panel, macro = build_panel(), build_macro()
y_all = logit_pct(trailing_loss_rate(panel, "credit_card"))
F = all_lags(macro).join(covid_dummy(macro.index)).assign(y_L4=y_all.shift(4))
data = F.join(y_all.rename("y")).loc["1995Q2":].dropna()
X_all, y = data.drop(columns="y"), data["y"]
macro_cols = [c for c in X_all.columns if c not in ("covid", "y_L4")]
print(X_all.shape)
```

## 1. In-sample fit always improves with more variables

```python
small = ["y_L4", "unemployment_chg_yoy_L2", "bank_rate_L8", "covid"]
for name, cols in [("4 features", small), ("51 features", list(X_all.columns))]:
    r = sm.OLS(y, sm.add_constant(X_all[cols])).fit()
    print(f"{name:>12}: R² = {r.rsquared:.3f}, adj R² = {r.rsquared_adj:.3f}, AIC = {r.aic:.1f}, BIC = {r.bic:.1f}")
```

R² can only go up when you add a variable. Adjusted R² and the information criteria **penalise complexity**:

$$\text{AIC} = -2\ln L + 2k \qquad \text{BIC} = -2\ln L + k\ln n$$

Lower is better. BIC's penalty ($\ln n \approx 4.8$ per parameter here) is harsher than AIC's (2), so BIC picks smaller models. BIC is consistent (picks the true model as n → ∞ if it is among the candidates); AIC aims at the best predictor.

**Look at the result: adjusted R² and AIC both *prefer* the 51-feature model.** Their penalties are too mild to stop overfitting at this ratio of features to observations. Only BIC strongly prefers the small model (−85 vs +5). The decisive test is the next one.

## 2. The honest test: data the model has not seen

For time series, **never shuffle**. A random K-fold split would train on 2010 and test on 2008, letting the model peek at the future. Use expanding windows: train on the past, test on the next block.

```python
tscv = TimeSeriesSplit(n_splits=5, test_size=16)

def cv_rmse(make_model, cols):
    errs = []
    for tr, te in tscv.split(X_all):
        m = make_model().fit(X_all.iloc[tr][cols], y.iloc[tr])
        pred = 100 / (1 + np.exp(-m.predict(X_all.iloc[te][cols])))
        actual = 100 / (1 + np.exp(-y.iloc[te]))
        errs.append(np.sqrt(np.mean((pred - actual) ** 2)))
    return np.mean(errs)

print(f"OLS, 4 features:  {cv_rmse(LinearRegression, small):.2f} pp")
print(f"OLS, 51 features: {cv_rmse(LinearRegression, list(X_all.columns)):.2f} pp")
```

The 51-feature model fits history better but forecasts about three times worse on quarters it has not seen (about 3.7 vs 1.2 percentage points of loss rate). That gap is the cost of overfitting, and it is invisible in-sample.

## 3. Searching over specifications

With a manageable number of candidates, fit them all and rank by BIC, subject to rules. Here, every single driver at every lag (the project does all combinations of up to three: about 19,000 models):

```python
rows = []
for d in DRIVERS:
    for lag in (0, 1, 2, 3, 4, 6, 8):
        cols = ["y_L4", "covid", f"{d}_L{lag}"]
        r = sm.OLS(y, sm.add_constant(X_all[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        b = r.params[f"{d}_L{lag}"]
        rows.append({"feature": f"{d}_L{lag}", "coef": b, "p": r.pvalues[f"{d}_L{lag}"], "bic": r.bic,
                     "sign_ok": np.sign(b) == DRIVERS[d]["sign"]})
ranked = pd.DataFrame(rows).sort_values("bic")
print(ranked.head(6).round(3).to_string(index=False))
```

Searching many models has its own risk: with enough candidates, something fits by chance (**data mining**). Defences: restrict candidates to economically sensible drivers, demand significance, prefer simple models, and validate on data not used for selection (lesson 9 re-runs the whole selection on pre-2008 data and tests it on the crisis).

## 4. Economic sign rules

A model can be statistically excellent and economically absurd. Some of the best-fitting single drivers above have the **wrong sign**, and are highly significant: the unemployment *level* two years earlier comes out negative, and so do *changes* in Bank Rate two to three quarters earlier. These are accidents of history: Bank Rate was cut to 0.5% just as losses peaked in 2009, and raised in 2022–23 while losses were at record lows. A stress test built on such a coefficient would conclude that a rate rise *reduces* losses. Banks therefore **reject any model whose coefficients contradict economics**, before looking at fit:

```python
print(ranked[~ranked.sign_ok].head(4)[["feature", "coef", "p"]].round(3).to_string(index=False))
```

## 5. Regularisation: shrink instead of select

Regularisation adds a penalty on coefficient size to the least-squares objective:

| Method | Objective | Effect |
|---|---|---|
| Ridge (L2) | $\lVert y - X\beta\rVert^2 + \lambda \sum \beta_j^2$ | Shrinks all coefficients; never exactly zero; great with collinear features |
| Lasso (L1) | $\lVert y - X\beta\rVert^2 + \lambda \sum \lvert\beta_j\rvert$ | Sets many coefficients exactly to zero: automatic feature selection |
| Elastic Net | mix of both | Lasso-like selection, more stable with correlated features |

**Standardise features first**, otherwise the penalty hits variables measured in small units harder. Choose λ by time-series cross-validation.

```python
lasso = make_pipeline(StandardScaler(), LassoCV(cv=TimeSeriesSplit(4), max_iter=50_000))
print(f"Ridge, 51 features: {cv_rmse(lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))), list(X_all.columns)):.2f} pp")
print(f"Lasso, 51 features: {cv_rmse(lambda: make_pipeline(StandardScaler(), LassoCV(cv=TimeSeriesSplit(4), max_iter=50_000)), list(X_all.columns)):.2f} pp")

lasso.fit(X_all, y)
coef = pd.Series(lasso[-1].coef_ / lasso[0].scale_, index=X_all.columns)
used = coef[macro_cols][coef[macro_cols].abs() > 1e-8]
wrong = [c for c in used.index if np.sign(used[c]) != DRIVERS[c.rsplit("_L", 1)[0]]["sign"]]
print(f"Lasso uses {len(used)} macro features; {len(wrong)} have the wrong economic sign")
```

## 6. Sign-constrained Lasso

Penalised models do not know economics either. A neat fix: multiply each feature by its required sign, then fit a Lasso with **all coefficients forced to be non-negative** (`positive=True`). It can only use a driver in the economically correct direction.

```python
from sklearn.linear_model import Lasso

sign = pd.Series({c: DRIVERS[c.rsplit("_L", 1)[0]]["sign"] for c in macro_cols} | {"covid": -1, "y_L4": 1})
Xs = X_all * sign[X_all.columns]
scaler = StandardScaler().fit(Xs)
pl = Lasso(alpha=0.01, positive=True, max_iter=50_000).fit(scaler.transform(Xs), y)
kept = pd.Series(pl.coef_, index=X_all.columns)
print("kept:", list(kept[macro_cols][kept[macro_cols] > 0].sort_values(ascending=False).index))
```

Ridge and Lasso both forecast much better than the unpenalised 51-feature OLS (about 2.4 and 1.3 pp), but the Lasso uses 17 drivers and **13 of them have the wrong sign**. The sign-constrained version keeps only correctly signed drivers, led by Bank Rate two years earlier and the change in unemployment two to three quarters earlier.

In [notebook 05](../notebooks/05_regularisation_panel.ipynb) the sign-constrained Lasso, free to choose among all 49 macro features, picks exactly the drivers the champion model uses: an independent check on the selection.

## Key takeaways
- In-sample fit always rewards complexity. Adjusted R² and even AIC can still prefer an overfit model; BIC and out-of-sample tests are stricter.
- Time-series cross-validation must respect time order.
- Searching many models risks fitting noise; constrain candidates and validate out of sample.
- Ridge shrinks, Lasso selects, Elastic Net mixes; standardise first; tune λ by CV.
- Economic sign rules (or sign-constrained estimation) are essential when a model will be used beyond the data, as in stress testing.

## Practice

**Q1.** Model A has 3 parameters and log-likelihood −50; model B has 6 parameters and log-likelihood −46; n = 125. Which does AIC prefer? BIC?

<details><summary>Answer</summary>

AIC: A = 100 + 6 = 106; B = 92 + 12 = 104 → B. BIC: A = 100 + 3 ln125 ≈ 114.5; B = 92 + 6 ln125 ≈ 121.0 → A. BIC's heavier penalty favours the smaller model.
</details>

**Q2.** Why is shuffled 5-fold cross-validation misleading for this data?

<details><summary>Answer</summary>

Quarters are autocorrelated and the trailing target overlaps, so a test quarter's neighbours (which share most of its information) sit in the training set. The model is effectively tested on data it has seen, and future information leaks into training. The CV error is far too optimistic.
</details>

**Q3.** Two features are almost identical. What do Ridge and Lasso each tend to do with them?

<details><summary>Answer</summary>

Ridge splits the weight roughly equally between them. Lasso tends to pick one and set the other to zero, somewhat arbitrarily, and the choice can flip between samples. Elastic Net keeps both with shared weight, which is more stable.
</details>

**Q4.** An unconstrained Lasso has the best cross-validated error but several wrong-signed coefficients. Should a bank use it for a stress test?

<details><summary>Answer</summary>

No. Cross-validation tests performance on historical conditions; a stress test projects conditions outside them. Wrong-signed coefficients can offset the stress exactly where it matters. In notebook 05 the unconstrained Lasso projected a card peak about half the champion's. Use a sign-constrained version or the rule-based champion.
</details>
