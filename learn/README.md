# Learn linear regression: from first principles to bank stress testing

A ten-lesson course that teaches linear regression **on real data from this project**: UK credit-card losses, unemployment, house prices and Bank Rate. Every lesson explains the idea, shows the maths only where it helps, runs real code, and ends with practice questions (answers hidden under each question).

You do not need to have read the rest of the repository. Each lesson links to the notebook where the same technique is used for real.

## How to use it

```bash
pip install -e ".[dev]"      # from the repository root
python -m ukstress.ingest.build_panel
```

Then read the lessons in order, running the code blocks in a Python session or a notebook. Code blocks within one lesson build on each other. They are executed automatically in CI (`tests/test_learn.py`), so they always run.

## Roadmap

| Level | Lesson | You will learn | Used for real in |
|---|---|---|---|
| 🟢 Beginner | [01 · The straight line](01_simple_regression.md) | What regression does, least squares, slope and intercept, R², correlation vs causation | [Notebook 01](../notebooks/01_eda.ipynb) |
| 🟢 Beginner | [02 · Many variables at once](02_multiple_regression.md) | Matrix form, normal equation, dummy variables, interactions, "holding others constant" (FWL) | [Notebook 02](../notebooks/02_ols_from_scratch.ipynb) |
| 🟡 Intermediate | [03 · How sure are we? Inference](03_inference.md) | Assumptions, standard errors, t and F tests, p-values, confidence vs prediction intervals | [Notebook 02](../notebooks/02_ols_from_scratch.ipynb) |
| 🟡 Intermediate | [04 · Checking the model: diagnostics](04_diagnostics.md) | Residual plots, heteroskedasticity, autocorrelation, normality, multicollinearity, influential points | [Notebook 04](../notebooks/04_diagnostics.ipynb) |
| 🟡 Intermediate | [05 · Fixing problems](05_remedies.md) | Transforms (log, logit, Box-Cox), robust standard errors, WLS, Huber regression, non-linear terms | [Notebook 04](../notebooks/04_diagnostics.ipynb) |
| 🟠 Advanced | [06 · Choosing a model](06_model_selection.md) | Overfitting, AIC/BIC, cross-validation for time series, Ridge, Lasso, Elastic Net, sign constraints | [Notebooks 03, 05](../notebooks/03_model_development.ipynb) |
| 🟠 Advanced | [07 · Regression with time series](07_time_series_regression.md) | Lags, stationarity, spurious regression, cointegration, lagged dependent variables, Newey–West | [Notebooks 01, 03](../notebooks/03_model_development.ipynb) |
| 🟠 Advanced | [08 · Panel data](08_panel_regression.md) | Pooling, fixed effects, slope homogeneity tests, clustered and Driscoll–Kraay errors | [Notebook 05](../notebooks/05_regularisation_panel.ipynb) |
| 🔴 Expert | [09 · Uncertainty and validation](09_uncertainty_validation.md) | Bootstrap, quantile regression, simulation, out-of-time backtests, benchmarks | [Notebooks 06, 07](../notebooks/06_validation.ipynb) |
| 🔴 Expert | [10 · Regression in industry](10_regression_in_industry.md) | Stress testing, model risk management (SS1/23), documentation, monitoring, communicating results | [docs/](../docs/) |

**Extras**
- [Cheat sheet](cheat_sheet.md): every formula and test on two pages
- [Interview questions](interview_questions.md): 40 questions on linear regression, from basic to quant-level, with model answers
- [Glossary](glossary.md)

## The running example

Throughout the course the question is the one this project answers: **how do UK credit-card losses respond to the economy?**

- **Target:** the card loss rate, i.e. the share of card balances written off in a year, in %
- **Drivers:** unemployment, its change over a year, GDP growth, house price growth, Bank Rate
- **Data:** 1993–2026, quarterly, from the Bank of England, ONS and HM Land Registry

Every concept is introduced because this problem needs it. For example, lesson 7 exists because regressing one trending series on another can produce nonsense, and this data trends.
