# Linear regression interview questions

[Course index](README.md) · 40 questions, grouped by level, with model answers. Lesson references in brackets.

## 🟢 Fundamentals

<details><summary><b>1. What does OLS minimise, and why squares?</b></summary>

The sum of squared residuals. Squares make errors positive, penalise large errors more, are differentiable, and give a closed-form solution; under normal errors OLS is also maximum likelihood. Absolute errors give median (quantile) regression instead. [01]
</details>

<details><summary><b>2. Derive the OLS estimator in matrix form.</b></summary>

Minimise (y − Xβ)'(y − Xβ). Gradient: −2X'(y − Xβ) = 0 ⇒ X'Xβ = X'y ⇒ β̂ = (X'X)⁻¹X'y, provided X has full column rank. [02]
</details>

<details><summary><b>3. Interpret a slope coefficient in a multiple regression.</b></summary>

The expected change in y for a one-unit increase in that regressor, holding the other regressors fixed (formally, the FWL partial effect), in the units of y and x. It is an association, not automatically causal. [02]
</details>

<details><summary><b>4. What is R², and why can it mislead?</b></summary>

The share of variance in y explained by the model, 1 − RSS/TSS. It never decreases when variables are added (so it rewards overfitting), says nothing about causality or bias, and can be high for spurious trending series. [01, 06, 07]
</details>

<details><summary><b>5. Correlation vs regression coefficient?</b></summary>

Correlation is unit-free and symmetric. The simple-regression slope is r·(s_y/s_x): in units of y per unit of x, and asymmetric (regressing x on y gives a different slope). [01]
</details>

<details><summary><b>6. What is the dummy-variable trap?</b></summary>

Including a dummy for every category plus an intercept makes the columns sum to the intercept: perfect collinearity, X'X singular. Drop one category (the baseline) or the intercept. [02]
</details>

<details><summary><b>7. How do you interpret a log-log coefficient? A log-level one?</b></summary>

Log-log: elasticity, % change in y per 1% change in x. Log-level (ln y on x): ≈ 100β% change in y per unit of x (exactly e^β − 1). [05]
</details>

<details><summary><b>8. What is an interaction term and why centre variables?</b></summary>

x₁·x₂ lets the effect of x₁ depend on x₂: ∂y/∂x₁ = β₁ + β₃x₂. Centring makes β₁ the effect at the average x₂ (not at x₂ = 0, which may be meaningless) and reduces collinearity with the main effects. [02]
</details>

<details><summary><b>9. Why compute OLS with QR rather than inverting X'X?</b></summary>

Forming X'X squares the condition number, losing numerical precision when regressors are nearly collinear. QR solves Rβ = Q'y directly and is stable. [02]
</details>

<details><summary><b>10. When would you use gradient descent instead of the normal equation?</b></summary>

When X is too large to factorise or doesn't fit in memory (stochastic / mini-batch GD), or in online learning. Features must be standardised and the learning rate tuned; it converges to the same solution for convex least squares. [02]
</details>

## 🟡 Assumptions, inference and diagnostics

<details><summary><b>11. State the Gauss–Markov assumptions. Which one matters most?</b></summary>

Linearity, full rank, exogeneity E[ε|X] = 0, homoskedasticity, no autocorrelation. Then OLS is BLUE. Exogeneity matters most: its failure biases the coefficients themselves; the others affect efficiency and standard errors. [03]
</details>

<details><summary><b>12. Does OLS require normally distributed errors?</b></summary>

No, not for unbiasedness or BLUE. Normality gives exact t/F distributions in small samples; in large samples the CLT makes the tests approximately valid anyway. [03]
</details>

<details><summary><b>13. What is a p-value? What is it not?</b></summary>

The probability, under the null, of a test statistic at least as extreme as observed. It is not the probability the null is true, and not a measure of effect size or importance. [03]
</details>

<details><summary><b>14. Confidence vs prediction interval?</b></summary>

The CI covers the mean response (parameter uncertainty only, shrinks with n). The PI covers one new outcome (adds σ², never shrinks to zero). Forecasts and stress tests need PIs. [03]
</details>

<details><summary><b>15. What is heteroskedasticity, how do you detect it, and what does it do?</b></summary>

Error variance changing across observations. Detect with residual-vs-fitted plots, Breusch–Pagan, White. Coefficients stay unbiased; classical SEs are wrong. Fix with HC (robust) SEs or WLS. [04, 05]
</details>

<details><summary><b>16. Why is Durbin–Watson invalid with a lagged dependent variable? What do you use instead?</b></summary>

DW is biased towards 2 when y_{t−1} is a regressor, hiding autocorrelation. Use Breusch–Godfrey (or Durbin's h). [04]
</details>

<details><summary><b>17. What is multicollinearity and how do you deal with it?</b></summary>

High correlation among regressors: large SEs, unstable coefficients, though no bias. Detect with VIF (> 5–10) or correlation matrices. Drop/combine variables, collect more data, or use Ridge. [04, 06]
</details>

<details><summary><b>18. Explain leverage, Cook's distance and DFBETAS.</b></summary>

Leverage: how extreme an observation's x-values are (hat-matrix diagonal). Cook's D: overall influence on all fitted values (combines leverage and residual size). DFBETAS: influence on each coefficient in SE units. [04]
</details>

<details><summary><b>19. You find an influential outlier. What do you do?</b></summary>

Check for data errors; understand whether it is a different regime (model it with a dummy) or a genuine extreme (keep it). Report results with and without it, or use robust regression (Huber). Never delete it silently. [04, 05]
</details>

<details><summary><b>20. Explain omitted-variable bias with the formula.</b></summary>

If the true model has x₁ and x₂ but x₂ is omitted, E[β̂₁] = β₁ + β₂δ, where δ is the slope of x₂ on x₁. Bias needs x₂ to matter (β₂ ≠ 0) and to be correlated with x₁ (δ ≠ 0). [02]
</details>

## 🟠 Model selection, regularisation, time series, panels

<details><summary><b>21. AIC vs BIC: which picks larger models and why?</b></summary>

AIC (penalty 2k) picks larger models than BIC (penalty k ln n, larger when n > 7). AIC targets predictive accuracy; BIC is consistent for the true model if it is among the candidates. [06]
</details>

<details><summary><b>22. How would you cross-validate a time-series regression?</b></summary>

Expanding (or rolling) window splits that respect time order: train on the past, test on the next block. Shuffled K-fold leaks the future and overstates performance, especially with autocorrelation or overlapping targets. [06]
</details>

<details><summary><b>23. Ridge vs Lasso vs Elastic Net?</b></summary>

Ridge (L2) shrinks all coefficients, handles collinearity, never zeroes. Lasso (L1) zeroes some (feature selection), but picks arbitrarily among correlated features. Elastic Net mixes both: selection with stability. Standardise first; tune λ by CV. [06]
</details>

<details><summary><b>24. Why does Lasso produce exact zeros but Ridge doesn't?</b></summary>

The L1 constraint region is a diamond with corners on the axes; the least-squares contours typically first touch it at a corner, where some coefficients are zero. The L2 region is a sphere with no corners. [06]
</details>

<details><summary><b>25. What is a spurious regression? How do you avoid it?</b></summary>

A significant relationship between unrelated non-stationary (trending) series. Test for unit roots (ADF/KPSS); difference the data, demonstrate cointegration (Engle–Granger), or model the dynamics with lagged y. [07]
</details>

<details><summary><b>26. What is cointegration?</b></summary>

Two (or more) non-stationary series whose linear combination is stationary: they share a long-run equilibrium, so a levels regression is meaningful. Test via stationarity of the levels-regression residuals (Engle–Granger) or Johansen. [07]
</details>

<details><summary><b>27. In y_t = βx_t + ρy_{t−1} + ε_t, what is the long-run effect of x?</b></summary>

β/(1 − ρ), for |ρ| < 1. The short-run effect is β. [07]
</details>

<details><summary><b>28. When are Newey–West standard errors needed? How do you choose the lag?</b></summary>

When errors are autocorrelated (and possibly heteroskedastic), e.g. time series or overlapping targets. Rule of thumb L ≈ 4(n/100)^(2/9), at least the overlap length; check conclusions are robust to L. [07]
</details>

<details><summary><b>29. Your target is a 12-month rolling sum observed monthly. What problems does that cause?</b></summary>

Consecutive observations share 11 months: MA(11) autocorrelation by construction, inflated t-statistics with classical SEs, misleading fit. Use HAC with ≥ 11 lags, non-overlapping data, or lag the dependent variable by 12 rather than 1. [07]
</details>

<details><summary><b>30. Fixed effects vs random effects?</b></summary>

FE: unit-specific intercepts, consistent even if they correlate with regressors; uses within-unit variation only. RE: treats them as random and uncorrelated with regressors; more efficient if that holds. The Hausman test compares them. [08]
</details>

<details><summary><b>31. How should you compute standard errors in a panel of firms over time?</b></summary>

Cluster at the level where shocks are shared: by firm for serial correlation within firm (needs many firms), by time for common shocks, two-way or Driscoll–Kraay for both. [08]
</details>

<details><summary><b>32. Why can't you include time fixed effects when estimating the effect of national unemployment on several portfolios?</b></summary>

National unemployment is identical for all portfolios in a quarter, so it is perfectly collinear with the quarter dummies and its coefficient is not identified. [08]
</details>

## 🔴 Validation, risk and industry

<details><summary><b>33. How would you validate a regression used for stress testing?</b></summary>

Conceptual soundness (signs, theory, diagnostics, stability), outcome analysis (out-of-time backtests through past crises, rolling-origin forecasts vs a naive benchmark), sensitivity analysis, challenger models, external benchmarks, and ongoing monitoring with triggers. [09, 10]
</details>

<details><summary><b>34. Why impose economic sign constraints if a wrong-signed coefficient fits better?</b></summary>

Stress tests project outside historical conditions. A historically coincidental wrong sign (e.g. rate cuts during the 2009 loss peak) would make a severe scenario look benign. Fit on history matters less than behaving sensibly under the scenario. [06]
</details>

<details><summary><b>35. What is a block bootstrap and when is it needed?</b></summary>

Resampling blocks of consecutive observations (rather than single ones) to estimate sampling uncertainty while preserving dependence. Needed for time series and overlapping data, where the ordinary bootstrap understates uncertainty. [09]
</details>

<details><summary><b>36. What does quantile regression tell you that OLS does not?</b></summary>

How drivers affect the distribution of y beyond its mean: e.g. whether unemployment raises the 90th percentile of losses more than the median. Robust to outliers in y; useful for tail risk. [09]
</details>

<details><summary><b>37. A model has negative skill against a no-change forecast. Is it useless?</b></summary>

Not necessarily. A stress model answers conditional "what if" questions; no-change cannot answer them. But the result must be reported, and the model should not be used for unconditional forecasting. [09]
</details>

<details><summary><b>38. How would you turn a portfolio loss-rate projection into a £ loss and an uncertainty range?</b></summary>

Multiply each year's projected rate by the balance (static or dynamic balance sheet) and sum. For the range, simulate parameter draws and residual shocks, compute the total in each simulation, and take percentiles of the simulated totals, not sums of pointwise bands. [09]
</details>

<details><summary><b>39. What is model risk, and name three mitigants.</b></summary>

The risk of adverse outcomes from incorrect or misused models. Mitigants: independent validation, conservative overlays or management adjustments, use restrictions, challenger models, ongoing monitoring with triggers, documentation. [10]
</details>

<details><summary><b>40. Explain this project in two minutes.</b></summary>

"I built the satellite models a bank uses for Bank of England stress tests. I modelled UK write-off rates on cards, personal loans, mortgages and corporate lending from Bank of England data back to 1993, as logit regressions on lagged macro drivers with the loss rate's own persistence and Newey–West errors. I selected models from about 19,000 candidates per portfolio using economic sign rules, significance, VIF and cointegration, then validated them with crisis backtests, bootstrap, quantile regression and challengers. Under the 2025 scenario, five-year write-offs are about £44bn (range £35–49bn), and the results sit at a consistent ratio to the Bank's own published impairment rates. The main limitations: sensitivities learned from essentially one severe cycle, and an 8% Bank Rate outside the historical data." [10]
</details>
