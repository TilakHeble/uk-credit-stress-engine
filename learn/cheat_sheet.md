# Linear regression cheat sheet

[Course index](README.md)

## Model and estimator

| | Formula |
|---|---|
| Model | $y = X\beta + \varepsilon$ |
| OLS | $\hat\beta = (X'X)^{-1}X'y$ (compute via QR / `lstsq`, not an explicit inverse) |
| Simple regression slope | $\hat\beta_1 = \text{Cov}(x,y)/\text{Var}(x)$ |
| Fitted / residuals | $\hat y = X\hat\beta = Hy$, $H = X(X'X)^{-1}X'$; $e = y - \hat y$ |
| Error variance | $\hat\sigma^2 = e'e/(n-k)$, *k* includes the intercept |
| Gradient descent | $\beta \leftarrow \beta - \eta\,\tfrac{2}{n}X'(X\beta - y)$; standardise features; stable if $\eta < 2/\lambda_{\max}(2X'X/n)$ |

## Fit

| Statistic | Formula | Note |
|---|---|---|
| $R^2$ | $1 - \text{RSS}/\text{TSS}$ | Never falls when a variable is added |
| Adjusted $R^2$ | $1 - (1-R^2)\frac{n-1}{n-k}$ | Mild penalty |
| AIC | $-2\ln L + 2k$ | Prediction-oriented |
| BIC | $-2\ln L + k\ln n$ | Harsher; picks smaller models |
| Gaussian $\ln L$ | $-\tfrac n2[\ln 2\pi + \ln(\text{RSS}/n) + 1]$ | |

## Inference

| | Formula |
|---|---|
| Classical covariance | $\hat\sigma^2(X'X)^{-1}$ |
| White (HC0) | $(X'X)^{-1}\big(\sum e_i^2 x_ix_i'\big)(X'X)^{-1}$; HC1 × $n/(n-k)$; HC3 adjusts for leverage |
| Newey–West | sandwich with $\sum_{l=1}^{L} (1-\tfrac{l}{L+1})$-weighted autocovariances of $x_te_t$; $L \approx 4(n/100)^{2/9}$ |
| t-test | $t = \hat\beta_j/\text{se}_j \sim t_{n-k}$ |
| CI | $\hat\beta_j \pm t_{0.975,n-k}\,\text{se}_j$ |
| Nested F | $F = \frac{(\text{RSS}_r - \text{RSS}_f)/q}{\text{RSS}_f/(n-k)} \sim F_{q,n-k}$; one restriction: $F = t^2$ |
| Mean CI at $x_0$ | $x_0'\hat\beta \pm t\sqrt{x_0'Vx_0}$ |
| Prediction interval | $x_0'\hat\beta \pm t\sqrt{x_0'Vx_0 + \hat\sigma^2}$ |

## Gauss–Markov (OLS is BLUE if…)
Linearity · full rank · $E[\varepsilon|X]=0$ (**the one that biases coefficients**) · constant variance · no autocorrelation. Normality only for exact small-sample tests.

## Diagnostics

| Problem | Test (H0) | Fix |
|---|---|---|
| Heteroskedasticity | Breusch–Pagan, White (H0: constant variance) | HC errors, WLS |
| Autocorrelation | Breusch–Godfrey, Ljung–Box (H0: none). DW invalid with lagged y | Newey–West, model dynamics |
| Non-normality | Jarque–Bera (H0: normal) | Transform; bootstrap intervals |
| Functional form | RESET (H0: linear form adequate) | Transforms, polynomials, hinges, interactions |
| Multicollinearity | VIF $= 1/(1-R_j^2)$; > 5–10 warns | Drop/combine, Ridge |
| Influence | Leverage $h_{ii}$ (avg $k/n$); Cook's D > $4/n$; \|DFBETAS\| > $2/\sqrt n$ | Investigate; Huber; with/without |
| Unit root | ADF (H0: unit root); KPSS (H0: stationary) | Difference; cointegration; dynamic model |
| Spurious regression | Engle–Granger (H0: no cointegration) | As above |
| Structural break | Chow (known date), CUSUM, recursive estimates | Dummies, sub-samples |

## Interpretation

| Model | Coefficient means |
|---|---|
| $y = \beta x$ | +β units of y per unit x |
| $\ln y = \beta x$ | ≈ +100β% in y per unit x (exactly $e^\beta - 1$) |
| $y = \beta \ln x$ | +β/100 units of y per 1% rise in x |
| $\ln y = \beta \ln x$ | elasticity: +β% in y per 1% in x |
| $\text{logit}(p) = \beta x$ | $\partial p/\partial x \approx \beta p(1-p)$ |
| Interaction $\beta_3 x_1x_2$ | $\partial y/\partial x_1 = \beta_1 + \beta_3 x_2$ (centre first) |
| Dynamic $\rho y_{t-s}$ | long-run effect $\beta/(1-\rho)$ |
| Omitted-variable bias | $E[\hat\beta_1^{\text{short}}] = \beta_1 + \beta_2\delta$ |

## Regularisation

| | Penalty | Behaviour |
|---|---|---|
| Ridge | $\lambda\sum\beta_j^2$ | Shrinks, keeps all; closed form $(X'X + \lambda I)^{-1}X'y$ |
| Lasso | $\lambda\sum\lvert\beta_j\rvert$ | Exact zeros: selection |
| Elastic Net | mix | Selection, stable with correlated features |

Standardise first; tune λ by **time-ordered** CV for time series.

## Panels
Fixed effects = unit dummies = within (demeaning). Test slope homogeneity before pooling. Errors: cluster where shocks are shared; Driscoll–Kraay for cross-sectional + serial correlation. Time FE absorb any regressor common to all units in a period.

## Uncertainty and validation
Block bootstrap for dependent data · quantile regression for tails (check loss) · simulate parameters and shocks · totals from simulated totals · out-of-time backtests · skill = 1 − RMSE(model)/RMSE(naive) · external benchmarks.
