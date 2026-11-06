"""Satellite models: portfolio loss rates as a linear function of lagged macro drivers.

Model (``dynamic=True`` adds the loss rate four quarters earlier, which does not overlap the
trailing four-quarter window):

    logit(LR4_t) = a + sum_j b_j * X_{j, t - lag_j} + c * COVID_t [+ rho * logit(LR4_{t-4})] + e_t

Standard errors are Newey-West (HAC): the trailing target overlaps by construction, so residuals
are autocorrelated and classical standard errors would overstate significance.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from ukstress.features.build import (
    DRIVERS,
    JUMP_OFF,
    LAG_GRID,
    covid_dummy,
    inv_logit_pct,
    lag_name,
    lagged,
    logit_pct,
    trailing_loss_rate,
)

HAC_LAGS = 4
SAMPLE_START = pd.Period("1994Q2", freq="Q")  # first quarter with a full trailing window


@dataclass(frozen=True)
class Spec:
    portfolio: str
    terms: tuple[tuple[str, int], ...]
    dynamic: bool = False
    covid: bool = True

    @property
    def names(self) -> list[str]:
        cols = ["const"] + [lag_name(d, k) for d, k in self.terms]
        if self.covid:
            cols.append("covid")
        if self.dynamic:
            cols.append("y_L4")
        return cols

    def describe(self) -> str:
        t = " + ".join(f"{d}[t-{k}]" for d, k in self.terms)
        return t + (" + y[t-4]" if self.dynamic else "")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["terms"] = [list(t) for t in self.terms]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Spec:
        return cls(d["portfolio"], tuple(tuple(t) for t in d["terms"]), d["dynamic"], d["covid"])


def target(panel: pd.DataFrame, portfolio: str) -> pd.Series:
    return pd.Series(logit_pct(trailing_loss_rate(panel, portfolio)), index=panel.index, name="y")


def design(spec: Spec, y: pd.Series, macro: pd.DataFrame, index=None) -> pd.DataFrame:
    """Regressor matrix for ``spec`` on ``index`` (default: index of ``y``)."""
    index = y.index if index is None else index
    X = lagged(macro, list(spec.terms)).reindex(index)
    X.insert(0, "const", 1.0)
    if spec.covid:
        X["covid"] = covid_dummy(index)
    if spec.dynamic:
        X["y_L4"] = y.shift(4).reindex(index)
    return X[spec.names]


def sample(spec: Spec, y, macro, start=SAMPLE_START, end=None):
    X = design(spec, y, macro)
    df = pd.concat([y.rename("y"), X], axis=1).loc[start:end].dropna()
    return df["y"], df[spec.names]


@dataclass
class FittedModel:
    spec: Spec
    params: pd.Series
    cov: pd.DataFrame
    resid: pd.Series
    nobs: int
    rsquared_adj: float
    aic: float
    bic: float
    sample: tuple[str, str]
    extra: dict = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X[self.params.index] @ self.params

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "params": self.params.to_dict(),
            "cov": self.cov.values.tolist(),
            "resid": {str(k): v for k, v in self.resid.items()},
            "nobs": self.nobs,
            "rsquared_adj": self.rsquared_adj,
            "aic": self.aic,
            "bic": self.bic,
            "sample": list(self.sample),
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FittedModel:
        spec = Spec.from_dict(d["spec"])
        params = pd.Series(d["params"])
        params = params[[c for c in spec.names if c in params.index]]
        cov = pd.DataFrame(d["cov"], index=params.index, columns=params.index)
        resid = pd.Series(d["resid"])
        resid.index = pd.PeriodIndex(resid.index, freq="Q")
        return cls(spec, params, cov, resid, d["nobs"], d["rsquared_adj"], d["aic"], d["bic"],
                   tuple(d["sample"]), d.get("extra", {}))


def fit(spec: Spec, y, macro, start=SAMPLE_START, end=None, hac_lags: int = HAC_LAGS):
    """OLS with Newey-West covariance. Returns (FittedModel, statsmodels results)."""
    yy, X = sample(spec, y, macro, start, end)
    if spec.covid and not X["covid"].any():  # sample ends before COVID: the dummy is all zeros
        X = X.drop(columns="covid")
    res = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    fm = FittedModel(spec, res.params, res.cov_params(), res.resid, int(res.nobs),
                     float(res.rsquared_adj), float(res.aic), float(res.bic),
                     (str(yy.index[0]), str(yy.index[-1])))
    return fm, res


def vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factors of the non-constant columns (diag of the inverse correlation)."""
    Z = X.drop(columns="const")
    if Z.shape[1] < 2:
        return pd.Series(1.0, index=Z.columns)
    c = np.corrcoef(Z.values, rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(c)), index=Z.columns)


def sign_ok(spec: Spec, params: pd.Series) -> bool:
    for d, k in spec.terms:
        if np.sign(params[lag_name(d, k)]) != DRIVERS[d]["sign"]:
            return False
    if spec.dynamic and not 0 < params["y_L4"] < 1:  # persistent but stable
        return False
    return True


def candidate_specs(portfolio: str, max_terms: int = 3, dynamic=(False, True), lags=LAG_GRID):
    """Every combination of up to ``max_terms`` distinct drivers, each at one lag.

    Unemployment level and change are one driver *family* and may not appear together (the change
    is a difference of the level, so including both makes the coefficients uninterpretable); the
    same applies to Bank Rate.
    """
    family = {"unemployment": "u", "unemployment_chg_yoy": "u", "bank_rate": "r",
              "bank_rate_chg_yoy": "r"}
    for n in range(1, max_terms + 1):
        for ds in itertools.combinations(DRIVERS, n):
            fams = [family.get(d, d) for d in ds]
            if len(set(fams)) < len(fams):
                continue
            for ks in itertools.product(lags, repeat=n):
                for dyn in dynamic:
                    yield Spec(portfolio, tuple(zip(ds, ks, strict=True)), dyn)


def search(portfolio: str, panel: pd.DataFrame, macro: pd.DataFrame, max_terms: int = 3,
           p_max: float = 0.05, vif_max: float = 5.0, end=None) -> pd.DataFrame:
    """Fit every candidate and record the checks a bank's model-selection rules apply.

    Cheap screening first (OLS by least squares, sign rules), then HAC inference, VIF and, for
    static models, an Engle-Granger cointegration test on the survivors only.
    """
    y = target(panel, portfolio)
    rows = []
    for spec in candidate_specs(portfolio, max_terms):
        yy, X = sample(spec, y, macro, end=end)
        if spec.covid and not X["covid"].any():  # sample ends before COVID
            X = X.drop(columns="covid")
        beta, *_ = np.linalg.lstsq(X.values, yy.values, rcond=None)
        params = pd.Series(beta, index=X.columns)
        row = {"spec": spec, "terms": len(spec.terms), "dynamic": spec.dynamic,
               "describe": spec.describe(), "sign_ok": sign_ok(spec, params)}
        if row["sign_ok"]:
            res = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
            macro_p = res.pvalues[[lag_name(d, k) for d, k in spec.terms]]
            row |= {"max_p": float(macro_p.max()), "max_vif": float(vif(X).max()),
                    "bic": float(res.bic), "aic": float(res.aic), "r2_adj": float(res.rsquared_adj),
                    "nobs": int(res.nobs)}
            if not spec.dynamic:
                row["coint_p"] = float(coint(yy, X.drop(columns=["const", "covid"], errors="ignore"),
                                             trend="c")[1])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["passes"] = (
        out.sign_ok
        & (out.max_p < p_max)
        & (out.max_vif < vif_max)
        & (out.dynamic | (out.get("coint_p", 1.0) < 0.10))
    )
    return out.sort_values(["passes", "bic"], ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------------------------------- projection

def project(fm: FittedModel, y_hist: pd.Series, macro_path: pd.DataFrame, start=None, end=None,
            n_sims: int = 0, block: int = 4, seed: int = 0) -> pd.DataFrame:
    """Project the loss rate over ``macro_path`` from the quarter after ``start`` (default jump-off).

    Dynamic models are solved recursively: y[t-4] is actual history while available, then the
    model's own projection. With ``n_sims`` > 0 the projection is simulated with (i) parameters drawn
    from N(b, V_HAC) and (ii) residuals resampled in blocks of ``block`` quarters, which preserves
    their autocorrelation. Returns the central path and simulation quantiles in loss-rate %.
    """
    start = JUMP_OFF if start is None else pd.Period(start, freq="Q")
    idx = macro_path.index[(macro_path.index > start)]
    if end is not None:
        idx = idx[idx <= pd.Period(end, freq="Q")]
    spec = fm.spec
    y = y_hist.copy().loc[:start]
    X = design(spec, y.reindex(y.index.union(idx)), macro_path, index=idx)[fm.params.index]

    def run(beta: pd.Series, shocks: np.ndarray) -> np.ndarray:
        yy = y.reindex(y.index.union(idx)).copy()
        out = np.empty(len(idx))
        for i, q in enumerate(idx):
            x = X.loc[q].copy()
            if spec.dynamic:
                x["y_L4"] = yy.loc[q - 4]
            out[i] = float(x.values @ beta.values) + shocks[i]
            yy.loc[q] = out[i]
        return out

    central = run(fm.params, np.zeros(len(idx)))
    result = pd.DataFrame({"central": inv_logit_pct(central)}, index=idx)
    if n_sims:
        rng = np.random.default_rng(seed)
        resid = fm.resid.values
        draws = rng.multivariate_normal(fm.params.values, fm.cov.values, size=n_sims)
        sims = np.empty((n_sims, len(idx)))
        n_blocks = int(np.ceil(len(idx) / block))
        for s in range(n_sims):
            starts = rng.integers(0, len(resid) - block + 1, n_blocks)
            shocks = np.concatenate([resid[a:a + block] for a in starts])[: len(idx)]
            sims[s] = run(pd.Series(draws[s], index=fm.params.index), shocks)
        rates = inv_logit_pct(sims)
        for q, name in [(0.05, "p05"), (0.25, "p25"), (0.5, "p50"), (0.75, "p75"), (0.95, "p95")]:
            result[name] = np.quantile(rates, q, axis=0)
        result.attrs["sims"] = rates
    return result


def save_models(models: dict[str, FittedModel], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({p: m.to_dict() for p, m in models.items()}, indent=1))


def load_models(path) -> dict[str, FittedModel]:
    return {p: FittedModel.from_dict(d) for p, d in json.loads(path.read_text()).items()}
