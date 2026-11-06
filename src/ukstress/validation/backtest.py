"""Out-of-time backtests: re-estimate on data up to a cutoff, then project with the macro path that
actually happened. This isolates model error from scenario error (perfect foresight of the economy),
which is the standard performance test for stress-testing satellite models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ukstress.features.build import inv_logit_pct, trailing_loss_rate
from ukstress.models.satellite import SAMPLE_START, Spec, fit, project, target

# Bank of England 2025 Bank Capital Stress Test, Financial Stability Report December 2025, Table 5.C:
# cumulative five-year impairment charge / average gross on-balance-sheet exposures.
BOE_2025_IMPAIRMENT_RATES = {
    "UK consumer credit": 23.3,
    "UK mortgages": 1.0,
    "Total UK": 3.8,
}
BOE_SOURCE = "https://www.bankofengland.co.uk/financial-stability-report/2025/december-2025"


def backtest(spec: Spec, panel: pd.DataFrame, macro: pd.DataFrame, cutoff: str, horizon: int = 20,
             n_sims: int = 500, seed: int = 0) -> pd.DataFrame:
    """Fit ``spec`` on data to ``cutoff``; project ``horizon`` quarters on the actual macro path."""
    y = target(panel, spec.portfolio)
    fm, _ = fit(spec, y, macro, start=SAMPLE_START, end=cutoff)
    end = pd.Period(cutoff, freq="Q") + horizon
    pr = project(fm, y, macro, start=cutoff, end=min(end, y.dropna().index.max()), n_sims=n_sims, seed=seed)
    pr["actual"] = trailing_loss_rate(panel, spec.portfolio).reindex(pr.index)
    pr.attrs["model"] = fm
    return pr


def naive_forecast(panel: pd.DataFrame, portfolio: str, cutoff: str, index: pd.PeriodIndex) -> pd.Series:
    """No-change benchmark: the loss rate stays at its cutoff value."""
    lr = trailing_loss_rate(panel, portfolio)
    return pd.Series(lr.loc[pd.Period(cutoff, freq="Q")], index=index)


def rolling_origin(spec: Spec, panel, macro, cutoffs, horizon: int = 8) -> pd.DataFrame:
    """Errors at a fixed horizon from many forecast origins: model vs the no-change benchmark."""
    rows = []
    for c in cutoffs:
        pr = backtest(spec, panel, macro, c, horizon=horizon, n_sims=0)
        if len(pr) < horizon or pr.actual.isna().any():
            continue
        naive = naive_forecast(panel, spec.portfolio, c, pr.index)
        rows.append({"cutoff": c, "actual": pr.actual.iloc[-1], "model": pr.central.iloc[-1],
                     "naive": naive.iloc[-1]})
    out = pd.DataFrame(rows).set_index("cutoff")
    out["model error"] = out.model - out.actual
    out["naive error"] = out.naive - out.actual
    return out


def summary(pr: pd.DataFrame) -> dict:
    """Peak, cumulative and error statistics of a backtest path."""
    err = pr.central - pr.actual
    inside = ((pr.actual >= pr.p05) & (pr.actual <= pr.p95)).mean() if "p05" in pr else np.nan
    return {
        "actual peak %": pr.actual.max(),
        "model peak %": pr.central.max(),
        "peak error, pp": pr.central.max() - pr.actual.max(),
        "RMSE, pp": float(np.sqrt(np.mean(err**2))),
        "mean error, pp (+ = over-predicts)": float(err.mean()),
        "share of quarters inside 90% band": inside,
    }


def cumulative_rate(path: pd.Series, years=(2025, 2026, 2027, 2028, 2029)) -> float:
    """Five-year cumulative loss rate: sum of each calendar year's trailing rate at Q4.

    The trailing rate at Q4 is that year's write-offs over its average balance, so the sum is the
    five-year write-offs relative to average balances: the Bank of England's impairment-rate basis.
    """
    return float(sum(path.loc[pd.Period(f"{y}Q4", freq="Q")] for y in years))


__all__ = ["backtest", "rolling_origin", "naive_forecast", "summary", "cumulative_rate",
           "BOE_2025_IMPAIRMENT_RATES", "BOE_SOURCE", "inv_logit_pct"]
