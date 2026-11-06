"""Model features: the target transform, lagged drivers and scenario paths.

The same functions build features for history and for scenarios, so a model can never be
projected on a variable defined differently from the one it was fitted on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ukstress.config import BCST_PATHS_FILE, PORTFOLIOS
from ukstress.ingest.readers import read_bcst_paths

# Candidate macro drivers. ``sign`` is the economically required direction of the effect on losses
# (banks reject models whose signs contradict economics, however well they fit).
DRIVERS: dict[str, dict] = {
    "unemployment": {"label": "Unemployment rate (%)", "sign": +1},
    "unemployment_chg_yoy": {"label": "Change in unemployment, y/y (pp)", "sign": +1},
    "gdp_growth_yoy": {"label": "Real GDP growth, y/y (%)", "sign": -1},
    "house_price_growth_yoy": {"label": "House price growth, y/y (%)", "sign": -1},
    "bank_rate": {"label": "Bank Rate (%)", "sign": +1},
    "bank_rate_chg_yoy": {"label": "Change in Bank Rate, y/y (pp)", "sign": +1},
    "cpi_inflation": {"label": "CPI inflation (%)", "sign": +1},
}
LAG_GRID = (0, 1, 2, 3, 4, 6, 8)
COVID = ("2020Q2", "2021Q4")  # furlough / payment-holiday regime
JUMP_OFF = pd.Period("2024Q4", freq="Q")  # last quarter before the 2025 stress scenario


def trailing_loss_rate(panel: pd.DataFrame, portfolio: str) -> pd.Series:
    """Write-offs over the last four quarters / average lagged balance, in %."""
    wo = panel[f"wo_{portfolio}"].rolling(4).sum()
    avg_bal = panel[f"bal_{portfolio}"].shift(1).rolling(4).mean()
    return (100 * wo / avg_bal).rename(f"lr4_{portfolio}")


def _keep_type(x, values):
    if isinstance(x, pd.Series):
        return pd.Series(values, index=x.index, name=x.name)
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(values, index=x.index, columns=x.columns)
    return values


def logit_pct(rate_pct):
    """Logit of a rate in %, clipped away from 0 and 100. Preserves pandas types."""
    p = np.clip(np.asarray(rate_pct, float) / 100, 1e-6, 1 - 1e-6)
    return _keep_type(rate_pct, np.log(p / (1 - p)))


def inv_logit_pct(z):
    """Inverse of ``logit_pct``: logit -> rate in %. Preserves pandas types."""
    return _keep_type(z, 100 / (1 + np.exp(-np.asarray(z, float))))


def covid_dummy(index: pd.PeriodIndex) -> pd.Series:
    s = pd.Series(0.0, index=index, name="covid")
    s[(index >= pd.Period(COVID[0])) & (index <= pd.Period(COVID[1]))] = 1.0
    return s


def lag_name(driver: str, lag: int) -> str:
    return f"{driver}_L{lag}"


def lagged(macro: pd.DataFrame, terms: list[tuple[str, int]]) -> pd.DataFrame:
    """Columns ``driver_Lk`` = driver shifted k quarters (driver at t-k explains losses at t)."""
    return pd.DataFrame({lag_name(d, k): macro[d].shift(k) for d, k in terms}, index=macro.index)


def all_lags(macro: pd.DataFrame, lags=LAG_GRID) -> pd.DataFrame:
    return lagged(macro, [(d, k) for d in DRIVERS for k in lags])


def scenario_macro(macro_history: pd.DataFrame, path_file=BCST_PATHS_FILE) -> pd.DataFrame:
    """Actual history to the 2024Q4 jump-off, then the Bank of England 2025 stress paths.

    Growth rates in the scenario quarters are computed from the Bank's own level series (history
    and projection share one vintage), so the first projected quarters compare like with like.
    """
    b = read_bcst_paths(path_file)
    sc = pd.DataFrame(index=b.index)
    sc["unemployment"] = b["UK unemployment rate"]
    sc["gdp_growth_yoy"] = 100 * b["UK real GDP"].pct_change(4)
    sc["house_price_growth_yoy"] = 100 * b["UK residential property price index"].pct_change(4)
    sc["bank_rate"] = b["Bank Rate"]
    sc["cpi_inflation"] = 100 * b["UK CPI"].pct_change(4)
    hist = macro_history.loc[:JUMP_OFF, list(sc.columns)]
    out = pd.concat([hist, sc.loc[sc.index > JUMP_OFF]])
    return add_changes(out)


def add_changes(macro: pd.DataFrame) -> pd.DataFrame:
    macro = macro.copy()
    macro["unemployment_chg_yoy"] = macro["unemployment"].diff(4)
    macro["bank_rate_chg_yoy"] = macro["bank_rate"].diff(4)
    return macro


def scaled_scenario(macro_path: pd.DataFrame, severity: float = 1.0) -> pd.DataFrame:
    """Scale every scenario deviation from the jump-off quarter by ``severity``.

    severity = 0 holds each driver at its 2024Q4 value; 1 is the Bank of England scenario;
    1.5 is 50% more severe. Changes are recomputed afterwards so they stay consistent.
    """
    base = ["unemployment", "gdp_growth_yoy", "house_price_growth_yoy", "bank_rate", "cpi_inflation"]
    out = macro_path.copy()
    proj = out.index > JUMP_OFF
    jump = out.loc[JUMP_OFF, base]
    out.loc[proj, base] = jump + severity * (out.loc[proj, base] - jump)
    out["bank_rate"] = out["bank_rate"].clip(lower=0)
    return add_changes(out)


def portfolio_labels() -> dict[str, str]:
    return {p: spec["label"] for p, spec in PORTFOLIOS.items()}
