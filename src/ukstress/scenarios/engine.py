"""Scenario engine: champion models -> loss-rate paths, five-year cumulative rates and £ losses.

Shared by notebook 07 and the dashboard, so both always report the same numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ukstress.config import PORTFOLIOS
from ukstress.features.build import JUMP_OFF, inv_logit_pct, logit_pct, trailing_loss_rate
from ukstress.models.satellite import FittedModel, design, project, target

YEARS = (2025, 2026, 2027, 2028, 2029)


def jump_off_gap(fm: FittedModel, panel: pd.DataFrame, macro: pd.DataFrame) -> float:
    """Logit gap between the actual loss rate and the model's fitted value at the jump-off quarter."""
    y = target(panel, fm.spec.portfolio)
    X = design(fm.spec, y, macro, index=pd.PeriodIndex([JUMP_OFF]))[fm.params.index]
    return float(y.loc[JUMP_OFF] - (X.values @ fm.params.values)[0])


def anchor(path_pct: pd.DataFrame, gap: float, quarters: int = 8) -> pd.DataFrame:
    """Jump-off adjustment: carry the jump-off gap into the projection, fading linearly to zero.

    Applied on the logit scale to every column (central and quantiles). Banks use such an adjustment
    so that a projection starts from the book's actual position rather than from the model's view of it.
    """
    h = np.arange(1, len(path_pct) + 1)
    w = np.clip(1 - (h - 1) / quarters, 0, 1)
    return inv_logit_pct(logit_pct(path_pct).add(gap * w, axis=0))


def run(models: dict[str, FittedModel], panel, macro, macro_path, n_sims: int = 0, anchored: bool = False,
        seed: int = 0) -> dict[str, pd.DataFrame]:
    out = {}
    for p, fm in models.items():
        pr = project(fm, target(panel, p), macro_path, n_sims=n_sims, seed=seed)
        cols = [c for c in pr.columns]
        if anchored:
            pr[cols] = anchor(pr[cols], jump_off_gap(fm, panel, macro))
        out[p] = pr
    return out


def cumulative(path: pd.Series, years=YEARS) -> float:
    """Five-year cumulative loss rate: sum of each year's trailing rate at Q4 (write-offs / avg balance)."""
    return float(sum(path.loc[pd.Period(f"{y}Q4", freq="Q")] for y in years))


def jump_off_balances(panel: pd.DataFrame) -> pd.Series:
    """Balances at the jump-off quarter, £bn (static balance sheet, as in the Bank of England test)."""
    return pd.Series({p: panel.loc[JUMP_OFF, f"bal_{p}"] / 1000 for p in PORTFOLIOS})


def summary_table(paths: dict[str, pd.DataFrame], panel: pd.DataFrame) -> pd.DataFrame:
    bal = jump_off_balances(panel)
    rows = {}
    for p, pr in paths.items():
        lr0 = trailing_loss_rate(panel, p).loc[JUMP_OFF]
        cum = cumulative(pr.central)
        row = {
            "loss rate at jump-off %": lr0,
            "peak loss rate %": pr.central.max(),
            "peak quarter": str(pr.central.idxmax()),
            "5-year cumulative %": cum,
            "balance £bn": bal[p],
            "5-year losses £bn": cum / 100 * bal[p],
            "losses if rates stayed at jump-off £bn": 5 * lr0 / 100 * bal[p],
        }
        if "p05" in pr:
            row["5-year cumulative 5th pct %"] = cumulative(pr.p05)
            row["5-year cumulative 95th pct %"] = cumulative(pr.p95)
            sims = pr.attrs.get("sims")
            if sims is not None:  # proper percentiles of the cumulative (not the sum of pointwise bands)
                q4 = [i for i, q in enumerate(pr.index) if q.quarter == 4 and q.year in YEARS]
                cum_sims = sims[:, q4].sum(axis=1)
                row["5-year cumulative 5th pct %"] = float(np.quantile(cum_sims, 0.05))
                row["5-year cumulative 95th pct %"] = float(np.quantile(cum_sims, 0.95))
        rows[p] = row
    t = pd.DataFrame(rows).T
    t["stress uplift £bn"] = t["5-year losses £bn"] - t["losses if rates stayed at jump-off £bn"]
    return t
