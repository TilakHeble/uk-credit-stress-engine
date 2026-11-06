"""Build the quarterly modelling panel from the raw downloads.

Output: ``data/processed/quarterly_panel.parquet`` with one row per quarter and

* ``wo_<portfolio>``    write-offs in the quarter, £m (sterling, not seasonally adjusted)
* ``bal_<portfolio>``   amount outstanding at quarter end, £m
* ``lr_<portfolio>``    annualised loss rate, %: 4 x write-offs(t) / balance(t-1) x 100
* macro columns         unemployment, GDP growth, inflation, Bank Rate, mortgage rate and changes

Run: ``python -m ukstress.ingest.build_panel``
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from ukstress.config import (
    BOE_DIR,
    BOE_RATES,
    FIRST_QUARTER,
    ONS_DIR,
    ONS_SERIES,
    PANEL_PATH,
    PORTFOLIOS,
    UKHPI_FILE,
)
from ukstress.ingest.contracts import PANEL_SCHEMA
from ukstress.ingest.readers import (
    read_bankstats_c21,
    read_boe_iadb,
    read_ons_timeseries,
    read_ukhpi,
)

log = logging.getLogger(__name__)

WRITE_OFF_FILE = BOE_DIR / "writeoffs.csv"  # database extract, 1993 onwards
BALANCE_FILE = BOE_DIR / "balances_rates.csv"
C21_FILE = BOE_DIR / "bankstats_C2_1_latest.xls"


def load_boe() -> pd.DataFrame:
    """Balances, rates and write-offs on one quarterly index. Write-off sources must agree."""
    boe = read_boe_iadb(BALANCE_FILE)
    if WRITE_OFF_FILE.exists():
        wo = read_boe_iadb(WRITE_OFF_FILE)
        overlap = [c for c in wo.columns if c in boe.columns]
        for c in overlap:  # the same series downloaded twice must be identical
            both = pd.concat([boe[c], wo[c]], axis=1).dropna()
            if not np.allclose(both.iloc[:, 0], both.iloc[:, 1]):
                raise ValueError(f"{c} differs between {BALANCE_FILE.name} and {WRITE_OFF_FILE.name}")  # noqa: E501
        boe = boe.combine_first(wo)
    else:
        log.warning("%s missing: card/other write-offs limited to Bankstats C2.1", WRITE_OFF_FILE)
    if C21_FILE.exists():  # fill and cross-check the latest quarters from the published table
        c21 = read_bankstats_c21(C21_FILE)
        for c in c21.columns.intersection(boe.columns):
            both = pd.concat([boe[c], c21[c]], axis=1).dropna()
            bad = (both.iloc[:, 0] - both.iloc[:, 1]).abs() > 0.5
            if bad.any():
                log.warning("%s: %d quarters revised in C2.1 vs database extract", c, bad.sum())
        boe = boe.combine_first(c21[[c for c in c21.columns if c.startswith("RPQTFH")]])
    return boe


def load_macro() -> pd.DataFrame:
    cols = {col: read_ons_timeseries(ONS_DIR / f, col) for f, col in ONS_SERIES.values()}
    cols["house_price_index"] = read_ukhpi(UKHPI_FILE)
    return pd.DataFrame(cols)


def build_panel() -> pd.DataFrame:
    boe, macro = load_boe(), load_macro()
    idx = pd.period_range(FIRST_QUARTER, max(boe.index.max(), macro.index.max()), freq="Q")
    boe, macro = boe.reindex(idx), macro.reindex(idx)
    panel = pd.DataFrame(index=idx)
    panel.index.name = "quarter"

    for name, spec in PORTFOLIOS.items():
        wo = boe.get(spec["write_off"], pd.Series(np.nan, index=idx))
        bal = boe[spec["balance"]]
        panel[f"wo_{name}"] = wo
        panel[f"bal_{name}"] = bal
        panel[f"lr_{name}"] = 400 * wo / bal.shift(1)  # annualised % of lagged balance

    # Macro drivers, in the units the 2025 stress scenario is expressed in
    panel["unemployment"] = macro["unemployment"]
    panel["gdp_growth_yoy"] = 100 * macro["real_gdp"].pct_change(4, fill_method=None)
    panel["cpi_inflation"] = macro["cpi_inflation"]
    hpi = macro["house_price_index"]
    panel["house_price_growth_yoy"] = 100 * hpi.pct_change(4, fill_method=None)
    for code, col in BOE_RATES.items():
        panel[col] = boe[code]
    panel["unemployment_chg_yoy"] = panel["unemployment"].diff(4)
    panel["bank_rate_chg_yoy"] = panel["bank_rate"].diff(4)

    # Keep quarters where at least one loss rate is observed
    lr_cols = [c for c in panel if c.startswith("lr_")]
    panel = panel.loc[panel[lr_cols].notna().any(axis=1)]
    return PANEL_SCHEMA.validate(panel)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    panel = build_panel()
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH)
    first, last = panel.index[0], panel.index[-1]
    log.info("wrote %s: %d quarters (%s to %s)", PANEL_PATH, len(panel), first, last)


if __name__ == "__main__":
    main()
