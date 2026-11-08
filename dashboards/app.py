"""UK Credit Stress Testing Engine: interactive scenario dashboard.

Run locally:  streamlit run dashboards/app.py
Everything is recomputed from the committed raw data and the saved champion models, so the
numbers here always match notebook 07.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ukstress.config import MODELS_DIR, PORTFOLIOS  # noqa: E402
from ukstress.features.build import (  # noqa: E402
    JUMP_OFF,
    add_changes,
    scaled_scenario,
    scenario_macro,
    trailing_loss_rate,
)
from ukstress.ingest.build_panel import build_macro, build_panel  # noqa: E402
from ukstress.models.satellite import load_models  # noqa: E402
from ukstress.scenarios.engine import jump_off_balances, run, summary_table  # noqa: E402
from ukstress.validation.backtest import BOE_2025_IMPAIRMENT_RATES  # noqa: E402

COLOURS = {"credit_card": "#2a78d6", "other_consumer": "#eb6834", "mortgage": "#1baf7a", "corporate": "#eda100"}
LABEL = {p: s["label"] for p, s in PORTFOLIOS.items()}
RATING = {"credit_card": "🟡 Amber (conservative)", "other_consumer": "🟡 Amber (weak)",
          "mortgage": "🟡 Amber", "corporate": "🟢 Green"}

st.set_page_config(page_title="UK Credit Stress Testing Engine", page_icon="📉", layout="wide")


@st.cache_data(show_spinner=False)
def load():
    panel, macro = build_panel(), build_macro()
    return panel, macro, scenario_macro(macro)


@st.cache_resource(show_spinner=False)
def models():
    return load_models(MODELS_DIR / "champions.json")


def build_path(base: pd.DataFrame, severity: float, du: float, dr: float, dh: float) -> pd.DataFrame:
    p = scaled_scenario(base, severity)
    proj = p.index > JUMP_OFF
    p.loc[proj, "unemployment"] += du
    p.loc[proj, "bank_rate"] = (p.loc[proj, "bank_rate"] + dr).clip(lower=0)
    p.loc[proj, "house_price_growth_yoy"] += dh
    return add_changes(p)


@st.cache_data(show_spinner="Running 500 simulations per portfolio…")
def simulate(severity, du, dr, dh, anchored, n_sims):
    panel, macro, base = load()
    path = build_path(base, severity, du, dr, dh)
    paths = run(models(), panel, macro, path, n_sims=n_sims, anchored=anchored, seed=1)
    table = summary_table(paths, panel)
    out = {}
    for p, pr in paths.items():
        df = pr[["central", "p05", "p25", "p75", "p95"]].copy()
        df.attrs = {}  # drop the raw simulation arrays before charting
        out[p] = df
    return path, out, table


panel, macro, base_path = load()

# ------------------------------------------------------------------------------------------- sidebar
st.sidebar.header("Scenario")
st.sidebar.caption("Start from the Bank of England 2025 Bank Capital Stress Test and adjust it.")
severity = st.sidebar.slider("Severity (1 = Bank of England scenario)", 0.0, 2.0, 1.0, 0.1,
                             help="Scales every deviation from the end-2024 economy. 0 holds the economy at end-2024.")
du = st.sidebar.slider("Extra unemployment (pp, whole projection)", -2.0, 3.0, 0.0, 0.25)
dr = st.sidebar.slider("Extra Bank Rate (pp, whole projection)", -4.0, 3.0, 0.0, 0.25)
dh = st.sidebar.slider("Extra house price growth (pp a year)", -10.0, 10.0, 0.0, 1.0)
anchored = st.sidebar.toggle("Jump-off adjustment", value=False,
                             help="Start each projection from the actual end-2024 loss rate (gap fades over 8 quarters).")
overlay = st.sidebar.toggle("Consumer credit-quality overlay", value=False,
                            help="Scale consumer projections by actual ÷ modelled losses over 2025–26 (validation finding).")
n_sims = st.sidebar.select_slider("Simulations", [200, 500, 1000], value=500)

path, paths, table = simulate(severity, du, dr, dh, anchored, n_sims)
OVERLAY = {"credit_card": 0.72, "other_consumer": 0.62}  # notebook 07, section 3
if overlay:
    for p, f in OVERLAY.items():
        paths[p] = paths[p] * f
        for c in ["5-year cumulative %", "5-year losses £bn", "5-year cumulative 5th pct %", "5-year cumulative 95th pct %",
                  "peak loss rate %"]:
            table.loc[p, c] = float(table.loc[p, c]) * f

# --------------------------------------------------------------------------------------------- header
st.title("UK Credit Stress Testing Engine")
st.markdown(
    "How much would UK lenders lose on **credit cards, other consumer credit, mortgages and corporate lending** "
    "in a severe recession? Satellite regression models of Bank of England write-off data, projected over 2025–29. "
    "[Code and documentation](https://github.com/TilakHeble/uk-credit-stress-engine)."
)

total = table["5-year losses £bn"].astype(float).sum()
bal = jump_off_balances(panel)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("5-yr losses", f"£{total:,.0f}bn", help="Five-year write-offs across the four portfolios")
c2.metric("Card peak", f"{float(table.loc['credit_card', 'peak loss rate %']):.1f}%",
          help=f"Peak annual card loss rate, from {trailing_loss_rate(panel, 'credit_card').loc[JUMP_OFF]:.1f}% at end-2024")
c3.metric("Corp. peak", f"{float(table.loc['corporate', 'peak loss rate %']):.1f}%", help="Peak annual corporate loss rate")
c4.metric("Unemp. peak", f"{path.loc[path.index > JUMP_OFF, 'unemployment'].max():.1f}%")
c5.metric("Bank Rate peak", f"{path.loc[path.index > JUMP_OFF, 'bank_rate'].max():.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(["Loss projections", "Scenario paths", "Losses in £", "Models and validation"])

# ------------------------------------------------------------------------------------- projections
with tab1:
    cols = st.columns(2)
    for i, p in enumerate(PORTFOLIOS):
        hist = trailing_loss_rate(panel, p).loc["2005Q1":].rename("rate").reset_index()
        hist["date"] = hist["quarter"].dt.to_timestamp(how="end")
        pr = paths[p].reset_index().rename(columns={"index": "quarter"})
        pr["date"] = pr["quarter"].dt.to_timestamp(how="end")
        base = alt.Chart(pr).encode(x=alt.X("date:T", title=None))
        band90 = base.mark_area(opacity=0.15, color=COLOURS[p]).encode(y=alt.Y("p05:Q", title="loss rate, % a year"), y2="p95:Q")
        band50 = base.mark_area(opacity=0.25, color=COLOURS[p]).encode(y="p25:Q", y2="p75:Q")
        line = base.mark_line(color=COLOURS[p], strokeWidth=2.5).encode(
            y="central:Q", tooltip=[alt.Tooltip("quarter:N", title="quarter"), alt.Tooltip("central:Q", format=".2f", title="projection %"),
                                    alt.Tooltip("p05:Q", format=".2f", title="5th pct"), alt.Tooltip("p95:Q", format=".2f", title="95th pct")])
        hline = alt.Chart(hist).mark_line(color="#a8a7a2", strokeWidth=1.6).encode(
            x="date:T", y="rate:Q", tooltip=[alt.Tooltip("quarter:N"), alt.Tooltip("rate:Q", format=".2f", title="actual %")])
        with cols[i % 2]:
            st.markdown(f"**{LABEL[p]}**: peak {float(table.loc[p, 'peak loss rate %']):.2f}% · "
                        f"five-year cumulative {float(table.loc[p, '5-year cumulative %']):.1f}% · {RATING[p]}")
            st.altair_chart((band90 + band50 + hline + line).properties(height=260), use_container_width=True)
    st.caption("Grey: actual trailing four-quarter loss rate. Line: projection. Bands: 50% and 90% simulation intervals "
               "(parameter uncertainty from the Newey–West covariance and block-bootstrapped residuals).")

# -------------------------------------------------------------------------------------- scenario
with tab2:
    show = path.loc["2015Q1":, ["unemployment", "bank_rate", "house_price_growth_yoy", "gdp_growth_yoy"]].copy()
    show.columns = ["Unemployment rate (%)", "Bank Rate (%)", "House price growth, y/y (%)", "Real GDP growth, y/y (%)"]
    long = show.reset_index().melt("quarter", var_name="driver", value_name="value")
    long["date"] = long["quarter"].dt.to_timestamp(how="end")
    long["part"] = np.where(long["quarter"] > JUMP_OFF, "scenario", "actual")
    ch = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None), y=alt.Y("value:Q", title=None),
        color=alt.Color("part:N", scale=alt.Scale(domain=["actual", "scenario"], range=["#a8a7a2", "#eb6834"]), title=None),
        tooltip=["quarter:N", "driver:N", alt.Tooltip("value:Q", format=".2f")],
    ).properties(height=200, width=420).facet(facet=alt.Facet("driver:N", title=None), columns=2).resolve_scale(y="independent")
    st.altair_chart(ch)
    st.caption("History to 2024 Q4 (ONS, Bank of England, HM Land Registry), then the Bank of England 2025 scenario as adjusted in the sidebar. "
               "The scenario's 8% Bank Rate is above anything in the 1993–2026 data the models were fitted on.")

# ------------------------------------------------------------------------------------------- £ losses
with tab3:
    t = table[["balance £bn", "5-year cumulative %", "5-year cumulative 5th pct %", "5-year cumulative 95th pct %",
               "5-year losses £bn", "losses if rates stayed at jump-off £bn", "stress uplift £bn"]].astype(float)
    t.index = [LABEL[p] for p in t.index]
    t.loc["Total"] = t.sum()
    for c in ["5-year cumulative %", "5-year cumulative 5th pct %", "5-year cumulative 95th pct %"]:
        t.loc["Total", c] = 100 * (t[c].iloc[:-1] / 100 * t["balance £bn"].iloc[:-1]).sum() / t.loc["Total", "balance £bn"]
    st.dataframe(t.style.format("{:,.2f}"), use_container_width=True)
    st.caption("Static balance sheet at end-2024, as in the Bank of England test. Write-offs net of recoveries, all UK banks "
               "and building societies. Total percentiles add portfolio percentiles, so they assume perfectly correlated errors.")
    cons = (float(table.loc["credit_card", "5-year cumulative %"]) * bal["credit_card"]
            + float(table.loc["other_consumer", "5-year cumulative %"]) * bal["other_consumer"]) / (bal["credit_card"] + bal["other_consumer"])
    st.markdown(
        f"**Benchmark.** Bank of England 2025 test, five-year *impairment* rates (FSR Dec 2025, Table 5.C): consumer credit "
        f"{BOE_2025_IMPAIRMENT_RATES['UK consumer credit']}%, mortgages {BOE_2025_IMPAIRMENT_RATES['UK mortgages']}%. "
        f"This model's five-year *write-off* rates: consumer credit {cons:.1f}%, mortgages "
        f"{float(table.loc['mortgage', '5-year cumulative %']):.2f}%. Impairments should exceed write-offs within five years, because IFRS 9 "
        "books lifetime losses before debts are written off."
    )

# ------------------------------------------------------------------------------------ documentation
with tab4:
    rows = []
    for p, fm in models().items():
        rows.append({"portfolio": LABEL[p], "drivers": fm.spec.describe(), "adj. R²": round(fm.rsquared_adj, 3),
                     "sample": f"{fm.sample[0]}–{fm.sample[1]}", "validation rating": RATING[p]})
    st.dataframe(pd.DataFrame(rows).set_index("portfolio"), use_container_width=True)
    st.markdown("""
- **Model form:** logit of the trailing four-quarter loss rate on lagged macro drivers, a COVID-regime control and the loss rate four quarters earlier. Newey–West standard errors.
- **Selection:** about 19,000 candidates per portfolio, filtered by economic sign rules, significance, VIF < 5 and stationarity; lowest BIC wins.
- **Validation:** out-of-time backtests through 2008 and 2020, live monitoring since 2024 Q4, block bootstrap, quantile regression, sensitivity analysis, challenger models.
- **Key limitations:** stress sensitivities are learned largely from one severe cycle (2008–10); an 8% Bank Rate is outside the fitting data; current consumer books appear less sensitive than history.

Full detail: [model development document](https://github.com/TilakHeble/uk-credit-stress-engine/blob/main/docs/model_development_document.md) ·
[validation report](https://github.com/TilakHeble/uk-credit-stress-engine/blob/main/docs/validation_report.md) ·
[executive summary](https://github.com/TilakHeble/uk-credit-stress-engine/blob/main/docs/executive_summary.md)
""")
