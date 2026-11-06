"""Readers for the raw Bank of England and ONS downloads.

Each reader returns a pandas object indexed by a quarterly ``PeriodIndex`` so that every source
joins on the same key. Readers never fill or interpolate: gaps stay visible until the panel build
decides how to treat them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_QUARTER_ROW = re.compile(r"^(\d{4}) Q([1-4])$")


def read_ons_timeseries(path: str | Path, name: str | None = None) -> pd.Series:
    """Read an ONS ``generator?format=csv`` download and keep the quarterly rows only.

    ONS files hold metadata rows, then annual, quarterly and monthly observations in one column.
    """
    raw = pd.read_csv(path, header=None, names=["key", "value"], dtype=str)
    meta = dict(zip(raw.key.head(8), raw.value.head(8), strict=False))
    rows = raw[raw.key.str.match(_QUARTER_ROW, na=False)]
    idx = pd.PeriodIndex([k.replace(" ", "") for k in rows.key], freq="Q")
    s = pd.Series(pd.to_numeric(rows.value, errors="coerce").values, index=idx)
    s.name = name or meta.get("CDID", Path(path).stem)
    s.attrs = {
        "title": meta.get("Title"),
        "cdid": meta.get("CDID"),
        "release": meta.get("Release date"),
    }
    if s.index.has_duplicates:
        raise ValueError(f"duplicate quarters in {path}")
    return s.sort_index()


def read_boe_iadb(path: str | Path) -> pd.DataFrame:
    """Read a Bank of England database CSV export (``CSVF=TN``: DATE + one column per series).

    Quarter-end rows are kept for quarterly series. Monthly series (e.g. quoted mortgage rates)
    are averaged within each quarter so every column is on a quarterly basis.
    """
    raw = pd.read_csv(path)
    raw["DATE"] = pd.to_datetime(raw["DATE"], format="%d %b %Y")
    raw = raw.set_index("DATE").apply(pd.to_numeric, errors="coerce")
    q = raw.index.to_period("Q")
    out = {}
    for col in raw.columns:
        s = raw[col].dropna()
        months_per_q = s.groupby(s.index.to_period("Q")).size()
        if (months_per_q > 1).any():  # monthly series -> quarterly mean
            out[col] = s.groupby(s.index.to_period("Q")).mean()
        else:  # quarterly series -> value at quarter end
            out[col] = s.groupby(s.index.to_period("Q")).last()
    df = pd.DataFrame(out).reindex(pd.period_range(q.min(), q.max(), freq="Q"))
    df.index.name = "quarter"
    return df


def read_bankstats_c21(path: str | Path) -> pd.DataFrame:
    """Read the sterling sheet of Bankstats Table C2.1 (write-offs, latest ~5 years).

    Used to cross-check the database extract; codes are rebuilt from the split header
    (prefix ``RPQ`` + per-column suffix such as ``TFHE``).
    """
    t = pd.read_excel(path, sheet_name="Sterling", header=None)
    code_row = t.index[t.iloc[:, 1].astype(str).eq("RPQ")][0]
    suffix = t.iloc[code_row]
    cols = {
        j: f"RPQ{suffix[j]}"
        for j in range(2, t.shape[1])
        if isinstance(suffix[j], str) and re.fullmatch(r"[A-Z0-9]{4}", suffix[j])
    }
    body = t.iloc[code_row + 1 :]
    # Labels carry the year only on Q1 (and the first row): "2021 Q1", "Q2", "Q3", ...
    labels = body.iloc[:, 1].astype(str).str.strip()
    year = labels.str.extract(r"^(\d{4})")[0].ffill()
    qtr = labels.str.extract(r"Q([1-4])$")[0]
    keep = qtr.notna() & year.notna()
    body = body[keep]
    idx = pd.PeriodIndex(year[keep] + "Q" + qtr[keep], freq="Q", name="quarter")
    df = body[list(cols)].rename(columns=cols).replace("-", 0).apply(pd.to_numeric, errors="coerce")
    df.index = idx
    return df


def read_ukhpi(path: str | Path) -> pd.Series:
    """HM Land Registry UK HPI download -> quarterly average of the all-property index.

    Before 1995 the index is quarterly and repeated on each month, so the quarterly mean is exact.
    """
    raw = pd.read_csv(path, usecols=["Period", "House price index All property types"])
    raw["quarter"] = pd.PeriodIndex(raw["Period"], freq="M").asfreq("Q")
    s = raw.groupby("quarter")["House price index All property types"].mean()
    s.name = "house_price_index"
    return s


def read_bcst_paths(path: str | Path, sheet: str = "Macroeconomic variables") -> pd.DataFrame:
    """Bank of England 2025 Bank Capital Stress Test variable paths (history + 2025-29 scenario).

    Returns one row per quarter with the variable names from the workbook as columns and a
    boolean ``is_projection`` column marking the scenario quarters.
    """
    t = pd.read_excel(path, sheet_name=sheet, header=None)
    names = t.iloc[1]
    keep = {j: str(names[j]).strip() for j in range(1, t.shape[1]) if isinstance(names[j], str)}
    labels = t.iloc[:, 0].astype(str).str.strip()
    proj_start = labels[labels.eq("Projections")].index[0]
    rows = labels.str.fullmatch(r"Q[1-4] \d{4}")
    body = t.loc[rows, list(keep)].rename(columns=keep).apply(pd.to_numeric, errors="coerce")
    body.index = pd.PeriodIndex(
        labels[rows].str.replace(r"Q([1-4]) (\d{4})", r"\2Q\1", regex=True),
        freq="Q",
        name="quarter",
    )
    body["is_projection"] = rows[rows].index > proj_start
    return body
