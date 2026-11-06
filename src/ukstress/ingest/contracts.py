"""Data contract for the modelling panel (Pandera).

The checks encode things that must be true of the data, so a bad download or a silent change in a
source file fails loudly at build time instead of quietly moving a model coefficient.
"""

import pandera.pandas as pa
from pandera.pandas import Check, Column

from ukstress.config import PORTFOLIOS

_rate = Column(float, Check.in_range(-1.0, 40.0), nullable=True)  # annualised loss rate, %
_money = Column(float, nullable=True)

_columns = {}
for p in PORTFOLIOS:
    _columns[f"wo_{p}"] = _money  # write-offs can be slightly negative (recoveries)
    _columns[f"bal_{p}"] = Column(float, Check.gt(0), nullable=True)
    _columns[f"lr_{p}"] = _rate

_columns |= {
    "unemployment": Column(float, Check.in_range(2, 15), nullable=True),
    "gdp_growth_yoy": Column(float, Check.in_range(-30, 30), nullable=True),
    "cpi_inflation": Column(float, Check.in_range(-3, 20), nullable=True),
    "house_price_growth_yoy": Column(float, Check.in_range(-30, 40), nullable=True),
    "bank_rate": Column(float, Check.in_range(0, 20), nullable=True),
    "mortgage_rate_2y_fixed": Column(float, Check.in_range(0, 20), nullable=True),
    "unemployment_chg_yoy": Column(float, nullable=True),
    "bank_rate_chg_yoy": Column(float, nullable=True),
}

PANEL_SCHEMA = pa.DataFrameSchema(
    _columns,
    index=pa.Index("period[Q-DEC]", unique=True, name="quarter"),
    strict=True,
    coerce=True,
    checks=[Check(lambda df: df.index.is_monotonic_increasing, error="quarters out of order")],
)
