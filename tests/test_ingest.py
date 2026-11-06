import numpy as np
import pandas as pd
import pandera.errors
import pytest

from ukstress.config import BOE_DIR, PORTFOLIOS
from ukstress.ingest.build_panel import build_panel
from ukstress.ingest.contracts import PANEL_SCHEMA
from ukstress.ingest.readers import read_bankstats_c21, read_boe_iadb, read_ons_timeseries


def test_ons_reader_keeps_only_quarters(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text(
        '"Title","Thing"\n"CDID","ABCD"\n"Source dataset ID","X"\n"PreUnit",""\n"Unit","%"\n'
        '"Release date","01-01-2026"\n"Next release","x"\n"Important notes",\n'
        '"2020","5"\n"2020 Q1","1"\n"2020 Q2","2"\n"2020 JAN","9"\n'
    )
    s = read_ons_timeseries(f)
    assert list(s.index.astype(str)) == ["2020Q1", "2020Q2"]
    assert s.tolist() == [1.0, 2.0] and s.name == "ABCD"


def test_boe_reader_averages_monthly_and_takes_quarter_end(tmp_path):
    f = tmp_path / "boe.csv"
    f.write_text(
        "DATE,QSER,MSER\n31 Jan 2020,,1\n29 Feb 2020,,2\n31 Mar 2020,10,3\n30 Apr 2020,,4\n"
        "31 May 2020,,5\n30 Jun 2020,20,6\n"
    )
    df = read_boe_iadb(f)
    assert df.loc[pd.Period("2020Q1"), "QSER"] == 10
    assert df.loc[pd.Period("2020Q1"), "MSER"] == pytest.approx(2.0)
    assert df.loc[pd.Period("2020Q2"), "MSER"] == pytest.approx(5.0)


real = pytest.mark.skipif(not (BOE_DIR / "balances_rates.csv").exists(), reason="raw data absent")


@real
def test_bankstats_table_agrees_with_database_extract():
    db = read_boe_iadb(BOE_DIR / "balances_rates.csv")
    c21 = read_bankstats_c21(BOE_DIR / "bankstats_C2_1_latest.xls")
    assert len(c21) >= 20  # ~5 years of quarters; year label appears only on Q1 rows
    for code in ["RPQTFHD", "RPQTFHB"]:
        both = pd.concat([db[code], c21[code]], axis=1).dropna()
        assert len(both) >= 20
        np.testing.assert_allclose(both.iloc[:, 0], both.iloc[:, 1])


@real
def test_panel_contract_and_loss_rate_definition():
    panel = build_panel()
    for p in PORTFOLIOS:
        expected = 400 * panel[f"wo_{p}"] / panel[f"bal_{p}"].shift(1)
        # first quarter's lagged balance lies before the panel window, so compare from row 2
        pd.testing.assert_series_equal(
            panel[f"lr_{p}"].iloc[1:], expected.iloc[1:], check_names=False
        )
    assert panel["unemployment"].notna().all()


def test_contract_rejects_impossible_values():
    idx = pd.period_range("2000Q1", periods=2, freq="Q", name="quarter")
    cols = {c: [1.0, 1.0] for c in PANEL_SCHEMA.columns}
    bad = pd.DataFrame(cols, index=idx)
    bad["unemployment"] = [50.0, 4.0]
    with pytest.raises(pandera.errors.SchemaError):
        PANEL_SCHEMA.validate(bad)
