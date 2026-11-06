import numpy as np
import pandas as pd
import pytest

from ukstress.features.build import (
    JUMP_OFF,
    add_changes,
    inv_logit_pct,
    logit_pct,
    scaled_scenario,
    scenario_macro,
)
from ukstress.models.satellite import (
    FittedModel,
    Spec,
    candidate_specs,
    design,
    fit,
    project,
    sign_ok,
    vif,
)


def toy(n=80, seed=1):
    idx = pd.period_range("2000Q1", periods=n, freq="Q")
    rng = np.random.default_rng(seed)
    macro = pd.DataFrame({
        "unemployment": 5 + np.cumsum(rng.normal(0, 0.2, n)),
        "gdp_growth_yoy": rng.normal(2, 1, n),
        "house_price_growth_yoy": rng.normal(4, 3, n),
        "bank_rate": np.clip(4 + np.cumsum(rng.normal(0, 0.2, n)), 0, None),
        "cpi_inflation": rng.normal(2, 0.5, n),
    }, index=idx)
    macro = add_changes(macro)
    y = -4 + 0.3 * macro["unemployment"].shift(2) + rng.normal(0, 0.05, n)
    return y.rename("y"), macro


def test_logit_roundtrip():
    r = np.array([0.01, 1.0, 5.0, 30.0])
    np.testing.assert_allclose(inv_logit_pct(logit_pct(r)), r)


def test_design_uses_lagged_values():
    y, macro = toy()
    spec = Spec("x", (("unemployment", 2),), dynamic=True)
    X = design(spec, y, macro)
    q = X.index[10]
    assert X.loc[q, "unemployment_L2"] == macro.loc[q - 2, "unemployment"]
    assert X.loc[q, "y_L4"] == y.loc[q - 4]
    assert list(X.columns) == ["const", "unemployment_L2", "covid", "y_L4"]


def test_fit_recovers_known_coefficient_and_sign_rules():
    y, macro = toy()
    fm, _ = fit(Spec("x", (("unemployment", 2),), covid=False), y, macro, start=None)
    assert fm.params["unemployment_L2"] == pytest.approx(0.3, abs=0.03)
    assert sign_ok(fm.spec, fm.params)
    flipped = fm.params.copy()
    flipped["unemployment_L2"] *= -1
    assert not sign_ok(fm.spec, flipped)


def test_candidate_specs_exclude_same_family():
    for s in candidate_specs("x", max_terms=2, lags=(0,)):
        ds = {d for d, _ in s.terms}
        assert not {"unemployment", "unemployment_chg_yoy"} <= ds
        assert not {"bank_rate", "bank_rate_chg_yoy"} <= ds


def test_vif_detects_collinearity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = pd.DataFrame({"const": 1.0, "a": a, "b": a + rng.normal(0, 0.01, 200), "c": rng.normal(size=200)})
    v = vif(X)
    assert v["a"] > 100 and v["c"] < 1.5


def test_static_projection_equals_prediction_and_serialisation_roundtrip():
    y, macro = toy()
    fm, _ = fit(Spec("x", (("unemployment", 2),), covid=False), y, macro, start=None)
    start = macro.index[60]
    pr = project(fm, y, macro, start=start)
    X = design(fm.spec, y, macro, index=pr.index)
    np.testing.assert_allclose(pr.central, inv_logit_pct(fm.predict(X)))
    fm2 = FittedModel.from_dict(fm.to_dict())
    pd.testing.assert_series_equal(fm.params, fm2.params)


def test_dynamic_projection_is_recursive():
    y, macro = toy()
    fm, _ = fit(Spec("x", (("unemployment", 2),), dynamic=True, covid=False), y, macro, start=None)
    start = macro.index[60]
    pr = project(fm, y, macro, start=start)
    q = pr.index[5]  # y[t-4] must be the model's own projection, not the actual value
    x = design(fm.spec, y, macro, index=[q]).iloc[0].copy()
    x["y_L4"] = logit_pct(pr.central.loc[q - 4])
    assert inv_logit_pct(x @ fm.params) == pytest.approx(pr.central.loc[q])


def test_simulated_bands_are_ordered():
    y, macro = toy()
    fm, _ = fit(Spec("x", (("unemployment", 2),), dynamic=True, covid=False), y, macro, start=None)
    pr = project(fm, y, macro, start=macro.index[60], n_sims=100)
    assert (pr.p05 <= pr.p50).all() and (pr.p50 <= pr.p95).all()


@pytest.fixture(scope="module")
def real_macro():
    from ukstress.config import BCST_PATHS_FILE

    if not BCST_PATHS_FILE.exists():
        pytest.skip("raw data absent")
    from ukstress.ingest.build_panel import build_macro

    return build_macro()


def test_scenario_path_spliced_at_jump_off(real_macro):
    path = scenario_macro(real_macro)
    assert path.index.max() == pd.Period("2029Q4")
    pd.testing.assert_series_equal(path.loc[:JUMP_OFF, "unemployment"],
                                   real_macro.loc[:JUMP_OFF, "unemployment"], check_names=False)
    assert path.loc["2027Q1", "unemployment"] == pytest.approx(8.5)
    assert path.loc["2025Q4", "bank_rate"] == pytest.approx(8.0)


def test_severity_scaling(real_macro):
    path = scenario_macro(real_macro)
    flat = scaled_scenario(path, 0.0)
    proj = flat.index > JUMP_OFF
    assert (flat.loc[proj, "unemployment"] == path.loc[JUMP_OFF, "unemployment"]).all()
    pd.testing.assert_frame_equal(scaled_scenario(path, 1.0), path)
