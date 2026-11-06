import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

from ukstress.models.ols_scratch import (
    add_constant,
    cooks_distance,
    fit_gradient_descent,
    fit_ols,
    leverage,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=(n, 3))
    y = 1.5 + x @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=1 + np.abs(x[:, 0]), size=n)
    return add_constant(x), y


@pytest.mark.parametrize("cov", ["nonrobust", "HC0", "HC1", "HAC"])
def test_matches_statsmodels(data, cov):
    X, y = data
    ours = fit_ols(X, y, cov_type=cov)
    kw = {"cov_kwds": {"maxlags": 4}} if cov == "HAC" else {}
    ref = sm.OLS(y, X).fit(cov_type=cov, use_t=True, **kw)  # t-based p-values, like ours
    np.testing.assert_allclose(ours.params, ref.params, rtol=1e-10)
    np.testing.assert_allclose(ours.bse, ref.bse, rtol=1e-10)
    np.testing.assert_allclose(ours.pvalues, ref.pvalues, rtol=1e-8, atol=1e-14)
    assert ours.rsquared == pytest.approx(ref.rsquared, rel=1e-12)
    assert ours.rsquared_adj == pytest.approx(ref.rsquared_adj, rel=1e-12)
    if cov == "nonrobust":
        assert ours.fvalue == pytest.approx(ref.fvalue, rel=1e-10)
        np.testing.assert_allclose(ours.conf_int(), ref.conf_int(), rtol=1e-10)


def test_influence_matches_statsmodels(data):
    X, y = data
    ours = fit_ols(X, y)
    infl = OLSInfluence(sm.OLS(y, X).fit())
    np.testing.assert_allclose(leverage(X), infl.hat_matrix_diag, rtol=1e-10)
    np.testing.assert_allclose(cooks_distance(ours, X), infl.cooks_distance[0], rtol=1e-10)


def test_gradient_descent_converges_to_normal_equation(data):
    X, y = data
    Xs = X.copy()
    Xs[:, 1:] = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0)
    beta_gd, hist = fit_gradient_descent(Xs, y, lr=0.1)
    np.testing.assert_allclose(beta_gd, fit_ols(Xs, y).params, atol=1e-5)
    assert np.all(np.diff(hist) <= 1e-12)  # loss never increases with a stable step size


def test_rejects_underdetermined():
    with pytest.raises(ValueError):
        fit_ols(np.ones((3, 3)), np.ones(3))


def test_nested_f_and_information_criteria_match_statsmodels(data):
    from ukstress.models.ols_scratch import f_test_nested, information_criteria

    X, y = data
    full, restricted = sm.OLS(y, X).fit(), sm.OLS(y, X[:, :2]).fit()
    f, p = f_test_nested(restricted.ssr, full.ssr, 2, int(full.df_resid))
    ref_f, ref_p, _ = full.compare_f_test(restricted)
    assert f == pytest.approx(ref_f) and p == pytest.approx(ref_p)
    aic, bic = information_criteria(full.ssr, len(y), X.shape[1])
    assert aic == pytest.approx(full.aic) and bic == pytest.approx(full.bic)
