"""Tests of the joint test of alphas in export_hw1_fits.py, on simulated data."""

import numpy as np
import pandas as pd
import pytest

import export_hw1_fits as export


def simulated(n_months: int, n_portfolios: int, alpha: float, seed: int = 0):
    """
    Simulate monthly factor and portfolio excess returns with a known alpha.

    Inputs:
        n_months (int): number of months T
        n_portfolios (int): number of portfolios N
        alpha (float): monthly alpha of every portfolio
        seed (int): random seed

    Returns (tuple[pd.DataFrame, pd.DataFrame]): excess returns (T x N), factors (T x 1).
    """
    rng = np.random.default_rng(seed)
    mkt = pd.DataFrame({"Mkt-RF": rng.normal(0.006, 0.045, n_months)})
    noise = rng.normal(0, 0.02, (n_months, n_portfolios))
    excess = pd.DataFrame(alpha + mkt.values * 1.1 + noise, columns=[f"P{i}" for i in range(n_portfolios)])
    return excess, mkt


def test_with_one_portfolio_the_joint_test_is_the_squared_alpha_t_stat():
    excess, mkt = simulated(n_months=120, n_portfolios=1, alpha=0.002)
    X = np.column_stack([np.ones(len(mkt)), mkt.values])
    b = np.linalg.lstsq(X, excess.values[:, 0], rcond=None)[0]
    e = excess.values[:, 0] - X @ b
    se_alpha = np.sqrt(e @ e / (len(e) - 2) * np.linalg.inv(X.T @ X)[0, 0])
    result = export.joint_alpha_test(excess, mkt)
    assert result["f_stat"] == pytest.approx((b[0] / se_alpha) ** 2)
    assert (result["df_num"], result["df_den"]) == (1, 120 - 1 - 1)


def test_with_one_portfolio_the_p_value_is_the_two_sided_t_test():
    from scipy import stats

    excess, mkt = simulated(n_months=60, n_portfolios=1, alpha=0.004, seed=3)
    result = export.joint_alpha_test(excess, mkt)
    t = np.sqrt(result["f_stat"])
    assert result["p_value"] == pytest.approx(2 * stats.t.sf(t, df=60 - 2))


def test_portfolio_order_does_not_change_the_joint_test():
    excess, mkt = simulated(n_months=60, n_portfolios=4, alpha=0.001, seed=7)
    forward = export.joint_alpha_test(excess, mkt)
    backward = export.joint_alpha_test(excess[excess.columns[::-1]], mkt)
    assert forward["f_stat"] == pytest.approx(backward["f_stat"])


def test_loading_stats_match_an_ols_by_hand():
    rng = np.random.default_rng(11)
    T = 90
    factors = pd.DataFrame(
        rng.normal(0.005, 0.03, (T, 3)), columns=["Mkt-RF", "SMB", "HML"]
    )
    y = pd.Series(0.001 + factors.to_numpy() @ [1.0, 0.2, 0.8] + rng.normal(0, 0.01, T))
    stats_ = export.loading_stats(y, factors)

    X = np.column_stack([np.ones(T), factors.to_numpy()])
    b = np.linalg.lstsq(X, y.to_numpy(), rcond=None)[0]
    e = y.to_numpy() - X @ b
    se = np.sqrt(e @ e / (T - 4) * np.diag(np.linalg.inv(X.T @ X)))
    assert stats_["hml_t"] == pytest.approx(b[3] / se[3])
    assert stats_["smb_t"] == pytest.approx(b[2] / se[2])
    assert stats_["mkt_t"] == pytest.approx(b[1] / se[1])

    # R-squared with HML left out: regress on the market and SMB only.
    Xs = X[:, :3]
    es = y.to_numpy() - Xs @ np.linalg.lstsq(Xs, y.to_numpy(), rcond=None)[0]
    tss = ((y - y.mean()) ** 2).sum()
    assert stats_["r_squared_mkt_smb"] == pytest.approx(1 - es @ es / tss)


def test_loading_stats_for_the_capm_have_only_a_market_t():
    rng = np.random.default_rng(12)
    mkt = pd.DataFrame({"Mkt-RF": rng.normal(0.005, 0.04, 60)})
    y = pd.Series(mkt["Mkt-RF"] * 1.2 + rng.normal(0, 0.01, 60))
    stats_ = export.loading_stats(y, mkt)
    assert stats_["mkt_t"] > 0
    assert np.isnan(stats_["smb_t"]) and np.isnan(stats_["hml_t"])
    assert np.isnan(stats_["r_squared_mkt_smb"])
