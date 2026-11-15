"""
Export the CAPM and FF3 fits of the six size/value portfolios from the HW 1 pipeline.

Reads the HW 1 pipeline's six Fama-French (1993) portfolios and Ken French's
factors through its own `calc_CAPM_and_FF3.load_test_assets`, runs its
`calc_capm_table` and `calc_ff3_table` on two windows (every common month, and
the last 60), runs a joint test that all six alphas are zero (Gibbons, Ross,
and Shanken 1989) for each model and window, and writes three derived CSVs
beside this script:

- `capm_ff3_fits.csv`: one row per window, model, and portfolio.
- `joint_alpha_tests.csv`: one row per window and model.
- `size_value_returns.csv`: the six portfolios' monthly returns.

Usage: python export_hw1_fits.py --hw1-dir ../hw1_rerun
"""

import argparse
import sys
from pathlib import Path

import finm
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
SIX = ["SL", "SME", "SH", "BL", "BME", "BH"]
RECENT_MONTHS = 60
CUTOFF = pd.Timestamp("2026-10-31")  # the class report's data cutoff
MODELS = {"CAPM": ["Mkt-RF"], "FF3": ["Mkt-RF", "SMB", "HML"]}


def load_hw1(hw1_dir: Path):
    """
    Import the HW 1 regression module from a local checkout.

    Inputs:
        hw1_dir (Path): root of the HW 1 checkout whose `doit` has run

    Returns (module): the HW 1 `calc_CAPM_and_FF3` module.
    """
    sys.path.insert(0, str(hw1_dir / "src"))
    import calc_CAPM_and_FF3

    return calc_CAPM_and_FF3


def six_portfolio_assets(hw1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the six portfolios and the factors on the months they share, through the cutoff.

    Inputs:
        hw1 (module): the HW 1 `calc_CAPM_and_FF3` module

    Returns (tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]): raw returns, factors, excess returns.
    """
    data_dir = hw1.DATA_DIR
    size_bm = pd.read_parquet(Path(data_dir) / "FF_1993_vwret.parquet").pivot(
        index="jdate", columns="sbport", values="vwret"
    )[SIX]
    size_bm.columns.name = None
    factors = pd.read_parquet(Path(data_dir) / "FF_FACTORS.parquet").rename(
        columns={"date": "jdate", "mktrf": "Mkt-RF", "smb": "SMB", "hml": "HML", "rf": "RF"}
    )
    factors = factors.set_index("jdate")[[*hw1.FACTORS, "RF"]].astype("float64")
    common = size_bm.dropna().index.intersection(factors.dropna().index)
    common = common[common <= CUTOFF]
    portfolios = size_bm.loc[common]
    factors = factors.loc[common]
    excess = portfolios.sub(factors["RF"], axis=0)
    return portfolios, factors, excess


def loading_stats(excess_one: pd.Series, factors: pd.DataFrame) -> dict:
    """
    The t-statistic of each factor loading, and the R-squared with HML left out.

    Uses the same regression routine as HW 1 (finm.run_factor_regression, OLS).

    Inputs:
        excess_one (pd.Series): one portfolio's monthly excess returns
        factors (pd.DataFrame): the model's factors on the same months (Mkt-RF, and SMB, HML for FF3)

    Returns (dict): mkt_t, smb_t, hml_t (NaN if not in the model), and r_squared_mkt_smb (NaN for CAPM).
    """
    full = finm.run_factor_regression(excess_one, factors)
    out = {
        "mkt_t": full.beta_tstats["Mkt-RF"],
        "smb_t": full.beta_tstats.get("SMB", np.nan),
        "hml_t": full.beta_tstats.get("HML", np.nan),
        "r_squared_mkt_smb": np.nan,
    }
    if "HML" in factors.columns:
        out["r_squared_mkt_smb"] = finm.run_factor_regression(
            excess_one, factors[["Mkt-RF", "SMB"]]
        ).r_squared
    return out


def fit_window(
    hw1, portfolios: pd.DataFrame, factors: pd.DataFrame, excess: pd.DataFrame, window: str
) -> pd.DataFrame:
    """
    Run the HW 1 CAPM and FF3 regressions on one window, in long format.

    Inputs:
        hw1 (module): the HW 1 `calc_CAPM_and_FF3` module
        portfolios (pd.DataFrame): raw monthly returns of the six portfolios on the window
        factors (pd.DataFrame): Mkt-RF, SMB, HML, RF on the same months
        excess (pd.DataFrame): portfolios minus RF on the same months
        window (str): the window's name, `full` or `last_60m`

    Returns (pd.DataFrame): one row per model and portfolio.
    """
    tables = {
        "CAPM": hw1.calc_capm_table(excess, factors),
        "FF3": hw1.calc_ff3_table(portfolios, factors),
    }
    rows = []
    for model, table in tables.items():
        for portfolio in SIX:
            r = table.loc[portfolio]
            rows.append(
                {
                    "window": window,
                    "window_start": portfolios.index.min().date(),
                    "window_end": portfolios.index.max().date(),
                    "n_months": len(portfolios),
                    "model": model,
                    "portfolio": portfolio,
                    "alpha_annual": r["Alpha (annual)"],
                    "alpha_t": r["t(Alpha)"],
                    "beta_mkt": r["Beta Mkt-RF"],
                    "beta_smb": r.get("Beta SMB"),
                    "beta_hml": r.get("Beta HML"),
                    "r_squared": r["R-squared"],
                    **loading_stats(excess[portfolio], factors[MODELS[model]]),
                }
            )
    return pd.DataFrame(rows)


def joint_alpha_test(excess: pd.DataFrame, factors: pd.DataFrame) -> dict:
    """
    Test that every portfolio's alpha is zero at once (Gibbons, Ross, and Shanken 1989).

    Inputs:
        excess (pd.DataFrame): monthly excess returns, T rows by N portfolios
        factors (pd.DataFrame): monthly factor returns on the same rows, T by K

    Returns (dict): f_stat, df_num (N), df_den (T - N - K), and p_value of the F test.
    """
    T, N = excess.shape
    K = factors.shape[1]
    X = np.column_stack([np.ones(T), factors.to_numpy()])
    B = np.linalg.lstsq(X, excess.to_numpy(), rcond=None)[0]
    alpha = B[0]
    resid = excess.to_numpy() - X @ B
    sigma = resid.T @ resid / T  # residual covariance across portfolios (MLE)
    mu = factors.to_numpy().mean(axis=0)
    omega = np.atleast_2d(np.cov(factors.to_numpy().T, ddof=0))  # factor covariance (MLE)
    f_stat = (
        (T - N - K)
        / N
        * (alpha @ np.linalg.solve(sigma, alpha))
        / (1 + mu @ np.linalg.solve(omega, mu))
    )
    df_den = T - N - K
    return {
        "f_stat": f_stat,
        "df_num": N,
        "df_den": df_den,
        "p_value": float(stats.f.sf(f_stat, N, df_den)),
    }


def build_fits(hw1, portfolios: pd.DataFrame, factors: pd.DataFrame, excess: pd.DataFrame) -> pd.DataFrame:
    """
    Fit both models on the full sample and on the last 60 months.

    Inputs:
        hw1 (module): the HW 1 `calc_CAPM_and_FF3` module
        portfolios (pd.DataFrame): raw monthly returns of the six portfolios
        factors (pd.DataFrame): Mkt-RF, SMB, HML, RF
        excess (pd.DataFrame): portfolios minus RF

    Returns (pd.DataFrame): 24 rows, window x model x portfolio.
    """
    recent = portfolios.index[-RECENT_MONTHS:]
    return pd.concat(
        [
            fit_window(hw1, portfolios, factors, excess, "full"),
            fit_window(
                hw1, portfolios.loc[recent], factors.loc[recent], excess.loc[recent], "last_60m"
            ),
        ],
        ignore_index=True,
    )


def build_joint_tests(factors: pd.DataFrame, excess: pd.DataFrame) -> pd.DataFrame:
    """
    Run the joint test of alphas for both models on the full sample and the last 60 months.

    Inputs:
        factors (pd.DataFrame): Mkt-RF, SMB, HML, RF
        excess (pd.DataFrame): the six portfolios' excess returns on the same months

    Returns (pd.DataFrame): 4 rows, window x model.
    """
    windows = {"full": excess.index, "last_60m": excess.index[-RECENT_MONTHS:]}
    rows = []
    for window, months in windows.items():
        for model, cols in MODELS.items():
            rows.append(
                {
                    "window": window,
                    "window_start": months.min().date(),
                    "window_end": months.max().date(),
                    "n_months": len(months),
                    "model": model,
                    **joint_alpha_test(excess.loc[months, SIX], factors.loc[months, cols]),
                }
            )
    return pd.DataFrame(rows)


def returns_long(portfolios: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the six portfolios' returns to one row per month and portfolio.

    Inputs:
        portfolios (pd.DataFrame): raw monthly returns, one column per portfolio

    Returns (pd.DataFrame): columns date, portfolio, ret.
    """
    out = portfolios.rename_axis("date").reset_index().melt(
        id_vars="date", var_name="portfolio", value_name="ret"
    )
    out["date"] = out["date"].dt.date
    return out.sort_values(["date", "portfolio"], key=_portfolio_order).reset_index(drop=True)


def _portfolio_order(col: pd.Series) -> pd.Series:
    """
    Sort key that puts portfolios in SL, SME, SH, BL, BME, BH order.

    Inputs:
        col (pd.Series): a column being sorted

    Returns (pd.Series): the column, or the portfolio's position in SIX.
    """
    return col.map(SIX.index) if col.name == "portfolio" else col


def main(hw1_dir: Path) -> None:
    """
    Write capm_ff3_fits.csv, joint_alpha_tests.csv, and size_value_returns.csv beside this script.

    Inputs:
        hw1_dir (Path): root of the HW 1 checkout whose `doit` has run

    Returns (None): writes three CSVs and prints a summary.
    """
    hw1 = load_hw1(hw1_dir)
    portfolios, factors, excess = six_portfolio_assets(hw1)
    fits = build_fits(hw1, portfolios, factors, excess)
    fits.to_csv(HERE / "capm_ff3_fits.csv", index=False, float_format="%.10g", lineterminator="\n")
    joint = build_joint_tests(factors, excess)
    joint.to_csv(
        HERE / "joint_alpha_tests.csv", index=False, float_format="%.10g", lineterminator="\n"
    )
    returns_long(portfolios).to_csv(
        HERE / "size_value_returns.csv", index=False, float_format="%.10g", lineterminator="\n"
    )
    print(f"months: {portfolios.index.min():%Y-%m} to {portfolios.index.max():%Y-%m} ({len(portfolios)})")
    print(fits.round(3).to_string())
    print(joint.round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw1-dir", type=Path, default=HERE.parent / "hw1_rerun")
    main(parser.parse_args().hw1_dir.resolve())
