# CAPM and FF3 Fits of the Six Size/Value Portfolios

Derived tables from my FINM 32800 HW 1 pipeline, published for the class
report ([finm-32800/finm_midterm_report](https://github.com/finm-32800/finm_midterm_report),
issue #116: *Does the CAPM price the size and value portfolios?*).

| File | ChartBook dataframe | What it holds |
|---|---|---|
| `capm_ff3_fits.csv` | `capm_ff3_fits` | Alpha, t(alpha), betas and their t-statistics, and R-squared (with and without HML) of the CAPM and the Fama-French three-factor model for each of the six size/book-to-market portfolios, over the full sample and the last 60 months |
| `joint_alpha_tests.csv` | `joint_alpha_tests` | A joint test that all six alphas are zero (Gibbons, Ross, and Shanken 1989), for each model and window |
| `size_value_returns.csv` | `size_value_returns` | The six portfolios' value-weighted monthly returns |

`chartbook.toml` documents every column. Data through September 2026, the last
month of CRSP's monthly stock file when the tables were exported.

## Derived data only

The portfolios are built from CRSP and Compustat (licensed through WRDS). This
repository holds only portfolio-level returns and regression statistics,
never a stock-level record. The factors and the risk-free rate are Ken French's.

## How the tables are made

`export_hw1_fits.py` runs against a local checkout of the HW 1 pipeline after
its `doit` has built the six portfolios and pulled the factors. It reuses the
HW 1 regression code (`calc_CAPM_and_FF3.py`) on each window, adds the joint
test of alphas, and writes the three CSVs. `test_export_hw1_fits.py` checks the
joint test (`pytest test_export_hw1_fits.py`). The HW 1 solutions themselves are not published.

```bash
python export_hw1_fits.py --hw1-dir ../hw1_rerun
```

The CSVs are committed, not built: ChartBook reads them as committed, so a
clone holds the data (`.gitattributes` keeps their line endings fixed, so
their hashes match on every OS).
