# raft-frl

Replication code for:

> Kim, T. (2026). Carrying regime uncertainty forward in cryptocurrency tail-risk forecasting. *Finance Research Letters*. https://doi.org/10.1016/j.frl.2026.110286

This repository contains the full replication package for the above paper. All tables and figures in the manuscript and supplementary material can be reproduced by running the scripts in order.

Implementation of HAR-RV, MS-HAR, Hard-HMM, and RAFT for cryptocurrency volatility and tail-risk forecasting using Binance Futures minute-level data (2020–2025).

Place all files in the same directory before running.

## Structure

```text
raft-frl/
├── config.yaml
├── 01_download.py                # Download OHLCV data + compute mean daily volume (Table 1)
├── 02_analysis.py                # Regime labeling, correlation decay, visualization (Table 2, Figure 1)
├── 03_forecast.py                # Rolling OOS forecast, DM tests, ablation (Tables 3–5)
├── 04_var_backtest.py            # VaR/CVaR backtest, Kupiec POF test (Table 6)
├── S1a_benchmark_forecasts.py    # Supplementary benchmark forecasts (Table S1a)
├── S1b_benchmark_var_backtest.py # Supplementary benchmark VaR/CVaR backtest (Table S1b)
├── S2_threshold_sensitivity.py   # Threshold-sensitivity analysis (Table S2)
├── S3_alt_partitions.py          # Alternative adverse-state evaluation (Table S3)
├── requirements.txt
├── data/                         # auto-created, git-ignored
└── results/                      # auto-created, git-ignored
```

## Data

Raw OHLCV data are sourced from the Binance Futures API and are not redistributed in this package. Running `01_download.py` will fetch and cache the full sample locally under `data/`. No API keys are required; all endpoints used are public REST APIs.

A successful run of `01_download.py` produces data covering 5 symbols over 3,156,480 one-minute timestamps (2020-01-01 to 2025-12-31).

## Setup

Tested with Python 3.12.4 on Windows.

```bash
pip install -r requirements.txt
```

## Usage

Approximate runtimes on a consumer desktop (8-core CPU, standard broadband):

| Script | Runtime |
|---|---|
| `01_download.py` | ~40 min |
| `02_analysis.py` | ~1 min |
| `03_forecast.py` | ~40 min |
| `04_var_backtest.py` | ~20 min |

```bash
python 01_download.py
python 02_analysis.py
python 03_forecast.py
python 04_var_backtest.py
```

### Supplementary analyses

The following scripts are optional and reproduce the supplementary benchmark, threshold-sensitivity, and alternative adverse-state results after the main preprocessing files have been generated.

```bash
python S1a_benchmark_forecasts.py
python S1b_benchmark_var_backtest.py
python S2_threshold_sensitivity.py
python S3_alt_partitions.py
```

### CLI flags

| Script | Flag | Description |
|---|---|---|
| `01_download.py` | `--force` | Re-download from scratch |
| `01_download.py` | `--skip-download` | Skip download, recompute Table 1 only |
| `02_analysis.py` | `--force` | Recompute even if cached files exist |
| All | `--config PATH` | Use a custom config file |

## Config

All parameters are in `config.yaml`.

| Field | Description |
|---|---|
| `paths.base` | Root directory for data and results (default: `.`) |
| `symbols` | Trading pairs |
| `download.start` / `end` | Date range |
| `analysis.rolling_vol_window` | Volatility rolling window (minutes) |
| `analysis.extreme_sigma` | Sigma threshold for extreme move detection |
| `analysis.regime_calib_hours` | Lookback for regime quantile calibration |
| `analysis.numpy_seed` | Random seed for reproducibility |
| `forecast.test_start` | OOS evaluation start date |
| `forecast.train_months` | Training window length |
| `forecast.roll_months` | Rolling window step |
| `forecast.n_workers` | Parallel workers (`null` = auto) |

## Reproducibility

The results from `03_forecast.py` and `04_var_backtest.py` were generated under the default parallel configuration. Minor numerical differences (typically within ±0.001) may occur across machines because of BLAS implementation differences and parallel execution order.

## Citation

If you use this code, please cite:

```bibtex
@article{kim2026carrying,
  title   = {Carrying regime uncertainty forward in cryptocurrency tail-risk forecasting},
  author  = {Kim, Taeyun},
  journal = {Finance Research Letters},
  year    = {2026},
  doi     = {10.1016/j.frl.2026.110286},
  note    = {Forthcoming},
  publisher = {Elsevier}
}
```