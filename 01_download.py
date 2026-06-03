"""
01_download.py  —  Replicates Table 1
Download minute-level OHLCV for USDT perpetual futures from Binance Futures
and compute mean daily USD trading volume.

Run order: 01 -> 02 -> 03 -> 04
"""

import os
import glob
import math
import time
import warnings
import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml
import pandas as pd
from binance.client import Client
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = (Path(path).parent / cfg["paths"]["base"]).resolve() if cfg["paths"]["base"] == "." else Path(cfg["paths"]["base"]).expanduser().resolve()
    cfg["_data_dir"]    = base / cfg["paths"]["data_dir"]
    cfg["_results_dir"] = base / cfg["paths"]["results_dir"]
    cfg["_data_dir"].mkdir(parents=True, exist_ok=True)
    cfg["_results_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def retry_call(fn, *args, **kwargs):
    """Retry indefinitely with a 5-second backoff on any exception
    (network errors, rate limits, transient Binance API failures)."""
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"{e} -> retry in 5s")
            time.sleep(5)


def fetch_window(client: Client, symbol: str, start_ms: int, end_ms: int,
                 interval: str, limit: int, sleep_sec: float) -> pd.DataFrame:
    """
    Fetch up to `limit` 1-minute klines for one symbol over [start_ms, end_ms].
    A fixed sleep after each call keeps the request rate within Binance limits.
    Returns a DataFrame indexed by UTC timestamp; missing minutes appear as NaN
    after reindexing in the caller.
    """
    kl = retry_call(
        client.get_klines,
        symbol=symbol,
        interval=interval,
        startTime=start_ms,
        endTime=end_ms,
        limit=limit,
    )
    time.sleep(sleep_sec)

    cols = ["timestamp", f"{symbol}_open", f"{symbol}_high",
            f"{symbol}_low", f"{symbol}_close", f"{symbol}_volume"]
    if not kl:
        return pd.DataFrame(columns=cols).set_index("timestamp")

    df = pd.DataFrame({
        "timestamp":        pd.to_datetime([k[0] for k in kl], unit="ms", utc=True),
        f"{symbol}_open":   pd.to_numeric([k[1] for k in kl], errors="coerce"),
        f"{symbol}_high":   pd.to_numeric([k[2] for k in kl], errors="coerce"),
        f"{symbol}_low":    pd.to_numeric([k[3] for k in kl], errors="coerce"),
        f"{symbol}_close":  pd.to_numeric([k[4] for k in kl], errors="coerce"),
        f"{symbol}_volume": pd.to_numeric([k[5] for k in kl], errors="coerce"),
    }).drop_duplicates(subset=["timestamp"]).set_index("timestamp")
    return df


def save_chunk(df: pd.DataFrame, part: int, data_dir: Path) -> int:
    """Write `df` to part{part}.csv and return the next part number."""
    outpath = data_dir / f"part{part}.csv"
    df.to_csv(outpath, index=False)
    print(f"Saved {outpath}")
    return part + 1


def download_parts(cfg: dict) -> None:
    """
    Download the full sample in fixed-size chunks written to part*.csv.
    If part files already exist the download resumes from where it left off,
    so interruptions do not require restarting from the beginning.
    All symbols are joined on a shared minute-level index within each window;
    missing minutes are retained as NaN rather than dropped.
    """
    symbols    = cfg["symbols"]
    data_dir   = cfg["_data_dir"]
    dl         = cfg["download"]
    chunk_size = dl["chunk_size"]
    limit      = dl["limit"]
    sleep_sec  = dl["sleep_sec"]
    interval   = Client.KLINE_INTERVAL_1MINUTE

    start_ms = dt_to_ms(datetime.fromisoformat(dl["start"]).replace(tzinfo=timezone.utc))
    end_ms   = dt_to_ms(datetime.fromisoformat(dl["end"]).replace(
                    hour=23, minute=59, second=59, tzinfo=timezone.utc))
    step_ms  = 60_000

    total_minutes = (end_ms - start_ms) // step_ms + 1
    total_windows = math.ceil(total_minutes / limit)

    existing = sorted(glob.glob(str(data_dir / "part*.csv")))
    if existing:
        last_part = int(Path(existing[-1]).stem.replace("part", ""))
        part      = last_part + 1
        skip_rows = last_part * chunk_size
        print(f"Resuming from part {part} (skipping {skip_rows:,} rows)")
    else:
        part      = 1
        skip_rows = 0

    client       = Client()
    buffer       = pd.DataFrame()
    skip_windows = skip_rows // limit

    for w in tqdm(range(total_windows), desc="Downloading 1m windows", dynamic_ncols=True):
        if w < skip_windows:
            continue

        w_start = start_ms + w * limit * step_ms
        if w_start > end_ms:
            break
        w_end = min(w_start + (limit - 1) * step_ms, end_ms)

        idx  = pd.date_range(
            start=pd.to_datetime(w_start, unit="ms", utc=True),
            end=pd.to_datetime(w_end,     unit="ms", utc=True),
            freq="1min",
        )
        wide = pd.DataFrame(index=idx)
        for sym in symbols:
            wide = wide.join(
                fetch_window(client, sym, w_start, w_end, interval, limit, sleep_sec)
                    .reindex(idx),
                how="left",
            )

        wide   = wide.reset_index().rename(columns={"index": "timestamp"})
        buffer = pd.concat([buffer, wide], ignore_index=True)

        while len(buffer) >= chunk_size:
            part   = save_chunk(buffer.iloc[:chunk_size].copy(), part, data_dir)
            buffer = buffer.iloc[chunk_size:].reset_index(drop=True)

    if len(buffer) > 0:
        save_chunk(buffer, part, data_dir)


def compute_mean_daily_volume(cfg: dict) -> None:
    """
    Compute mean daily USD volume per asset and write Table 1.
    Volume is converted to USD using the contemporaneous close price before
    aggregating to daily totals.
    """
    symbols     = cfg["symbols"]
    data_dir    = cfg["_data_dir"]
    results_dir = cfg["_results_dir"]

    part_files = sorted(glob.glob(str(data_dir / "part*.csv")))
    if not part_files:
        print(f"No part files in {data_dir}")
        return

    vol_cols   = [f"{s}_volume" for s in symbols]
    close_cols = [f"{s}_close"  for s in symbols]
    agg        = None

    for fpath in part_files:
        df = pd.read_csv(
            fpath,
            usecols=["timestamp"] + vol_cols + close_cols,
            parse_dates=["timestamp"],
        )
        df["date"] = df["timestamp"].dt.date

        usd_data = {"date": df["date"]}
        for sym in symbols:
            usd_data[f"{sym}_usd_vol"] = df[f"{sym}_volume"] * df[f"{sym}_close"]

        daily = pd.DataFrame(usd_data).groupby("date").sum()
        agg   = daily if agg is None else agg.add(daily, fill_value=0)

    mean_daily = agg.mean()

    print("=== Mean Daily Volume (USDT) ===")
    for sym in symbols:
        print(f"  {sym:10s}: ${mean_daily[f'{sym}_usd_vol']:>20,.0f}")

    result = pd.DataFrame({
        "symbol":              symbols,
        "mean_daily_vol_usdt": [mean_daily[f"{s}_usd_vol"] for s in symbols],
    })
    outpath = results_dir / "table1_mean_daily_vol.csv"
    result.to_csv(outpath, index=False)
    print(f"Saved {outpath}")


def parse_args():
    p = argparse.ArgumentParser(description="Download Binance OHLCV and compute daily volume")
    p.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"), help="Path to config.yaml")
    p.add_argument("--force",  action="store_true",   help="Re-download even if parts exist")
    p.add_argument("--skip-download", action="store_true", help="Skip download, only compute volume")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.force:
        for f in glob.glob(str(cfg["_data_dir"] / "part*.csv")):
            os.remove(f)
        print("Cleared existing part files (--force)")

    if not args.skip_download:
        download_parts(cfg)

    compute_mean_daily_volume(cfg)


if __name__ == "__main__":
    main()