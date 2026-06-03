"""
02_analysis.py  —  Replicates Table 2 and Figure 1
Constructs hourly realized volatility and cross-asset correlation from minute-level
OHLCV data, assigns regime labels, measures correlation impulse responses, and
produces Figure 1.

Regime labels require contemporaneous cross-asset correlation and are therefore
unavailable to any real-time classifier.  At test time they are applied ex-post
solely to stratify forecast errors (Tables 3 and 6).

Run order: 01 -> 02 -> 03 -> 04
"""

import glob
import argparse
import warnings
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from matplotlib import font_manager
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 22, "axes.labelsize": 22,
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 18, "figure.titlesize": 26,
})

SHORT = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "XRPUSDT": "XRP",
         "LTCUSDT": "LTC", "LINKUSDT": "LINK"}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = (Path(path).parent / cfg["paths"]["base"]).resolve() if cfg["paths"]["base"] == "." else Path(cfg["paths"]["base"]).expanduser().resolve()
    cfg["_data_dir"]    = base / cfg["paths"]["data_dir"]
    cfg["_results_dir"] = base / cfg["paths"]["results_dir"]
    cfg["_results_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def load_all_parts(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "part*.csv")))
    if not files:
        raise FileNotFoundError(f"No part*.csv in {data_dir}")
    print(f"[1/5] Loading {len(files)} files...")
    df = pd.concat(
        [pd.read_csv(f, parse_dates=["timestamp"]) for f in files],
        ignore_index=True,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} rows | {df['timestamp'].min()} -> {df['timestamp'].max()}")
    return df


def compute_hourly(df: pd.DataFrame, cfg: dict, results_dir: Path,
                   force: bool = False) -> pd.DataFrame:
    """
    Aggregate minute-level returns to hourly realized volatility and cross-asset
    correlation, following Eq. (1) of the paper.

    Per-asset hourly realized volatility is stored as sqrt(sum r_{i,s}^2) over
    the 60 intra-hour minutes (i.e., in standard-deviation units, not variance
    units).  The cross-asset mean log realized variance target y_{h+1} in Eq. (1)
    is constructed in 03_forecast.py by squaring and log-transforming these values.

    Cross-asset correlation is the mean pairwise Pearson correlation of the same
    minute-level log-returns across all five assets within each hour.
    Contemporaneous correlation is used solely for regime labeling and is never
    available to any real-time classifier.

    Results are cached to hourly_stats.parquet; pass --force to recompute.
    """
    cache_path = results_dir / "hourly_stats.parquet"
    if cache_path.exists() and not force:
        print("[2/5] Loading cached hourly_stats.parquet")
        return pd.read_parquet(cache_path)

    print("[2/5] Computing minute-level features and aggregating to hourly...")
    symbols  = cfg["symbols"]
    an       = cfg["analysis"]
    rv_win   = an["rolling_vol_window"]
    ex_sigma = an["extreme_sigma"]
    shorts   = [SHORT[s] for s in symbols]
    ret_cols = []

    df = df.copy()
    df["hour_block"] = df["timestamp"].dt.floor("h")

    for sym in symbols:
        s       = SHORT[sym]
        c       = df[f"{sym}_close"]
        log_ret = np.log(c / c.shift(1))
        # Extreme-move flag: |r| > ex_sigma * rolling std.
        # The hourly mean of this flag (extreme_move) feeds into the
        # Single-Asset Stress classifier in label_regimes().
        roll_std           = log_ret.rolling(rv_win, min_periods=10).std()
        df[f"{s}_log_ret"] = log_ret
        df[f"{s}_extreme"] = (log_ret.abs() > ex_sigma * roll_std).astype(np.float32)
        df[f"{s}_close"]   = c.values
        ret_cols.append(f"{s}_log_ret")

    def _agg_group(grp):
        rec = {}
        for s in shorts:
            rets   = grp[f"{s}_log_ret"].dropna().values
            closes = grp[f"{s}_close"].dropna().values
            # Stored as sqrt(sum r^2): standard-deviation scale, not variance scale.
            rec[f"{s}_hvol"]         = float(np.sqrt(np.sum(rets ** 2))) if len(rets) > 0 else 0.0
            rec[f"{s}_extreme_move"] = float(grp[f"{s}_extreme"].mean())
            if len(closes) >= 2:
                rec[f"{s}_hret"] = float(np.log(closes[-1] / closes[0]))
            elif len(rets) > 0:
                rec[f"{s}_hret"] = float(np.sum(rets))
            else:
                rec[f"{s}_hret"] = 0.0

        mat = grp[ret_cols].dropna()
        if len(mat) >= 10:
            cm       = mat.corr().values
            mask_tri = np.triu(np.ones(cm.shape, dtype=bool), k=1)
            rec["cross_corr"] = float(np.nanmean(cm[mask_tri]))
        else:
            rec["cross_corr"] = np.nan  # too few observations to estimate reliably
        return pd.Series(rec)

    hourly = (
        df.groupby("hour_block", sort=True)
          .apply(_agg_group)
          .reset_index()
    )
    hourly["hour_block"] = pd.to_datetime(hourly["hour_block"], utc=True)

    hvol_cols = [f"{s}_hvol" for s in shorts]
    hret_cols = [f"{s}_hret" for s in shorts]
    hourly["avg_hvol"]        = hourly[hvol_cols].mean(axis=1)
    # Annualized cross-asset mean hourly volatility (fraction, not percent).
    # Multiplied by 100 in build_summary() for Table 2.
    hourly["avg_hvol_annual"] = hourly["avg_hvol"] * np.sqrt(365 * 24)
    hourly["port_hret"]       = hourly[hret_cols].mean(axis=1)

    hourly.to_parquet(cache_path, index=False)
    print(f"  Saved {cache_path} ({len(hourly):,} hours)")
    return hourly


def label_regimes(hourly: pd.DataFrame, cfg: dict, results_dir: Path,
                  force: bool = False) -> pd.DataFrame:
    """
    Assign hourly regime labels using rolling 6-month calibration windows (Section 2).
    Labels use contemporaneous cross-asset correlation and are applied ex-post only,
    to stratify forecast errors in Tables 3 and 6.

    Contagion (all three conditions jointly):
        (i)   at least one asset above its rolling 95th-percentile realized volatility,
        (ii)  cross-asset correlation above its rolling 90th percentile,
        (iii) at least three assets above their individual rolling 80th-percentile volatility.
    Single-Asset Stress (non-Contagion hours satisfying either):
        stress1: at least one asset above p80 vol AND correlation above p80, or
        stress2: at least one asset above p95 vol (regardless of correlation).
    Normal: all remaining hours.

    min_periods=720 ≈ 30 days burn-in before the first label is assigned.
    Results are cached to regimes_hourly.parquet; pass --force to recompute.
    """
    cache_path = results_dir / "regimes_hourly.parquet"
    if cache_path.exists() and not force:
        print("[3/5] Loading cached regimes_hourly.parquet")
        return pd.read_parquet(cache_path)

    print("[3/5] Labeling hourly regimes...")
    calib_h   = cfg["analysis"]["regime_calib_hours"]
    shorts    = [SHORT[s] for s in cfg["symbols"]]
    hvol_cols = [f"{s}_hvol" for s in shorts]

    hourly = hourly.copy().sort_values("hour_block").reset_index(drop=True)
    n      = len(hourly)

    p80_hvol = hourly[hvol_cols].apply(
        lambda c: c.rolling(calib_h, min_periods=720).quantile(0.80))
    p95_hvol = hourly[hvol_cols].apply(
        lambda c: c.rolling(calib_h, min_periods=720).quantile(0.95))
    corr_p80 = hourly["cross_corr"].rolling(calib_h, min_periods=720).quantile(0.80)
    corr_p90 = hourly["cross_corr"].rolling(calib_h, min_periods=720).quantile(0.90)

    above_p80 = (hourly[hvol_cols].values > p80_hvol.values).sum(axis=1)
    above_p95 = (hourly[hvol_cols].values > p95_hvol.values).sum(axis=1)

    corr    = hourly["cross_corr"].values
    cp80    = corr_p80.values
    cp90    = corr_p90.values
    regimes = np.full(n, "Normal", dtype=object)

    # Contagion: all three joint conditions must hold (Section 2)
    contagion = (above_p95 >= 1) & (corr > cp90) & (above_p80 >= 3) & np.isfinite(cp90)
    regimes[contagion] = "Contagion"

    # stress1: joint vol + correlation criterion; stress2: standalone extreme-vol criterion
    stress1 = ~contagion & (above_p80 >= 1) & (corr > cp80) & np.isfinite(cp80)
    stress2 = ~contagion & ~stress1 & (above_p95 >= 1)
    regimes[stress1 | stress2] = "Single-Asset Stress"

    hourly["regime"] = regimes
    hourly.to_parquet(cache_path, index=False)

    dist = pd.Series(regimes[720:]).value_counts(normalize=True).round(3).sort_index()
    print(f"  Regime distribution (after burn-in):\n{dist.to_string()}")
    return hourly


def measure_correlation_decay(hourly: pd.DataFrame):
    """
    Estimate mean cross-asset correlation impulse responses after volatility shocks,
    stratified by regime (Figure 1a, Table 2).

    A shock is defined as avg_hvol exceeding its full-sample 95th percentile.
    The pre-shock baseline uses hours -24 to -6 relative to shock onset to avoid
    contamination from the run-up period.  The 50%-decay point is located by linear
    interpolation between the first pair of consecutive lags that straddles the
    halfway level, and is the key quantity contrasting Contagion (1.8h) with
    Normal (4.4h) in Table 2.
    """
    print("[4/5] Measuring cross-correlation decay after shocks...")
    hourly     = hourly.copy().sort_values("hour_block").reset_index(drop=True)
    corr_arr   = hourly["cross_corr"].values
    regime_arr = hourly["regime"].values
    n          = len(hourly)
    p95_vol    = hourly["avg_hvol"].quantile(0.95)
    shock_mask = hourly["avg_hvol"].values > p95_vol

    max_lead           = 24
    pre_start, pre_end = -24, -6   # relative offsets from shock index (both negative)
    results, impulse_responses = {}, {}

    for regime_name in ["Normal", "Single-Asset Stress", "Contagion"]:
        shock_idx    = np.flatnonzero(shock_mask & (regime_arr == regime_name)).tolist()
        corr_profile = np.zeros(max_lead + 1)
        baseline_sum = 0.0
        count        = 0

        for si in shock_idx:
            # Require enough lead and lag history on both sides of the shock
            if si + max_lead >= n or si + pre_start < 0:
                continue
            bl = corr_arr[si + pre_start: si + pre_end]
            bl = bl[np.isfinite(bl)]
            if len(bl) < 6:
                continue
            for lag in range(max_lead + 1):
                v = corr_arr[si + lag]
                if np.isfinite(v):
                    corr_profile[lag] += v
            baseline_sum += bl.mean()
            count        += 1

        if count > 5:
            avg_profile  = corr_profile / count
            avg_baseline = baseline_sum / count
            impulse_responses[regime_name] = {
                "lags":     np.arange(max_lead + 1),
                "profile":  avg_profile.copy(),
                "baseline": avg_baseline,
            }
            # Locate the 50%-decay crossing by linear interpolation
            peak      = avg_profile[0]
            threshold = avg_baseline + 0.5 * (peak - avg_baseline)
            decay_h   = float(max_lead)
            for lag in range(1, max_lead + 1):
                if avg_profile[lag] < threshold:
                    prev    = avg_profile[lag - 1]
                    curr    = avg_profile[lag]
                    frac    = (prev - threshold) / (prev - curr) if prev != curr else 0.0
                    decay_h = (lag - 1) + frac
                    break
            print(f"  {regime_name}: n={count}, baseline={avg_baseline:.3f}, "
                     f"peak={peak:.3f}, 50%-decay={decay_h:.1f}h")
            results[regime_name] = {
                "n_events":         count,
                "baseline_corr":    round(avg_baseline, 3),
                "peak_corr":        round(peak, 3),
                "corr_decay_hours": round(decay_h, 2),
            }
        else:
            print(f"  {regime_name}: too few events ({count}), skipping")
            results[regime_name] = {"n_events": count}

    return results, impulse_responses


def build_summary(hourly: pd.DataFrame, corr_decay: dict,
                  results_dir: Path) -> pd.DataFrame:
    """
    Compile regime frequency, annualized vol, cross-asset correlation, and
    correlation 50%-decay time into Table 2.

    avg_hvol_annual is stored in fraction units (e.g., 0.60 = 60% annualized vol);
    multiplying by 100 converts to the percentage representation used in Table 2.
    """
    h           = hourly.iloc[720:].copy()  # drop burn-in observations
    total_hours = len(h)
    rows        = []

    for rn in ["Normal", "Single-Asset Stress", "Contagion"]:
        mask            = h["regime"] == rn
        freq            = int(mask.sum()) / total_hours * 100
        # avg_hvol_annual is in fraction units; *100 converts to percent for Table 2
        avg_hvol_annual = float(h.loc[mask, "avg_hvol_annual"].mean() * 100) if mask.any() else np.nan
        avg_corr        = float(h.loc[mask, "cross_corr"].mean())             if mask.any() else np.nan
        cd = corr_decay.get(rn, {})
        rows.append({
            "Regime":           rn,
            "Frequency":        f"{freq:.0f}%",
            "Annual Vol":       f"{int(round(avg_hvol_annual))}%",
            "Cross-Asset Corr": f"{avg_corr:.2f}",
            "Corr Decay (h)":   f"{cd['corr_decay_hours']:.1f}" if "corr_decay_hours" in cd else "N/A",
        })

    summary = pd.DataFrame(rows)
    outpath = results_dir / "table2_regime_summary.csv"
    summary.to_csv(outpath, index=False)
    print(f"Saved {outpath}")
    return summary


def save_table2(summary: pd.DataFrame) -> None:
    """
    Print Table 2 in a styled terminal format consistent with Tables 3--6.
    """
    print("\n" + "=" * 86)
    print("TABLE 2: Regime Characteristics")
    print("=" * 86)

    hdr = (
        f"{'Regime':<24} "
        f"{'Freq':>6} "
        f"{'Ann. Vol':>10} "
        f"{'Cross-Corr':>12} "
        f"{'Shock Decay':>12}"
    )
    print(hdr)
    print("-" * 86)

    for _, row in summary.iterrows():
        freq_str = str(row["Frequency"])
        vol_str = str(row["Annual Vol"])
        corr_str = str(row["Cross-Asset Corr"])
        decay_raw = row["Corr Decay (h)"]
        decay_str = f"{decay_raw} h" if decay_raw != "N/A" else "N/A"

        print(
            f"{str(row['Regime']):<24} "
            f"{freq_str:>6} "
            f"{vol_str:>10} "
            f"{corr_str:>12} "
            f"{decay_str:>12}"
        )


def create_visualizations(hourly: pd.DataFrame, corr_decay: dict,
                          impulse_responses: dict, results_dir: Path,
                          seed: int = 42) -> None:
    """
    Produce Figure 1.
    Panel (a): mean cross-asset correlation impulse response by regime.
    Panel (b): scatter of annualized hourly vol vs. cross-asset correlation,
               3,000 observations per regime sampled with a fixed seed.
    """
    print("[5/5] Creating visualizations...")
    np.random.seed(seed)
    h      = hourly.iloc[720:].copy()
    colors = {
        "Normal":              "#2ecc71",
        "Single-Asset Stress": "#f39c12",
        "Contagion":           "#e74c3c",
    }
    order = ["Normal", "Single-Asset Stress", "Contagion"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    ax = axes[0]
    for rn in order:
        if rn not in impulse_responses:
            continue
        ir       = impulse_responses[rn]
        profile  = ir["profile"]
        baseline = ir["baseline"]
        ax.plot(ir["lags"], profile, linewidth=2.8, label=rn,
                color=colors[rn], linestyle="solid")
        ax.axhline(baseline, linestyle="--", alpha=0.4,
                   color=colors[rn], linewidth=1.75)

        if rn == "Contagion":
            decay_h = corr_decay.get("Contagion", {}).get("corr_decay_hours")
            half_v  = baseline + 0.5 * (profile[0] - baseline)
            if decay_h is not None and decay_h <= 12:
                ax.axvline(x=decay_h, color="#7B3F00", linestyle=":", linewidth=2.5, alpha=0.85, zorder=3)
                ax.axhline(y=half_v,  color="#7B3F00", linestyle=":", linewidth=2.5, alpha=0.75, zorder=3)
                ax.annotate(
                    f"50%-decay:\n{decay_h:.1f}h",
                    xy=(decay_h, half_v),
                    xytext=(decay_h + 2.0, half_v + 0.0125),
                    ha="center",
                    color="#7B3F00",
                    fontproperties=font_manager.FontProperties(
                        family="Times New Roman", weight="bold", size=18),
                    arrowprops=dict(arrowstyle="->", color="#7B3F00", lw=1.5),
                )

    ax.set_xlabel("Hours after shock")
    ax.set_ylabel("Cross-asset correlation")
    ax.set_title("(a) Correlation impulse response", fontweight="bold")
    ax.set_xlim(0, 12)
    ax.legend(loc="upper right", framealpha=0.88)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for rn in order:
        mask   = h["regime"] == rn
        sample = h[mask].sample(min(3000, int(mask.sum())), random_state=seed)
        ax.scatter(
            sample["avg_hvol_annual"] * 100,
            sample["cross_corr"],
            c=colors[rn], label=rn,
            s=10, alpha=0.25,
            zorder=order.index(rn) + 1,
        )

    ax.set_xlabel("Annualized hourly vol (%)")
    ax.set_xscale("log")
    ax.set_ylabel("Cross-asset correlation")
    ax.set_title("(b) Volatility vs correlation by regime", fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.legend(markerscale=4, framealpha=0.88)
    ax.grid(True, alpha=0.3)

    figpath = results_dir / "figure1_regime_analysis.pdf"
    plt.savefig(figpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved {figpath}")


def parse_args():
    p = argparse.ArgumentParser(description="Regime analysis of crypto OHLCV data")
    p.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"), help="Path to config.yaml")
    p.add_argument("--force",  action="store_true",
                   help="Recompute even if cached parquet files exist")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    np.random.seed(cfg["analysis"]["numpy_seed"])

    data_dir    = cfg["_data_dir"]
    results_dir = cfg["_results_dir"]

    raw    = load_all_parts(data_dir)
    hourly = compute_hourly(raw, cfg, results_dir, force=args.force)
    hourly = label_regimes(hourly, cfg, results_dir, force=args.force)
    corr_decay, impulse_responses = measure_correlation_decay(hourly)
    summary = build_summary(hourly, corr_decay, results_dir)
    save_table2(summary)
    create_visualizations(hourly, corr_decay, impulse_responses, results_dir,
                          seed=cfg["analysis"]["numpy_seed"])
    print(f"Done. All outputs in: {results_dir}")


if __name__ == "__main__":
    main()