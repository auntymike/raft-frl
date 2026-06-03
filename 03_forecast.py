"""
03_forecast.py  —  Replicates Tables 3, 4, and 5
Rolling out-of-sample evaluation of Persistence, HAR-RV, MS-HAR, Hard-HMM, and RAFT.

Training windows: 24 months.  Test windows: 12-month increments from test_start,
yielding four non-overlapping periods over 2022-2025.  All models receive the same
HAR feature vector x_h = (1, logRV^(1)_h, logRV^(24)_h, logRV^(168)_h) and the same
training information.

The central design difference: RAFT carries regime uncertainty forward as a
continuous posterior (Eq. 3); MS-HAR resolves it with a hard threshold rule at
prediction time; Hard-HMM uses the same HMM-HAR machinery as RAFT but collapses
the filtered posterior to a single regime at the forecast step.  The Diebold-Mariano
tests are reported as a standalone Table 4.  The ablation in Table 5 decomposes the
contagion-hour QLIKE gain into the contributions of soft weighting, frequency-based
initialization, and regime-specific dynamics.

Run order: 01 -> 02 -> 03 -> 04
"""

import os
import warnings
import argparse
from itertools import permutations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from tqdm import tqdm

warnings.filterwarnings("ignore")

SHORTS     = ["BTC", "ETH", "XRP", "LTC", "LINK"]
HVOL_COLS  = [f"{s}_hvol" for s in SHORTS]
EM_COLS    = [f"{s}_extreme_move" for s in SHORTS]
LABEL_MAP  = {"Normal": 0, "Single-Asset Stress": 1, "Contagion": 2}
INT_TO_LBL = {v: k for k, v in LABEL_MAP.items()}

REGIME_NORMAL    = 0
REGIME_STRESS    = 1
REGIME_CONTAGION = 2

_CFG = {}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = (Path(path).parent / cfg["paths"]["base"]).resolve() if cfg["paths"]["base"] == "." else Path(cfg["paths"]["base"]).expanduser().resolve()
    cfg["_results_dir"] = base / cfg["paths"]["results_dir"]
    cfg["_results_dir"].mkdir(parents=True, exist_ok=True)
    _CFG.update(cfg)
    return cfg


def _fc() -> dict:
    return _CFG["forecast"]


def _eps() -> float:
    return _fc()["eps"]


def load_data(cfg: dict) -> pd.DataFrame:
    """
    Merge hourly RV stats with regime labels and build the HAR feature vector.

    The prediction target is the cross-asset mean log realized variance y_{h+1},
    Eq. (1).  All lagged features are shifted by one period so that no feature
    uses information from hour h+1 or later.

    hvol columns from 02_analysis.py are in standard-deviation units
    (sqrt(sum r^2)); squaring and log-transforming them yields the log realized
    variance target used in Eq. (1).
    """
    print("[1/4] Loading data...")
    results_dir = cfg["_results_dir"]
    hourly  = pd.read_parquet(results_dir / "hourly_stats.parquet")
    regimes = pd.read_parquet(results_dir / "regimes_hourly.parquet")

    for df_ in (hourly, regimes):
        df_["hour_block"] = pd.to_datetime(df_["hour_block"], utc=True)

    df = (hourly
          .merge(regimes[["hour_block", "regime"]].rename(columns={"regime": "regime_fi"}),
                 on="hour_block", how="left")
          .sort_values("hour_block").reset_index(drop=True))

    eps = _eps()
    # Eq. (1): cross-asset mean log realized variance.
    # hvol = sqrt(sum r^2), so hvol^2 = sum r^2 = realized variance per asset.
    df["log_rv"] = np.log(df[HVOL_COLS].pow(2).clip(lower=eps)).mean(axis=1)

    y = df["log_rv"]
    df["rv_lag1"]   = y.shift(1)
    df["rv_lag24"]  = y.shift(1).rolling(24,  min_periods=24).mean()
    df["rv_lag168"] = y.shift(1).rolling(168, min_periods=168).mean()

    # Cross-sectional vol statistics used by MS-HAR's threshold rule
    hvol_mat = df[HVOL_COLS].values
    df["cs_hvol_std"]    = np.nanstd(hvol_mat,  axis=1)
    df["cs_hvol_max"]    = np.nanmax(hvol_mat,  axis=1)
    df["cs_hvol_std_l1"] = df["cs_hvol_std"].shift(1)
    df["cs_hvol_max_l1"] = df["cs_hvol_max"].shift(1)

    em_avail = [c for c in EM_COLS if c in df.columns]
    if em_avail:
        df["extreme_move_mean"] = df[em_avail].mean(axis=1)
    else:
        # Fallback: construct from rolling z-scores when the pre-computed column is absent
        roll_std = df[HVOL_COLS].rolling(720, min_periods=50).std()
        roll_mu  = df[HVOL_COLS].rolling(720, min_periods=50).mean()
        df["extreme_move_mean"] = (
            (df[HVOL_COLS] > (roll_mu + 2 * roll_std)).astype(float).mean(axis=1))
    df["extreme_move_l1"] = df["extreme_move_mean"].shift(1)

    for s in SHORTS:
        df[f"{s}_hvol_l1"] = df[f"{s}_hvol"].shift(1)
    df["x_corr"] = df["cross_corr"].shift(1)

    df = df.dropna(subset=[
        "rv_lag1", "rv_lag24", "rv_lag168",
        "cs_hvol_std_l1", "cs_hvol_max_l1", "extreme_move_l1",
    ]).reset_index(drop=True)

    df["hb_tz"] = df["hour_block"].dt.tz_convert(None)
    print(f"  {len(df):,} obs | {df['hour_block'].min()} -> {df['hour_block'].max()}")
    return df


def compute_thresh_quantiles(df_tr: pd.DataFrame) -> dict:
    """
    Calibrate MS-HAR threshold percentiles on the training window.
    Thresholds are recomputed for each rolling window so they adapt to the
    volatility regime prevailing during training.
    """
    return {
        "tw_q95_rv_max": np.nanpercentile(df_tr["cs_hvol_max_l1"].values, 95),
        "tw_q90_cs_std": np.nanpercentile(df_tr["cs_hvol_std_l1"].values, 90),
        "tw_q80_em":     np.nanpercentile(df_tr["extreme_move_l1"].values, 80),
        "per_asset_q80": {
            s: np.nanpercentile(df_tr[f"{s}_hvol_l1"].values, 80)
            for s in SHORTS if f"{s}_hvol_l1" in df_tr.columns
        },
    }


def threshold_regime(df: pd.DataFrame, thr: dict) -> np.ndarray:
    """
    Hard regime assignment used by MS-HAR at prediction time (Section 3).

    Because contemporaneous cross-asset return correlation is unavailable at
    forecast time, the rule uses lagged cross-sectional volatility dispersion
    as a proxy for condition (ii).

    Returns an integer array with values in {REGIME_NORMAL, REGIME_STRESS,
    REGIME_CONTAGION} (i.e., {0, 1, 2}).
    """
    per_q80    = thr["per_asset_q80"]
    n_above_80 = sum(
        (df[f"{s}_hvol_l1"].values > per_q80[s]).astype(int)
        for s in SHORTS if f"{s}_hvol_l1" in df.columns and s in per_q80
    )
    contagion_mask = (
        (df["cs_hvol_max_l1"].values > thr["tw_q95_rv_max"]) &
        (df["cs_hvol_std_l1"].values > thr["tw_q90_cs_std"]) &
        (n_above_80 >= 3)
    )
    stress_mask = (
        ~contagion_mask &
        (df["extreme_move_l1"].values > thr["tw_q80_em"])
    )
    out = np.zeros(len(df), dtype=int)
    out[stress_mask]    = REGIME_STRESS
    out[contagion_mask] = REGIME_CONTAGION
    return out


def log_score(y: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> float:
    """Average log predictive density under a Gaussian forecast (secondary metric)."""
    s2 = np.maximum(sigma2, _eps())
    return float(np.mean(-0.5 * np.log(2 * np.pi * s2) - 0.5 * (y - mu) ** 2 / s2))


def qlike_vec(mu: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Element-wise QLIKE (Patton, 2011): ratio - log(ratio) - 1,
    where ratio = sigma^2 / h = exp(y) / exp(mu) is the realized-to-forecast
    variance ratio.  Both y and mu are log-variance, so exp() recovers the
    variance scale before computing the ratio.
    The asymmetric penalty on variance underestimation (ratio > 1 penalised more
    than ratio < 1) makes this the primary loss for a tail-risk evaluation.
    """
    h     = np.exp(np.clip(mu, -30, 30))
    act   = np.exp(np.clip(y,  -30, 30))
    ratio = act / np.maximum(h, _eps())
    return ratio - np.log(np.maximum(ratio, _eps())) - 1


def qlike(mu: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(qlike_vec(mu, y)))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> float:
    """
    Two-sided Diebold-Mariano test (Diebold and Mariano, 1995) with Newey-West
    HAC standard errors.  Bandwidth h = floor(T^{1/3}) follows the rule-of-thumb
    in Newey and West (1987).  Returns the two-tailed p-value.
    A negative mean of (loss_a - loss_b) indicates model A is more accurate.
    """
    d = loss_a - loss_b
    n = len(d)
    if n < 2:
        return np.nan
    h      = max(1, int(n ** (1 / 3)))
    nw_var = np.var(d, ddof=1)
    for lag in range(1, h + 1):
        c = np.cov(d[lag:], d[:-lag])
        if c.shape == (2, 2):
            nw_var += 2 * (1 - lag / (h + 1)) * c[0, 1]
    stat = np.mean(d) / np.sqrt(max(nw_var, _eps()) / n)
    return float(2 * (1 - stats.norm.cdf(abs(stat))))


def _align_losses(preds_a, preds_b, regime_filter=None):
    """Merge two prediction sets on matching hour_block timestamps before computing
    per-observation QLIKE losses for Diebold-Mariano tests."""
    def to_frame(preds, rf):
        frames = []
        for p in preds:
            mask = (p["regime"] == rf) if rf else np.ones(len(p["y"]), dtype=bool)
            if mask.sum() == 0:
                continue
            frames.append(pd.DataFrame({
                "hb": p["hb"][mask],
                "mu": p["mu"][mask],
                "y":  p["y"][mask],
            }))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    fa = to_frame(preds_a, regime_filter)
    fb = to_frame(preds_b, regime_filter)
    if fa.empty or fb.empty:
        return None, None
    m = fa.merge(fb, on="hb", suffixes=("_a", "_b"))
    if len(m) < 2:
        return None, None
    return (qlike_vec(m["mu_a"].values, m["y_a"].values),
            qlike_vec(m["mu_b"].values, m["y_b"].values))


def _har_X(df: pd.DataFrame) -> np.ndarray:
    """HAR feature matrix: x_h = (1, logRV^(1)_h, logRV^(24)_h, logRV^(168)_h)."""
    return np.column_stack([np.ones(len(df)),
                            df["rv_lag1"].values,
                            df["rv_lag24"].values,
                            df["rv_lag168"].values])


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """
    Gamma-weighted least squares for the EM M-step.
    A small ridge (eps * I) is added to X'WX to prevent near-singular systems
    when a regime carries very low posterior mass (Appendix A).
    Returns (beta, sigma2) where sigma2 is the gamma-weighted residual variance.
    """
    eps = _eps()
    sw  = w.sum()
    if sw < eps:
        return np.linalg.lstsq(X, y, rcond=None)[0], np.var(y) + eps
    Xw   = X * w[:, None]
    XtWX = Xw.T @ X + eps * np.eye(X.shape[1])
    XtWy = Xw.T @ y
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return beta, max(float(np.dot(w, resid ** 2) / sw), eps)


def _pi_init(K: int, diag: float = 0.95) -> np.ndarray:
    """Initialize transition matrix with strong diagonal persistence."""
    Pi = np.full((K, K), (1.0 - diag) / (K - 1))
    np.fill_diagonal(Pi, diag)
    return Pi


def _pi_regularize(Pi: np.ndarray, K: int) -> np.ndarray:
    """
    Shrink the M-step transition matrix toward a diagonal prior (Appendix A).
    pi_diag_prior and pi_prior_weight are read from config["forecast"].
    """
    fc    = _fc()
    diag  = fc["pi_diag_prior"]
    w     = fc["pi_prior_weight"]
    prior = np.full((K, K), (1.0 - diag) / (K - 1))
    np.fill_diagonal(prior, diag)
    Pi_reg = (1.0 - w) * Pi + w * prior
    Pi_reg /= Pi_reg.sum(axis=1, keepdims=True)
    return Pi_reg


def _freq_init(k_label: np.ndarray, K: int) -> np.ndarray:
    """
    Compute empirical regime frequencies from training-window labels.
    Used to anchor gamma^(0) in Eq. (4).  Returns a length-K probability vector.
    """
    freq = np.maximum(np.bincount(k_label, minlength=K).astype(float), _eps())
    return freq / freq.sum()


def _log_emission(y_t, x_t, beta, sigma2, K):
    """Gaussian log-emission log p(y_t | s_t = k) for each of the K components."""
    eps = _eps()
    out = np.empty(K)
    for k in range(K):
        out[k] = stats.norm.logpdf(y_t, x_t @ beta[k], np.sqrt(max(sigma2[k], eps)))
    return out


def _baum_welch(y: np.ndarray, X: np.ndarray,
                beta: np.ndarray, sigma2: np.ndarray,
                Pi: np.ndarray, K: int) -> tuple:
    """
    E-step of Baum-Welch EM: compute posterior responsibilities gamma_{t,k}
    and expected transition counts xi via the forward-backward algorithm.

    All quantities are maintained in log-domain via logsumexp to prevent
    underflow in long sequences (Appendix A).  Rows with a non-finite
    log-normalizer are reset to the uniform distribution over K states.

    Returns:
        gamma  : (T, K) smoothed posterior state probabilities
        xi_sum : (K, K) sum of expected transition counts across t = 0..T-2
    """
    eps      = _eps()
    T        = len(y)
    log_Pi   = np.log(np.clip(Pi, eps, 1.0))
    log_init = np.log(np.full(K, 1.0 / K))  # uniform initial state distribution

    # Forward pass: log alpha_{t,k} = log P(y_{1:t}, s_t = k)
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = log_init + _log_emission(y[0], X[0], beta, sigma2, K)
    for t in range(1, T):
        log_et = _log_emission(y[t], X[t], beta, sigma2, K)
        for k in range(K):
            log_alpha[t, k] = logsumexp(log_alpha[t-1] + log_Pi[:, k]) + log_et[k]

    # Backward pass: log beta_{t,k} = log P(y_{t+1:T} | s_t = k)
    log_bwd = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_et1 = _log_emission(y[t+1], X[t+1], beta, sigma2, K)
        for k in range(K):
            log_bwd[t, k] = logsumexp(log_Pi[k, :] + log_et1 + log_bwd[t+1])

    log_gamma = log_alpha + log_bwd
    log_norm  = logsumexp(log_gamma, axis=1, keepdims=True)
    valid     = np.isfinite(log_norm[:, 0])
    log_gamma[valid]  -= log_norm[valid]
    log_gamma[~valid]  = np.log(1.0 / K)
    gamma = np.exp(log_gamma)

    # Expected transition counts xi_{t,j,k} = P(s_t = j, s_{t+1} = k | y_{1:T})
    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        log_et1    = _log_emission(y[t+1], X[t+1], beta, sigma2, K)
        log_norm_t = logsumexp(log_alpha[t])
        for j in range(K):
            for kk in range(K):
                lx = (log_alpha[t, j] + log_Pi[j, kk]
                      + log_et1[kk] + log_bwd[t+1, kk] - log_norm_t)
                xi[t, j, kk] = np.exp(lx) if np.isfinite(lx) else 0.0
        rs = xi[t].sum()
        xi[t] = xi[t] / rs if rs > eps else 1.0 / (K * K)

    return gamma, xi.sum(axis=0)


def _forward_gamma(X_te: np.ndarray, y_te: np.ndarray,
                   beta: np.ndarray, sigma2: np.ndarray,
                   Pi: np.ndarray, K: int) -> np.ndarray:
    """
    HMM forward filtering at test time: compute the one-step-ahead posterior
    gamma_{h,k} = P(s_h = k | y_0, ..., y_{h-1}), Eq. (3).

    Emission at step t uses y_{t-1}, so gamma_{h,k} is strictly causal
    (does not incorporate y_h).  Returns a (T, K) array.
    """
    eps      = _eps()
    T        = len(X_te)
    log_Pi   = np.log(np.clip(Pi, eps, 1.0))

    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = np.log(np.full(K, 1.0 / K))  # uniform prior at t = 0
    for t in range(1, T):
        log_et_prev = _log_emission(y_te[t-1], X_te[t-1], beta, sigma2, K)
        for k in range(K):
            log_alpha[t, k] = logsumexp(log_alpha[t-1] + log_Pi[:, k]) + log_et_prev[k]

    log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
    valid    = np.isfinite(log_norm[:, 0])
    log_alpha[valid]  -= log_norm[valid]
    log_alpha[~valid]  = np.log(1.0 / K)
    return np.exp(log_alpha)


def _sort_by_sigma2(beta, sigma2, Pi):
    """Order HMM states by ascending residual variance.
    Used for MS-HAR and RAFT_noLabel where no label anchor is available (Appendix A)."""
    o = np.argsort(sigma2)
    return beta[o], sigma2[o], Pi[np.ix_(o, o)]


def _sort_by_label_alignment(beta, sigma2, Pi, gamma, k_label, K):
    """
    Resolve HMM label-switching by finding the state permutation pi* that
    maximises average within-regime posterior mass, Eq. (A.1).

    This ensures that 'state 2' consistently corresponds to Contagion across
    rolling windows, which is necessary for coherent out-of-sample evaluation.
    """
    score_mat = np.zeros((K, K))
    for i in range(K):
        mask = k_label == i
        score_mat[i] = gamma[mask].mean(axis=0) if mask.sum() > 0 else 1.0 / K
    best_perm = max(permutations(range(K)),
                    key=lambda p: sum(score_mat[i, p[i]] for i in range(K)))
    bp = list(best_perm)
    return beta[bp], sigma2[bp], Pi[np.ix_(bp, bp)]


def _em_loop(y, X, gamma, Pi, K, beta, sigma2):
    """
    Baum-Welch EM: starting from the current gamma, alternate M-step then E-step
    for em_iter iterations, regularizing Pi after each M-step (Appendix A).
    Returns updated (gamma, Pi, beta, sigma2).
    """
    eps = _eps()
    for _ in range(_fc()["em_iter"]):
        for k in range(K):
            beta[k], sigma2[k] = _wls(X, y, gamma[:, k])
        gamma, xi_sum = _baum_welch(y, X, beta, sigma2, Pi, K)
        Pi_new = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + eps)
        Pi = _pi_regularize(Pi_new, K)
    return gamma, Pi, beta, sigma2


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class Persistence:
    """Naive benchmark: forecast = last observed log realized variance."""
    name = "Persistence"

    def fit(self, df, thresholds=None, k_tr=None):
        self.sigma2_ = float(np.var(df["log_rv"].values - df["rv_lag1"].values))

    def predict(self, df, thresholds=None, k_te=None):
        return df["rv_lag1"].values, np.full(len(df), self.sigma2_)


class HARRV:
    """HAR-RV (Corsi, 2009): regime-agnostic benchmark, estimated by OLS."""
    name = "HAR-RV"

    def fit(self, df, thresholds=None, k_tr=None):
        X = _har_X(df)
        y = df["log_rv"].values
        self.beta_, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.sigma2_   = float(np.var(y - X @ self.beta_))

    def predict(self, df, thresholds=None, k_te=None):
        return _har_X(df) @ self.beta_, np.full(len(df), self.sigma2_)


class MSHAR:
    """
    Markov-Switching HAR with hard regime assignment at prediction time (Section 3).

    Parameters are estimated by Baum-Welch EM with uniform gamma^(0) (no label
    anchor).  At test time threshold_regime() assigns each hour to exactly one
    regime before selecting the corresponding (beta_k, sigma2_k).
    """
    name = "MS-HAR"

    def fit(self, df, thresholds=None, k_tr=None):
        K     = _fc()["n_regimes"]
        y     = df["log_rv"].values
        X     = _har_X(df)
        T     = len(y)
        var_y = np.var(y)
        gamma  = np.full((T, K), 1.0 / K)
        Pi     = _pi_init(K)
        beta   = np.zeros((K, X.shape[1]))
        sigma2 = np.array([var_y * 0.3, var_y * 1.0, var_y * 3.0])
        gamma, Pi, beta, sigma2 = _em_loop(y, X, gamma, Pi, K, beta, sigma2)
        self.beta_, self.sigma2_, self.Pi_ = _sort_by_sigma2(beta, sigma2, Pi)

    def predict(self, df, thresholds=None, k_te=None):
        X     = _har_X(df)
        T     = len(df)
        k_arr = k_te if k_te is not None else threshold_regime(df, thresholds)
        return (np.array([X[t] @ self.beta_[k_arr[t]] for t in range(T)]),
                np.array([self.sigma2_[k_arr[t]] for t in range(T)]))


class RAFT:
    """
    Regime-Adaptive Forecasting with Temporal Filtering (Section 3).

    Training: Baum-Welch EM initialized from empirical regime frequencies (Eq. 4);
    label-switching resolved by score-based permutation search, Eq. (A.1).
    Prediction: HMM forward recursion (Eq. 3) produces soft posterior gamma_{h,k};
    point forecast is the posterior-weighted mixture mean, Eq. (2).
    """
    name     = "RAFT"
    _use_lbl = True

    def fit(self, df, thresholds=None, k_tr=None):
        K     = _fc()["n_regimes"]
        y     = df["log_rv"].values
        X     = _har_X(df)
        T     = len(y)
        var_y = np.var(y)

        if self._use_lbl and k_tr is not None:
            k_label   = k_tr
            init_dist = _freq_init(k_tr, K)  # Eq. (4): frequency-anchored initialization
        else:
            k_label   = np.zeros(T, dtype=int)
            init_dist = np.full(K, 1.0 / K)

        gamma  = np.tile(init_dist, (T, 1))
        Pi     = _pi_init(K)
        beta   = np.zeros((K, X.shape[1]))
        sigma2 = np.array([var_y * 0.3, var_y * 1.0, var_y * 3.0])

        gamma, Pi, beta, sigma2 = _em_loop(y, X, gamma, Pi, K, beta, sigma2)

        # Warn if the three components have collapsed toward indistinguishable variances
        s2_ratio = sigma2.max() / max(sigma2.min(), _eps())
        if s2_ratio < _fc()["collapse_ratio_thresh"]:
            print(f"  [COLLAPSE] {self.name}: sigma2_ratio={s2_ratio:.3f}")

        self.beta_, self.sigma2_, self.Pi_ = _sort_by_label_alignment(
            beta, sigma2, Pi, gamma, k_label, K)

    def predict(self, df, thresholds=None, k_te=None):
        K        = _fc()["n_regimes"]
        X_te     = _har_X(df)
        y_te     = df["log_rv"].values
        gamma_te = _forward_gamma(X_te, y_te, self.beta_, self.sigma2_, self.Pi_, K)  # Eq.(3)
        # Eq.(2): posterior-weighted mixture mean
        beta_preds = np.stack([X_te @ self.beta_[k] for k in range(K)], axis=1)
        mu         = (gamma_te * beta_preds).sum(axis=1)
        s2         = gamma_te @ self.sigma2_
        return mu, s2


class HardHMM(RAFT):
    """
    Hard-HMM: the hard-decision counterpart to RAFT.

    In the rolling-window evaluation, Hard-HMM is implemented by reusing RAFT's
    fitted HMM-HAR parameters and filtered posteriors, then collapsing the
    forecast-time posterior to the single most likely regime,
    k_hat = argmax_k gamma_{h,k}. The class definition is retained for naming
    and table-ordering consistency.
    """
    name = "Hard-HMM"

    def predict(self, df, thresholds=None, k_te=None):
        K        = _fc()["n_regimes"]
        X_te     = _har_X(df)
        y_te     = df["log_rv"].values
        gamma_te = _forward_gamma(X_te, y_te, self.beta_, self.sigma2_, self.Pi_, K)
        k_hat    = np.argmax(gamma_te, axis=1)
        mu       = np.array([X_te[t] @ self.beta_[k_hat[t]] for t in range(len(df))])
        s2       = np.array([self.sigma2_[k_hat[t]] for t in range(len(df))])
        return mu, s2


class RAFTnoLabel(RAFT):
    """
    Ablation (Table 5, row 3): RAFT with uniform gamma^(0) instead of
    frequency-based initialization.  Isolates the contribution of Eq. (4).
    State ordering falls back to ascending sigma2 (no label anchor available).
    """
    name     = "RAFT_noLabel"
    _use_lbl = False

    def fit(self, df, thresholds=None, k_tr=None):
        K     = _fc()["n_regimes"]
        y     = df["log_rv"].values
        X     = _har_X(df)
        T     = len(y)
        var_y = np.var(y)
        gamma  = np.full((T, K), 1.0 / K)
        Pi     = _pi_init(K)
        beta   = np.zeros((K, X.shape[1]))
        sigma2 = np.array([var_y * 0.3, var_y * 1.0, var_y * 3.0])
        gamma, Pi, beta, sigma2 = _em_loop(y, X, gamma, Pi, K, beta, sigma2)
        self.beta_, self.sigma2_, self.Pi_ = _sort_by_sigma2(beta, sigma2, Pi)


class RAFTnoDelta(RAFT):
    """
    Ablation (Table 5, row 4): pooled (regime-invariant) beta, per-regime sigma2_k.
    In each EM iteration, beta is estimated by pooled OLS and broadcast to all K
    components; per-regime sigma2_k is updated by gamma-weighted WLS as usual.
    """
    name     = "RAFT_noDelta"
    _use_lbl = True

    def fit(self, df, thresholds=None, k_tr=None):
        K     = _fc()["n_regimes"]
        y     = df["log_rv"].values
        X     = _har_X(df)
        T     = len(y)
        var_y = np.var(y)
        eps   = _eps()

        if self._use_lbl and k_tr is not None:
            k_label   = k_tr
            init_dist = _freq_init(k_tr, K)
        else:
            k_label   = np.zeros(T, dtype=int)
            init_dist = np.full(K, 1.0 / K)

        gamma  = np.tile(init_dist, (T, 1))
        Pi     = _pi_init(K)
        sigma2 = np.array([var_y * 0.3, var_y * 1.0, var_y * 3.0])

        for _ in range(_fc()["em_iter"]):
            # Pooled OLS: single beta shared across all regimes
            beta_p, _ = _wls(X, y, np.ones(T))
            beta      = np.tile(beta_p, (K, 1))
            # Per-regime sigma2: gamma-weighted residual variance for each component
            for k in range(K):
                _, sigma2[k] = _wls(X, y, gamma[:, k])
            gamma, xi_sum = _baum_welch(y, X, beta, sigma2, Pi, K)
            Pi_new = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + eps)
            Pi = _pi_regularize(Pi_new, K)

        self.beta_, self.sigma2_, self.Pi_ = _sort_by_label_alignment(
            beta, sigma2, Pi, gamma, k_label, K)


TABLE3_MODELS = [Persistence, HARRV, MSHAR, HardHMM, RAFT]
TABLE5_MODELS = [RAFT, HardHMM, RAFTnoLabel, RAFTnoDelta]
# Display order follows the paper tables; run order fits RAFT before deriving
# the shared-fit Hard-HMM counterpart to avoid an extra EM run.
MODEL_CLASSES = list({m.name: m for m in TABLE3_MODELS + TABLE5_MODELS}.values())
MODEL_RUN_CLASSES = [Persistence, HARRV, MSHAR, RAFT, RAFTnoLabel, RAFTnoDelta]


def _predict_hard_hmm_from_raft(df_te: pd.DataFrame, beta: np.ndarray,
                                sigma2: np.ndarray, Pi: np.ndarray):
    """Construct Hard-HMM forecasts from the fitted RAFT parameters only.

    This preserves a like-for-like comparison with RAFT by sharing the same
    HMM-HAR fit and filtered posteriors, differing only in replacing the soft
    posterior average with the hard decision k_hat = argmax_k gamma_{h,k}.
    """
    K        = _fc()["n_regimes"]
    X_te     = _har_X(df_te)
    y_te     = df_te["log_rv"].values
    gamma_te = _forward_gamma(X_te, y_te, beta, sigma2, Pi, K)
    k_hat    = np.argmax(gamma_te, axis=1)
    mu       = np.array([X_te[t] @ beta[k_hat[t]] for t in range(len(df_te))])
    s2       = np.array([sigma2[k_hat[t]] for t in range(len(df_te))])
    return mu, s2


def _run_one_window(args):
    """
    Fit and evaluate all models on a single rolling window.
    Called via ProcessPoolExecutor; cfg_snapshot is passed explicitly because
    worker processes do not share the parent's _CFG global.
    """
    ws, df_tr_raw, df_te_raw, cfg_snapshot = args
    _CFG.update(cfg_snapshot)

    med   = df_tr_raw["x_corr"].median()
    df_tr = df_tr_raw.copy()
    df_te = df_te_raw.copy()
    # x_corr (lagged cross-asset correlation) is not used in any model's
    # prediction step; fillna here prevents NaN propagation if the column
    # is later referenced in diagnostics.
    df_tr["x_corr"] = df_tr["x_corr"].fillna(med)
    df_te["x_corr"] = df_te["x_corr"].fillna(med)

    thresholds  = compute_thresh_quantiles(df_tr)
    k_tr        = df_tr["regime_fi"].map(LABEL_MAP).fillna(REGIME_NORMAL).values.astype(int)
    k_te_thresh = threshold_regime(df_te, thresholds)
    # Held-out labels: used only for ex-post error stratification, not for any model's prediction
    regime_te   = df_te["regime_fi"].fillna("Normal").values.astype(str) \
                  if "regime_fi" in df_te.columns \
                  else np.array([INT_TO_LBL[k] for k in k_te_thresh])
    hb_te       = df_te["hb_tz"].values

    out = {}
    raft_fit = None
    for Cls in MODEL_RUN_CLASSES:
        m = Cls()
        try:
            m.fit(df_tr, thresholds=thresholds, k_tr=k_tr)
            mu, sigma2 = m.predict(df_te, thresholds=thresholds, k_te=k_te_thresh)
            entry = {
                "hb":     hb_te,
                "mu":     mu,
                "sigma2": sigma2,
                "y":      df_te["log_rv"].values,
                "regime": regime_te,
                "ws":     str(ws.date()),
            }
            if m.name == "MS-HAR":
                entry["k_pred"] = k_te_thresh  # retained for confusion-matrix analysis
            if m.name == "RAFT":
                entry["beta_hat"] = np.array(m.beta_, copy=True)
                entry["sigma2_hat"] = np.array(m.sigma2_, copy=True)
                entry["Pi_hat"] = np.array(m.Pi_, copy=True)
                raft_fit = {
                    "beta": np.array(m.beta_, copy=True),
                    "sigma2": np.array(m.sigma2_, copy=True),
                    "Pi": np.array(m.Pi_, copy=True),
                }
            out[m.name] = entry
        except Exception as e:
            print(f"{m.name} @ {ws.date()}: {e}")

    if raft_fit is not None:
        mu_hard, sigma2_hard = _predict_hard_hmm_from_raft(
            df_te, raft_fit["beta"], raft_fit["sigma2"], raft_fit["Pi"]
        )
        out["Hard-HMM"] = {
            "hb":     hb_te,
            "mu":     mu_hard,
            "sigma2": sigma2_hard,
            "y":      df_te["log_rv"].values,
            "regime": regime_te,
            "ws":     str(ws.date()),
        }
    else:
        print(f"Hard-HMM @ {ws.date()}: skipped because RAFT fit was unavailable")
    return out


def print_mshar_confusion(results: dict) -> None:
    """Print a confusion matrix of MS-HAR threshold predictions vs. held-out labels."""
    preds = results.get("MS-HAR", [])
    if not preds:
        return

    labels = ["Normal", "Single-Asset Stress", "Contagion"]
    cm     = np.zeros((3, 3), dtype=int)

    for p in preds:
        if "k_pred" not in p:
            continue
        for actual_str, pred_int in zip(p["regime"], p["k_pred"]):
            actual_int = LABEL_MAP.get(actual_str, REGIME_NORMAL)
            cm[actual_int, pred_int] += 1

    total = cm.sum()
    if total == 0:
        return

    col_w   = 22
    lbl_w   = 24
    print("\n" + "=" * 95)
    print("MS-HAR Regime Classification: Held-Out Label vs Threshold Prediction")
    print("=" * 95)
    print(f"{'':>{lbl_w}}    Predicted by MS-HAR threshold")
    print(f"{'Actual (held-out)':>{lbl_w}}  "
          + "  ".join(f"{l:>{col_w}}" for l in ["Normal", "Stress", "Contagion"]))
    print("-" * 95)

    for i, act_lbl in enumerate(labels):
        row_total = cm[i].sum()
        cells = [f"{cm[i,j]:>7,} ({100*cm[i,j]/row_total:5.1f}%)" for j in range(3)]
        print(f"  {act_lbl:<{lbl_w-2}}  " + "  ".join(f"{c:>{col_w}}" for c in cells)
              + f"   n={row_total:>6,}")

    print("-" * 95)
    cells = [f"{v:>7,} ({100*v/total:5.1f}%)" for v in cm.sum(axis=0)]
    print(f"  {'Total':<{lbl_w-2}}  " + "  ".join(f"{c:>{col_w}}" for c in cells)
          + f"   n={total:>6,}")
    print("=" * 95)


def print_raft_parameter_summary(results: dict) -> None:
    """Print RAFT parameter means averaged across rolling windows."""
    preds = results.get("RAFT", [])
    if not preds:
        return

    beta_list = [p["beta_hat"] for p in preds if "beta_hat" in p]
    s2_list   = [p["sigma2_hat"] for p in preds if "sigma2_hat" in p]
    Pi_list   = [p["Pi_hat"] for p in preds if "Pi_hat" in p]

    if not beta_list or not s2_list or not Pi_list:
        print("\n[RAFT params] No stored parameter estimates found.")
        return

    beta_mean = np.mean(np.stack(beta_list, axis=0), axis=0)   # (K, 4)
    s2_mean   = np.mean(np.stack(s2_list, axis=0), axis=0)     # (K,)
    Pi_mean   = np.mean(np.stack(Pi_list, axis=0), axis=0)     # (K, K)

    regime_names = ["Normal", "Single-Asset Stress", "Contagion"]
    feature_names = ["Intercept", "log-RV^(1)", "log-RV^(24)", "log-RV^(168)"]

    print("\n" + "=" * 95)
    print("RAFT parameter summary (mean across 4 rolling estimation windows)")
    print("=" * 95)

    for k, rname in enumerate(regime_names):
        print(f"\n[{rname}]")
        print(f"  sigma2 = {s2_mean[k]:.3f}")
        for j, fname in enumerate(feature_names):
            print(f"  beta[{fname}] = {beta_mean[k, j]: .3f}")


def rolling_oos(df: pd.DataFrame, cfg: dict) -> dict:
    """Execute the rolling OOS evaluation across all windows in parallel."""
    print("[2/4] Rolling OOS evaluation...")
    fc       = cfg["forecast"]
    test_end = df["hb_tz"].max()
    ts       = pd.Timestamp(fc["test_start"])
    windows  = []

    while ts <= test_end:
        we       = min(ts + pd.DateOffset(months=fc["roll_months"])
                       - pd.Timedelta(hours=1), test_end)
        tr_start = ts - pd.DateOffset(months=fc["train_months"])
        df_tr    = df[(df["hb_tz"] >= tr_start) & (df["hb_tz"] < ts)].copy()
        df_te    = df[(df["hb_tz"] >= ts) & (df["hb_tz"] <= we)].copy()
        if len(df_tr) >= 500 and len(df_te) >= 10:
            windows.append((ts, df_tr, df_te, dict(cfg)))
        ts += pd.DateOffset(months=fc["roll_months"])

    n_workers = fc["n_workers"] or min(len(windows), os.cpu_count() or 4)
    results   = {Cls().name: [] for Cls in MODEL_CLASSES}

    regime_arr = np.concatenate([w[2]["regime_fi"].fillna("Normal").values for w in windows])
    unique, counts = np.unique(regime_arr, return_counts=True)
    print(f"  Regime hour counts across {len(windows)} test windows (total={len(regime_arr):,}):")
    for r, c in sorted(zip(unique, counts), key=lambda x: LABEL_MAP.get(x[0], 99)):
        print(f"    {r:<30} n={c:>7,}")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_run_one_window, w): w[0] for w in windows}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="  windows"):
            ws = futs[fut]
            try:
                for name, vals in fut.result().items():
                    results[name].append(vals)
            except Exception as e:
                print(f"window {ws.date()} failed: {e}")
    for name in results:
        results[name].sort(key=lambda x: x["ws"])
    return results


def _concat_preds(preds, regime_filter=None):
    """Concatenate prediction arrays across rolling windows, optionally
    restricted to a single regime stratum."""
    hbs, mus, ys, s2s = [], [], [], []
    for p in preds:
        mask = (p["regime"] == regime_filter) if regime_filter \
               else np.ones(len(p["y"]), dtype=bool)
        if mask.sum() == 0:
            continue
        hbs.append(p["hb"][mask])
        mus.append(p["mu"][mask])
        ys.append(p["y"][mask])
        s2s.append(p["sigma2"][mask])
    if not mus:
        return None, None, None, None
    return (np.concatenate(hbs), np.concatenate(mus),
            np.concatenate(ys),  np.concatenate(s2s))



def aggregate_metrics(results: dict, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute QLIKE and Log-Score by regime stratum (Table 3) and
    collect Diebold-Mariano tests for the standalone Table 4."""
    print("[3/4] Aggregating metrics...")
    groups = {
        "All Hours":                 None,
        "Contagion Hours":           "Contagion",
        "Single-Asset Stress Hours": "Single-Asset Stress",
        "Normal Hours":              "Normal",
    }
    rows = []
    for Cls in TABLE3_MODELS:
        name  = Cls().name
        preds = results.get(name, [])
        if not preds:
            continue
        row = {"Model": name}
        for label, rf in groups.items():
            _, mu, y, s2 = _concat_preds(preds, rf)
            if mu is None or len(mu) < 5:
                row[f"{label}_LogS"]  = np.nan
                row[f"{label}_QLIKE"] = np.nan
            else:
                row[f"{label}_LogS"]  = round(log_score(y, mu, s2), 4)
                row[f"{label}_QLIKE"] = round(qlike(mu, y), 4)
        rows.append(row)

    table = pd.DataFrame(rows)

    pr = results.get("RAFT",      [])
    ph = results.get("HAR-RV",    [])
    pm = results.get("MS-HAR",    [])
    phh = results.get("Hard-HMM", [])

    dm_rows = []
    dm_specs = [
        ("vs. HAR-RV", ph),
        ("vs. MS-HAR", pm),
        ("vs. Hard-HMM", phh),
    ]
    strata = [
        ("All", None, "All"),
        ("Cont", "Contagion", "Cont"),
        ("Stress", "Single-Asset Stress", "Stress"),
        ("Norm", "Normal", "Norm"),
    ]

    for comparison_label, preds_b in dm_specs:
        if not pr or not preds_b:
            continue
        row = {"Comparison": comparison_label}
        for _, rf, short_label in strata:
            la, lb = _align_losses(pr, preds_b, rf)
            if la is None:
                row[f"{short_label} d"] = np.nan
                row[f"{short_label} p"] = np.nan
                continue
            d = float(np.mean(la - lb))
            pv = diebold_mariano(la, lb)
            row[f"{short_label} d"] = round(d, 4)
            row[f"{short_label} p"] = pv
        dm_rows.append(row)

    dm_table = pd.DataFrame(dm_rows, columns=[
        "Comparison",
        "All d", "All p",
        "Cont d", "Cont p",
        "Stress d", "Stress p",
        "Norm d", "Norm p",
    ])

    csv_cols = [
        "Model",
        "All Hours_QLIKE",                 "All Hours_LogS",
        "Contagion Hours_QLIKE",           "Contagion Hours_LogS",
        "Single-Asset Stress Hours_QLIKE", "Single-Asset Stress Hours_LogS",
        "Normal Hours_QLIKE",              "Normal Hours_LogS",
    ]
    out = results_dir / "table3_oos_performance.csv"
    table[[c for c in csv_cols if c in table.columns]].to_csv(out, index=False)
    print(f"  Saved {out}")

    dm_out = results_dir / "table4_dm_tests.csv"
    if not dm_table.empty:
        dm_out_table = dm_table.copy()
        for col in ["All p", "Cont p", "Stress p", "Norm p"]:
            dm_out_table[col] = dm_out_table[col].apply(_fmt_p_csv)
        dm_out_table.to_csv(dm_out, index=False)
        print(f"  Saved {dm_out}")
    else:
        print("  No DM test results were available to save.")

    return table, dm_table




def _fmt_p_print(p) -> str:
    if np.isnan(p):
        return "   nan"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _fmt_p_csv(p) -> str:
    if np.isnan(p):
        return "nan"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def save_table3(table: pd.DataFrame) -> None:
    print("\n" + "=" * 110)
    print("TABLE 3: Out-of-Sample Forecasting Performance (QLIKE lower = better, Log-S higher = better)")
    print("=" * 110)

    hdr = (f"{'Model':<14} "
           f"{'All QLIKE':>10} {'All Log-S':>10} "
           f"{'Cont QLIKE':>11} {'Cont Log-S':>11} "
           f"{'Stress QLIKE':>13} {'Stress Log-S':>13} "
           f"{'Norm QLIKE':>11} {'Norm Log-S':>11}")
    print(hdr)

    for _, row in table.iterrows():
        print(
            f"{row['Model']:<14} "
            f"{row.get('All Hours_QLIKE',                 np.nan):>10.3f} "
            f"{row.get('All Hours_LogS',                  np.nan):>10.3f} "
            f"{row.get('Contagion Hours_QLIKE',           np.nan):>11.3f} "
            f"{row.get('Contagion Hours_LogS',            np.nan):>11.3f} "
            f"{row.get('Single-Asset Stress Hours_QLIKE', np.nan):>13.3f} "
            f"{row.get('Single-Asset Stress Hours_LogS',  np.nan):>13.3f} "
            f"{row.get('Normal Hours_QLIKE',              np.nan):>11.3f} "
            f"{row.get('Normal Hours_LogS',               np.nan):>11.3f}")



def save_table4_dm(dm_table: pd.DataFrame) -> None:
    """Print Table 4: Diebold-Mariano test results (QLIKE) in wide format."""
    print("\n" + "=" * 126)
    print("TABLE 4: Diebold-Mariano Tests (QLIKE)")
    print("=" * 126)

    if dm_table.empty:
        print("  No DM test results available.")
        return

    hdr = (
        f"{'Comparison':<16} "
        f"{'All d':>8} {'All p':>8} "
        f"{'Cont d':>9} {'Cont p':>8} "
        f"{'Stress d':>10} {'Stress p':>10} "
        f"{'Norm d':>8} {'Norm p':>8}"
    )
    print(hdr)
    print("-" * 126)

    for _, row in dm_table.iterrows():
        print(
            f"{str(row['Comparison']):<16} "
            f"{float(row['All d']):>8.3f} "
            f"{_fmt_p_print(row['All p']):>8} "
            f"{float(row['Cont d']):>9.3f} "
            f"{_fmt_p_print(row['Cont p']):>8} "
            f"{float(row['Stress d']):>10.3f} "
            f"{_fmt_p_print(row['Stress p']):>10} "
            f"{float(row['Norm d']):>8.3f} "
            f"{_fmt_p_print(row['Norm p']):>8}"
        )


def save_table5(results: dict, results_dir: Path) -> None:
    """Compute contagion-hour QLIKE for each ablation variant (Table 5)."""
    print("[4/4] Ablation (Table 5)...")
    ablation_order = [
        ("RAFT",         "Full RAFT (soft predict, freq. init)"),
        ("Hard-HMM",     "Hard-HMM: hard predict, freq. init"),
        ("RAFT_noLabel", "RAFT_noLabel: soft predict, uniform γ init"),
        ("RAFT_noDelta", "RAFT_noDelta: soft predict, pooled β, freq. init"),
    ]
    full_q = None
    rows   = []
    for name, label in ablation_order:
        _, mu, y, _ = _concat_preds(results.get(name, []), "Contagion")
        q = qlike(mu, y) if mu is not None and len(mu) > 0 else np.nan
        if name == "RAFT":
            full_q = q
        delta = q - full_q if (full_q is not None and not np.isnan(q) and name != "RAFT") else np.nan
        rows.append({"Variant": label, "QLIKE": q, "Delta vs Full RAFT": delta})

    print("\n" + "=" * 80)
    print("TABLE 5: Ablation — Contagion Hours QLIKE")
    print(f"  {'Variant':<54}  {'QLIKE':>7}  {'Delta':>7}")
    print("=" * 80)
    for r in rows:
        q_str     = f"{r['QLIKE']:.3f}" if not np.isnan(r['QLIKE']) else "   nan"
        delta_str = f"{r['Delta vs Full RAFT']:+.3f}" if not np.isnan(r['Delta vs Full RAFT']) else "      —"
        print(f"  {r['Variant']:<54}  {q_str:>7}  {delta_str:>7}")

    out = results_dir / "table5_ablation.csv"
    df_out = pd.DataFrame(rows)
    df_out["QLIKE"]              = df_out["QLIKE"].round(3)
    df_out["Delta vs Full RAFT"] = df_out["Delta vs Full RAFT"].round(3)
    df_out.to_csv(out, index=False)
    print(f"  Saved {out}")


def parse_args():
    p = argparse.ArgumentParser(description="RAFT rolling OOS forecast")
    p.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"),
                   help="Path to config.yaml")
    return p.parse_args()


def main():
    # One thread per process; window-level parallelism is handled by ProcessPoolExecutor
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(var, "1")
    args        = parse_args()
    cfg         = load_config(args.config)
    results_dir = cfg["_results_dir"]

    df      = load_data(cfg)
    results = rolling_oos(df, cfg)
    table, dm_table = aggregate_metrics(results, results_dir)
    save_table3(table)
    save_table4_dm(dm_table)
    print_mshar_confusion(results)
    print_raft_parameter_summary(results)
    save_table5(results, results_dir)
    print(f"\nDone. Outputs in: {results_dir}")


if __name__ == "__main__":
    main()