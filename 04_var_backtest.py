"""
04_var_backtest.py  —  Replicates Table 6
Rolling out-of-sample VaR and CVaR backtest for Persistence, HAR-RV,
MS-HAR, Hard-HMM, and RAFT.

VaR and CVaR for RAFT are computed from the full three-component Gaussian
mixture predictive distribution. Single-component benchmarks reduce to the
standard Gaussian quantile and truncated-normal expectation. Coverage is
assessed with the Kupiec POF test, and tail severity is measured by MAE-CVaR.

Run order: 01 -> 02 -> 03 -> 04
"""

import os
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations

import yaml
import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import logsumexp
from scipy.integrate import quad
from tqdm import tqdm

warnings.filterwarnings("ignore")

SHORTS = ["BTC", "ETH", "XRP", "LTC", "LINK"]
HVOL_COLS = [f"{s}_hvol" for s in SHORTS]
EM_COLS = [f"{s}_extreme_move" for s in SHORTS]
LABEL_MAP = {"Normal": 0, "Single-Asset Stress": 1, "Contagion": 2}
INT_TO_LBL = {v: k for k, v in LABEL_MAP.items()}

REGIME_NORMAL = 0
REGIME_STRESS = 1
REGIME_CONTAGION = 2

_CFG: dict = {}


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
    """Load hourly RV stats, merge regime labels, and build HAR features on the log-RV scale."""
    print("Loading data...")
    rd = cfg["_results_dir"]
    hourly = pd.read_parquet(rd / "hourly_stats.parquet")
    regimes = pd.read_parquet(rd / "regimes_hourly.parquet")
    for d in (hourly, regimes):
        d["hour_block"] = pd.to_datetime(d["hour_block"], utc=True)

    df = (hourly
          .merge(regimes[["hour_block", "regime"]].rename(columns={"regime": "regime_fi"}),
                 on="hour_block", how="left")
          .sort_values("hour_block").reset_index(drop=True))

    eps = _eps()
    df["log_rv"] = np.log(df[HVOL_COLS].pow(2).clip(lower=eps)).mean(axis=1)
    y = df["log_rv"]
    df["rv_lag1"] = y.shift(1)
    df["rv_lag24"] = y.shift(1).rolling(24, min_periods=24).mean()
    df["rv_lag168"] = y.shift(1).rolling(168, min_periods=168).mean()

    hvol_mat = df[HVOL_COLS].values
    df["cs_hvol_std"] = np.nanstd(hvol_mat, axis=1)
    df["cs_hvol_max"] = np.nanmax(hvol_mat, axis=1)
    df["cs_hvol_std_l1"] = df["cs_hvol_std"].shift(1)
    df["cs_hvol_max_l1"] = df["cs_hvol_max"].shift(1)

    em_avail = [c for c in EM_COLS if c in df.columns]
    if em_avail:
        df["extreme_move_mean"] = df[em_avail].mean(axis=1)
    else:
        roll_std = df[HVOL_COLS].rolling(720, min_periods=50).std()
        roll_mu = df[HVOL_COLS].rolling(720, min_periods=50).mean()
        df["extreme_move_mean"] = ((df[HVOL_COLS] > (roll_mu + 2 * roll_std)).astype(float).mean(axis=1))
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
        "tw_q80_em": np.nanpercentile(df_tr["extreme_move_l1"].values, 80),
        "per_asset_q80": {
            s: np.nanpercentile(df_tr[f"{s}_hvol_l1"].values, 80)
            for s in SHORTS if f"{s}_hvol_l1" in df_tr.columns
        },
    }


def threshold_regime(df: pd.DataFrame, thr: dict) -> np.ndarray:
    """
    Hard regime assignment for MS-HAR at prediction time (Section 3).

    Because contemporaneous cross-asset return correlation is unavailable at
    forecast time, the rule uses lagged cross-sectional volatility dispersion
    as a proxy for condition (ii).

    Returns an integer array with values in {REGIME_NORMAL, REGIME_STRESS,
    REGIME_CONTAGION} (i.e., {0, 1, 2}).
    """
    per_q80 = thr["per_asset_q80"]
    n_above_80 = sum(
        (df[f"{s}_hvol_l1"].values > per_q80[s]).astype(int)
        for s in SHORTS if f"{s}_hvol_l1" in df.columns and s in per_q80
    )
    contagion_mask = (
        (df["cs_hvol_max_l1"].values > thr["tw_q95_rv_max"]) &
        (df["cs_hvol_std_l1"].values > thr["tw_q90_cs_std"]) &
        (n_above_80 >= 3)
    )
    stress_mask = ~contagion_mask & (df["extreme_move_l1"].values > thr["tw_q80_em"])
    out = np.zeros(len(df), dtype=int)
    out[stress_mask] = REGIME_STRESS
    out[contagion_mask] = REGIME_CONTAGION
    return out


def _har_X(df: pd.DataFrame) -> np.ndarray:
    """HAR feature matrix: x_h = (1, logRV^(1)_h, logRV^(24)_h, logRV^(168)_h)."""
    return np.column_stack([
        np.ones(len(df)),
        df["rv_lag1"].values,
        df["rv_lag24"].values,
        df["rv_lag168"].values,
    ])


def _wls(X, y, w):
    """
    Gamma-weighted least squares for the EM M-step.
    A small ridge (eps * I) is added to X'WX to prevent near-singular systems
    when a regime carries very low posterior mass (Appendix A).
    Returns (beta, sigma2) where sigma2 is the gamma-weighted residual variance.
    """
    eps = _eps()
    sw = w.sum()
    if sw < eps:
        return np.linalg.lstsq(X, y, rcond=None)[0], np.var(y) + eps
    Xw = X * w[:, None]
    XtWX = Xw.T @ X + eps * np.eye(X.shape[1])
    XtWy = Xw.T @ y
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return beta, max(float(np.dot(w, resid ** 2) / sw), eps)


def _pi_init(K, diag=0.95):
    """Initialize transition matrix with strong diagonal persistence."""
    Pi = np.full((K, K), (1.0 - diag) / (K - 1))
    np.fill_diagonal(Pi, diag)
    return Pi


def _pi_regularize(Pi, K):
    """Shrink the M-step transition matrix toward a diagonal prior (Appendix A)."""
    fc = _fc()
    diag = fc["pi_diag_prior"]
    w = fc["pi_prior_weight"]
    prior = np.full((K, K), (1.0 - diag) / (K - 1))
    np.fill_diagonal(prior, diag)
    Pi_reg = (1.0 - w) * Pi + w * prior
    Pi_reg /= Pi_reg.sum(axis=1, keepdims=True)
    return Pi_reg


def _freq_init(k_label, K):
    """Compute empirical regime frequencies from training-window labels (Eq. 4)."""
    freq = np.maximum(np.bincount(k_label, minlength=K).astype(float), _eps())
    return freq / freq.sum()


def _log_emission(y_t, x_t, beta, sigma2, K):
    """Gaussian log-emission log p(y_t | s_t = k) for each component."""
    eps = _eps()
    return np.array([
        stats.norm.logpdf(y_t, x_t @ beta[k], np.sqrt(max(sigma2[k], eps)))
        for k in range(K)
    ])


def _baum_welch(y, X, beta, sigma2, Pi, K):
    """
    E-step of Baum-Welch EM: compute posterior responsibilities gamma_{t,k}
    and expected transition counts xi via the forward-backward algorithm.

    All quantities are maintained in log-domain via logsumexp to prevent
    underflow in long sequences (Appendix A). Rows with a non-finite
    log-normalizer are reset to the uniform distribution over K states.
    """
    eps = _eps()
    T = len(y)
    log_Pi = np.log(np.clip(Pi, eps, 1.0))

    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = np.log(np.full(K, 1.0 / K)) + _log_emission(y[0], X[0], beta, sigma2, K)
    for t in range(1, T):
        log_et = _log_emission(y[t], X[t], beta, sigma2, K)
        for k in range(K):
            log_alpha[t, k] = logsumexp(log_alpha[t - 1] + log_Pi[:, k]) + log_et[k]

    log_bwd = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_et1 = _log_emission(y[t + 1], X[t + 1], beta, sigma2, K)
        for k in range(K):
            log_bwd[t, k] = logsumexp(log_Pi[k, :] + log_et1 + log_bwd[t + 1])

    log_gamma = log_alpha + log_bwd
    log_norm = logsumexp(log_gamma, axis=1, keepdims=True)
    valid = np.isfinite(log_norm[:, 0])
    log_gamma[valid] -= log_norm[valid]
    log_gamma[~valid] = np.log(1.0 / K)
    gamma = np.exp(log_gamma)

    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        log_et1 = _log_emission(y[t + 1], X[t + 1], beta, sigma2, K)
        log_norm_t = logsumexp(log_alpha[t])
        for j in range(K):
            for kk in range(K):
                lx = (log_alpha[t, j] + log_Pi[j, kk] + log_et1[kk] + log_bwd[t + 1, kk] - log_norm_t)
                xi[t, j, kk] = np.exp(lx) if np.isfinite(lx) else 0.0
        rs = xi[t].sum()
        xi[t] = xi[t] / rs if rs > eps else 1.0 / (K * K)

    return gamma, xi.sum(axis=0)


def _forward_gamma(X_te, y_te, beta, sigma2, Pi, K):
    """
    HMM forward filtering at test time: compute the one-step-ahead posterior
    gamma_{h,k} = P(s_h = k | y_0, ..., y_{h-1}), Eq. (3).

    Emission at step t uses y_{t-1}, so gamma_{h,k} is strictly causal
    (does not incorporate y_h). Returns a (T, K) array.
    """
    eps = _eps()
    T = len(X_te)
    log_Pi = np.log(np.clip(Pi, eps, 1.0))

    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = np.log(np.full(K, 1.0 / K))
    for t in range(1, T):
        log_et_prev = _log_emission(y_te[t - 1], X_te[t - 1], beta, sigma2, K)
        for k in range(K):
            log_alpha[t, k] = logsumexp(log_alpha[t - 1] + log_Pi[:, k]) + log_et_prev[k]

    log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
    valid = np.isfinite(log_norm[:, 0])
    log_alpha[valid] -= log_norm[valid]
    log_alpha[~valid] = np.log(1.0 / K)
    return np.exp(log_alpha)


def _sort_by_sigma2(beta, sigma2, Pi):
    """Order HMM states by ascending residual variance when no label anchor is available."""
    o = np.argsort(sigma2)
    return beta[o], sigma2[o], Pi[np.ix_(o, o)]


def _sort_by_label_alignment(beta, sigma2, Pi, gamma, k_label, K):
    """Resolve HMM label-switching by maximizing within-regime posterior mass, Eq. (A.1)."""
    score_mat = np.zeros((K, K))
    for i in range(K):
        mask = k_label == i
        score_mat[i] = gamma[mask].mean(axis=0) if mask.sum() > 0 else 1.0 / K
    best = max(permutations(range(K)),
               key=lambda p: sum(score_mat[i, p[i]] for i in range(K)))
    bp = list(best)
    return beta[bp], sigma2[bp], Pi[np.ix_(bp, bp)]


def _em_loop(y, X, gamma, Pi, K, beta, sigma2):
    """Baum-Welch EM: alternate between the weighted WLS M-step and the E-step."""
    for _ in range(_fc()["em_iter"]):
        for k in range(K):
            beta[k], sigma2[k] = _wls(X, y, gamma[:, k])
        gamma, xi_sum = _baum_welch(y, X, beta, sigma2, Pi, K)
        Pi_new = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + _eps())
        Pi = _pi_regularize(Pi_new, K)
    return gamma, Pi, beta, sigma2



class Persistence:
    """Naive benchmark: forecast = last observed log realized variance."""
    name = "Persistence"

    def fit(self, df, thresholds=None, k_tr=None):
        self.sigma2_ = float(np.var(df["log_rv"].values - df["rv_lag1"].values))

    def predict(self, df, thresholds=None, k_te=None):
        return df["rv_lag1"].values, np.full(len(df), self.sigma2_)

    def forward_gamma(self, X_te, y_te, K):
        return None


class HARRV:
    """HAR-RV (Corsi, 2009): regime-agnostic benchmark, estimated by OLS."""
    name = "HAR-RV"

    def fit(self, df, thresholds=None, k_tr=None):
        X = _har_X(df)
        y = df["log_rv"].values
        self.beta_, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.sigma2_ = float(np.var(y - X @ self.beta_))

    def predict(self, df, thresholds=None, k_te=None):
        return _har_X(df) @ self.beta_, np.full(len(df), self.sigma2_)

    def forward_gamma(self, X_te, y_te, K):
        return None


class MSHAR:
    """
    Markov-Switching HAR with hard regime assignment at prediction time (Section 3).

    Parameters are estimated by Baum-Welch EM with uniform gamma^(0) (no label
    anchor). At test time threshold_regime() assigns each hour to exactly one
    regime before selecting the corresponding (beta_k, sigma2_k).
    """
    name = "MS-HAR"

    def fit(self, df, thresholds=None, k_tr=None):
        K = _fc()["n_regimes"]
        y = df["log_rv"].values
        X = _har_X(df)
        T = len(y)
        var_y = np.var(y)
        gamma = np.full((T, K), 1.0 / K)
        Pi = _pi_init(K)
        beta = np.zeros((K, X.shape[1]))
        sigma2 = np.array([var_y * 0.3, var_y * 1.0, var_y * 3.0])
        gamma, Pi, beta, sigma2 = _em_loop(y, X, gamma, Pi, K, beta, sigma2)
        self.beta_, self.sigma2_, self.Pi_ = _sort_by_sigma2(beta, sigma2, Pi)

    def predict(self, df, thresholds=None, k_te=None):
        X = _har_X(df)
        k_arr = k_te if k_te is not None else threshold_regime(df, thresholds)
        mu = np.array([X[t] @ self.beta_[k_arr[t]] for t in range(len(df))])
        return mu, np.array([self.sigma2_[k_arr[t]] for t in range(len(df))])

    def forward_gamma(self, X_te, y_te, K):
        return None


class RAFT:
    """
    Regime-Adaptive Forecasting with Temporal Filtering (Section 3).

    Training: Baum-Welch EM initialized from empirical regime frequencies (Eq. 4);
    label-switching resolved by score-based permutation search, Eq. (A.1).
    Prediction: HMM forward recursion (Eq. 3) produces soft posterior gamma_{h,k};
    point forecast is the posterior-weighted mixture mean, Eq. (2).
    """
    name = "RAFT"
    _use_lbl = True

    def fit(self, df, thresholds=None, k_tr=None):
        K = _fc()["n_regimes"]
        y = df["log_rv"].values
        X = _har_X(df)
        T = len(y)
        var_y = np.var(y)

        if self._use_lbl and k_tr is not None:
            k_label = k_tr
            init_dist = _freq_init(k_tr, K)
        else:
            k_label = np.zeros(T, dtype=int)
            init_dist = np.full(K, 1.0 / K)

        gamma = np.tile(init_dist, (T, 1))
        Pi = _pi_init(K)
        beta = np.zeros((K, X.shape[1]))
        sigma2 = np.array([var_y * 0.3, var_y * 1.0, var_y * 3.0])
        gamma, Pi, beta, sigma2 = _em_loop(y, X, gamma, Pi, K, beta, sigma2)

        s2_ratio = sigma2.max() / max(sigma2.min(), _eps())
        if s2_ratio < _fc()["collapse_ratio_thresh"]:
            print(f"  [COLLAPSE] {self.name}: sigma2_ratio={s2_ratio:.3f}")

        self.beta_, self.sigma2_, self.Pi_ = _sort_by_label_alignment(beta, sigma2, Pi, gamma, k_label, K)

    def predict(self, df, thresholds=None, k_te=None):
        K = _fc()["n_regimes"]
        X_te = _har_X(df)
        y_te = df["log_rv"].values
        gamma_te = _forward_gamma(X_te, y_te, self.beta_, self.sigma2_, self.Pi_, K)
        beta_hat = np.stack([X_te @ self.beta_[k] for k in range(K)], axis=1)
        mu = (gamma_te * beta_hat).sum(axis=1)
        return mu, gamma_te @ self.sigma2_

    def forward_gamma(self, X_te, y_te, K):
        return _forward_gamma(X_te, y_te, self.beta_, self.sigma2_, self.Pi_, K)


MODEL_ORDER = ["Persistence", "HAR-RV", "MS-HAR", "Hard-HMM", "RAFT"]
# Hard-HMM is produced from the fitted RAFT model, so only four models are fit directly.
MODEL_FIT_CLASSES = [Persistence, HARRV, MSHAR, RAFT]



def _mixture_cdf(q: float, weights, mus, sigmas) -> float:
    return float(np.sum(weights * stats.norm.cdf(q, mus, sigmas)))


def mixture_var(weights: np.ndarray, mus: np.ndarray, sigmas: np.ndarray, level: float) -> float:
    """Compute mixture VaR_p as the root of F(q)=p using Brent's method."""
    lo = float(np.min(mus) - 6 * np.max(sigmas))
    hi = float(np.max(mus) + 6 * np.max(sigmas))
    try:
        return optimize.brentq(
            lambda q: _mixture_cdf(q, weights, mus, sigmas) - level,
            lo, hi, xtol=1e-8, maxiter=200,
        )
    except ValueError:
        return float(np.dot(weights, mus) + np.dot(weights, sigmas) * stats.norm.ppf(level))


def mixture_cvar(weights: np.ndarray, mus: np.ndarray, sigmas: np.ndarray, level: float, var_q: float) -> float:
    """Compute mixture CVaR_p as E[y | y >= VaR_p] by tail integration (Eq. B.2)."""
    tail = 1.0 - level
    if tail < _eps():
        return var_q
    numer, _ = quad(
        lambda q: q * float(np.sum(weights * stats.norm.pdf(q, mus, sigmas))),
        var_q, np.inf, limit=200,
    )
    return numer / tail


def kupiec_pof(violations: int, n: int, level: float):
    """Kupiec (1995) unconditional coverage test."""
    p0 = 1.0 - level
    x = violations
    if x == 0:
        lr = -2.0 * n * np.log(1.0 - p0)
    elif x == n:
        lr = -2.0 * n * np.log(p0)
    else:
        p_hat = x / n
        lr = -2.0 * (x * np.log(p0 / p_hat) + (n - x) * np.log((1 - p0) / (1 - p_hat)))
    return float(lr), float(1.0 - stats.chi2.cdf(lr, df=1))


def _compute_var_cvar_batch(model, X_te, y_te, mu, sigma2, K):
    """Compute per-hour VaR and CVaR at 95% and 99% for one fitted model."""
    eps = _eps()
    gamma_te = None if model is None else model.forward_gamma(X_te, y_te, K)
    is_mixture = gamma_te is not None
    T = len(y_te)

    var95 = np.empty(T)
    var99 = np.empty(T)
    cvar95 = np.empty(T)
    cvar99 = np.empty(T)

    for t in range(T):
        if is_mixture:
            w_t = gamma_te[t]
            m_t = np.array([X_te[t] @ model.beta_[k] for k in range(K)])
            s_t = np.sqrt(np.maximum(model.sigma2_, eps))
        else:
            w_t = np.array([1.0])
            m_t = np.array([mu[t]])
            s_t = np.array([np.sqrt(max(float(sigma2[t]), eps))])

        v95 = mixture_var(w_t, m_t, s_t, 0.95)
        v99 = mixture_var(w_t, m_t, s_t, 0.99)
        var95[t] = v95
        var99[t] = v99
        cvar95[t] = mixture_cvar(w_t, m_t, s_t, 0.95, v95)
        cvar99[t] = mixture_cvar(w_t, m_t, s_t, 0.99, v99)

    return var95, var99, cvar95, cvar99


def _run_one_window(args):
    """
    Run one rolling train/test window and return model-specific VaR/CVaR backtest
    results for that window.
    """
    ws, df_tr_raw, df_te_raw, cfg_snapshot = args
    _CFG.update(cfg_snapshot)
    K = _fc()["n_regimes"]

    med = df_tr_raw["x_corr"].median()
    df_tr = df_tr_raw.copy()
    df_te = df_te_raw.copy()
    df_tr["x_corr"] = df_tr["x_corr"].fillna(med)
    df_te["x_corr"] = df_te["x_corr"].fillna(med)

    thresholds = compute_thresh_quantiles(df_tr)
    k_tr = df_tr["regime_fi"].map(LABEL_MAP).fillna(REGIME_NORMAL).values.astype(int)
    k_te_thresh = threshold_regime(df_te, thresholds)
    regime_te = (df_te["regime_fi"].fillna("Normal").values.astype(str)
                 if "regime_fi" in df_te.columns
                 else np.array([INT_TO_LBL[k] for k in k_te_thresh]))

    y_te = df_te["log_rv"].values
    X_te = _har_X(df_te)
    hb_te = df_te["hb_tz"].values

    out = {}
    for Cls in MODEL_FIT_CLASSES:
        m = Cls()
        try:
            m.fit(df_tr, thresholds=thresholds, k_tr=k_tr)
            mu, sigma2 = m.predict(df_te, thresholds=thresholds, k_te=k_te_thresh)
            var95, var99, cvar95, cvar99 = _compute_var_cvar_batch(m, X_te, y_te, mu, sigma2, K)
            out[m.name] = {
                "hb": hb_te, "y": y_te, "mu": mu, "sigma2": sigma2,
                "var95": var95, "var99": var99, "cvar95": cvar95, "cvar99": cvar99,
                "regime": regime_te, "ws": str(ws.date()),
            }

            if m.name == "RAFT":
                # Hard-HMM uses the fitted RAFT model, but replaces soft posterior weighting
                # with hard state selection via argmax(gamma_te).
                gamma_te = m.forward_gamma(X_te, y_te, K)
                k_hat = np.argmax(gamma_te, axis=1)
                mu_hard = np.array([X_te[t] @ m.beta_[k_hat[t]] for t in range(len(df_te))])
                s2_hard = np.array([m.sigma2_[k_hat[t]] for t in range(len(df_te))])
                var95_h, var99_h, cvar95_h, cvar99_h = _compute_var_cvar_batch(None, X_te, y_te, mu_hard, s2_hard, K)
                out["Hard-HMM"] = {
                    "hb": hb_te, "y": y_te, "mu": mu_hard, "sigma2": s2_hard,
                    "var95": var95_h, "var99": var99_h, "cvar95": cvar95_h, "cvar99": cvar99_h,
                    "regime": regime_te, "ws": str(ws.date()),
                }
        except Exception as e:
            print(f"{m.name} @ {ws.date()}: {e}")
    return out


def rolling_oos_var(df: pd.DataFrame, cfg: dict) -> dict:
    """Run the rolling out-of-sample VaR/CVaR evaluation in parallel."""
    print("Rolling OOS VaR/CVaR evaluation...")
    fc = cfg["forecast"]
    test_end = df["hb_tz"].max()
    ts = pd.Timestamp(fc["test_start"])
    windows = []

    while ts <= test_end:
        we = min(ts + pd.DateOffset(months=fc["roll_months"]) - pd.Timedelta(hours=1), test_end)
        tr_start = ts - pd.DateOffset(months=fc["train_months"])
        df_tr = df[(df["hb_tz"] >= tr_start) & (df["hb_tz"] < ts)].copy()
        df_te = df[(df["hb_tz"] >= ts) & (df["hb_tz"] <= we)].copy()
        if len(df_tr) >= 500 and len(df_te) >= 10:
            windows.append((ts, df_tr, df_te, dict(cfg)))
        ts += pd.DateOffset(months=fc["roll_months"])

    n_workers = fc["n_workers"] or min(len(windows), os.cpu_count() or 4)
    results = {name: [] for name in MODEL_ORDER}
    print(f"  {len(windows)} rolling windows, {n_workers} workers")

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


def compute_var_metrics(results: dict) -> pd.DataFrame:
    """Aggregate violation rates, Kupiec p-values, and MAE-CVaR by regime stratum."""
    print("Computing VaR/CVaR metrics...")
    groups = {
        "All": None,
        "Contagion": "Contagion",
        "Stress": "Single-Asset Stress",
        "Normal": "Normal",
    }
    rows = []
    for model_name in MODEL_ORDER:
        preds = results.get(model_name, [])
        if not preds:
            continue
        for grp, rf in groups.items():
            arrs = {k: [] for k in ("y", "var95", "var99", "cvar95", "cvar99")}
            for p in preds:
                mask = (p["regime"] == rf) if rf else np.ones(len(p["y"]), dtype=bool)
                if mask.sum() == 0:
                    continue
                for k in arrs:
                    arrs[k].append(p[k][mask])
            if not arrs["y"]:
                continue

            y = np.concatenate(arrs["y"])
            var95 = np.concatenate(arrs["var95"])
            var99 = np.concatenate(arrs["var99"])
            cvar95 = np.concatenate(arrs["cvar95"])
            cvar99 = np.concatenate(arrs["cvar99"])
            n = len(y)

            viol95 = int((y > var95).sum())
            viol99 = int((y > var99).sum())
            _, pv95 = kupiec_pof(viol95, n, 0.95)
            _, pv99 = kupiec_pof(viol99, n, 0.99)
            exc95 = y[y > var95] - cvar95[y > var95]
            exc99 = y[y > var99] - cvar99[y > var99]
            mae_cvar95 = float(np.mean(np.abs(exc95))) if len(exc95) > 0 else np.nan
            mae_cvar99 = float(np.mean(np.abs(exc99))) if len(exc99) > 0 else np.nan

            rows.append({
                "Model": model_name,
                "Regime": grp,
                "n": n,
                "ViolRate95": round(viol95 / n * 100, 1),
                "Kupiec_p95": pv95,
                "ViolRate99": round(viol99 / n * 100, 1),
                "Kupiec_p99": pv99,
                "MAE_CVaR95": round(mae_cvar95, 2) if not np.isnan(mae_cvar95) else np.nan,
                "MAE_CVaR99": round(mae_cvar99, 2) if not np.isnan(mae_cvar99) else np.nan,
            })
    return pd.DataFrame(rows)


def _fmt_p(p) -> str:
    if np.isnan(p):
        return "     nan"
    if p < 0.001:
        return "  <0.001"
    return f"{p:8.3f}"


def save_table6(df_metrics: pd.DataFrame, results_dir: Path) -> None:
    print("\n" + "=" * 120)
    print("TABLE 6: VaR/CVaR Backtest — Violation Rates and Kupiec POF Test")
    print("  Nominal: 5.0% at 95%-VaR, 1.0% at 99%-VaR  |  Kupiec p < 0.05 = rejection")
    print("=" * 120)

    for grp in ["All", "Contagion", "Stress", "Normal"]:
        sub = df_metrics[df_metrics["Regime"] == grp].copy()
        if sub.empty:
            continue
        sub["Model"] = pd.Categorical(sub["Model"], categories=MODEL_ORDER, ordered=True)
        sub = sub.sort_values("Model")
        print(f"\n  [{grp} Hours]")
        print(f"  {'Model':<14} "
              f"{'Viol95%':>8} {'p95':>8} {'MAE-CVaR95':>12} "
              f"{'Viol99%':>8} {'p99':>8} {'MAE-CVaR99':>12}")
        for _, row in sub.iterrows():
            print(f"  {row['Model']:<14} "
                  f"{row['ViolRate95']:>7.1f}% {_fmt_p(row['Kupiec_p95'])} "
                  f"{row['MAE_CVaR95']:>12.2f} "
                  f"{row['ViolRate99']:>7.1f}% {_fmt_p(row['Kupiec_p99'])} "
                  f"{row['MAE_CVaR99']:>12.2f}")

    def fmt_pval(p):
        if np.isnan(p):
            return "nan"
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    out_df = df_metrics.copy()
    out_df["Model"] = pd.Categorical(out_df["Model"], categories=MODEL_ORDER, ordered=True)
    out_df["Regime"] = pd.Categorical(out_df["Regime"], categories=["All", "Contagion", "Stress", "Normal"], ordered=True)
    out_df = out_df.sort_values(["Regime", "Model"])
    out_df["Kupiec_p95"] = out_df["Kupiec_p95"].apply(fmt_pval)
    out_df["Kupiec_p99"] = out_df["Kupiec_p99"].apply(fmt_pval)
    out_df = out_df[["Model", "Regime", "ViolRate95", "Kupiec_p95", "MAE_CVaR95",
                     "ViolRate99", "Kupiec_p99", "MAE_CVaR99"]]
    out = results_dir / "table6_var_backtest.csv"
    out_df.to_csv(out, index=False)
    print(f"\nSaved {out}")


def parse_args():
    p = argparse.ArgumentParser(description="RAFT VaR/CVaR backtest")
    p.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    return p.parse_args()


def main():
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(var, "1")
    args = parse_args()
    cfg = load_config(args.config)
    results_dir = cfg["_results_dir"]

    df = load_data(cfg)
    results = rolling_oos_var(df, cfg)
    df_metrics = compute_var_metrics(results)
    df_metrics.to_csv(results_dir / "table6_var_backtest.csv", index=False)
    save_table6(df_metrics, results_dir)
    print(f"\nDone. Outputs in: {results_dir}")


if __name__ == "__main__":
    main()