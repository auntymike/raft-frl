"""
S3_alt_partitions.py  —  Alternative adverse-state evaluation
Evaluates all five models in three adverse-state subsets defined without
reference to the paper's Contagion taxonomy and without contemporaneous
cross-asset correlation. Thresholds are derived from each rolling training
window, and the subsets are evaluated ex post using the same stratification
design as the main manuscript. Produces table_s3_alt_partitions.csv.
Run independently after the main preprocessing files have been generated.
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
from scipy import stats, optimize
from scipy.special import logsumexp
warnings.filterwarnings('ignore')

SHORTS    = ['BTC', 'ETH', 'XRP', 'LTC', 'LINK']
HVOL_COLS = [f'{s}_hvol' for s in SHORTS]
HRET_COLS = [f'{s}_hret' for s in SHORTS]
EM_COLS   = [f'{s}_extreme_move' for s in SHORTS]
LABEL_MAP = {'Normal': 0, 'Single-Asset Stress': 1, 'Contagion': 2}
REGIME_NORMAL    = 0
REGIME_STRESS    = 1
REGIME_CONTAGION = 2
_CFG = {}
BASELINE_SCHEME = {'rv_max': 95.0, 'corr': 90.0, 'asset': 80.0}
MODEL_ORDER = ['Persistence', 'HAR-RV', 'MS-HAR', 'Hard-HMM', 'RAFT']
PARTITIONS  = {
    'High realized volatility':      'high_realized_rv',
    'Large negative market return':  'large_negative_return',
    'Large absolute market return':  'large_abs_return',
}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = (Path(path).parent / cfg['paths']['base']).resolve() \
           if cfg['paths']['base'] == '.' \
           else Path(cfg['paths']['base']).expanduser().resolve()
    cfg['_results_dir'] = base / cfg['paths']['results_dir']
    cfg['_results_dir'].mkdir(parents=True, exist_ok=True)
    _CFG.update(cfg)
    return cfg

def _fc() -> dict:
    return _CFG['forecast']

def _eps() -> float:
    return _fc()['eps']


def load_data(cfg: dict) -> pd.DataFrame:
    rd      = cfg['_results_dir']
    hourly  = pd.read_parquet(rd / 'hourly_stats.parquet')
    regimes = pd.read_parquet(rd / 'regimes_hourly.parquet')
    for d in (hourly, regimes):
        d['hour_block'] = pd.to_datetime(d['hour_block'], utc=True)
    df = (hourly
          .merge(regimes[['hour_block', 'regime']].rename(columns={'regime': 'regime_fi'}),
                 on='hour_block', how='left')
          .sort_values('hour_block')
          .reset_index(drop=True))
    eps = _eps()
    df['log_rv']    = np.log(df[HVOL_COLS].pow(2).clip(lower=eps)).mean(axis=1)
    y = df['log_rv']
    df['rv_lag1']   = y.shift(1)
    df['rv_lag24']  = y.shift(1).rolling(24,  min_periods=24).mean()
    df['rv_lag168'] = y.shift(1).rolling(168, min_periods=168).mean()
    hvol_mat = df[HVOL_COLS].values
    df['cs_hvol_std']    = np.nanstd(hvol_mat, axis=1)
    df['cs_hvol_max']    = np.nanmax(hvol_mat, axis=1)
    df['cs_hvol_std_l1'] = df['cs_hvol_std'].shift(1)
    df['cs_hvol_max_l1'] = df['cs_hvol_max'].shift(1)
    em_avail = [c for c in EM_COLS if c in df.columns]
    if em_avail:
        df['extreme_move_mean'] = df[em_avail].mean(axis=1)
    else:
        roll_std = df[HVOL_COLS].rolling(720, min_periods=50).std()
        roll_mu  = df[HVOL_COLS].rolling(720, min_periods=50).mean()
        df['extreme_move_mean'] = (df[HVOL_COLS] > roll_mu + 2 * roll_std).astype(float).mean(axis=1)
    df['extreme_move_l1'] = df['extreme_move_mean'].shift(1)
    for s in SHORTS:
        df[f'{s}_hvol_l1'] = df[f'{s}_hvol'].shift(1)
    df['x_corr'] = df['cross_corr'].shift(1)
    if 'port_hret' not in df.columns:
        if all(c in df.columns for c in HRET_COLS):
            df['port_hret'] = df[HRET_COLS].mean(axis=1)
        else:
            raise ValueError(
                'Need port_hret or per-asset hret columns '
                '(BTC_hret, ETH_hret, ...) for return-based adverse partitions.'
            )
    df = df.dropna(subset=['rv_lag1', 'rv_lag24', 'rv_lag168',
                            'cs_hvol_std_l1', 'cs_hvol_max_l1',
                            'extreme_move_l1', 'port_hret']).reset_index(drop=True)
    df['hb_tz'] = df['hour_block'].dt.tz_convert(None)
    return df


def build_scheme_labels(df: pd.DataFrame, scheme: dict, calib_h: int) -> pd.DataFrame:
    out = df.copy().sort_values('hour_block').reset_index(drop=True)
    p_rv          = out[HVOL_COLS].apply(lambda c: c.rolling(calib_h, min_periods=720).quantile(scheme['rv_max'] / 100.0))
    above_rv      = (out[HVOL_COLS].values > p_rv.values).sum(axis=1)
    p_corr        = out['cross_corr'].rolling(calib_h, min_periods=720).quantile(scheme['corr'] / 100.0)
    above_corr    = out['cross_corr'].values > p_corr.values
    p_asset       = out[HVOL_COLS].apply(lambda c: c.rolling(calib_h, min_periods=720).quantile(scheme['asset'] / 100.0))
    above_asset   = (out[HVOL_COLS].values > p_asset.values).sum(axis=1)
    p_corr80      = out['cross_corr'].rolling(calib_h, min_periods=720).quantile(0.80)
    p_asset80     = out[HVOL_COLS].apply(lambda c: c.rolling(calib_h, min_periods=720).quantile(0.80))
    above_asset80 = (out[HVOL_COLS].values > p_asset80.values).sum(axis=1)
    regimes   = np.full(len(out), 'Normal', dtype=object)
    contagion = (above_rv >= 1) & above_corr & (above_asset >= 3) & np.isfinite(p_corr.values)
    stress1   = ~contagion & (above_asset80 >= 1) & (out['cross_corr'].values > p_corr80.values) & np.isfinite(p_corr80.values)
    stress2   = ~contagion & ~stress1 & (above_rv >= 1)
    regimes[stress1 | stress2] = 'Single-Asset Stress'
    regimes[contagion]         = 'Contagion'
    out['scheme_regime'] = regimes
    return out


def compute_thresh_quantiles(df_tr: pd.DataFrame, scheme: dict) -> dict:
    return {
        'tw_q_rv_max': np.nanpercentile(df_tr['cs_hvol_max_l1'].values, scheme['rv_max']),
        'tw_q_cs_std': np.nanpercentile(df_tr['cs_hvol_std_l1'].values, scheme['corr']),
        'tw_q80_em':   np.nanpercentile(df_tr['extreme_move_l1'].values, 80.0),
        'per_asset_q': {s: np.nanpercentile(df_tr[f'{s}_hvol_l1'].values, scheme['asset'])
                        for s in SHORTS if f'{s}_hvol_l1' in df_tr.columns},
    }


def threshold_regime(df: pd.DataFrame, thr: dict) -> np.ndarray:
    per_q   = thr['per_asset_q']
    n_above = sum((df[f'{s}_hvol_l1'].values > per_q[s]).astype(int)
                  for s in SHORTS if f'{s}_hvol_l1' in df.columns and s in per_q)
    contagion_mask = ((df['cs_hvol_max_l1'].values > thr['tw_q_rv_max']) &
                      (df['cs_hvol_std_l1'].values  > thr['tw_q_cs_std']) &
                      (n_above >= 3))
    stress_mask = (~contagion_mask) & (df['extreme_move_l1'].values > thr['tw_q80_em'])
    out = np.zeros(len(df), dtype=int)
    out[stress_mask]    = REGIME_STRESS
    out[contagion_mask] = REGIME_CONTAGION
    return out


def _har_X(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)),
                            df['rv_lag1'].values,
                            df['rv_lag24'].values,
                            df['rv_lag168'].values])

def _safe_lstsq(X, y):
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if len(y) == 0:
        raise ValueError('No finite rows')
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta
    except np.linalg.LinAlgError:
        return np.linalg.pinv(X) @ y

def _wls(X, y, w):
    eps  = _eps()
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(w)
    X, y, w = X[mask], y[mask], w[mask]
    if len(y) == 0:
        raise ValueError('No finite rows for WLS')
    sw = w.sum()
    if sw < eps:
        return _safe_lstsq(X, y), np.var(y) + eps
    Xw   = X * w[:, None]
    XtWX = Xw.T @ X + eps * np.eye(X.shape[1])
    XtWy = Xw.T @ y
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtWX) @ XtWy
    resid = y - X @ beta
    return beta, max(float(np.dot(w, resid ** 2) / sw), eps)

def _pi_init(K, diag=0.95):
    Pi = np.full((K, K), (1.0 - diag) / (K - 1))
    np.fill_diagonal(Pi, diag)
    return Pi

def _pi_regularize(Pi, K):
    fc   = _fc()
    diag, w = fc['pi_diag_prior'], fc['pi_prior_weight']
    prior = np.full((K, K), (1.0 - diag) / (K - 1))
    np.fill_diagonal(prior, diag)
    Pi_reg  = (1.0 - w) * Pi + w * prior
    Pi_reg /= Pi_reg.sum(axis=1, keepdims=True)
    return Pi_reg

def _freq_init(k_label, K):
    freq = np.maximum(np.bincount(k_label, minlength=K).astype(float), _eps())
    return freq / freq.sum()

def _log_emission(y_t, x_t, beta, sigma2, K):
    eps = _eps()
    out = np.empty(K)
    for k in range(K):
        out[k] = stats.norm.logpdf(y_t, x_t @ beta[k], np.sqrt(max(sigma2[k], eps)))
    return out

def _baum_welch(y, X, beta, sigma2, Pi, K):
    eps = _eps()
    T   = len(y)
    log_Pi    = np.log(np.clip(Pi, eps, 1.0))
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = np.log(np.full(K, 1.0 / K)) + _log_emission(y[0], X[0], beta, sigma2, K)
    for t in range(1, T):
        log_et = _log_emission(y[t], X[t], beta, sigma2, K)
        for k in range(K):
            log_alpha[t, k] = logsumexp(log_alpha[t-1] + log_Pi[:, k]) + log_et[k]
    log_bwd = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_et1 = _log_emission(y[t+1], X[t+1], beta, sigma2, K)
        for k in range(K):
            log_bwd[t, k] = logsumexp(log_Pi[k, :] + log_et1 + log_bwd[t+1])
    log_gamma = log_alpha + log_bwd
    log_norm  = logsumexp(log_gamma, axis=1, keepdims=True)
    valid = np.isfinite(log_norm[:, 0])
    log_gamma[valid]  -= log_norm[valid]
    log_gamma[~valid]  = np.log(1.0 / K)
    gamma = np.exp(log_gamma)
    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        log_et1    = _log_emission(y[t+1], X[t+1], beta, sigma2, K)
        log_norm_t = logsumexp(log_alpha[t])
        for j in range(K):
            for kk in range(K):
                lx = (log_alpha[t, j] + log_Pi[j, kk] +
                      log_et1[kk] + log_bwd[t+1, kk] - log_norm_t)
                xi[t, j, kk] = np.exp(lx) if np.isfinite(lx) else 0.0
        rs     = xi[t].sum()
        xi[t]  = xi[t] / rs if rs > eps else 1.0 / (K * K)
    return gamma, xi.sum(axis=0)

def _forward_gamma(X_te, y_te, beta, sigma2, Pi, K):
    eps = _eps()
    T   = len(X_te)
    log_Pi    = np.log(np.clip(Pi, eps, 1.0))
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = np.log(np.full(K, 1.0 / K))
    for t in range(1, T):
        log_et_prev = _log_emission(y_te[t-1], X_te[t-1], beta, sigma2, K)
        for k in range(K):
            log_alpha[t, k] = logsumexp(log_alpha[t-1] + log_Pi[:, k]) + log_et_prev[k]
    log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
    valid = np.isfinite(log_norm[:, 0])
    log_alpha[valid]  -= log_norm[valid]
    log_alpha[~valid]  = np.log(1.0 / K)
    return np.exp(log_alpha)

def _sort_by_sigma2(beta, sigma2, Pi):
    o = np.argsort(sigma2)
    return beta[o], sigma2[o], Pi[np.ix_(o, o)]

def _sort_by_label_alignment(beta, sigma2, Pi, gamma, k_label, K):
    score_mat = np.zeros((K, K))
    for i in range(K):
        mask = k_label == i
        score_mat[i] = gamma[mask].mean(axis=0) if mask.sum() > 0 else 1.0 / K
    best_perm = max(permutations(range(K)),
                    key=lambda p: sum(score_mat[i, p[i]] for i in range(K)))
    bp = list(best_perm)
    return beta[bp], sigma2[bp], Pi[np.ix_(bp, bp)]

def _em_loop(y, X, gamma, Pi, K, beta, sigma2):
    eps = _eps()
    for _ in range(_fc()['em_iter']):
        for k in range(K):
            beta[k], sigma2[k] = _wls(X, y, gamma[:, k])
        gamma, xi_sum = _baum_welch(y, X, beta, sigma2, Pi, K)
        Pi_new = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + eps)
        Pi     = _pi_regularize(Pi_new, K)
    return gamma, Pi, beta, sigma2

def _predict_hard_hmm(df_te, beta, sigma2, Pi):
    K        = _fc()['n_regimes']
    X_te     = _har_X(df_te)
    y_te     = df_te['log_rv'].values
    gamma_te = _forward_gamma(X_te, y_te, beta, sigma2, Pi, K)
    k_hat    = np.argmax(gamma_te, axis=1)
    mu = np.array([X_te[t] @ beta[k_hat[t]] for t in range(len(df_te))])
    s2 = np.array([sigma2[k_hat[t]] for t in range(len(df_te))])
    return mu, s2

def qlike(mu, y):
    h     = np.exp(np.clip(mu, -30, 30))
    act   = np.exp(np.clip(y,  -30, 30))
    ratio = act / np.maximum(h, _eps())
    return float(np.mean(ratio - np.log(np.maximum(ratio, _eps())) - 1))

def log_score(y, mu, sigma2):
    s2 = np.maximum(sigma2, _eps())
    return float(np.mean(-0.5 * np.log(2 * np.pi * s2) - 0.5 * (y - mu) ** 2 / s2))

def _mixture_var_row(weights, means, sigma2, level=0.99):
    """Numerically invert a Gaussian-mixture CDF for one forecast origin."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(weights.sum(), _eps())
    means   = np.asarray(means, dtype=float)
    sigma   = np.sqrt(np.maximum(np.asarray(sigma2, dtype=float), _eps()))
    lo = float(np.min(means - 8.0 * sigma))
    hi = float(np.max(means + 8.0 * sigma))
    def cdf_minus(q):
        return float(np.sum(weights * stats.norm.cdf((q - means) / sigma)) - level)
    try:
        return float(optimize.brentq(cdf_minus, lo, hi, maxiter=100))
    except Exception:
        return float(np.sum(weights * means) + np.sum(weights * sigma) * stats.norm.ppf(level))

def _var99_viol_percent(selected: dict) -> float:
    """Return 99%-VaR violation percentage for Gaussian or RAFT mixture forecasts."""
    y = selected['y']
    if 'gamma' in selected:
        var99 = np.array([
            _mixture_var_row(selected['gamma'][i],
                             selected['component_mu'][i],
                             selected['component_sigma2'][i],
                             level=0.99)
            for i in range(len(y))
        ])
    else:
        var99 = (selected['mu'] +
                 np.sqrt(np.maximum(selected['sigma2'], _eps())) * stats.norm.ppf(0.99))
    return float(100.0 * np.mean(y > var99))


def _run_one_window(args):
    ws, df_tr_raw, df_te_raw, scheme, cfg_snapshot = args
    _CFG.update(cfg_snapshot)
    K   = _fc()['n_regimes']
    med = df_tr_raw['x_corr'].median()
    df_tr = df_tr_raw.copy()
    df_te = df_te_raw.copy()
    df_tr['x_corr'] = df_tr['x_corr'].fillna(med)
    df_te['x_corr'] = df_te['x_corr'].fillna(med)

    thresholds      = compute_thresh_quantiles(df_tr, scheme)
    k_tr            = df_tr['scheme_regime'].map(LABEL_MAP).fillna(REGIME_NORMAL).values.astype(int)
    k_te_thresh     = threshold_regime(df_te, thresholds)

    rv95_train     = np.nanpercentile(df_tr['log_rv'].values, 95.0)
    negret5_train  = np.nanpercentile(df_tr['port_hret'].values, 5.0)
    absret95_train = np.nanpercentile(np.abs(df_tr['port_hret'].values), 95.0)

    high_rv_mask   = df_te['log_rv'].values    > rv95_train
    neg_ret_mask   = df_te['port_hret'].values < negret5_train
    abs_ret_mask   = np.abs(df_te['port_hret'].values) > absret95_train

    out      = {name: None for name in MODEL_ORDER}
    raft_fit = None

    def _store(name, mu, s2, extra=None):
        entry = {
            'mu': mu, 'sigma2': s2,
            'y':  df_te['log_rv'].values,
            'high_realized_rv':     high_rv_mask,
            'large_negative_return': neg_ret_mask,
            'large_abs_return':      abs_ret_mask,
        }
        if extra:
            entry.update(extra)
        out[name] = entry

    try:
        resid_pers = (df_tr['log_rv'] - df_tr['rv_lag1']).dropna()
        s2_pers    = float(np.var(resid_pers)) or _eps()
        _store('Persistence', df_te['rv_lag1'].values, np.full(len(df_te), s2_pers))
    except Exception as e:
        print(f'  Persistence @ {ws}: {e}')

    try:
        X = _har_X(df_tr); y = df_tr['log_rv'].values
        beta_har = _safe_lstsq(X, y)
        resid    = y - X @ beta_har
        s2_har   = float(np.var(resid[np.isfinite(resid)])) or _eps()
        _store('HAR-RV', _har_X(df_te) @ beta_har, np.full(len(df_te), s2_har))
    except Exception as e:
        print(f'  HAR-RV @ {ws}: {e}')

    try:
        X = _har_X(df_tr); y = df_tr['log_rv'].values
        var_y  = np.var(y[np.isfinite(y)])
        gamma  = np.full((len(y), K), 1.0 / K)
        Pi     = _pi_init(K)
        beta   = np.zeros((K, X.shape[1]))
        sigma2 = np.array([var_y * 0.3, var_y * 1.0, var_y * 3.0])
        gamma, Pi, beta, sigma2 = _em_loop(y, X, gamma, Pi, K, beta, sigma2)
        beta, sigma2, Pi = _sort_by_sigma2(beta, sigma2, Pi)
        X_te = _har_X(df_te)
        _store('MS-HAR',
               np.array([X_te[t] @ beta[k_te_thresh[t]] for t in range(len(df_te))]),
               np.array([sigma2[k_te_thresh[t]]          for t in range(len(df_te))]))
    except Exception as e:
        print(f'  MS-HAR @ {ws}: {e}')

    try:
        X = _har_X(df_tr); y = df_tr['log_rv'].values
        var_y     = np.var(y[np.isfinite(y)])
        init_dist = _freq_init(k_tr, K)
        gamma     = np.tile(init_dist, (len(y), 1))
        Pi        = _pi_init(K)
        beta      = np.zeros((K, X.shape[1]))
        sigma2    = np.array([var_y * 0.3, var_y * 1.0, var_y * 3.0])
        gamma, Pi, beta, sigma2 = _em_loop(y, X, gamma, Pi, K, beta, sigma2)
        beta, sigma2, Pi = _sort_by_label_alignment(beta, sigma2, Pi, gamma, k_tr, K)
        X_te       = _har_X(df_te)
        y_te       = df_te['log_rv'].values
        gamma_te   = _forward_gamma(X_te, y_te, beta, sigma2, Pi, K)
        beta_preds = np.stack([X_te @ beta[k] for k in range(K)], axis=1)
        _store('RAFT',
               (gamma_te * beta_preds).sum(axis=1),
               gamma_te @ sigma2,
               extra={
                   'gamma':            gamma_te,
                   'component_mu':     beta_preds,
                   'component_sigma2': np.tile(sigma2, (len(df_te), 1)),
               })
        raft_fit = {'beta': beta.copy(), 'sigma2': sigma2.copy(), 'Pi': Pi.copy()}
    except Exception as e:
        print(f'  RAFT @ {ws}: {e}')

    if raft_fit is not None:
        try:
            mu_hard, s2_hard = _predict_hard_hmm(df_te, raft_fit['beta'],
                                                  raft_fit['sigma2'], raft_fit['Pi'])
            _store('Hard-HMM', mu_hard, s2_hard)
        except Exception as e:
            print(f'  Hard-HMM @ {ws}: {e}')

    return out


def _collect_results(df_labeled: pd.DataFrame, cfg: dict) -> dict:
    fc       = cfg['forecast']
    test_end = df_labeled['hb_tz'].max()
    ts       = pd.Timestamp(fc['test_start'])
    windows  = []
    while ts <= test_end:
        we       = min(ts + pd.DateOffset(months=fc['roll_months']) - pd.Timedelta(hours=1), test_end)
        tr_start = ts - pd.DateOffset(months=fc['train_months'])
        df_tr    = df_labeled[(df_labeled['hb_tz'] >= tr_start) & (df_labeled['hb_tz'] < ts)].copy()
        df_te    = df_labeled[(df_labeled['hb_tz'] >= ts) & (df_labeled['hb_tz'] <= we)].copy()
        if len(df_tr) >= 500 and len(df_te) >= 10:
            windows.append((ts, df_tr, df_te))
        ts += pd.DateOffset(months=fc['roll_months'])
    results   = {name: [] for name in MODEL_ORDER}
    n_workers = cfg['forecast']['n_workers'] or min(len(windows), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_run_one_window, (ws, df_tr, df_te, BASELINE_SCHEME, dict(cfg))): ws
                for ws, df_tr, df_te in windows}
        for fut in as_completed(futs):
            ws = futs[fut]
            try:
                for name, entry in fut.result().items():
                    if entry is not None:
                        results[name].append(entry)
            except Exception as e:
                print(f'window {ws.date()} failed: {e}')
    return results


def _select(preds: list, mask_key: str) -> dict | None:
    mus, ys, s2s = [], [], []
    gammas, comp_mus, comp_s2s = [], [], []
    has_mixture = False
    for p in preds:
        if mask_key not in p:
            continue
        mask = np.asarray(p[mask_key], dtype=bool)
        if mask.sum() == 0:
            continue
        mus.append(p['mu'][mask])
        ys.append(p['y'][mask])
        s2s.append(p['sigma2'][mask])
        if 'gamma' in p and 'component_mu' in p and 'component_sigma2' in p:
            has_mixture = True
            gammas.append(p['gamma'][mask])
            comp_mus.append(p['component_mu'][mask])
            comp_s2s.append(p['component_sigma2'][mask])
    if not mus:
        return None
    out = {'mu': np.concatenate(mus), 'y': np.concatenate(ys), 'sigma2': np.concatenate(s2s)}
    if has_mixture and gammas:
        out['gamma']            = np.vstack(gammas)
        out['component_mu']     = np.vstack(comp_mus)
        out['component_sigma2'] = np.vstack(comp_s2s)
    return out


def build_table(results: dict) -> pd.DataFrame:
    rows = []
    for part_name, mask_key in PARTITIONS.items():
        for model_name in MODEL_ORDER:
            sel = _select(results.get(model_name, []), mask_key)
            if sel is None or len(sel['y']) < 5:
                rows.append({'Partition': part_name, 'Model': model_name,
                             'n': 0, 'QLIKE': np.nan,
                             'Log-S': np.nan, '99%-VaR Viol.%': np.nan})
                continue
            rows.append({
                'Partition':      part_name,
                'Model':          model_name,
                'n':              int(len(sel['y'])),
                'QLIKE':          round(qlike(sel['mu'], sel['y']), 3),
                'Log-S':          round(log_score(sel['y'], sel['mu'], sel['sigma2']), 3),
                '99%-VaR Viol.%': round(_var99_viol_percent(sel), 1),
            })
    return pd.DataFrame(rows)


def print_table(table: pd.DataFrame) -> None:
    sep = '=' * 82
    hdr = f"{'Model':<12} {'n':>7} {'QLIKE':>8} {'Log-S':>8} {'99%-VaR Viol.%':>16}"
    print(f'\n{sep}')
    print('TABLE S3: Alternative Adverse-State Partitions')
    print(sep)
    for part_name in PARTITIONS:
        sub = table[table['Partition'] == part_name]
        if sub.empty:
            continue
        print(f'\nPartition: {part_name}  (n={int(sub["n"].iloc[0])})')
        print('-' * 82)
        print(hdr)
        for _, row in sub.iterrows():
            print(f"{row['Model']:<12} {int(row['n']):>7d} "
                  f"{row['QLIKE']:>8.3f} {row['Log-S']:>8.3f} "
                  f"{row['99%-VaR Viol.%']:>16.1f}")


def parse_args():
    p = argparse.ArgumentParser(description='Table S3: alternative adverse-state partitions')
    p.add_argument('--config', default=str(Path(__file__).parent / 'config.yaml'))
    return p.parse_args()


def main():
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ.setdefault(var, '1')
    args = parse_args()
    cfg  = load_config(args.config)
    df   = load_data(cfg)

    calib_h    = cfg['analysis']['regime_calib_hours']
    df_labeled = build_scheme_labels(df, BASELINE_SCHEME, calib_h)

    print('Running baseline-scheme windows...')
    results = _collect_results(df_labeled, cfg)

    table = build_table(results)
    out   = cfg['_results_dir'] / 'table_s3_alt_partitions.csv'
    table.to_csv(out, index=False)
    print(f'Saved {out}')
    print_table(table)
    print(f'\nDone. Outputs in: {cfg["_results_dir"]}')


if __name__ == '__main__':
    main()