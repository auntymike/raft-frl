"""
S1a_benchmark_forecasts.py  —  Supplementary benchmark forecasts
Replicates the supplementary out-of-sample comparison for HAR-t, HAR-CJ,
and Realized GARCH(1,1) using the same rolling train/test design as the
main manuscript. Produces forecast-level outputs for QLIKE and log-score
evaluation over the 2022–2025 out-of-sample period.
Run independently after the main preprocessing files have been generated.
"""
import os
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import numpy as np
import pandas as pd
from scipy import stats, optimize
warnings.filterwarnings('ignore')
SHORTS = ['BTC', 'ETH', 'XRP', 'LTC', 'LINK']
HVOL_COLS = [f'{s}_hvol' for s in SHORTS]
EM_COLS = [f'{s}_extreme_move' for s in SHORTS]
LABEL_MAP = {'Normal': 0, 'Single-Asset Stress': 1, 'Contagion': 2}
INT_TO_LBL = {v: k for k, v in LABEL_MAP.items()}
REGIME_NORMAL = 0
REGIME_STRESS = 1
REGIME_CONTAGION = 2
_CFG = {}

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = (Path(path).parent / cfg['paths']['base']).resolve() if cfg['paths']['base'] == '.' else Path(cfg['paths']['base']).expanduser().resolve()
    cfg['_results_dir'] = base / cfg['paths']['results_dir']
    cfg['_results_dir'].mkdir(parents=True, exist_ok=True)
    _CFG.update(cfg)
    return cfg

def _fc() -> dict:
    return _CFG['forecast']

def _eps() -> float:
    return _fc()['eps']

def load_data(cfg: dict) -> pd.DataFrame:
    results_dir = cfg['_results_dir']
    hourly = pd.read_parquet(results_dir / 'hourly_stats.parquet')
    regimes = pd.read_parquet(results_dir / 'regimes_hourly.parquet')
    for df_ in (hourly, regimes):
        df_['hour_block'] = pd.to_datetime(df_['hour_block'], utc=True)
    df = hourly.merge(regimes[['hour_block', 'regime']].rename(columns={'regime': 'regime_fi'}), on='hour_block', how='left').sort_values('hour_block').reset_index(drop=True)
    eps = _eps()
    df['log_rv'] = np.log(df[HVOL_COLS].pow(2).clip(lower=eps)).mean(axis=1)
    y = df['log_rv']
    df['rv_lag1'] = y.shift(1)
    df['rv_lag24'] = y.shift(1).rolling(24, min_periods=24).mean()
    df['rv_lag168'] = y.shift(1).rolling(168, min_periods=168).mean()
    candidate_jump_cols = ['jump_log_rv', 'log_rv_jump', 'jump_proxy', 'cj_jump', 'jump_component']
    candidate_cont_cols = ['cont_log_rv', 'log_rv_cont', 'cont_proxy', 'cj_cont', 'continuous_component']
    jump_col = next((c for c in candidate_jump_cols if c in df.columns), None)
    cont_col = next((c for c in candidate_cont_cols if c in df.columns), None)
    if jump_col is not None and cont_col is not None:
        df['cj_jump_raw'] = df[jump_col].astype(float)
        df['cj_cont_raw'] = df[cont_col].astype(float)
    else:
        pi_half = np.pi / 2.0
        rv_mat       = df[HVOL_COLS].pow(2).clip(lower=eps).values
        hvol_mat_raw = df[HVOL_COLS].clip(lower=0.0).values
        bv_mat       = pi_half * hvol_mat_raw * np.roll(hvol_mat_raw, 1, axis=0)
        bv_mat[0]    = rv_mat[0]
        jump_mat     = np.maximum(rv_mat - bv_mat, 0.0)
        cont_mat     = rv_mat - jump_mat
        df['cj_jump_raw'] = np.log(np.maximum(jump_mat.mean(axis=1), eps))
        df['cj_cont_raw'] = np.log(np.maximum(cont_mat.mean(axis=1), eps))
    df['cj_cont_lag1'] = df['cj_cont_raw'].shift(1)
    df['cj_jump_lag1'] = df['cj_jump_raw'].shift(1)
    df['cj_cont_lag24'] = df['cj_cont_raw'].shift(1).rolling(24, min_periods=24).mean()
    df['cj_jump_lag24'] = df['cj_jump_raw'].shift(1).rolling(24, min_periods=24).mean()
    df['cj_cont_lag168'] = df['cj_cont_raw'].shift(1).rolling(168, min_periods=168).mean()
    df['cj_jump_lag168'] = df['cj_jump_raw'].shift(1).rolling(168, min_periods=168).mean()
    hvol_mat = df[HVOL_COLS].values
    df['cs_hvol_std'] = np.nanstd(hvol_mat, axis=1)
    df['cs_hvol_max'] = np.nanmax(hvol_mat, axis=1)
    df['cs_hvol_std_l1'] = df['cs_hvol_std'].shift(1)
    df['cs_hvol_max_l1'] = df['cs_hvol_max'].shift(1)
    em_avail = [c for c in EM_COLS if c in df.columns]
    if em_avail:
        df['extreme_move_mean'] = df[em_avail].mean(axis=1)
    else:
        roll_std = df[HVOL_COLS].rolling(720, min_periods=50).std()
        roll_mu = df[HVOL_COLS].rolling(720, min_periods=50).mean()
        df['extreme_move_mean'] = (df[HVOL_COLS] > roll_mu + 2 * roll_std).astype(float).mean(axis=1)
    df['extreme_move_l1'] = df['extreme_move_mean'].shift(1)
    df = df.dropna(subset=['rv_lag1', 'rv_lag24', 'rv_lag168', 'cj_cont_lag1', 'cj_jump_lag1',
                           'cj_cont_lag24', 'cj_jump_lag24', 'cj_cont_lag168', 'cj_jump_lag168']).reset_index(drop=True)
    df['hb_tz'] = df['hour_block'].dt.tz_convert(None)
    return df

def qlike_vec(mu: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = np.exp(np.clip(mu, -30, 30))
    act = np.exp(np.clip(y, -30, 30))
    ratio = act / np.maximum(h, _eps())
    return ratio - np.log(np.maximum(ratio, _eps())) - 1

def qlike(mu: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(qlike_vec(mu, y)))

def _gaussian_logpdf(y: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    s2 = np.maximum(sigma2, _eps())
    return -0.5 * np.log(2 * np.pi * s2) - 0.5 * (y - mu) ** 2 / s2

def _har_X(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df['rv_lag1'].values, df['rv_lag24'].values, df['rv_lag168'].values])

def _harcj_X(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df['cj_cont_lag1'].values, df['cj_jump_lag1'].values,
                            df['cj_cont_lag24'].values, df['cj_jump_lag24'].values,
                            df['cj_cont_lag168'].values, df['cj_jump_lag168'].values])

def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    eps = _eps()
    sw = w.sum()
    if sw < eps:
        return (np.linalg.lstsq(X, y, rcond=None)[0], np.var(y) + eps)
    Xw = X * w[:, None]
    XtWX = Xw.T @ X + eps * np.eye(X.shape[1])
    XtWy = Xw.T @ y
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return (beta, max(float(np.dot(w, resid ** 2) / sw), eps))

class HART:
    name = 'HAR-t'

    def fit(self, df, k_tr=None):
        X = _har_X(df)
        y = df['log_rv'].values
        self.beta_, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ self.beta_
        var0 = max(float(np.var(resid)), _eps())
        def nll(theta):
            log_s2, log_dfm2 = theta
            s2 = np.exp(log_s2)
            nu = 2.0 + np.exp(log_dfm2)
            scale = np.sqrt(s2 * max((nu - 2.0) / nu, _eps()))
            ll = stats.t.logpdf(resid / scale, df=nu) - np.log(scale)
            return -float(np.sum(ll))
        x0 = np.array([np.log(var0), np.log(8.0)])
        opt = optimize.minimize(nll, x0, method='L-BFGS-B')
        if opt.success:
            self.sigma2_ = float(np.exp(opt.x[0]))
            self.nu_ = float(2.0 + np.exp(opt.x[1]))
        else:
            self.sigma2_ = var0
            self.nu_ = 10.0

    def predict(self, df):
        X = _har_X(df)
        mu = X @ self.beta_
        scale = np.sqrt(self.sigma2_ * max((self.nu_ - 2.0) / self.nu_, _eps()))
        logpdf = stats.t.logpdf((df['log_rv'].values - mu) / scale, df=self.nu_) - np.log(scale)
        return {'mu': mu, 'sigma2': np.full(len(df), self.sigma2_), 'logpdf': logpdf}

class HARCJ:
    name = 'HAR-CJ'

    def fit(self, df, k_tr=None):
        X = _harcj_X(df)
        y = df['log_rv'].values
        self.beta_, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ self.beta_
        self.sigma2_ = max(float(np.var(resid)), _eps())

    def predict(self, df):
        X = _harcj_X(df)
        mu = X @ self.beta_
        logpdf = _gaussian_logpdf(df['log_rv'].values, mu, np.full(len(df), self.sigma2_))
        return {'mu': mu, 'sigma2': np.full(len(df), self.sigma2_), 'logpdf': logpdf}

class RealizedGARCH:
    name = 'Realized GARCH(1,1)'

    @staticmethod
    def _filter(log_rv: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
        T = len(log_rv)
        log_h = np.empty(T)
        log_h[0] = np.mean(log_rv)
        for t in range(1, T):
            log_h[t] = omega + beta * log_h[t - 1] + alpha * log_rv[t - 1]
        return log_h

    @staticmethod
    def _nll(params: np.ndarray, log_rv: np.ndarray) -> float:
        omega, alpha, beta, xi, phi, log_s2u = params
        sigma2_u = np.exp(log_s2u)
        log_h = RealizedGARCH._filter(log_rv, omega, alpha, beta)
        u = log_rv - (xi + phi * log_h)
        ll = -0.5 * (np.log(2.0 * np.pi * sigma2_u) + u ** 2 / sigma2_u)
        return -float(np.sum(ll))

    def fit(self, df, k_tr=None):
        log_rv = df['log_rv'].values.astype(float)
        mu0 = float(np.mean(log_rv))
        var0 = float(np.var(log_rv))
        bounds = [(-20.0, 20.0), (1e-6, 0.999), (1e-6, 0.999),
                  (-20.0, 20.0), (0.01, 5.0), (-15.0, 5.0)]
        def penalised(params):
            pen = 1e6 * max(0.0, params[1] + params[2] - 0.9999) ** 2
            return RealizedGARCH._nll(params, log_rv) + pen
        log_s2u0 = np.log(max(var0 * 0.5, _eps()))
        starts = [
            [mu0 * (1 - 0.85 - 0.10), 0.10, 0.85, 0.0, 1.0, log_s2u0],
            [mu0 * (1 - 0.70 - 0.20), 0.20, 0.70, 0.0, 1.0, log_s2u0],
            [mu0 * (1 - 0.50 - 0.40), 0.40, 0.50, 0.0, 1.0, log_s2u0],
        ]
        best, best_val = None, np.inf
        for x0 in starts:
            opt = optimize.minimize(penalised, x0, method='L-BFGS-B', bounds=bounds,
                                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-9})
            val = opt.fun if opt.success else penalised(opt.x)
            if val < best_val:
                best_val, best = val, opt.x
        p = best if best is not None else np.array(starts[0])
        self.omega_, self.alpha_, self.beta_ = float(p[0]), float(p[1]), float(p[2])
        self.xi_, self.phi_ = float(p[3]), float(p[4])
        self.sigma2_u_ = float(np.exp(p[5]))
        log_h_tr = self._filter(log_rv, self.omega_, self.alpha_, self.beta_)
        self._last_log_h = float(log_h_tr[-1])
        self._last_log_rv = float(log_rv[-1])

    def predict(self, df):
        log_rv_te = df['log_rv'].values.astype(float)
        T = len(log_rv_te)
        log_h = np.empty(T)
        log_h[0] = self.omega_ + self.beta_ * self._last_log_h + self.alpha_ * self._last_log_rv
        for t in range(1, T):
            log_h[t] = self.omega_ + self.beta_ * log_h[t - 1] + self.alpha_ * log_rv_te[t - 1]
        mu = self.xi_ + self.phi_ * log_h
        sigma2 = np.full(T, self.sigma2_u_)
        logpdf = _gaussian_logpdf(log_rv_te, mu, sigma2)
        return {'mu': mu, 'sigma2': sigma2, 'logpdf': logpdf}

SUPP_MODELS = [HART, HARCJ, RealizedGARCH]

def _run_one_window(args):
    ws, df_tr, df_te, cfg_snapshot = args
    _CFG.update(cfg_snapshot)
    k_tr = df_tr['regime_fi'].map(LABEL_MAP).fillna(REGIME_NORMAL).values.astype(int)
    regime_te = df_te['regime_fi'].fillna('Normal').values.astype(str) if 'regime_fi' in df_te.columns else np.full(len(df_te), 'Normal')
    hb_te = df_te['hb_tz'].values
    out = {}
    for Cls in SUPP_MODELS:
        m = Cls()
        try:
            m.fit(df_tr, k_tr=k_tr)
            pred = m.predict(df_te)
            entry = {'hb': hb_te, 'mu': pred['mu'], 'sigma2': pred['sigma2'],
                     'logpdf': pred['logpdf'], 'y': df_te['log_rv'].values,
                     'regime': regime_te, 'ws': str(ws.date())}
            if m.name == 'HAR-t':
                entry['nu_hat'] = m.nu_
            if m.name == 'Realized GARCH(1,1)':
                entry['alpha_hat'] = m.alpha_
                entry['beta_hat']  = m.beta_
            out[m.name] = entry
        except Exception as e:
            print(f'{m.name} @ {ws.date()}: {e}')
    return out

def rolling_oos(df: pd.DataFrame, cfg: dict) -> dict:
    fc = cfg['forecast']
    test_end = df['hb_tz'].max()
    ts = pd.Timestamp(fc['test_start'])
    windows = []
    while ts <= test_end:
        we = min(ts + pd.DateOffset(months=fc['roll_months']) - pd.Timedelta(hours=1), test_end)
        tr_start = ts - pd.DateOffset(months=fc['train_months'])
        df_tr = df[(df['hb_tz'] >= tr_start) & (df['hb_tz'] < ts)].copy()
        df_te = df[(df['hb_tz'] >= ts) & (df['hb_tz'] <= we)].copy()
        if len(df_tr) >= 500 and len(df_te) >= 10:
            windows.append((ts, df_tr, df_te, dict(cfg)))
        ts += pd.DateOffset(months=fc['roll_months'])
    n_workers = fc['n_workers'] or min(len(windows), os.cpu_count() or 4)
    results = {Cls().name: [] for Cls in SUPP_MODELS}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_run_one_window, w): w[0] for w in windows}
        for fut in as_completed(futs):
            ws = futs[fut]
            try:
                for name, vals in fut.result().items():
                    results[name].append(vals)
            except Exception as e:
                print(f'window {ws.date()} failed: {e}')
    for name in results:
        results[name].sort(key=lambda x: x['ws'])
    return results

def _concat_preds(preds, regime_filter=None):
    mus, ys, logpdfs = [], [], []
    for p in preds:
        mask = p['regime'] == regime_filter if regime_filter else np.ones(len(p['y']), dtype=bool)
        if mask.sum() == 0:
            continue
        mus.append(p['mu'][mask])
        ys.append(p['y'][mask])
        logpdfs.append(p['logpdf'][mask])
    if not mus:
        return (None, None, None)
    return (np.concatenate(mus), np.concatenate(ys), np.concatenate(logpdfs))

def aggregate_metrics(results: dict, results_dir: Path) -> pd.DataFrame:
    groups = {'All Hours': None, 'Contagion Hours': 'Contagion',
              'Single-Asset Stress Hours': 'Single-Asset Stress', 'Normal Hours': 'Normal'}
    rows = []
    for Cls in SUPP_MODELS:
        name = Cls().name
        preds = results.get(name, [])
        if not preds:
            continue
        row = {'Model': name}
        for label, rf in groups.items():
            mu, y, logpdf = _concat_preds(preds, rf)
            if mu is None or len(mu) < 5:
                row[f'{label}_QLIKE'] = np.nan
                row[f'{label}_LogS'] = np.nan
            else:
                row[f'{label}_QLIKE'] = round(qlike(mu, y), 4)
                row[f'{label}_LogS'] = round(float(np.mean(logpdf)), 4)
        rows.append(row)
    table = pd.DataFrame(rows)
    order = ['HAR-t', 'HAR-CJ', 'Realized GARCH(1,1)']
    table['Model'] = pd.Categorical(table['Model'], categories=order, ordered=True)
    table = table.sort_values('Model').reset_index(drop=True)
    out = results_dir / 'table_s1a_additional_benchmarks.csv'
    table.to_csv(out, index=False)
    print(f'  Saved {out}')
    return table

def save_table_s1a(table: pd.DataFrame) -> None:
    print('\n' + '=' * 118)
    print('TABLE S1a: Supplementary Additional Benchmark Results')
    print('=' * 118)
    hdr = (f"{'Model':<24} {'All QLIKE':>10} {'All Log-S':>10} {'Cont QLIKE':>11} "
           f"{'Cont Log-S':>11} {'Stress QLIKE':>13} {'Stress Log-S':>13} "
           f"{'Norm QLIKE':>11} {'Norm Log-S':>11}")
    print(hdr)
    for _, row in table.iterrows():
        print(f"{str(row['Model']):<24} "
              f"{row.get('All Hours_QLIKE', np.nan):>10.3f} "
              f"{row.get('All Hours_LogS', np.nan):>10.3f} "
              f"{row.get('Contagion Hours_QLIKE', np.nan):>11.3f} "
              f"{row.get('Contagion Hours_LogS', np.nan):>11.3f} "
              f"{row.get('Single-Asset Stress Hours_QLIKE', np.nan):>13.3f} "
              f"{row.get('Single-Asset Stress Hours_LogS', np.nan):>13.3f} "
              f"{row.get('Normal Hours_QLIKE', np.nan):>11.3f} "
              f"{row.get('Normal Hours_LogS', np.nan):>11.3f}")

def parse_args():
    p = argparse.ArgumentParser(description='Supplementary benchmark comparison')
    p.add_argument('--config', default=str(Path(__file__).parent / 'config.yaml'), help='Path to config.yaml')
    return p.parse_args()

def main():
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ.setdefault(var, '1')
    args = parse_args()
    cfg = load_config(args.config)
    df = load_data(cfg)
    results = rolling_oos(df, cfg)
    table = aggregate_metrics(results, cfg['_results_dir'])
    save_table_s1a(table)
    hart_preds = results.get('HAR-t', [])
    nu_vals = [p['nu_hat'] for p in hart_preds if 'nu_hat' in p]
    if nu_vals:
        print(f"\nHAR-t mean estimated dof across windows: {np.mean(nu_vals):.3f}")
    rgarch_preds = results.get('Realized GARCH(1,1)', [])
    if rgarch_preds:
        alpha_vals = [p['alpha_hat'] for p in rgarch_preds if 'alpha_hat' in p]
        beta_vals  = [p['beta_hat']  for p in rgarch_preds if 'beta_hat'  in p]
        if alpha_vals:
            print(f"Realized GARCH(1,1) mean alpha across windows: {np.mean(alpha_vals):.3f}")
            print(f"Realized GARCH(1,1) mean beta  across windows: {np.mean(beta_vals):.3f}")
            print(f"Realized GARCH(1,1) mean alpha+beta:           {np.mean([a+b for a,b in zip(alpha_vals,beta_vals)]):.3f}")
    print(f"\nDone. Outputs in: {cfg['_results_dir']}")

if __name__ == '__main__':
    main()