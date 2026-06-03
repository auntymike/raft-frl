"""
S1b_benchmark_var_backtest.py  —  Supplementary benchmark VaR/CVaR backtest
Replicates the supplementary 95%/99% VaR and CVaR backtests for HAR-t,
HAR-CJ, and Realized GARCH(1,1) under the same rolling-window design as
the main manuscript. Produces regime-stratified backtest outputs for
comparison with the main-model results reported in the supplement.
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
    em_avail = [c for c in EM_COLS if c in df.columns]
    if em_avail:
        df['extreme_move_mean'] = df[em_avail].mean(axis=1)
    else:
        roll_std = df[HVOL_COLS].rolling(720, min_periods=50).std()
        roll_mu = df[HVOL_COLS].rolling(720, min_periods=50).mean()
        df['extreme_move_mean'] = (df[HVOL_COLS] > roll_mu + 2 * roll_std).astype(float).mean(axis=1)
    df = df.dropna(subset=['rv_lag1', 'rv_lag24', 'rv_lag168', 'cj_cont_lag1', 'cj_jump_lag1',
                           'cj_cont_lag24', 'cj_jump_lag24', 'cj_cont_lag168', 'cj_jump_lag168']).reset_index(drop=True)
    df['hb_tz'] = df['hour_block'].dt.tz_convert(None)
    return df

def kupiec_pof(violations: int, n: int, level: float) -> float:
    p0 = 1.0 - level
    x = violations
    if x == 0:
        lr = -2.0 * n * np.log(1.0 - p0)
    elif x == n:
        lr = -2.0 * n * np.log(p0)
    else:
        p_hat = x / n
        lr = -2.0 * (x * np.log(p0 / p_hat) + (n - x) * np.log((1 - p0) / (1 - p_hat)))
    return float(1.0 - stats.chi2.cdf(lr, df=1))

def _gaussian_var_cvar(mu: np.ndarray, sigma2: np.ndarray, level: float):
    sigma = np.sqrt(np.maximum(sigma2, _eps()))
    z = stats.norm.ppf(level)
    var = mu + sigma * z
    cvar = mu + sigma * stats.norm.pdf(z) / (1.0 - level)
    return var, cvar

def _t_var_cvar(mu: np.ndarray, sigma2: np.ndarray, nu: float, level: float):
    scale = np.sqrt(np.maximum(sigma2, _eps()) * max((nu - 2.0) / nu, _eps()))
    z = stats.t.ppf(level, df=nu)
    var = mu + scale * z
    cvar = mu + scale * stats.t.pdf(z, df=nu) * (nu + z ** 2) / ((nu - 1.0) * (1.0 - level))
    return var, cvar

def _gaussian_logpdf(y: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    s2 = np.maximum(sigma2, _eps())
    return -0.5 * np.log(2 * np.pi * s2) - 0.5 * (y - mu) ** 2 / s2

def _har_X(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df['rv_lag1'].values, df['rv_lag24'].values, df['rv_lag168'].values])

def _harcj_X(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df['cj_cont_lag1'].values, df['cj_jump_lag1'].values,
                            df['cj_cont_lag24'].values, df['cj_jump_lag24'].values,
                            df['cj_cont_lag168'].values, df['cj_jump_lag168'].values])


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
        opt = optimize.minimize(nll, [np.log(var0), np.log(8.0)], method='L-BFGS-B')
        if opt.success:
            self.sigma2_ = float(np.exp(opt.x[0]))
            self.nu_ = float(2.0 + np.exp(opt.x[1]))
        else:
            self.sigma2_ = var0
            self.nu_ = 10.0

    def predict(self, df):
        X = _har_X(df)
        mu = X @ self.beta_
        return {'mu': mu, 'sigma2': np.full(len(df), self.sigma2_)}


class HARCJ:
    name = 'HAR-CJ'

    def fit(self, df, k_tr=None):
        X = _harcj_X(df)
        y = df['log_rv'].values
        self.beta_, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ self.beta_
        self.sigma2_ = max(float(np.var(resid)), _eps())

    def predict(self, df):
        mu = _harcj_X(df) @ self.beta_
        return {'mu': mu, 'sigma2': np.full(len(df), self.sigma2_)}


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
        return {'mu': mu, 'sigma2': np.full(T, self.sigma2_u_)}


SUPP_MODELS = [HART, HARCJ, RealizedGARCH]

def _run_one_window(args):
    ws, df_tr, df_te, cfg_snapshot = args
    _CFG.update(cfg_snapshot)
    k_tr = df_tr['regime_fi'].map(LABEL_MAP).fillna(0).values.astype(int)
    regime_te = df_te['regime_fi'].fillna('Normal').values.astype(str) if 'regime_fi' in df_te.columns else np.full(len(df_te), 'Normal')
    y_te = df_te['log_rv'].values
    hb_te = df_te['hb_tz'].values
    out = {}
    for Cls in SUPP_MODELS:
        m = Cls()
        try:
            m.fit(df_tr, k_tr=k_tr)
            pred = m.predict(df_te)
            mu, sigma2 = pred['mu'], pred['sigma2']
            if hasattr(m, 'nu_'):
                var95, cvar95 = _t_var_cvar(mu, sigma2, m.nu_, 0.95)
                var99, cvar99 = _t_var_cvar(mu, sigma2, m.nu_, 0.99)
            else:
                var95, cvar95 = _gaussian_var_cvar(mu, sigma2, 0.95)
                var99, cvar99 = _gaussian_var_cvar(mu, sigma2, 0.99)
            entry = {'hb': hb_te, 'y': y_te, 'mu': mu, 'sigma2': sigma2,
                     'var95': var95, 'var99': var99, 'cvar95': cvar95, 'cvar99': cvar99,
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

def compute_var_metrics(results: dict) -> pd.DataFrame:
    groups = {'All': None, 'Contagion': 'Contagion',
              'Stress': 'Single-Asset Stress', 'Normal': 'Normal'}
    order = ['HAR-t', 'HAR-CJ', 'Realized GARCH(1,1)']
    rows = []
    for model_name in order:
        preds = results.get(model_name, [])
        if not preds:
            continue
        for grp, rf in groups.items():
            arrs = {k: [] for k in ('y', 'var95', 'var99', 'cvar95', 'cvar99')}
            for p in preds:
                mask = (p['regime'] == rf) if rf else np.ones(len(p['y']), dtype=bool)
                if mask.sum() == 0:
                    continue
                for k in arrs:
                    arrs[k].append(p[k][mask])
            if not arrs['y']:
                continue
            y     = np.concatenate(arrs['y'])
            var95 = np.concatenate(arrs['var95'])
            var99 = np.concatenate(arrs['var99'])
            cvar95 = np.concatenate(arrs['cvar95'])
            cvar99 = np.concatenate(arrs['cvar99'])
            n = len(y)
            viol95 = int((y > var95).sum())
            viol99 = int((y > var99).sum())
            exc95 = y[y > var95] - cvar95[y > var95]
            exc99 = y[y > var99] - cvar99[y > var99]
            rows.append({
                'Model': model_name, 'Regime': grp, 'n': n,
                'ViolRate95': round(viol95 / n * 100, 1),
                'Kupiec_p95': kupiec_pof(viol95, n, 0.95),
                'MAE_CVaR95': round(float(np.mean(np.abs(exc95))), 2) if len(exc95) > 0 else np.nan,
                'ViolRate99': round(viol99 / n * 100, 1),
                'Kupiec_p99': kupiec_pof(viol99, n, 0.99),
                'MAE_CVaR99': round(float(np.mean(np.abs(exc99))), 2) if len(exc99) > 0 else np.nan,
            })
    return pd.DataFrame(rows)

def save_table(df_metrics: pd.DataFrame, results_dir: Path) -> None:
    order = ['HAR-t', 'HAR-CJ', 'Realized GARCH(1,1)']
    def _fmt_p(p):
        if np.isnan(p): return '     nan'
        if p < 0.001:   return '  <0.001'
        return f'{p:8.3f}'
    print('\n' + '=' * 110)
    print('TABLE S1b: Supplementary VaR Backtest — Additional Benchmarks')
    print('  Nominal: 5.0% at 95%-VaR, 1.0% at 99%-VaR  |  Kupiec p < 0.05 = rejection')
    print('=' * 110)
    for grp in ['All', 'Contagion', 'Stress', 'Normal']:
        sub = df_metrics[df_metrics['Regime'] == grp].copy()
        if sub.empty:
            continue
        sub['Model'] = pd.Categorical(sub['Model'], categories=order, ordered=True)
        sub = sub.sort_values('Model')
        print(f'\n  [{grp} Hours]')
        print(f"  {'Model':<24} {'Viol95%':>8} {'p95':>8} {'MAE-CVaR95':>12} {'Viol99%':>8} {'p99':>8} {'MAE-CVaR99':>12}")
        for _, row in sub.iterrows():
            print(f"  {row['Model']:<24} "
                  f"{row['ViolRate95']:>7.1f}% {_fmt_p(row['Kupiec_p95'])} "
                  f"{row['MAE_CVaR95']:>12.2f} "
                  f"{row['ViolRate99']:>7.1f}% {_fmt_p(row['Kupiec_p99'])} "
                  f"{row['MAE_CVaR99']:>12.2f}")
    def fmt_pval(p):
        if np.isnan(p): return 'nan'
        if p < 0.001:   return '<0.001'
        return f'{p:.3f}'
    out_df = df_metrics.copy()
    out_df['Kupiec_p95'] = out_df['Kupiec_p95'].apply(fmt_pval)
    out_df['Kupiec_p99'] = out_df['Kupiec_p99'].apply(fmt_pval)
    out = results_dir / 'table_s1b_additional_benchmarks.csv'
    out_df[['Model', 'Regime', 'ViolRate95', 'Kupiec_p95', 'MAE_CVaR95',
            'ViolRate99', 'Kupiec_p99', 'MAE_CVaR99']].to_csv(out, index=False)
    print(f'\n  Saved {out}')

def parse_args():
    p = argparse.ArgumentParser(description='Supplementary VaR backtest')
    p.add_argument('--config', default=str(Path(__file__).parent / 'config.yaml'))
    return p.parse_args()

def main():
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ.setdefault(var, '1')
    args = parse_args()
    cfg = load_config(args.config)
    df = load_data(cfg)
    results = rolling_oos(df, cfg)
    df_metrics = compute_var_metrics(results)
    save_table(df_metrics, cfg['_results_dir'])
    hart_preds = results.get('HAR-t', [])
    nu_vals = [p['nu_hat'] for p in hart_preds if 'nu_hat' in p]
    if nu_vals:
        print(f'\nHAR-t mean estimated dof across windows: {np.mean(nu_vals):.3f}')
    rgarch_preds = results.get('Realized GARCH(1,1)', [])
    if rgarch_preds:
        alpha_vals = [p['alpha_hat'] for p in rgarch_preds if 'alpha_hat' in p]
        beta_vals  = [p['beta_hat']  for p in rgarch_preds if 'beta_hat'  in p]
        if alpha_vals:
            print(f'Realized GARCH(1,1) mean alpha across windows: {np.mean(alpha_vals):.3f}')
            print(f'Realized GARCH(1,1) mean beta  across windows: {np.mean(beta_vals):.3f}')
            print(f'Realized GARCH(1,1) mean alpha+beta:           {np.mean([a+b for a,b in zip(alpha_vals,beta_vals)]):.3f}')
    print(f'\nDone. Outputs in: {cfg["_results_dir"]}')

if __name__ == '__main__':
    main()