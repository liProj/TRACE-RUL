"""Unit-level cross-fitting of the TRACE expert pool on one N-CMAPSS sub-dataset.

usage: run_trace_ncmapss.py DS01 SEED [--experts aidgna,cnn,gru,lgbm,tabpfn] [--tag trace] [--no_age]
Folds = groups of development units (3 folds); inside a fold one training unit is held out for early
stopping.  Every window of every development unit gets an out-of-fold prediction; test windows get the
mean over fold models.  Same windows (50 sampled rows, stride 10) and labels as the reference.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncmapss import build
import experts as E

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def add_age(X, cyc, mean_life):
    return np.concatenate([X, (cyc / mean_life)[..., None].astype(np.float32)], -1)


def features(X, cyc, mean_life):
    t = np.arange(X.shape[1]) - (X.shape[1] - 1) / 2
    slope = np.einsum('t,ntc->nc', t, X - X.mean(1, keepdims=True)) / (t ** 2).sum()
    return np.concatenate([X.mean(1), X.std(1), X[:, -1], slope,
                           (cyc[:, -1:] / mean_life), cyc[:, -1:]], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ds'); ap.add_argument('seed', type=int)
    ap.add_argument('--experts', default='aidgna,cnn,gru,lgbm,tabpfn')
    ap.add_argument('--tag', default='trace')
    ap.add_argument('--no_age', action='store_true')
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--louo', action='store_true', help='leave-one-unit-out, fixed budget, no inner unit')
    a = ap.parse_args()
    out_dir = f'{ROOT}/results/trace_n/{a.tag}'
    os.makedirs(out_dir, exist_ok=True)
    out = f'{out_dir}/{a.ds}_s{a.seed}.npz'
    d = build(a.ds)
    cap, ml = float(d['max_life']), float(d['mean_life'])
    Wtr, Wte = add_age(d['X_tr'], d['cyc_tr'], ml), add_age(d['X_te'], d['cyc_te'], ml)
    Ftr, Fte = features(d['X_tr'], d['cyc_tr'], ml), features(d['X_te'], d['cyc_te'], ml)
    if a.no_age:
        Wtr[..., -1] = 0; Wte[..., -1] = 0; Ftr[:, -2:] = 0; Fte[:, -2:] = 0
    ytr, unit = d['y_tr'], d['unit_tr']
    units = np.unique(unit)
    rng = np.random.default_rng(a.seed)
    groups = [np.array([u]) for u in units] if a.louo else np.array_split(rng.permutation(units), a.folds)
    a.folds = len(groups)
    names = a.experts.split(',')
    res = {f'oof_{n}': np.full(len(ytr), np.nan, np.float32) for n in names}
    res.update({f'test_{n}': np.zeros(len(Wte), np.float32) for n in names})
    log = {n: [] for n in names}
    t0 = time.time()
    for k, g in enumerate(groups):
        trn_u = np.setdiff1d(units, g)
        iv_u = rng.choice(trn_u, 1)
        m_it, m_iv, m_oof = np.isin(unit, np.setdiff1d(trn_u, iv_u)), np.isin(unit, iv_u), np.isin(unit, g)
        for n in names:
            s = a.seed * 100 + k
            if a.louo and n in E.NETS:
                f, v = E.fit_net_fixed(n, Wtr[~m_oof], ytr[~m_oof], cap, s), float('nan')
                po, pt = f(Wtr[m_oof]), f(Wte)
            elif a.louo and n == 'lgbm':
                f, v = E.fit_lgbm_fixed(Ftr[~m_oof], ytr[~m_oof], s), float('nan')
                po, pt = f(Ftr[m_oof]), f(Fte)
            elif a.louo and n == 'tabpfn':
                f, v = E.fit_tabpfn(Ftr[~m_oof], ytr[~m_oof], Ftr[m_oof][:10], ytr[m_oof][:10], s)
                po, pt = f(Ftr[m_oof]), f(Fte)
            elif n in E.NETS:
                f, v = E.fit_net(n, Wtr[m_it], ytr[m_it], Wtr[m_iv], ytr[m_iv], cap, s, epochs=60, bs=512)
                po, pt = f(Wtr[m_oof]), f(Wte)
            elif n == 'lgbm':
                f, v = E.fit_lgbm(Ftr[m_it], ytr[m_it], Ftr[m_iv], ytr[m_iv], s)
                po, pt = f(Ftr[m_oof]), f(Fte)
            elif n == 'tabpfn':
                f, v = E.fit_tabpfn(Ftr[m_it], ytr[m_it], Ftr[m_iv], ytr[m_iv], s)
                po, pt = f(Ftr[m_oof]), f(Fte)
            res[f'oof_{n}'][m_oof] = np.clip(po, 0, cap)
            res[f'test_{n}'] += np.clip(pt, 0, cap) / a.folds
            log[n].append(v)
            print(f'{a.ds} s{a.seed} fold{k} {n} inner-val RMSE {v:.2f} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    np.savez(out, y_tr=ytr, unit_tr=unit, c_tr=d['cyc_tr'][:, -1], y_te=d['y_te'], unit_te=d['unit_te'],
             c_te=d['cyc_te'][:, -1], cap=cap, mean_life=ml, **res)
    json.dump({'inner_val_rmse': log, 'minutes': (time.time() - t0) / 60, 'groups': [list(map(int, g)) for g in groups]},
              open(out.replace('.npz', '.json'), 'w'))
    print('saved', out)


if __name__ == '__main__':
    main()
