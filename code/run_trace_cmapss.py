"""Engine-level 5-fold cross-fitting of the TRACE expert pool on one C-MAPSS subset.

usage: run_trace_cmapss.py FD001 SEED [--experts aidgna,cnn,gru,lgbm,tabpfn] [--tag trace]
                           [--no_drift] [--no_age]
Writes results/trace/<tag>/<sub>_s<seed>.npz with OOF predictions for every training sample
(every cycle of every training engine) and fold-averaged predictions for every test sample.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_data import load
import experts as E

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CAP = 125.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sub'); ap.add_argument('seed', type=int)
    ap.add_argument('--experts', default='aidgna,cnn,gru,lgbm,tabpfn')
    ap.add_argument('--tag', default='trace')
    ap.add_argument('--no_drift', action='store_true')
    ap.add_argument('--no_age', action='store_true')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--noise_test', type=float, default=0.0)
    a = ap.parse_args()
    out_dir = f'{ROOT}/results/trace/{a.tag}'
    os.makedirs(out_dir, exist_ok=True)
    out = f'{out_dir}/{a.sub}_s{a.seed}.npz'
    tr, te, mean_life = load(a.sub, noise_test=a.noise_test, noise_seed=a.seed)
    use_drift = not a.no_drift
    Wtr, Wte = tr.windows(use_drift), te.windows(use_drift)
    Ftr, Fte = tr.features(), te.features()
    if a.no_age:  # age channel and the two age/time features are removed (set to a constant)
        Wtr[..., -1] = 0; Wte[..., -1] = 0
        Ftr[:, -2:] = 0; Fte[:, -2:] = 0
    if not use_drift:  # drop the drift-vs-baseline block of the tabular features (columns 56:70)
        Ftr[:, 56:70] = 0; Fte[:, 56:70] = 0
    ytr, eng = tr.targets(), tr.sample_engine()
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(tr.ids))
    fold_of = np.empty(len(tr.ids), int)
    for k, chunk in enumerate(np.array_split(perm, a.folds)):
        fold_of[chunk] = k
    names = a.experts.split(',')
    res = {f'oof_{n}': np.full(len(ytr), np.nan, np.float32) for n in names}
    res.update({f'test_{n}': np.zeros(len(Wte), np.float32) for n in names})
    log = {n: [] for n in names}
    t0 = time.time()
    for k in range(a.folds):
        trn_e = np.where(fold_of != k)[0]
        inner = rng.permutation(trn_e)
        n_iv = max(2, len(inner) // 10)
        iv_e, it_e = inner[:n_iv], inner[n_iv:]
        m_it, m_iv, m_oof = np.isin(eng, it_e), np.isin(eng, iv_e), fold_of[eng] == k
        for n in names:
            s = a.seed * 100 + k
            if n in E.NETS:
                f, v = E.fit_net(n, Wtr[m_it], ytr[m_it], Wtr[m_iv], ytr[m_iv], CAP, s)
                po, pt = f(Wtr[m_oof]), f(Wte)
            elif n == 'lgbm':
                f, v = E.fit_lgbm(Ftr[m_it], ytr[m_it], Ftr[m_iv], ytr[m_iv], s)
                po, pt = f(Ftr[m_oof]), f(Fte)
            elif n == 'tabpfn':
                f, v = E.fit_tabpfn(Ftr[m_it], ytr[m_it], Ftr[m_iv], ytr[m_iv], s)
                po, pt = f(Ftr[m_oof]), f(Fte)
            res[f'oof_{n}'][m_oof] = np.clip(po, 0, CAP)
            res[f'test_{n}'] += np.clip(pt, 0, CAP) / a.folds
            log[n].append(v)
            print(f'{a.sub} s{a.seed} fold{k} {n} inner-val RMSE {v:.2f} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    np.savez(out, y_tr=ytr, eng_tr=eng, t_tr=tr.sample_t(), y_te=te.targets(), eng_te=te.sample_engine(),
             t_te=te.sample_t(), n_te=te.n, ids_te=te.ids, fold_of=fold_of, **res)
    json.dump({'inner_val_rmse': log, 'minutes': (time.time() - t0) / 60}, open(out.replace('.npz', '.json'), 'w'))
    print('saved', out)


if __name__ == '__main__':
    main()
