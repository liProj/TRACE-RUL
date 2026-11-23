"""Stack + TT + test evaluation for the C-MAPSS cross-fitting outputs.

usage: eval_cmapss.py TAG [SUBS] [SEEDS]
Writes results/trace/<tag>/<sub>_s<seed>_eval.json and the final per-engine test predictions.
Evaluation follows the reference exactly: the last cycle of each test engine, RMSE and summed Score.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_post import fit_stack, fit_tt, tt_filter_fast, rmse
from cmapss import score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CAP = 125.0


def oof_mask(d):
    eng, t = d['eng_tr'], d['t_tr']
    n = np.bincount(eng)
    uncapped = n[eng] - t - 1
    return (t >= 30) & (uncapped <= 200)


def last_idx(d):
    return np.r_[np.cumsum(d['n_te']) - 1]


def evaluate(path, extra=()):
    d = dict(np.load(path))
    for e in extra:  # merge experts computed under other tags (same seed => same folds)
        if os.path.exists(e):
            d.update({k: v for k, v in np.load(e).items() if k.startswith(('oof_', 'test_'))})
    names = [k[4:] for k in d if k.startswith('oof_')]
    m = oof_mask(d)
    ytr, ctr, utr = d['y_tr'], d['t_tr'] + 1.0, d['eng_tr']
    cte, ute = d['t_te'] + 1.0, d['eng_te']
    li = last_idx(d)
    yte = d['y_te'][li]
    out, final = {}, {}

    def add(name, poof, pte):
        tt = fit_tt(poof, ctr, utr, ytr, m, CAP)
        fte = tt_filter_fast(pte, cte, ute, tt['gamma'], tt['beta'], tt['tau'], CAP)
        out[name] = dict(raw_rmse=rmse(pte[li], yte), raw_score=score(pte[li], yte),
                         tt_rmse=rmse(fte[li], yte), tt_score=score(fte[li], yte),
                         oof_raw=rmse(poof[m], ytr[m]), **{f'tt_{k}': v for k, v in tt.items()})
        final[name] = (pte[li], fte[li])

    for n in names:
        add(n, d[f'oof_{n}'], d[f'test_{n}'])
    P = np.stack([d[f'oof_{n}'] for n in names], 1)
    T = np.stack([d[f'test_{n}'] for n in names], 1)
    w = fit_stack(P, ytr, m)
    add('stack', P @ w, T @ w)
    out['stack']['weights'] = dict(zip(names, map(float, w)))
    add('mean', P.mean(1), T.mean(1))
    if 'tabpfn' in names:  # like-for-like reference for the 4-expert ablation runs
        keep = [i for i, n in enumerate(names) if n != 'tabpfn']
        w4 = fit_stack(P[:, keep], ytr, m)
        add('stack4', P[:, keep] @ w4, T[:, keep] @ w4)
        add('mean4', P[:, keep].mean(1), T[:, keep].mean(1))
    json.dump(out, open(path.replace('.npz', '_eval.json'), 'w'), indent=1)
    np.savez(path.replace('.npz', '_final.npz'), y=yte, ids=d['ids_te'],
             **{f'{k}_raw': v[0] for k, v in final.items()}, **{f'{k}_tt': v[1] for k, v in final.items()})
    return out


if __name__ == '__main__':
    tag = sys.argv[1]
    subs = sys.argv[2].split(',') if len(sys.argv) > 2 else ['FD001', 'FD002', 'FD003', 'FD004']
    seeds = [int(s) for s in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0, 1, 2, 3, 4]
    for sub in subs:
        for s in seeds:
            p = f'{ROOT}/results/trace/{tag}/{sub}_s{s}.npz'
            if not os.path.exists(p):
                continue
            o = evaluate(p, extra=[os.path.join(os.path.dirname(os.path.dirname(p)), tag + '_pfn', os.path.basename(p))])
            print(sub, s, ' | '.join(f"{k}: {v['raw_rmse']:.2f}->{v['tt_rmse']:.2f} (oof {v['oof_raw']:.2f}->{v['tt_oof_rmse']:.2f})"
                                     for k, v in o.items()))
