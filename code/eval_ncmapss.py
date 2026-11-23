"""Stack + TT + test evaluation for N-CMAPSS cross-fitting outputs (every test window, RMSE and mean Score,
as in the reference).  Windows of a unit are stored in time order, so TT runs causally over them.

usage: eval_ncmapss.py TAG [DS list] [SEEDS]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_post import fit_stack, fit_tt, tt_filter_fast, rmse
from cmapss import score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DSS = ['DS01', 'DS02', 'DS03', 'DS04', 'DS05', 'DS06', 'DS07']


def evaluate(path, extra=(), suffix=''):
    d = dict(np.load(path))
    for e in extra:  # merge experts computed under other tags (same seed => same folds)
        if os.path.exists(e):
            d.update({k: v for k, v in np.load(e).items() if k.startswith(('oof_', 'test_'))})
    cap = float(d['cap'])
    names = [k[4:] for k in d if k.startswith('oof_')]
    ytr, ctr, utr = d['y_tr'], d['c_tr'], d['unit_tr']
    yte, cte, ute = d['y_te'], d['c_te'], d['unit_te']
    m = np.ones(len(ytr), bool)
    out, final = {}, {}

    def add(name, poof, pte):
        tt = fit_tt(poof, ctr, utr, ytr, m, cap)
        fte = tt_filter_fast(pte, cte, ute, tt['gamma'], tt['beta'], tt['tau'], cap)
        out[name] = dict(raw_rmse=rmse(pte, yte), raw_score=score(pte, yte, mean=True), tt_rmse=rmse(fte, yte),
                         tt_score=score(fte, yte, mean=True), oof_raw=rmse(poof, ytr),
                         **{f'tt_{k}': v for k, v in tt.items()})
        final[name] = (pte, fte)

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
    json.dump(out, open(path.replace('.npz', f'_eval{suffix}.json'), 'w'), indent=1)
    np.savez(path.replace('.npz', f'_final{suffix}.npz'), y=yte, unit=ute, c=cte,
             **{f'{k}_raw': v[0] for k, v in final.items()}, **{f'{k}_tt': v[1] for k, v in final.items()})
    return out


if __name__ == '__main__':
    tag = sys.argv[1]
    dss = sys.argv[2].split(',') if len(sys.argv) > 2 else DSS
    seeds = [int(s) for s in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0, 1, 2, 3, 4]
    for ds in dss:
        for s in seeds:
            p = f'{ROOT}/results/trace_n/{tag}/{ds}_s{s}.npz'
            if os.path.exists(p):
                o = evaluate(p)  # headline: the 4-expert pool (design_log section 9)
                pf = os.path.join(os.path.dirname(os.path.dirname(p)), tag + '_pfn', os.path.basename(p))
                if os.path.exists(pf):  # TabPFN cost/benefit ablation (DS01 seed 0 only)
                    evaluate(p, extra=[pf], suffix='_pfn')
                print(ds, s, ' | '.join(f"{k}: {v['raw_rmse']:.2f}->{v['tt_rmse']:.2f}" for k, v in o.items()))
