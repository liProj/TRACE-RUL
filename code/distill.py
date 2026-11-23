"""TRACE-S: distil the stacked expert ensemble into one compact 1D-CNN (~19k parameters).

Targets: 0.5 * hard label + 0.5 * teacher, where the teacher is the NNLS stack of the OUT-OF-FOLD expert
predictions (so the student never sees a teacher prediction made on its own training engine by a model
that was fitted on it).  The student's TT parameters are the teacher stack's (fitted on OOF).

usage: distill.py cmapss FD001 SEED [--tag trace]  |  distill.py ncmapss DS01 SEED [--tag trace]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experts as E
from trace_post import fit_stack, tt_filter_fast, rmse
from cmapss import score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LAM = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bench'); ap.add_argument('sub'); ap.add_argument('seed', type=int)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    a.tag = a.tag or ('trace' if a.bench == 'cmapss' else 'trace_louo')
    if a.bench == 'cmapss':
        from trace_data import load
        from eval_cmapss import oof_mask
        base = f'{ROOT}/results/trace/{a.tag}/{a.sub}_s{a.seed}'
        d = dict(np.load(base + '.npz'))
        ev = json.load(open(base + '_eval.json'))
        tr, te, _ = load(a.sub)
        Wtr, Wte = tr.windows(True), te.windows(True)
        cap, unit_tr, c_te, u_te = 125.0, d['eng_tr'], d['t_te'] + 1.0, d['eng_te']
        mask = oof_mask(d)
        li = np.cumsum(d['n_te']) - 1
    else:
        from ncmapss import build
        from run_trace_ncmapss import add_age
        base = f'{ROOT}/results/trace_n/{a.tag}/{a.sub}_s{a.seed}'
        d = dict(np.load(base + '.npz'))
        ev = json.load(open(base + '_eval.json'))
        raw = build(a.sub)
        cap, ml = float(d['cap']), float(d['mean_life'])
        Wtr, Wte = add_age(raw['X_tr'], raw['cyc_tr'], ml), add_age(raw['X_te'], raw['cyc_te'], ml)
        unit_tr, c_te, u_te = d['unit_tr'], d['c_te'], d['unit_te']
        mask = np.ones(len(d['y_tr']), bool)
        li = np.arange(len(d['y_te']))
    pfn = os.path.join(os.path.dirname(os.path.dirname(base)), a.tag + '_pfn', os.path.basename(base) + '.npz')
    if os.path.exists(pfn):
        d.update({k: v for k, v in np.load(pfn).items() if k.startswith(('oof_', 'test_'))})
    names = list(ev['stack']['weights'])
    # C-MAPSS: NNLS stack; N-CMAPSS: uniform mean (design_log section 7)
    ens = 'stack'  # NNLS stack on both benchmarks (design_log section 8 supersedes section 7)
    w = np.array([ev['stack']['weights'][n] for n in names]) if ens == 'stack' else np.ones(len(names)) / len(names)
    teacher = np.stack([d[f'oof_{n}'] for n in names], 1) @ w
    y = d['y_tr']
    target = LAM * y + (1 - LAM) * teacher
    rng = np.random.default_rng(a.seed)
    units = np.unique(unit_tr)
    iv = rng.choice(units, max(1, len(units) // 10), replace=False)
    m_iv = np.isin(unit_tr, iv)
    t0 = time.time()
    if a.bench == 'cmapss':  # same training rule as the C-MAPSS experts (engine-level early stopping)
        model, _ = E.fit_net('cnn', Wtr[~m_iv], y[~m_iv], Wtr[m_iv], y[m_iv], cap, a.seed + 7,
                             soft=target[~m_iv], return_model=True, bs=256, epochs=80)
    else:  # same rule as the N-CMAPSS experts after amendment 2: all units, fixed budget
        model, _ = E.fit_net_fixed('cnn', Wtr, y, cap, a.seed + 7, soft=target, return_model=True)
    p = np.clip(E.predict_net(model, torch.tensor(Wte, device=E.DEV)) * cap, 0, cap)
    s = ev[ens]
    f = tt_filter_fast(p, c_te, u_te, s['tt_gamma'], s['tt_beta'], s['tt_tau'], cap)
    yte = d['y_te'][li]
    mean = a.bench != 'cmapss'
    res = dict(raw_rmse=rmse(p[li], yte), raw_score=score(p[li], yte, mean=mean), tt_rmse=rmse(f[li], yte),
               tt_score=score(f[li], yte, mean=mean), params=E.n_params(model), minutes=(time.time() - t0) / 60)
    json.dump(res, open(base + '_student.json', 'w'))
    torch.save(model.state_dict(), base + '_student.pt')
    np.savez(base + '_student.npz', raw=p[li], tt=f[li], y=yte)
    print(a.sub, a.seed, json.dumps(res))


if __name__ == '__main__':
    main()
