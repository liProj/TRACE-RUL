"""Aggregate every result into results/summary.json (the single source for tables, figures and prose).

Headline rule (design_log sections 3 and 7): C-MAPSS TRACE = NNLS stack + TT; N-CMAPSS TRACE = uniform mean + TT.
Significance: paired bootstrap over test engines (C-MAPSS) or over (unit, cycle) blocks (N-CMAPSS) of
  mean_s RMSE(TRACE seed s) - mean_s RMSE(AIDGN rerun seed s)   (expected single-run difference, no implicit ensembling).
"""
import glob, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmapss import rmse, score
from trace_post import tt_filter_fast

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R = f'{ROOT}/results'
SUBS = ['FD001', 'FD002', 'FD003', 'FD004']
DSS = ['DS01', 'DS02', 'DS03', 'DS04', 'DS05', 'DS06', 'DS07']
B = 2000
TAGN = 'trace_louo'   # N-CMAPSS protocol after amendment 2 (design_log section 8)


def ms(xs):
    xs = [x for x in xs if x is not None]
    return dict(mean=float(np.mean(xs)), sd=float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, n=len(xs),
                values=[float(x) for x in xs]) if xs else None


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def seeds_of(pattern):
    return sorted(int(p.split('_s')[-1].split('_')[0].split('.')[0]) for p in glob.glob(pattern))


def boot_diff(P_a, P_b, y, groups, mean_score=False, seed=0):
    """P_a, P_b: lists (over seeds) of prediction vectors; returns mean diff and 95% CI of RMSE and Score."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in ug}
    def stat(ix):
        ra = np.mean([rmse(p[ix], y[ix]) for p in P_a]); rb = np.mean([rmse(p[ix], y[ix]) for p in P_b])
        sa = np.mean([score(p[ix], y[ix], mean_score) for p in P_a]); sb = np.mean([score(p[ix], y[ix], mean_score) for p in P_b])
        return ra - rb, sa - sb
    full = stat(np.arange(len(y)))
    bs = []
    for _ in range(B):
        g = rng.choice(ug, len(ug), replace=True)
        bs.append(stat(np.concatenate([idx_of[x] for x in g])))
    bs = np.array(bs)
    return dict(d_rmse=full[0], ci_rmse=[float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
                p_rmse=float(np.mean(bs[:, 0] >= 0)), d_score=full[1],
                ci_score=[float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
                p_score=float(np.mean(bs[:, 1] >= 0)))


def aidgn_allcycles(sub, pt):
    """faithful AIDGN checkpoint applied to the window ending at EVERY test cycle (reference input format)."""
    from trace_data import load
    from aidgn import AIDGN
    _, te, _ = load(sub)
    W = te.windows(use_drift=False)
    m = AIDGN(); m.load_state_dict(torch.load(pt, map_location='cpu')); m.eval()
    with torch.no_grad():
        p = np.concatenate([m(torch.tensor(W[i:i + 8192, :, :14]), torch.tensor(W[i:i + 8192, :, 14:] * 1e-3)).numpy().ravel()
                            for i in range(0, len(W), 8192)]) * 125
    return p, te


def cmapss():
    out = {}
    for sub in SUBS:
        o = {}
        for tag in ['aidgn', 'aidgn_noage']:
            js = [jload(p) for p in sorted(glob.glob(f'{R}/runs/{tag}/{sub}_s*.json'))]
            o[tag] = dict(rmse=ms([j['rmse'] for j in js]), score=ms([j['score'] for j in js])) if js else None
        ev = {s: jload(f'{R}/trace/trace/{sub}_s{s}_eval.json') for s in seeds_of(f'{R}/trace/trace/{sub}_s*_eval.json')}
        if ev:
            keys = list(next(iter(ev.values())).keys())
            o['trace'] = {k: {m: ms([e[k][m] for e in ev.values() if k in e]) for m in ['raw_rmse', 'raw_score', 'tt_rmse', 'tt_score', 'oof_raw', 'tt_oof_rmse']}
                          for k in set().union(*ev.values())}
            o['trace_weights'] = {n: ms([e['stack']['weights'].get(n, 0.0) for e in ev.values()]) for n in ev[min(ev)]['stack']['weights']}
            o['trace_tt_params'] = {s: {k: e['stack'][f'tt_{k}'] for k in ['gamma', 'beta', 'tau']} for s, e in ev.items()}
            o['headline'] = dict(rmse=o['trace']['stack']['tt_rmse'], score=o['trace']['stack']['tt_score'])
        st = [jload(p) for p in sorted(glob.glob(f'{R}/trace/trace/{sub}_s*_student.json'))]
        if st:
            o['student'] = {m: ms([j[m] for j in st]) for m in ['raw_rmse', 'raw_score', 'tt_rmse', 'tt_score']}
            o['student']['params'] = st[0]['params']
        for ab in ['trace_nodrift', 'trace_noage', 'trace_noise', 'trace_noise05']:
            evs = {s: jload(f'{R}/trace/{ab}/{sub}_s{s}_eval.json') for s in seeds_of(f'{R}/trace/{ab}/{sub}_s*_eval.json')}
            if evs:
                o[ab] = {k: {m: ms([e[k][m] for e in evs.values() if k in e]) for m in ['raw_rmse', 'raw_score', 'tt_rmse', 'tt_score']}
                         for k in ['stack', 'mean', 'aidgna', 'cnn', 'gru', 'lgbm']}
                # like-for-like reference: main run, same seeds, 4-expert stack
                o[ab]['ref_stack4'] = {m: ms([ev[s]['stack4'][m] for s in evs if s in ev and 'stack4' in ev[s]])
                                       for m in ['tt_rmse', 'tt_score', 'raw_rmse', 'raw_score']}
        for a_, key in [('0.1', 'aidgn_noise'), ('0.5', 'aidgn_noise05')]:
            nz = jload(f'{R}/noise_aidgn_a{a_}.json')
            if nz:
                o[key] = dict(rmse=ms([v['rmse'] for k, v in nz.items() if k.startswith(sub)]),
                              score=ms([v['score'] for k, v in nz.items() if k.startswith(sub)]))
        # significance vs faithful AIDGN rerun (common seeds)
        seeds = [s for s in ev if os.path.exists(f'{R}/runs/aidgn/{sub}_s{s}.npz')]
        if seeds:
            Pa = [np.load(f'{R}/trace/trace/{sub}_s{s}_final.npz')['stack_tt'] for s in seeds]
            y = np.load(f'{R}/trace/trace/{sub}_s{seeds[0]}_final.npz')['y']
            Pb = [np.load(f'{R}/runs/aidgn/{sub}_s{s}.npz')['pred'] for s in seeds]
            o['sig_vs_rerun'] = boot_diff(Pa, Pb, y, np.arange(len(y)))
            o['sig_vs_rerun']['seeds'] = seeds
            # TT applied to the faithful AIDGN (parameters of the TRACE stack of the same seed, fitted on OOF)
            tt_r, tt_s, raw_r = [], [], []
            for s in seeds:
                p, te = aidgn_allcycles(sub, f'{R}/runs/aidgn/{sub}_s{s}.pt')
                li = np.cumsum(te.n) - 1
                prm = o['trace_tt_params'][s]
                f = tt_filter_fast(p, te.sample_t() + 1.0, te.sample_engine(), prm['gamma'], prm['beta'], prm['tau'], 125.0)
                raw_r.append(rmse(p[li], y)); tt_r.append(rmse(f[li], y)); tt_s.append(score(f[li], y))
            o['aidgn_tt'] = dict(raw_rmse=ms(raw_r), tt_rmse=ms(tt_r), tt_score=ms(tt_s))
        out[sub] = o
    return out


def ncmapss():
    out = {}
    for ds in DSS:
        o = {}
        ck = jload(f'{R}/runs/aidgn_n/{ds}_ckpt.json')
        o['released_ckpt'] = ck
        js = [jload(p) for p in sorted(glob.glob(f'{R}/runs/aidgn_n/{ds}_s*.json'))]
        o['aidgn'] = dict(rmse=ms([j['rmse'] for j in js]), score=ms([j['score_mean'] for j in js])) if js else None
        ev = {s: jload(f'{R}/trace_n/{TAGN}/{ds}_s{s}_eval.json') for s in seeds_of(f'{R}/trace_n/{TAGN}/{ds}_s*_eval.json')}
        if ev:
            keys = list(next(iter(ev.values())).keys())
            o['trace'] = {k: {m: ms([e[k][m] for e in ev.values() if k in e]) for m in ['raw_rmse', 'raw_score', 'tt_rmse', 'tt_score', 'oof_raw', 'tt_oof_rmse']}
                          for k in keys}
            o['trace_tt_params'] = {s: {k: e['stack'][f'tt_{k}'] for k in ['gamma', 'beta', 'tau']} for s, e in ev.items()}
            o['trace_weights'] = {n: ms([e['stack']['weights'].get(n, 0.0) for e in ev.values()]) for n in ev[min(ev)]['stack']['weights']}
            o['headline'] = dict(rmse=o['trace']['stack']['tt_rmse'], score=o['trace']['stack']['tt_score'])
        ep = jload(f'{R}/trace_n/{TAGN}/{ds}_s0_eval_pfn.json')
        if ep:
            o['tabpfn_ablation'] = {k: {m: ep[k][m] for m in ['raw_rmse', 'tt_rmse', 'tt_score']} for k in ['stack', 'stack4', 'tabpfn']}
            o['tabpfn_ablation']['weights'] = ep['stack']['weights']
        old = jload(f'{R}/trace_n/trace/{ds}_s0_eval.json')  # original 3-fold protocol (seed 0, before amendment 2)
        if old:
            o['orig3fold'] = {k: {m: old[k][m] for m in ['raw_rmse', 'tt_rmse', 'tt_score']} for k in ['stack', 'mean']}
        st = [jload(p) for p in sorted(glob.glob(f'{R}/trace_n/{TAGN}/{ds}_s*_student.json'))]
        if st:
            o['student'] = {m: ms([j[m] for j in st]) for m in ['raw_rmse', 'raw_score', 'tt_rmse', 'tt_score']}
            o['student']['params'] = st[0]['params']
        evn = {s: jload(f'{R}/trace_n/{TAGN}_noage/{ds}_s{s}_eval.json') for s in seeds_of(f'{R}/trace_n/{TAGN}_noage/{ds}_s*_eval.json')}
        if evn:
            o['trace_noage'] = {k: {m: ms([e[k][m] for e in evn.values()]) for m in ['raw_rmse', 'raw_score', 'tt_rmse', 'tt_score']}
                                for k in ['stack', 'mean', 'aidgna', 'cnn', 'gru', 'lgbm']}
            o['trace_noage']['ref_stack'] = {m: ms([ev[s]['stack'][m] for s in evn if s in ev])
                                             for m in ['tt_rmse', 'tt_score', 'raw_rmse', 'raw_score']}
        seeds = [s for s in ev if os.path.exists(f'{R}/runs/aidgn_n/{ds}_s{s}.npz')]
        if seeds:
            fa = [np.load(f'{R}/trace_n/{TAGN}/{ds}_s{s}_final.npz') for s in seeds]
            fb = [np.load(f'{R}/runs/aidgn_n/{ds}_s{s}.npz') for s in seeds]
            y = fa[0]['y']
            blocks = fa[0]['unit'] * 1000 + fa[0]['c']
            o['sig_vs_rerun'] = boot_diff([f['stack_tt'] for f in fa], [f['pred'] for f in fb], y, blocks, mean_score=True)
            o['sig_vs_rerun']['seeds'] = seeds
            tt_r, tt_s = [], []
            for s, f in zip(seeds, fb):  # TT applied to the faithful AIDGN (TRACE OOF parameters of the same seed)
                prm = o['trace_tt_params'][s]
                g = tt_filter_fast(f['pred'], f['cyc'], f['unit'], prm['gamma'], prm['beta'], prm['tau'], float(np.load(f'{R}/trace_n/{TAGN}/{ds}_s{s}.npz')['cap']))
                tt_r.append(rmse(g, y)); tt_s.append(score(g, y, True))
            o['aidgn_tt'] = dict(tt_rmse=ms(tt_r), tt_score=ms(tt_s))
        out[ds] = o
    return out


if __name__ == '__main__':
    S = dict(printed=json.load(open(f'{ROOT}/data/printed.json')), gate_ckpt=jload(f'{R}/gate_ckpt.json'),
             age_only=jload(f'{R}/age_only.json'), cmapss=cmapss(), ncmapss=ncmapss())
    json.dump(S, open(f'{R}/summary.json', 'w'), indent=1)
    for sub, o in S['cmapss'].items():
        h = o.get('headline')
        print(sub, 'TRACE', h and f"{h['rmse']['mean']:.2f}+-{h['rmse']['sd']:.2f} / {h['score']['mean']:.1f}",
              'rerun', o['aidgn'] and f"{o['aidgn']['rmse']['mean']:.2f}", 'printed', S['printed']['CMAPSS']['AIDGN'][sub]['rmse'])
    for ds, o in S['ncmapss'].items():
        h = o.get('headline')
        print(ds, 'TRACE', h and f"{h['rmse']['mean']:.2f}+-{h['rmse']['sd']:.2f} / {h['score']['mean']:.2f}",
              'rerun', o['aidgn'] and f"{o['aidgn']['rmse']['mean']:.2f}", 'printed', S['printed']['NCMAPSS']['AIDGN'][ds]['rmse'])
