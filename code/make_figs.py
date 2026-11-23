"""All paper figures (PDF, vector) from results/summary.json and the stored prediction files.
Colour = entity, fixed order (validated reference palette of the dataviz skill):
TRACE blue, AIDGN rerun orange, AIDGN printed aqua, age-only yellow, best printed baseline magenta, student violet."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R, FIG = f'{ROOT}/results', f'{ROOT}/figures'
os.makedirs(FIG, exist_ok=True)
C = dict(trace='#2a78d6', rerun='#eb6834', printed='#1baf7a', age='#eda100', base='#e87ba4', student='#6250d6',
         ink='#0b0b0b', ink2='#52514e', grid='#e4e3df')
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': C['ink2'], 'axes.linewidth': 0.6,
                     'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.5, 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
                     'font.family': 'DejaVu Sans', 'lines.linewidth': 1.5})
S = json.load(open(f'{R}/summary.json'))
PR = S['printed']
SUBS = ['FD001', 'FD002', 'FD003', 'FD004']
DSS = ['DS01', 'DS02', 'DS03', 'DS04', 'DS05', 'DS06', 'DS07']


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'{FIG}/{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('figure', name)


def seeds(pattern):
    return sorted(int(p.split('_s')[-1].split('_')[0].split('.')[0]) for p in glob.glob(pattern))


def best_baseline(bench, sub):
    rows = {k: v[sub] for k, v in PR[bench].items() if k != 'AIDGN'}
    k = min(rows, key=lambda n: rows[n]['rmse'])
    return k, rows[k]


# ---------------------------------------------------------------- 1 gate: released vs re-implemented
def fig_gate():
    import torch
    from cmapss import load_official
    from aidgn import AIDGN
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.9))
    for ax, sub in zip(axs, SUBS):
        d = load_official(sub)
        m = AIDGN(); m.load_state_dict(torch.load(f'{ROOT}/repo/AIDGN-RUL/logs/{sub}/AIDGN/exp0/best_checkpoint.pth', map_location='cpu')); m.eval()
        with torch.no_grad():
            p = m(torch.tensor(d['Xte']), torch.tensor(d['Ite'])).numpy().ravel() * 125
        rel = np.load(f'{ROOT}/repo/AIDGN-RUL/logs/{sub}/AIDGN/exp0/result.npz')['test_preds'].ravel()
        ax.plot([0, 130], [0, 130], color=C['ink2'], lw=0.8, ls='--')
        ax.scatter(rel, p, s=9, color=C['rerun'], edgecolor='white', linewidth=0.4)
        ax.set_title(f'{sub}\nmax |diff| = {np.abs(p - rel).max():.1e}')
        ax.set_xlabel('released prediction'); ax.set_xlim(0, 130); ax.set_ylim(0, 130)
    axs[0].set_ylabel('our re-implementation')
    save(fig, 'fig_gate')


# ---------------------------------------------------------------- 2 FD003 per-engine predictions
def fig_engines(sub='FD003'):
    f = np.load(f'{R}/trace/trace/{sub}_s0_final.npz')
    a = np.load(f'{R}/runs/aidgn/{sub}_s0.npz')
    y, pt, pa = f['y'], f['stack_tt'], a['pred']
    o = np.argsort(y)
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.fill_between(x, y[o] - 13, y[o] + 10, color=C['grid'], alpha=0.8, lw=0, label='non-penalised band')
    ax.plot(x, y[o], color=C['ink'], lw=1.2, label='true RUL')
    ax.scatter(x, pa[o], s=10, color=C['rerun'], edgecolor='white', linewidth=0.4, label=f'AIDGN rerun (RMSE {np.sqrt(np.mean((pa - y) ** 2)):.2f})', zorder=3)
    ax.scatter(x, pt[o], s=10, marker='D', color=C['trace'], edgecolor='white', linewidth=0.4, label=f'TRACE (RMSE {np.sqrt(np.mean((pt - y) ** 2)):.2f})', zorder=4)
    ax.set_xlabel(f'{sub} test engines sorted by true RUL'); ax.set_ylabel('RUL (cycles)')
    ax.legend(ncol=4, loc='upper left', frameon=False)
    save(fig, 'fig_engines')


# ---------------------------------------------------------------- 3 error histograms
def fig_hist():
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.9), sharey=True)
    bins = np.arange(-60, 61, 5)
    for ax, sub in zip(axs, SUBS):
        et = np.concatenate([(lambda f: f['stack_tt'] - f['y'])(np.load(p)) for p in sorted(glob.glob(f'{R}/trace/trace/{sub}_s*_final.npz'))])
        ea = np.concatenate([(lambda f: f['pred'] - f['true'])(np.load(p)) for p in sorted(glob.glob(f'{R}/runs/aidgn/{sub}_s*.npz'))])
        ax.axvspan(-13, 10, color=C['grid'], alpha=0.8, lw=0)
        ax.hist(ea, bins=bins, density=True, histtype='step', color=C['rerun'], lw=1.5, label='AIDGN rerun')
        ax.hist(et, bins=bins, density=True, histtype='step', color=C['trace'], lw=1.5, label='TRACE')
        inside = lambda e: 100 * np.mean((e >= -13) & (e <= 10))
        ax.set_title(f'{sub}\nin band: TRACE {inside(et):.0f}%, AIDGN {inside(ea):.0f}%')
        ax.set_xlabel('error  $\\hat y - y$')
    axs[0].set_ylabel('density'); axs[0].legend(frameon=False, loc='upper left')
    save(fig, 'fig_hist')


# ---------------------------------------------------------------- 4 N-CMAPSS trajectories
def fig_ntraj():
    show = [('DS01', 9), ('DS02', 11), ('DS04', 8), ('DS07', 9)]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=True)
    for ax, (ds, u) in zip(axs, show):
        f = np.load(f'{R}/trace_n/trace_louo/{ds}_s0_final.npz')
        a = np.load(f'{R}/runs/aidgn_n/{ds}_s0.npz') if os.path.exists(f'{R}/runs/aidgn_n/{ds}_s0.npz') else None
        m = f['unit'] == u
        c = f['c'][m]
        ax.plot(c, f['y'][m], color=C['ink'], lw=1.2, label='true')
        if a is not None:
            ax.scatter(a['cyc'][m], a['pred'][m], s=1, color=C['rerun'], alpha=0.25, rasterized=True, label='AIDGN rerun')
        ax.scatter(c, f['stack_raw'][m], s=1, color=C['trace'], alpha=0.2, rasterized=True, label='TRACE, no TT')
        ax.plot(c, f['stack_tt'][m], color=C['trace'], lw=1.5, label='TRACE')
        ax.set_title(f'{ds} unit {u}'); ax.set_xlabel('flight cycle')
    axs[0].set_ylabel('RUL (cycles)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False, markerscale=6)
    save(fig, 'fig_ntraj')


# ---------------------------------------------------------------- 5 age-only critique
def fig_ageonly():
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    x = np.arange(len(DSS)); w = 0.16
    series = [('best printed baseline (no age input)', C['base'], [best_baseline('NCMAPSS', d)[1]['rmse'] for d in DSS]),
              ('age only, no sensors', C['age'], [S['age_only'][d]['rmse'] for d in DSS]),
              ('AIDGN printed', C['printed'], [PR['NCMAPSS']['AIDGN'][d]['rmse'] for d in DSS]),
              ('AIDGN rerun', C['rerun'], [(S['ncmapss'][d].get('aidgn') or {}).get('rmse', {}).get('mean', np.nan) if S['ncmapss'][d].get('aidgn') else np.nan for d in DSS]),
              ('TRACE', C['trace'], [S['ncmapss'][d]['headline']['rmse']['mean'] if S['ncmapss'][d].get('headline') else np.nan for d in DSS])]
    for i, (lab, col, v) in enumerate(series):
        ax.bar(x + (i - 2) * w, v, w * 0.92, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(DSS); ax.set_ylabel('test RMSE (cycles)')
    ax.legend(ncol=5, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.18))
    save(fig, 'fig_ageonly')


# ---------------------------------------------------------------- 6 ablation (C-MAPSS)
def fig_ablation():
    labels, vals = [], []
    def g(sub, *path):
        o = S['cmapss'][sub]
        for p in path:
            o = o.get(p) if isinstance(o, dict) else None
            if o is None:
                return np.nan
        return o['mean']
    rows = [('AIDGN rerun', lambda s: S['cmapss'][s]['aidgn']['rmse']['mean'], C['rerun']),
            ('AIDGN-A (expert)', lambda s: g(s, 'trace', 'aidgna', 'raw_rmse'), C['ink2']),
            ('CNN (expert)', lambda s: g(s, 'trace', 'cnn', 'raw_rmse'), C['ink2']),
            ('GRU (expert)', lambda s: g(s, 'trace', 'gru', 'raw_rmse'), C['ink2']),
            ('LightGBM (expert)', lambda s: g(s, 'trace', 'lgbm', 'raw_rmse'), C['ink2']),
            ('TabPFN v2 (expert)', lambda s: g(s, 'trace', 'tabpfn', 'raw_rmse'), C['ink2']),
            ('uniform mean', lambda s: g(s, 'trace', 'mean', 'raw_rmse'), C['age']),
            ('stack w/o TabPFN', lambda s: g(s, 'trace', 'stack4', 'tt_rmse'), C['base']),
            ('stack w/o TT', lambda s: g(s, 'trace', 'stack', 'raw_rmse'), C['student']),
            ('TRACE', lambda s: g(s, 'trace', 'stack', 'tt_rmse'), C['trace'])]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=True)
    for ax, sub in zip(axs, SUBS):
        v = [f(sub) for _, f, _ in rows]
        ax.barh(np.arange(len(rows)), v, color=[c for *_, c in rows], height=0.7)
        lo = np.nanmin(v)
        ax.set_xlim(lo * 0.85, np.nanmax(v) * 1.03)
        ax.axvline(PR['CMAPSS']['AIDGN'][sub]['rmse'], color=C['printed'], ls='--', lw=1)
        ax.set_title(sub); ax.set_xlabel('RMSE')
    axs[0].set_yticks(np.arange(len(rows))); axs[0].set_yticklabels([r[0] for r in rows]); axs[0].invert_yaxis()
    save(fig, 'fig_ablation')


# ---------------------------------------------------------------- 7 noise robustness
def fig_noise():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.1), sharey=True)
    x = np.arange(4); w = 0.2
    for ax, (alpha, ka, kt) in zip(axs, [('0.1', 'aidgn_noise', 'trace_noise'), ('0.5', 'aidgn_noise05', 'trace_noise05')]):
        def v(*p):
            out = []
            for s in SUBS:
                o = S['cmapss'][s]
                for q in p:
                    o = o.get(q) if isinstance(o, dict) else None
                out.append(o['mean'] if isinstance(o, dict) and 'mean' in o else np.nan)
            return out
        ax.bar(x - 1.5 * w, v('aidgn', 'rmse'), w * 0.92, color=C['rerun'], label='AIDGN rerun, clean')
        ax.bar(x - 0.5 * w, v(ka, 'rmse'), w * 0.92, color=C['rerun'], alpha=0.45, hatch='////', edgecolor='white', label='AIDGN rerun, noisy')
        ax.bar(x + 0.5 * w, v(kt, 'ref_stack4', 'tt_rmse'), w * 0.92, color=C['trace'], label='TRACE, clean')
        ax.bar(x + 1.5 * w, v(kt, 'stack', 'tt_rmse'), w * 0.92, color=C['trace'], alpha=0.45, hatch='////', edgecolor='white', label='TRACE, noisy')
        ax.set_xticks(x); ax.set_xticklabels(SUBS); ax.set_title(f'test noise $\\alpha$ = {alpha}')
    axs[0].set_ylabel('RMSE')
    axs[0].legend(ncol=4, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.12))
    save(fig, 'fig_noise')


# ---------------------------------------------------------------- 8 efficiency bubbles
def fig_efficiency():
    e = json.load(open(f'{R}/efficiency.json'))
    o = S['cmapss']['FD003']
    pts = [('AIDGN rerun', e['aidgn_cuda_ms'], o['aidgn']['rmse']['mean'], e['aidgn_params'], C['rerun']),
           ('TRACE-S (student)', e['student_cuda_ms'], o['student']['tt_rmse']['mean'], e['student_params'], C['student']),
           ('TRACE (5 folds x 5 experts)', e['nets5fold_cuda_ms'] + e['lgbm5fold_cpu_ms'] + e.get('tabpfn5fold_cuda_ms', 0),
            o['headline']['rmse']['mean'], sum(e['expert_params'].values()) * 5, C['trace'])]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for lab, t, r, p, col in pts:
        ax.scatter(t, r, s=30 + p / 400, color=col, alpha=0.85, edgecolor='white', linewidth=1)
        ax.annotate(f'{lab}\n{p / 1000:.0f}k params', (t, r), textcoords='offset points', xytext=(8, -2), fontsize=6.5, color=C['ink'])
    ax.set_xscale('log'); ax.set_xlabel('latency for 100 engines (ms, GPU, log)'); ax.set_ylabel('FD003 RMSE')
    ax.axhline(PR['CMAPSS']['AIDGN']['FD003']['rmse'], color=C['printed'], ls='--', lw=1)
    ax.text(ax.get_xlim()[0] * 1.1, PR['CMAPSS']['AIDGN']['FD003']['rmse'] + 0.05, 'AIDGN printed', fontsize=6.5, color=C['ink2'])
    save(fig, 'fig_efficiency')


# ---------------------------------------------------------------- 9 stacking weights
def fig_weights():
    names = list(S['cmapss']['FD001']['trace_weights'])
    M = np.array([[S['cmapss'][s]['trace_weights'][n]['mean'] for n in names] for s in SUBS])
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=max(0.6, M.max()))
    ax.set_xticks(range(len(names))); ax.set_xticklabels(['AIDGN-A', 'CNN', 'GRU', 'LightGBM', 'TabPFN'][:len(names)], rotation=20)
    ax.set_yticks(range(4)); ax.set_yticklabels(SUBS); ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.5, color='white' if M[i, j] > 0.35 else C['ink'])
    fig.colorbar(im, ax=ax, fraction=0.04)
    save(fig, 'fig_weights')


# ---------------------------------------------------------------- 10 TT sensitivity on OOF
def fig_tt():
    from trace_post import tt_filter_fast, rmse
    gam = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]
    bet = [round(0.1 * i, 1) for i in range(11)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for ax, (bench, sub) in zip(axs, [('n', 'DS01'), ('n', 'DS04')]):
        d = dict(np.load(f'{R}/trace_n/trace_louo/{sub}_s0.npz'))
        pf = f'{R}/trace_n/trace_louo_pfn/{sub}_s0.npz'
        if os.path.exists(pf):
            d.update({k: v for k, v in np.load(pf).items() if k.startswith('oof_')})
        ev = json.load(open(f'{R}/trace_n/trace_louo/{sub}_s0_eval.json'))['stack']
        p = sum(wt * d[f'oof_{n}'] for n, wt in ev['weights'].items())
        cap = float(d['cap']); tau = ev['tt_tau']
        G = np.array([[rmse(tt_filter_fast(p, d['c_tr'], d['unit_tr'], g, b, tau, cap), d['y_tr']) for g in gam] for b in bet])
        im = ax.imshow(G, origin='lower', aspect='auto', cmap='Blues_r')
        ax.set_xticks(range(len(gam))); ax.set_xticklabels([str(g) for g in gam], rotation=60)
        ax.set_yticks(range(len(bet))); ax.set_yticklabels([str(b) for b in bet]); ax.grid(False)
        ax.set_xlabel('forgetting factor $\\gamma$'); ax.set_ylabel('blend $\\beta$')
        i, j = np.unravel_index(np.argmin(G), G.shape)
        ax.scatter([j], [i], marker='*', s=60, color=C['age'], edgecolor=C['ink'], linewidth=0.5)
        ax.set_title(f'{sub}: OOF RMSE {G.min():.2f} (no TT {G[-1].min():.2f})')
        fig.colorbar(im, ax=ax, fraction=0.04)
    save(fig, 'fig_tt')


# ---------------------------------------------------------------- 11 seed spread vs printed
def fig_seeds():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.1), gridspec_kw=dict(width_ratios=[4, 7]))
    for ax, bench, keys in [(axs[0], 'cmapss', SUBS), (axs[1], 'ncmapss', DSS)]:
        P = PR['CMAPSS' if bench == 'cmapss' else 'NCMAPSS']['AIDGN']
        for i, k in enumerate(keys):
            o = S[bench][k]
            ax.errorbar(i - 0.2, P[k]['rmse'], yerr=P[k]['rmse_sd'], fmt='s', ms=4, color=C['printed'], capsize=2, label='AIDGN printed (mean $\\pm$ sd)' if i == 0 else None)
            if o.get('aidgn'):
                v = o['aidgn']['rmse']['values']
                ax.scatter(np.full(len(v), i), v, s=12, color=C['rerun'], edgecolor='white', linewidth=0.4, label='AIDGN rerun seeds' if i == 0 else None)
            if o.get('headline'):
                v = o['headline']['rmse']['values']
                ax.scatter(np.full(len(v), i + 0.2), v, s=12, marker='D', color=C['trace'], edgecolor='white', linewidth=0.4, label='TRACE seeds' if i == 0 else None)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys); ax.set_ylabel('test RMSE')
    axs[1].legend(frameon=False, ncol=3, loc='lower right', bbox_to_anchor=(1, 1.0))
    save(fig, 'fig_seeds')


# ---------------------------------------------------------------- 12 per-cycle error vs true RUL (C-MAPSS, all cut points of the test engines)
def fig_error_vs_rul():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.0))
    for ax, sub in zip(axs, ['FD001', 'FD004']):
        d = np.load(f'{R}/trace/trace/{sub}_s0.npz')
        ev = json.load(open(f'{R}/trace/trace/{sub}_s0_eval.json'))
        w = ev['stack']['weights']
        p = sum(w[n] * d[f'test_{n}'] for n in w if f'test_{n}' in d)
        y = d['y_te']
        bins = np.arange(0, 130, 10)
        idx = np.digitize(y, bins)
        r = [np.sqrt(np.mean((p[idx == k] - y[idx == k]) ** 2)) if np.any(idx == k) else np.nan for k in range(1, len(bins) + 1)]
        ax.plot(bins + 5, r, marker='o', ms=4, color=C['trace'], label='TRACE (all test cut points)')
        ax.set_xlabel('true RUL (cycles, binned)'); ax.set_ylabel('RMSE'); ax.set_title(sub)
    axs[0].legend(frameon=False)
    save(fig, 'fig_err_rul')


if __name__ == '__main__':
    which = sys.argv[1:] or ['gate', 'engines', 'hist', 'ntraj', 'ageonly', 'ablation', 'noise', 'efficiency', 'weights', 'tt', 'seeds', 'error_vs_rul']
    for w in which:
        try:
            globals()[f'fig_{w}']()
        except Exception as ex:
            print('FAILED', w, repr(ex))
