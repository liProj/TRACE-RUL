"""LaTeX tables + paper/numbers.tex macros from results/summary.json.  Missing values become \\pending{...}
so the build gate (check_paper.py) refuses to finish while anything is still running."""
import json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
P = f'{ROOT}/paper'
S = json.load(open(f'{ROOT}/results/summary.json'))
E = json.load(open(f'{ROOT}/results/efficiency.json')) if os.path.exists(f'{ROOT}/results/efficiency.json') else {}
PR = S['printed']
SUBS = ['FD001', 'FD002', 'FD003', 'FD004']
DSS = ['DS01', 'DS02', 'DS03', 'DS04', 'DS05', 'DS06', 'DS07']
L = {**{s: 'FD' + 'abcd'[i] for i, s in enumerate(SUBS)}, **{d: 'DS' + 'abcdefg'[i] for i, d in enumerate(DSS)}}
MAC = {}


def mac(name, val, fmt='{:.2f}'):
    MAC[name] = r'\pending{' + name + '}' if val is None or (isinstance(val, float) and np.isnan(val)) else (fmt.format(val) if not isinstance(val, str) else val)


def get(o, *path):
    for p in path:
        if not isinstance(o, dict) or o.get(p) is None:
            return None
        o = o[p]
    return o


def pm(o, nd=2, sd=True):
    if o is None:
        return '--'
    if isinstance(o, dict) and 'mean' in o:
        return f"{o['mean']:.{nd}f}" + (f"$\\pm${o['sd']:.{nd}f}" if sd and o.get('n', 1) > 1 else '')
    return f'{o:.{nd}f}'


def mean_of(o):
    return None if o is None else (o['mean'] if isinstance(o, dict) else o)


# ------------------------------------------------------------------ C-MAPSS main table
def tab_cmapss():
    rows = []
    for name, v in PR['CMAPSS'].items():
        if name == 'AIDGN':
            continue
        rows.append((name, [v[s]['rmse'] for s in SUBS], [v[s]['score'] for s in SUBS],
                     [f"{v[s]['rmse']:.2f}$\\pm${v[s]['rmse_sd']:.2f}" for s in SUBS], [f"{v[s]['score']:.1f}$\\pm${v[s]['score_sd']:.1f}" for s in SUBS], 'printed'))
    a = PR['CMAPSS']['AIDGN']
    rows.append(('AIDGN-RUL~\\cite{abbas2026aidgn}', [a[s]['rmse'] for s in SUBS], [a[s]['score'] for s in SUBS],
                 [f"{a[s]['rmse']:.2f}$\\pm${a[s]['rmse_sd']:.2f}" for s in SUBS], [f"{a[s]['score']:.1f}$\\pm${a[s]['score_sd']:.1f}" for s in SUBS], 'printed'))
    C = S['cmapss']
    def ours(label, f_r, f_s):
        r = [f_r(s) for s in SUBS]; sc = [f_s(s) for s in SUBS]
        rows.append((label, [mean_of(x) for x in r], [mean_of(x) for x in sc], [pm(x) for x in r], [pm(x, 1) for x in sc], 'ours'))
    ours('Age only (no sensors)', lambda s: S['age_only'][s]['rmse'], lambda s: S['age_only'][s]['score'])
    ours('AIDGN-RUL, our rerun', lambda s: get(C[s], 'aidgn', 'rmse'), lambda s: get(C[s], 'aidgn', 'score'))
    ours('TRACE-S (student)', lambda s: get(C[s], 'student', 'tt_rmse'), lambda s: get(C[s], 'student', 'tt_score'))
    ours('\\textbf{TRACE (ours)}', lambda s: get(C[s], 'headline', 'rmse'), lambda s: get(C[s], 'headline', 'score'))
    best_r = [np.nanmin([r[1][i] if r[1][i] is not None else np.nan for r in rows]) for i in range(4)]
    best_s = [np.nanmin([r[2][i] if r[2][i] is not None else np.nan for r in rows]) for i in range(4)]
    out = [r'\begin{table*}[!t]\centering\caption{Comparison on C-MAPSS (last cycle of every test engine). Printed rows are copied',
           r'from Table~II of~\cite{abbas2026aidgn}; our rows are mean$\pm$sd over seeds (AIDGN rerun: five; TRACE: three complete',
           r'cross-fitting pipelines). Score is summed over engines. Best in bold.}\label{tab:cmapss}',
           r'\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lcccccccc}\toprule',
           r'& \multicolumn{4}{c}{RMSE} & \multicolumn{4}{c}{Score}\\\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
           r'Model & FD001 & FD002 & FD003 & FD004 & FD001 & FD002 & FD003 & FD004\\\midrule']
    for name, rv, sv, rs, ss, kind in rows:
        if name.startswith('Age only'):
            out.append(r'\midrule')
        cells = [(f'\\textbf{{{x}}}' if rv[i] is not None and abs(rv[i] - best_r[i]) < 1e-9 else x) for i, x in enumerate(rs)]
        cells += [(f'\\textbf{{{x}}}' if sv[i] is not None and abs(sv[i] - best_s[i]) < 1e-9 else x) for i, x in enumerate(ss)]
        out.append(name + ' & ' + ' & '.join(cells) + r'\\')
    out += [r'\bottomrule\end{tabular}\end{table*}']
    open(f'{P}/tab_cmapss.tex', 'w').write('\n'.join(out))


# ------------------------------------------------------------------ N-CMAPSS main table
def tab_ncmapss():
    rows = []
    for name, v in PR['NCMAPSS'].items():
        rows.append(('AIDGN-RUL~\\cite{abbas2026aidgn}' if name == 'AIDGN' else name,
                     [v[d]['rmse'] for d in DSS] + [v['Avg']['rmse']], [v[d]['score'] for d in DSS] + [v['Avg']['score']], 'printed'))
    N = S['ncmapss']
    def ours(label, fr, fs):
        r = [mean_of(fr(d)) for d in DSS]; s = [mean_of(fs(d)) for d in DSS]
        ra = float(np.mean(r)) if None not in r else None; sa = float(np.mean(s)) if None not in s else None
        rows.append((label, r + [ra], s + [sa], 'ours'))
    ours('Age only (no sensors)', lambda d: S['age_only'][d]['rmse'], lambda d: S['age_only'][d]['score'])
    ours('AIDGN-RUL, released weights', lambda d: get(N[d], 'released_ckpt', 'rmse'), lambda d: get(N[d], 'released_ckpt', 'score_mean'))
    ours('AIDGN-RUL, our rerun', lambda d: get(N[d], 'aidgn', 'rmse'), lambda d: get(N[d], 'aidgn', 'score'))
    ours('TRACE-S (student)', lambda d: get(N[d], 'student', 'tt_rmse'), lambda d: get(N[d], 'student', 'tt_score'))
    ours('\\textbf{TRACE (ours)}', lambda d: get(N[d], 'headline', 'rmse'), lambda d: get(N[d], 'headline', 'score'))
    best = [[np.nanmin([r[k][i] if r[k][i] is not None else np.nan for r in rows]) for i in range(8)] for k in (1, 2)]
    out = [r'\begin{table*}[!t]\centering\caption{Comparison on N-CMAPSS (every window of the test units, Score averaged over',
           r'windows). Printed rows from Table~III of~\cite{abbas2026aidgn} (their $\pm$sd omitted for space); our rows are means',
           r'over three seeds, standard deviations in the supplementary JSON. Best in bold.}\label{tab:ncmapss}',
           r'\scriptsize\setlength{\tabcolsep}{2.6pt}\begin{tabular}{llcccccccc}\toprule',
           r'Model & Metric & DS01 & DS02 & DS03 & DS04 & DS05 & DS06 & DS07 & Avg\\\midrule']
    for name, rv, sv, kind in rows:
        if name.startswith('Age only'):
            out.append(r'\midrule')
        for k, (lab, v) in enumerate((('RMSE', rv), ('Score', sv))):
            cells = []
            for i, x in enumerate(v):
                s = '--' if x is None else f'{x:.2f}'
                if x is not None and abs(x - best[k][i]) < 1e-9:
                    s = f'\\textbf{{{s}}}'
                cells.append(s)
            out.append((name if k == 0 else '') + f' & {lab} & ' + ' & '.join(cells) + r'\\')
    out += [r'\bottomrule\end{tabular}\end{table*}']
    open(f'{P}/tab_ncmapss.tex', 'w').write('\n'.join(out))


# ------------------------------------------------------------------ reproduction gate
def tab_gate():
    G = S['gate_ckpt']
    out = [r'\begin{table}[!t]\centering\caption{Reproduction of AIDGN-RUL. ``Released'': the result files shipped with the code;',
           r'``ckpt'': the released weights run through our re-implementation; ``rerun'': our training from scratch',
           r'(mean$\pm$sd, 5 seeds C-MAPSS / 3 seeds N-CMAPSS). RMSE.}\label{tab:gate}',
           r'\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lcccc}\toprule',
           r'Subset & Printed & Released & ckpt (ours) & Rerun (ours)\\\midrule']
    for s in SUBS:
        a = PR['CMAPSS']['AIDGN'][s]
        out.append(f"{s} & {a['rmse']:.2f}$\\pm${a['rmse_sd']:.2f} & {G[s]['released_npz_rmse']:.2f} & {G[s]['ours_rmse']:.2f} & {pm(get(S['cmapss'][s], 'aidgn', 'rmse'))}\\\\")
    out.append(r'\midrule')
    for d in DSS:
        a = PR['NCMAPSS']['AIDGN'][d]; ck = S['ncmapss'][d]['released_ckpt'] or {}
        out.append(f"{d} & {a['rmse']:.2f}$\\pm${a['rmse_sd']:.2f} & {ck.get('released_rmse', float('nan')):.2f} & {ck.get('rmse', float('nan')):.2f} & {pm(get(S['ncmapss'][d], 'aidgn', 'rmse'))}\\\\")
    out += [r'\bottomrule\end{tabular}\end{table}']
    open(f'{P}/tab_gate.tex', 'w').write('\n'.join(out))


# ------------------------------------------------------------------ ablations
def tab_ablation():
    C = S['cmapss']
    rows = [('AIDGN rerun (reference input)', lambda s: get(C[s], 'aidgn', 'rmse'), lambda s: get(C[s], 'aidgn', 'score')),
            ('AIDGN rerun, age removed', lambda s: get(C[s], 'aidgn_noage', 'rmse'), lambda s: get(C[s], 'aidgn_noage', 'score')),
            ('AIDGN rerun + TT', lambda s: get(C[s], 'aidgn_tt', 'tt_rmse'), lambda s: get(C[s], 'aidgn_tt', 'tt_score')),
            (None, None, None)]
    for k, lab in [('aidgna', 'AIDGN-A (age unscaled, drift)'), ('cnn', '1D-CNN'), ('gru', 'GRU'), ('lgbm', 'LightGBM'), ('tabpfn', 'TabPFN v2')]:
        rows.append((f'Expert: {lab}', (lambda k: lambda s: get(C[s], 'trace', k, 'raw_rmse'))(k), (lambda k: lambda s: get(C[s], 'trace', k, 'raw_score'))(k)))
    rows += [(None, None, None),
             ('Uniform mean of experts', lambda s: get(C[s], 'trace', 'mean', 'raw_rmse'), lambda s: get(C[s], 'trace', 'mean', 'raw_score')),
             ('NNLS stack, no TT', lambda s: get(C[s], 'trace', 'stack', 'raw_rmse'), lambda s: get(C[s], 'trace', 'stack', 'raw_score')),
             ('NNLS stack w/o TabPFN, + TT', lambda s: get(C[s], 'trace', 'stack4', 'tt_rmse'), lambda s: get(C[s], 'trace', 'stack4', 'tt_score')),
             ('\\textbf{TRACE} (NNLS stack + TT)', lambda s: get(C[s], 'headline', 'rmse'), lambda s: get(C[s], 'headline', 'score')),
             (None, None, None),
             ('w/o TabPFN, seed 0 (reference)', lambda s: get(C[s], 'trace_nodrift', 'ref_stack4', 'tt_rmse'), lambda s: get(C[s], 'trace_nodrift', 'ref_stack4', 'tt_score')),
             ('\\quad -- drift channels', lambda s: get(C[s], 'trace_nodrift', 'stack', 'tt_rmse'), lambda s: get(C[s], 'trace_nodrift', 'stack', 'tt_score')),
             ('\\quad -- age channel', lambda s: get(C[s], 'trace_noage', 'stack', 'tt_rmse'), lambda s: get(C[s], 'trace_noage', 'stack', 'tt_score'))]
    out = [r'\begin{table*}[!t]\centering\caption{Ablation on C-MAPSS (mean$\pm$sd over seeds; RMSE / summed Score).',
           r'TT = trajectory-consistent tracking. The last block is run without TabPFN on seed 0 and compared with the',
           r'main run restricted to the same four experts and seed.}\label{tab:ablation}',
           r'\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lcccccccc}\toprule',
           r'& \multicolumn{4}{c}{RMSE} & \multicolumn{4}{c}{Score}\\\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
           r'Variant & FD001 & FD002 & FD003 & FD004 & FD001 & FD002 & FD003 & FD004\\\midrule']
    for lab, fr, fs in rows:
        if lab is None:
            out.append(r'\midrule'); continue
        out.append(lab + ' & ' + ' & '.join(pm(fr(s)) for s in SUBS) + ' & ' + ' & '.join(pm(fs(s), 1) for s in SUBS) + r'\\')
    out += [r'\bottomrule\end{tabular}\end{table*}']
    open(f'{P}/tab_ablation.tex', 'w').write('\n'.join(out))


def tab_ablation_n():
    N = S['ncmapss']
    rows = [('AIDGN rerun', lambda d: get(N[d], 'aidgn', 'rmse')),
            ('AIDGN rerun + TT', lambda d: get(N[d], 'aidgn_tt', 'tt_rmse')),
            (None, None)]
    for k, lab in [('aidgna', 'AIDGN-A'), ('cnn', '1D-CNN'), ('gru', 'GRU'), ('lgbm', 'LightGBM')]:
        rows.append((f'Expert: {lab}', (lambda k: lambda d: get(N[d], 'trace', k, 'raw_rmse'))(k)))
        rows.append((f'Expert: {lab} + TT', (lambda k: lambda d: get(N[d], 'trace', k, 'tt_rmse'))(k)))
    rows += [(None, None),
             ('Uniform mean, no TT', lambda d: get(N[d], 'trace', 'mean', 'raw_rmse')),
             ('Uniform mean + TT (amendment 1)', lambda d: get(N[d], 'trace', 'mean', 'tt_rmse')),
             ('NNLS stack, no TT', lambda d: get(N[d], 'trace', 'stack', 'raw_rmse')),
             ('\\textbf{TRACE} (NNLS stack + TT)', lambda d: get(N[d], 'headline', 'rmse')),
             (None, None),
             ('TabPFN v2 alone + TT (seed 0)', lambda d: get(N[d], 'tabpfn_ablation', 'tabpfn', 'tt_rmse')),
             ('Stack + TabPFN + TT (seed 0)', lambda d: get(N[d], 'tabpfn_ablation', 'stack', 'tt_rmse')),
             ('Stack + TT, same seed (reference)', lambda d: get(N[d], 'tabpfn_ablation', 'stack4', 'tt_rmse')),
             (None, None),
             ('3-fold protocol, mean + TT (seed 0)', lambda d: get(N[d], 'orig3fold', 'mean', 'tt_rmse')),
             ('3-fold protocol, stack + TT (seed 0)', lambda d: get(N[d], 'orig3fold', 'stack', 'tt_rmse')),
             (None, None),
             ('TRACE, seed 0 (reference)', lambda d: get(N[d], 'trace_noage', 'ref_stack', 'tt_rmse')),
             ('\\quad -- age channel', lambda d: get(N[d], 'trace_noage', 'stack', 'tt_rmse'))]
    out = [r'\begin{table*}[!t]\centering\caption{Ablation on N-CMAPSS (test RMSE, mean over seeds). The 3-fold rows are the protocol in force before amendment~2 (Section~\ref{sec:amend}); ``--'' marks sub-datasets it was not run on.}\label{tab:ablation_n}',
           r'\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lcccccccc}\toprule',
           r'Variant & DS01 & DS02 & DS03 & DS04 & DS05 & DS06 & DS07 & Avg\\\midrule']
    for lab, fr in rows:
        if lab is None:
            out.append(r'\midrule'); continue
        v = [mean_of(fr(d)) for d in DSS]
        avg = f'{np.mean(v):.2f}' if None not in v else '--'
        out.append(lab + ' & ' + ' & '.join('--' if x is None else f'{x:.2f}' for x in v) + f' & {avg}' + r'\\')
    out += [r'\bottomrule\end{tabular}\end{table*}']
    open(f'{P}/tab_ablation_n.tex', 'w').write('\n'.join(out))


def tab_sig():
    out = [r'\begin{table}[!t]\centering\caption{Paired bootstrap (2000 resamples of test engines, or of unit-cycle blocks on',
           r'N-CMAPSS) of the expected single-run difference TRACE $-$ AIDGN rerun over common seeds. Negative = TRACE better.}',
           r'\label{tab:sig}\scriptsize\setlength{\tabcolsep}{2.5pt}\begin{tabular}{lcccc}\toprule',
           r'Subset & $\Delta$RMSE & 95\% CI & $\Delta$Score & 95\% CI\\\midrule']
    for bench, keys, nd in [('cmapss', SUBS, 1), ('ncmapss', DSS, 2)]:
        for k in keys:
            g = S[bench][k].get('sig_vs_rerun')
            if g is None:
                out.append(f'{k} & -- & -- & -- & --\\\\'); continue
            out.append(f"{k} & {g['d_rmse']:.2f} & [{g['ci_rmse'][0]:.2f}, {g['ci_rmse'][1]:.2f}] & {g['d_score']:.{nd}f} & "
                       f"[{g['ci_score'][0]:.{nd}f}, {g['ci_score'][1]:.{nd}f}]\\\\")
        if bench == 'cmapss':
            out.append(r'\midrule')
    out += [r'\bottomrule\end{tabular}\end{table}']
    open(f'{P}/tab_sig.tex', 'w').write('\n'.join(out))


def tab_noise():
    C = S['cmapss']
    out = [r'\begin{table}[!t]\centering\caption{Robustness to high-pass Gaussian sensor noise at test time ($\alpha=0.1$ unless stated, protocol of',
           r'\cite{abbas2026aidgn}; clean training). RMSE; TRACE without TabPFN, seed 0.}\label{tab:noise}',
           r'\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lcccc}\toprule',
           r'Model / test input & FD001 & FD002 & FD003 & FD004\\\midrule']
    for lab, f in [('AIDGN printed, clean', lambda s: PR['CMAPSS']['AIDGN'][s]['rmse']),
                   ('AIDGN rerun, clean', lambda s: get(C[s], 'aidgn', 'rmse')),
                   ('AIDGN rerun, noisy', lambda s: get(C[s], 'aidgn_noise', 'rmse')),
                   ('AIDGN rerun, noisy ($\\alpha=0.5$)', lambda s: get(C[s], 'aidgn_noise05', 'rmse')),
                   ('TRACE, clean', lambda s: get(C[s], 'trace_noise', 'ref_stack4', 'tt_rmse')),
                   ('TRACE, noisy', lambda s: get(C[s], 'trace_noise', 'stack', 'tt_rmse')),
                   ('TRACE, noisy ($\\alpha=0.5$)', lambda s: get(C[s], 'trace_noise05', 'stack', 'tt_rmse'))]:
        out.append(lab + ' & ' + ' & '.join(pm(f(s), sd=False) for s in SUBS) + r'\\')
    out += [r'\bottomrule\end{tabular}\end{table}']
    open(f'{P}/tab_noise.tex', 'w').write('\n'.join(out))


def tab_eff():
    if not E:
        open(f'{P}/tab_eff.tex', 'w').write(r'\begin{table}[!t]\caption{Efficiency \pending{eff}}\label{tab:eff}\end{table}'); return
    o = S['cmapss']['FD003']
    rows = [('AIDGN-RUL', E['aidgn_params'], E['aidgn_cuda_ms'], E['aidgn_cpu_ms'], get(o, 'aidgn', 'rmse')),
            ('TRACE-S (student)', E['student_params'], E['student_cuda_ms'], E['student_cpu_ms'], get(o, 'student', 'tt_rmse')),
            ('TRACE networks (15 fwd)', sum(E['expert_params'].values()) * 5, E['nets5fold_cuda_ms'], E['nets5fold_cpu_ms'], None),
            ('+ LightGBM (5 folds)', None, None, E['lgbm5fold_cpu_ms'], None),
            ('+ TabPFN v2 (5 folds)', E.get('tabpfn_params'), E.get('tabpfn5fold_cuda_ms'), None, get(o, 'headline', 'rmse'))]
    out = [r'\begin{table}[!t]\centering\caption{Parameters and latency for the 100 FD003 test engines (mean of repeated passes;',
           r'GPU shared with other jobs).}\label{tab:eff}\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lcccc}\toprule',
           r'Model & Params & GPU ms & CPU ms & RMSE\\\midrule']
    f = lambda x, fmt: '--' if x is None else fmt.format(x)
    for lab, p, g, c, r in rows:
        out.append(f"{lab} & {f(p, '{:,}')} & {f(g, '{:.2f}')} & {f(c, '{:.2f}')} & {pm(r, sd=False)}\\\\")
    out += [r'\bottomrule\end{tabular}\end{table}']
    open(f'{P}/tab_eff.tex', 'w').write('\n'.join(out))


# ------------------------------------------------------------------ macros
def macros():
    C, N = S['cmapss'], S['ncmapss']
    rel_r, rel_s = [], []
    for s in SUBS:
        t = get(C[s], 'headline', 'rmse', 'mean'); ts = get(C[s], 'headline', 'score', 'mean')
        p = PR['CMAPSS']['AIDGN'][s]
        mac(f'Tr{L[s]}', t); mac(f'TrS{L[s]}', ts, '{:.1f}')
        mac(f'Pr{L[s]}', p['rmse']); mac(f'PrS{L[s]}', p['score'], '{:.1f}')
        mac(f'Re{L[s]}', get(C[s], 'aidgn', 'rmse', 'mean')); mac(f'ReS{L[s]}', get(C[s], 'aidgn', 'score', 'mean'), '{:.1f}')
        mac(f'Na{L[s]}', get(C[s], 'aidgn_noage', 'rmse', 'mean'))
        mac(f'St{L[s]}', get(C[s], 'student', 'tt_rmse', 'mean'))
        mac(f'Age{L[s]}', S['age_only'][s]['rmse'])
        mac(f'Gck{L[s]}', S['gate_ckpt'][s]['ours_rmse']); mac(f'Gz{L[s]}', S['gate_ckpt'][s]['age_zeroed_rmse'])
        if t is not None:
            rel_r.append(100 * (p['rmse'] - t) / p['rmse']); rel_s.append(100 * (p['score'] - ts) / p['score'])
            mac(f'Imp{L[s]}', rel_r[-1], '{:.1f}'); mac(f'ImpS{L[s]}', rel_s[-1], '{:.1f}')
        g = get(C[s], 'sig_vs_rerun')
        mac(f'Sig{L[s]}', g and g['d_rmse']); mac(f'SigLo{L[s]}', g and g['ci_rmse'][0]); mac(f'SigHi{L[s]}', g and g['ci_rmse'][1])
    mac('ImpCavg', float(np.mean(rel_r)) if len(rel_r) == 4 else None, '{:.1f}')
    mac('ImpCSavg', float(np.mean(rel_s)) if len(rel_s) == 4 else None, '{:.1f}')
    mac('PrCavg', float(np.mean([PR['CMAPSS']['AIDGN'][s]['rmse'] for s in SUBS])))
    tv = [get(C[s], 'headline', 'rmse', 'mean') for s in SUBS]
    mac('TrCavg', float(np.mean(tv)) if None not in tv else None)
    rn, rns, tn, tns, wins = [], [], [], [], 0
    for d in DSS:
        t = get(N[d], 'headline', 'rmse', 'mean'); ts = get(N[d], 'headline', 'score', 'mean')
        p = PR['NCMAPSS']['AIDGN'][d]
        mac(f'Tr{L[d]}', t); mac(f'TrS{L[d]}', ts); mac(f'Pr{L[d]}', p['rmse']); mac(f'PrS{L[d]}', p['score'])
        mac(f'Re{L[d]}', get(N[d], 'aidgn', 'rmse', 'mean')); mac(f'Age{L[d]}', S['age_only'][d]['rmse'])
        mac(f'AgeS{L[d]}', S['age_only'][d]['score'])
        mac(f'St{L[d]}', get(N[d], 'student', 'tt_rmse', 'mean'))
        mac(f'ReTT{L[d]}', get(N[d], 'aidgn_tt', 'tt_rmse', 'mean'))
        best_b = min(v[d]['rmse'] for k, v in PR['NCMAPSS'].items() if k != 'AIDGN')
        mac(f'Bb{L[d]}', best_b)
        if t is not None:
            tn.append(t); tns.append(ts); rn.append(p['rmse']); rns.append(p['score'])
            wins += int(t < min(v[d]['rmse'] for v in PR['NCMAPSS'].values()))
    rr = [get(N[d], 'aidgn', 'rmse', 'mean') for d in DSS]; rt = [get(N[d], 'aidgn_tt', 'tt_rmse', 'mean') for d in DSS]
    mac('ReNavg', float(np.mean(rr)) if None not in rr else None); mac('ReTTNavg', float(np.mean(rt)) if None not in rt else None)
    mac('TrNavg', float(np.mean(tn)) if len(tn) == 7 else None); mac('TrNSavg', float(np.mean(tns)) if len(tns) == 7 else None)
    mac('PrNavg', PR['NCMAPSS']['AIDGN']['Avg']['rmse']); mac('PrNSavg', PR['NCMAPSS']['AIDGN']['Avg']['score'])
    if len(tn) == 7:
        mac('ImpNavg', 100 * (PR['NCMAPSS']['AIDGN']['Avg']['rmse'] - np.mean(tn)) / PR['NCMAPSS']['AIDGN']['Avg']['rmse'], '{:.1f}')
        mac('ImpNSavg', 100 * (PR['NCMAPSS']['AIDGN']['Avg']['score'] - np.mean(tns)) / PR['NCMAPSS']['AIDGN']['Avg']['score'], '{:.1f}')
        mac('WinsN', str(wins))
    else:
        mac('ImpNavg', None); mac('ImpNSavg', None); mac('WinsN', None)
    ages = [S['age_only'][d]['rmse'] for d in DSS]
    mac('AgeBeatsBaselines', str(sum(S['age_only'][d]['rmse'] < min(v[d]['rmse'] for k, v in PR['NCMAPSS'].items() if k != 'AIDGN') for d in DSS)))
    cells = [(S['age_only'][d]['rmse'] < v[d]['rmse']) for k, v in PR['NCMAPSS'].items() if k != 'AIDGN' for d in DSS]
    mac('AgeBeatsCells', str(int(sum(cells)))); mac('AgeCells', str(len(cells)))
    mac('AgeBeatsAidgn', str(sum(S['age_only'][d]['score'] < PR['NCMAPSS']['AIDGN'][d]['score'] for d in DSS)))
    mac('AgeNavg', float(np.mean(ages))); mac('AgeNmin', float(min(ages))); mac('AgeNmax', float(max(ages)))
    agc = [S['age_only'][s]['rmse'] for s in SUBS]
    mac('AgeCmin', float(min(agc))); mac('AgeCmax', float(max(agc)))
    if E:
        mac('ParAidgn', f"{E['aidgn_params']:,}".replace(',', '{,}')); mac('ParStudent', f"{E['student_params']:,}".replace(',', '{,}'))
    else:
        mac('ParAidgn', None); mac('ParStudent', None)
    with open(f'{P}/numbers.tex', 'w') as f:
        f.write('% generated by code/make_tables.py -- do not edit\n')
        for k, v in MAC.items():
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')


if __name__ == '__main__':
    for fn in (tab_cmapss, tab_ncmapss, tab_gate, tab_ablation, tab_ablation_n, tab_sig, tab_noise, tab_eff, macros):
        fn()
    print('pending:', [k for k, v in MAC.items() if 'pending' in v])
