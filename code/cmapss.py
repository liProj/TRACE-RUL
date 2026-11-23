"""C-MAPSS loading.

`load_official` re-implements get_cmapss_data_ of the AIDGN-RUL release
(CMAPSS_Related/load_data_CMAPSS.py) line by line: piecewise RUL capped at 125,
drop of the 7 constant sensors + settings 2/3, per-operating-condition min-max
scaling fitted on train (setting1 rounded to 1 decimal), windows of length L,
random 10% window-level validation split (random_state=42), last window per
test engine with front padding, and the aging prior cycle/mean_train_life*eps.

`load_frames` returns the same normalised frames per engine (full trajectories)
for our own models, which use the whole monitoring history.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATA = os.environ.get('TRACE_CMAPSS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'repo', 'AIDGN-RUL', 'CMAPSS'))
COLS = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + ['s%d' % i for i in range(1, 22)]
DROP = ['setting2', 'setting3', 's1', 's5', 's6', 's10', 's16', 's18', 's19']
SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']


def _read(kind, sub):
    df = pd.read_csv(f'{DATA}/{kind}_{sub}.txt', header=None, sep=r'\s+')
    if kind != 'RUL':
        df.columns = COLS
    return df


def _kink(n_rows, max_cycle, maxlife):
    knee = max_cycle - maxlife
    return [maxlife if i < knee else max_cycle - i - 1 for i in range(n_rows)]


def _frames(sub, maxlife=125):
    tr, te, rul = _read('train', sub), _read('test', sub), _read('RUL', sub)
    lab, lives = [], []
    for e in sorted(set(tr.engine_id)):
        d = tr[tr.engine_id == e]
        mc = d.cycle.max()
        lab += _kink(len(d), mc, maxlife)
        lives.append(mc)
    tr['RUL'] = lab
    lab = []
    for e in sorted(set(te.engine_id)):
        d = te[te.engine_id == e]
        mc = d.cycle.max() + int(rul.iloc[e - 1, 0])
        lab += _kink(len(d), mc, maxlife)
    te['RUL'] = lab
    tr, te = tr.drop(columns=DROP), te.drop(columns=DROP)
    tr['setting1'] = tr['setting1'].round(1)
    te['setting1'] = te['setting1'].round(1)
    # per-operating-condition min-max scaling, statistics from train only
    tr_n, te_n = tr.copy(), te.copy()
    for key, g in tr.groupby('setting1'):
        sc = MinMaxScaler().fit(g[SENSORS + ['RUL']])
        tr_n.loc[g.index, SENSORS] = sc.transform(g[SENSORS + ['RUL']])[:, :len(SENSORS)]
        gt = te[te.setting1 == key]
        if len(gt):
            te_n.loc[gt.index, SENSORS] = sc.transform(gt[SENSORS + ['RUL']])[:, :len(SENSORS)]
    return tr_n, te_n, float(np.mean(lives)), tr, te


def load_frames(sub, maxlife=125):
    """Normalised frames with columns engine_id, cycle, setting1, 14 sensors, RUL (+ raw copies)."""
    tr_n, te_n, mean_life, tr_raw, te_raw = _frames(sub, maxlife)
    return tr_n, te_n, mean_life, tr_raw, te_raw


def load_official(sub, L=50, maxlife=125, eps=1e-3, validation=0.1, noise_test=0.0, noise_seed=0):
    tr, te, mean_life, _, _ = _frames(sub, maxlife)
    if noise_test > 0:
        from trace_data import hp_noise
        te = hp_noise(te, noise_test, noise_seed)
    tr, te = tr.drop(columns='setting1'), te.drop(columns='setting1')
    X, Y, I = [], [], []
    for e in sorted(set(tr.engine_id)):
        d = tr[tr.engine_id == e]
        s, c, r = d[SENSORS].values.astype(np.float32), d.cycle.values, d.RUL.values
        for st in range(len(d) - L + 1):
            X.append(s[st:st + L]); Y.append(r[st + L - 1]); I.append(c[st:st + L])
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)[:, None]
    I = (np.asarray(I, np.float32)[..., None] / mean_life) * eps
    Xtr, Xva, ytr, yva, Itr, Iva = train_test_split(X, Y, I, test_size=validation, random_state=42)
    Xt, Yt, It, ids = [], [], [], []
    for e in sorted(set(te.engine_id)):
        d = te[te.engine_id == e]
        if len(d) < L:  # front padding with the first row, as in the release
            d = pd.concat([d.iloc[[0]]] * (L - len(d)) + [d])
        d = d.iloc[-L:]
        Xt.append(d[SENSORS].values); Yt.append(d.RUL.values[-1]); It.append(d.cycle.values); ids.append(e)
    Xt = np.asarray(Xt, np.float32)
    Yt = np.asarray(Yt, np.float32)[:, None]
    It = (np.asarray(It, np.float32)[..., None] / mean_life) * eps
    return dict(Xtr=Xtr, ytr=ytr, Itr=Itr, Xva=Xva, yva=yva, Iva=Iva, Xte=Xt, yte=Yt, Ite=It,
                test_ids=np.array(ids), mean_life=mean_life)


def rmse(p, g):
    p, g = np.ravel(p), np.ravel(g)
    return float(np.sqrt(np.mean((p - g) ** 2)))


def score(p, g, mean=False):
    d = np.ravel(p) - np.ravel(g)
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(s.mean() if mean else s.sum())
