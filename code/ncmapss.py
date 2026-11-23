"""N-CMAPSS loading, re-implementing N_CMAPSS_Related/{N_CMAPSS_create_sample,sample_creator_auto,
N_CMAPSS_load_data}.py of the AIDGN-RUL release:
  * df_all = [W (4), X_s (14), T40, P30, RUL, unit, cycle] of dev+test rows, then every 10th row (0.1 Hz),
  * RUL clipped at int(0.8 * max RUL of the file),
  * units split by File_DevUnits_TestUnits.csv,
  * MinMaxScaler(-1, 1) fitted on the training rows of the 20 channels (sorted column order, as
    pandas Index.difference returns it), applied to the test rows,
  * windows of 50 sampled rows with stride 10 inside each unit, label = RUL of the last row,
    aging prior = cycle of every row * 1e-5,
  * random 10% window-level validation split (random_state=42), test = every window of the test units.
Cached per file as data/ncmapss/cache/<DS>.npz.
"""
import os, glob
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import MinMaxScaler

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
H5DIR = os.environ.get('TRACE_NCMAPSS_DIR', f'{ROOT}/data/ncmapss/h5')
CACHE = f'{ROOT}/data/ncmapss/cache'
UNITS = {  # File_DevUnits_TestUnits.csv of the release
    'DS01': ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10]), 'DS02': ([2, 5, 10, 16, 18, 20], [11, 14, 15]),
    'DS03': ([1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15]), 'DS04': ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10]),
    'DS05': ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10]), 'DS06': ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10]),
    'DS07': ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10])}
L, STRIDE, SAMPLING, RATE = 50, 10, 10, 0.8


def h5_path(ds):
    return [p for p in glob.glob(f'{H5DIR}/**/*.h5', recursive=True) if f'_{ds}' in os.path.basename(p)][0]


def frame(ds):
    with h5py.File(h5_path(ds), 'r') as f:
        g = lambda k: np.concatenate([np.array(f[f'{k}_dev']), np.array(f[f'{k}_test'])])
        W, Xs, Xv, Y, A = g('W'), g('X_s'), g('X_v'), g('Y'), g('A')
        wv = [x.decode() if isinstance(x, bytes) else str(x) for x in np.array(f['W_var'])]
        xv = [x.decode() if isinstance(x, bytes) else str(x) for x in np.array(f['X_s_var'])]
        av = [x.decode() if isinstance(x, bytes) else str(x) for x in np.array(f['A_var'])]
    df = pd.concat([pd.DataFrame(W, columns=wv), pd.DataFrame(Xs, columns=xv),
                    pd.DataFrame(Xv[:, 0:2], columns=['T40', 'P30']), pd.DataFrame(Y, columns=['RUL']),
                    pd.DataFrame(A, columns=av).drop(columns=['Fc', 'hs'])], axis=1)
    return df[::SAMPLING]


def build(ds):
    os.makedirs(CACHE, exist_ok=True)
    cp = f'{CACHE}/{ds}.npz'
    if os.path.exists(cp):
        return dict(np.load(cp, allow_pickle=True))
    df = frame(ds)
    max_life = int(RATE * df['RUL'].max())
    df['RUL'] = df['RUL'].clip(upper=max_life)
    dev, test = UNITS[ds]
    tr = pd.concat([df[df.unit == float(u)] for u in dev]).reset_index(drop=True)
    te = pd.concat([df[df.unit == float(u)] for u in test]).reset_index(drop=True)
    cols = tr.columns.difference(['RUL', 'unit', 'cycle'])
    sc = MinMaxScaler(feature_range=(-1, 1)).fit(tr[cols])
    tr[cols] = sc.transform(tr[cols]); te[cols] = sc.transform(te[cols])
    out = dict(cols=np.array(list(cols)), max_life=max_life)
    for name, d, units in (('tr', tr, dev), ('te', te, test)):
        X, Y, C, U, ROW = [], [], [], [], []
        for u in units:
            du = d[d.unit == float(u)]
            x, y, c = du[cols].values.astype(np.float32), du.RUL.values.astype(np.float32), du.cycle.values.astype(np.float32)
            n = int((len(du) - L) / STRIDE) + 1
            st = np.arange(n) * STRIDE
            X.append(np.stack([x[s:s + L] for s in st])); Y.append(y[st + L - 1])
            C.append(np.stack([c[s:s + L] for s in st])); U.append(np.full(n, u)); ROW.append(st + L - 1)
        out.update({f'X_{name}': np.concatenate(X), f'y_{name}': np.concatenate(Y), f'cyc_{name}': np.concatenate(C),
                    f'unit_{name}': np.concatenate(U), f'row_{name}': np.concatenate(ROW)})
    # mean life of the training units (for our unscaled age channel)
    out['mean_life'] = float(np.mean([tr[tr.unit == float(u)].cycle.max() for u in dev]))
    np.savez(cp, **out)
    return out


def official_split(d):
    """window-level random 10% validation exactly as in the release (random_state=42)."""
    from sklearn.model_selection import train_test_split
    I = d['cyc_tr'][..., None] * 1e-5
    return train_test_split(d['X_tr'], I, d['y_tr'], test_size=0.1, random_state=42)
