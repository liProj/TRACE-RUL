"""RQ4: parameters and inference latency on the FD003 test set (100 engines, last window each), mirroring the
reference protocol (mean over 500 forward passes, data loading excluded).  The GPU is shared with other jobs,
so we also report CPU (4 threads) latency, which is less affected by contention.
TRACE (full) = 5 folds x {AIDGN-A, CNN, GRU} forward passes + 5 x LightGBM + 5 x TabPFN predictions."""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experts as E
from aidgn import AIDGN
from trace_data import load

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
torch.set_num_threads(4)
tr, te, _ = load('FD003')
W = te.windows(True)
li = np.cumsum(te.n) - 1
Wl = torch.tensor(W[li])
F = te.features()[li]


def lat(fn, reps=500):
    for _ in range(20):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


out = {}
for dev in ['cuda', 'cpu']:
    a = AIDGN().to(dev).eval()
    x, occ = Wl[..., :14].to(dev), (Wl[..., -1:] * 1e-3).to(dev)
    s = E.CNN1D(29, 50).to(dev).eval()
    nets = [E.AIDGNA(29, 50).to(dev).eval(), E.CNN1D(29, 50).to(dev).eval(), E.GRUNet(29, 50).to(dev).eval()]
    w = Wl.to(dev)
    with torch.no_grad():
        out[f'aidgn_{dev}_ms'] = lat(lambda: a(x, occ))
        out[f'student_{dev}_ms'] = lat(lambda: s(w))
        out[f'nets5fold_{dev}_ms'] = lat(lambda: [n(w) for n in nets for _ in range(5)], reps=100)
out['aidgn_params'] = E.n_params(AIDGN())
out['student_params'] = E.n_params(E.CNN1D(29, 50))
out['expert_params'] = {k: E.n_params(c(29, 50)) for k, c in E.NETS.items()}
# tree + foundation-model experts (CPU / GPU wall time for one prediction call on the 100 rows, 5 folds)
import lightgbm as lgb
y = tr.targets()
Ftr = tr.features()
g = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, n_jobs=4, verbose=-1).fit(Ftr, y)
out['lgbm5fold_cpu_ms'] = lat(lambda: [g.predict(F) for _ in range(5)], reps=50)
out['lgbm_trees'] = 600
try:
    from tabpfn import TabPFNRegressor
    r = TabPFNRegressor(model_path=E.TABPFN_CKPT, device='cuda', n_estimators=2, ignore_pretraining_limits=True)
    idx = np.random.default_rng(0).choice(len(Ftr), 4000, replace=False)
    r.fit(Ftr[idx], y[idx])
    out['tabpfn5fold_cuda_ms'] = lat(lambda: [r.predict(F) for _ in range(5)], reps=5)
    out['tabpfn_params'] = int(sum(p.numel() for p in r.model_.parameters())) if hasattr(r, 'model_') else None
except Exception as e:  # pragma: no cover
    out['tabpfn_error'] = str(e)
json.dump(out, open(f'{ROOT}/results/efficiency.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
