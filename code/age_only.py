"""How much does the elapsed cycle count alone explain?  Predictor: RUL_hat = clip(mean train life - cycle, 0, cap),
no sensor at all.  C-MAPSS: last cycle of each test engine (summed Score); N-CMAPSS: every test window (mean Score)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmapss import load_frames, rmse, score
from ncmapss import build
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
out = {}
for sub in ['FD001', 'FD002', 'FD003', 'FD004']:
    tr, te, ml, _, _ = load_frames(sub)
    last = te.groupby('engine_id').tail(1)
    p = np.clip(ml - last.cycle.values, 0, 125)
    out[sub] = dict(rmse=rmse(p, last.RUL.values), score=score(p, last.RUL.values), mean_life=ml)
for ds in ['DS01', 'DS02', 'DS03', 'DS04', 'DS05', 'DS06', 'DS07']:
    d = build(ds)
    p = np.clip(d['mean_life'] - d['cyc_te'][:, -1], 0, float(d['max_life']))
    lives = {int(u): float(d['cyc_te'][d['unit_te'] == u, -1].max()) for u in np.unique(d['unit_te'])}
    out[ds] = dict(rmse=rmse(p, d['y_te']), score=score(p, d['y_te'], mean=True), mean_train_life=float(d['mean_life']),
                   test_unit_lives=lives)
for k, v in out.items():
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
json.dump(out, open(f'{ROOT}/results/age_only.json', 'w'), indent=1)
