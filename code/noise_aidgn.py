"""Noise robustness (setting 2 of the reference, clean training + noisy test, alpha=0.1) for the
faithful AIDGN reruns: reload each saved checkpoint and score it on noisy test windows."""
import json, os, sys, glob
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmapss import load_official, rmse, score
from aidgn import AIDGN

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
out = {}
for pt in sorted(glob.glob(f'{ROOT}/results/runs/aidgn/FD00*_s*.pt')):
    sub, seed = os.path.basename(pt)[:5], int(pt.split('_s')[-1][:-3])
    d = load_official(sub, noise_test=ALPHA, noise_seed=seed)
    m = AIDGN(); m.load_state_dict(torch.load(pt, map_location='cpu')); m.eval()
    with torch.no_grad():
        p = m(torch.tensor(d['Xte']), torch.tensor(d['Ite'])).numpy().ravel() * 125
    out[f'{sub}_s{seed}'] = dict(rmse=rmse(p, d['yte']), score=score(p, d['yte']))
    print(sub, seed, out[f'{sub}_s{seed}'])
json.dump(out, open(f'{ROOT}/results/noise_aidgn_a{ALPHA}.json', 'w'), indent=1)
