"""Gate step 2: load the released AIDGN checkpoints into our re-implementation and
score them on the test set built by our re-implemented pipeline. Compare with
the released result.npz and with the printed Table II."""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmapss import load_official, rmse, score
from aidgn import AIDGN

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PRINTED = {'FD001': (11.46, 248.49), 'FD002': (12.63, 663.80), 'FD003': (10.52, 202.87), 'FD004': (13.03, 762.25)}
out = {}
for sub in ['FD001', 'FD002', 'FD003', 'FD004']:
    d = load_official(sub)
    m = AIDGN()
    sd = torch.load(f'{ROOT}/repo/AIDGN-RUL/logs/{sub}/AIDGN/exp0/best_checkpoint.pth', map_location='cpu')
    m.load_state_dict(sd, strict=True)
    m.eval()
    with torch.no_grad():
        p = m(torch.tensor(d['Xte']), torch.tensor(d['Ite'])).numpy().ravel() * 125
        p0 = AIDGN(use_age=False)
        p0.load_state_dict(sd); p0.eval()
        pz = p0(torch.tensor(d['Xte']), torch.tensor(d['Ite'])).numpy().ravel() * 125
    rel = np.load(f'{ROOT}/repo/AIDGN-RUL/logs/{sub}/AIDGN/exp0/result.npz')
    rp, rg = rel['test_preds'].ravel(), rel['test_trues'].ravel()
    out[sub] = dict(ours_rmse=rmse(p, d['yte']), ours_score=score(p, d['yte']),
                    released_npz_rmse=rmse(rp, rg), released_npz_score=score(rp, rg),
                    max_abs_pred_diff_vs_npz=float(np.abs(p - rp).max()),
                    labels_identical=bool(np.allclose(rg, d['yte'].ravel())),
                    age_zeroed_rmse=rmse(pz, d['yte']), age_zeroed_score=score(pz, d['yte']),
                    printed_rmse=PRINTED[sub][0], printed_score=PRINTED[sub][1])
    print(sub, json.dumps(out[sub]))
os.makedirs(f'{ROOT}/results', exist_ok=True)
json.dump(out, open(f'{ROOT}/results/gate_ckpt.json', 'w'), indent=1)
