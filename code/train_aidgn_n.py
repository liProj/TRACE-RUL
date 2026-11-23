"""Faithful AIDGN on N-CMAPSS (release settings: AATE 16, d_model 128, 20 channels, window 50,
stride 10, lr DS01 5e-3 / DS02 5e-4 / others 1e-3, batch 128, MSE on RUL/max_life, early stopping 10,
LR x0.3 after 10 bad epochs, window-level random validation).  Test = every window of the test units,
RMSE and MEAN Score.  Also: --ckpt evaluates the released checkpoint instead of training.

usage: train_aidgn_n.py DS01 SEED [--tag aidgn_n] [--epochs 300] [--ckpt]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch import nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncmapss import build, official_split
from aidgn import AIDGN
from cmapss import rmse, score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LR = {'DS01': 5e-3, 'DS02': 5e-4}


@torch.no_grad()
def predict(m, X, I, dev, bs=4096):
    m.eval()
    return torch.cat([m(torch.tensor(X[i:i + bs], device=dev), torch.tensor(I[i:i + bs], device=dev)).cpu()
                      for i in range(0, len(X), bs)]).numpy().ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ds'); ap.add_argument('seed', type=int)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--tag', default='aidgn_n')
    ap.add_argument('--ckpt', action='store_true')
    a = ap.parse_args()
    dev = torch.device('cuda')
    d = build(a.ds)
    ml = float(d['max_life'])
    Ite = d['cyc_te'][..., None].astype(np.float32) * 1e-5
    m = AIDGN(feature_num=20, sequence_len=50, hidden_dim=128, AATE_dim=16).to(dev)
    od = f'{ROOT}/results/runs/{a.tag}'
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    if a.ckpt:
        sd = torch.load(f'{ROOT}/repo/AIDGN-RUL/logs/{a.ds}/AIGCN/exp0/best_checkpoint.pth', map_location=dev)
        m.load_state_dict(sd, strict=True)
        p = predict(m, d['X_te'], Ite, dev) * ml
        rel = np.load(f'{ROOT}/repo/AIDGN-RUL/logs/{a.ds}/AIGCN/exp0/result.npz')
        res = dict(ds=a.ds, rmse=rmse(p, d['y_te']), score_mean=score(p, d['y_te'], mean=True), n_windows=len(p),
                   released_n=int(rel['test_preds'].size), released_rmse=rmse(rel['test_preds'], rel['test_trues']),
                   released_score_mean=score(rel['test_preds'], rel['test_trues'], mean=True),
                   label_multiset_equal=bool(np.allclose(np.sort(rel['test_trues'].ravel()), np.sort(d['y_te']))))
        json.dump(res, open(f'{od}/{a.ds}_ckpt.json', 'w'))
        print(json.dumps(res))
        return
    Xtr, Xva, Itr, Iva, ytr, yva = official_split(d)
    T = lambda z: torch.tensor(z, device=dev)
    Xtr, Xva, Itr, Iva = T(Xtr), T(Xva), T(Itr.astype(np.float32)), T(Iva.astype(np.float32))
    ytr, yva = T(ytr[:, None] / ml), T(yva[:, None] / ml)
    torch.manual_seed(a.seed)
    m = AIDGN(feature_num=20, sequence_len=50, hidden_dim=128, AATE_dim=16).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=LR.get(a.ds, 1e-3))
    crit = nn.MSELoss()
    g = torch.Generator().manual_seed(a.seed)
    best, bad, lr_best, lr_bad, state = np.inf, 0, np.inf, 0, None
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(len(Xtr), generator=g).to(dev)
        for i in range(0, len(perm) - len(perm) % 128, 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = crit(m(Xtr[b], Itr[b]), ytr[b])
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = float(np.mean([crit(m(Xva[i:i + 128], Iva[i:i + 128]), yva[i:i + 128]).item()
                               for i in range(0, len(Xva) - len(Xva) % 128, 128)]))
        if v < best:
            best, bad, state = v, 0, {k: x.detach().clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
        if lr_best == np.inf:
            lr_best, lr_bad = v, 1
        elif v >= lr_best:
            lr_bad += 1
        else:
            lr_best, lr_bad = v, 0
        if lr_bad == 10:
            for pg in opt.param_groups:
                pg['lr'] *= 0.3
            lr_bad = 0
        if bad >= 10:
            break
    m.load_state_dict(state)
    p = predict(m, d['X_te'], Ite, dev) * ml
    res = dict(ds=a.ds, seed=a.seed, rmse=rmse(p, d['y_te']), score_mean=score(p, d['y_te'], mean=True),
               epochs=ep + 1, minutes=(time.time() - t0) / 60)
    np.savez(f'{od}/{a.ds}_s{a.seed}.npz', pred=p, true=d['y_te'], unit=d['unit_te'], cyc=d['cyc_te'][:, -1])
    torch.save(state, f'{od}/{a.ds}_s{a.seed}.pt')
    json.dump(res, open(f'{od}/{a.ds}_s{a.seed}.json', 'w'))
    print(json.dumps(res))


if __name__ == '__main__':
    main()
