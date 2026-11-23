"""Faithful AIDGN training loop (Experiment/Experiment.py of the release):
Adam(lr), MSE on RUL/125, batch 128 with drop_last, window-level random 10% validation,
early stopping (patience 10) on validation MSE, LR x0.3 after 10 non-improving epochs,
best-validation checkpoint evaluated on the last window of each test engine.

usage: train_aidgn.py SUB SEED [--epochs 300] [--tag aidgn] [--no_age]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch import nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmapss import load_official, rmse, score
from aidgn import AIDGN

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LR = {'FD001': 1e-3, 'FD002': 1e-3, 'FD003': 1e-3, 'FD004': 1e-3}  # configs/hparams.py


def batches(n, bs, shuffle, drop_last, g=None):
    idx = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
    stop = n - n % bs if drop_last else n
    for i in range(0, stop, bs):
        yield idx[i:i + bs]


def predict(model, X, I, dev, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(X[i:i + bs].to(dev), I[i:i + bs].to(dev)).cpu())
    return torch.cat(out).numpy().ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sub'); ap.add_argument('seed', type=int)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--tag', default='aidgn')
    ap.add_argument('--no_age', action='store_true')
    a = ap.parse_args()
    dev = torch.device('cuda')
    d = load_official(a.sub)
    T = {k: torch.tensor(v) for k, v in d.items() if isinstance(v, np.ndarray) and v.dtype == np.float32}
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = AIDGN(use_age=not a.no_age).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR[a.sub])
    crit = nn.MSELoss()
    g = torch.Generator().manual_seed(a.seed)
    best, bad, lr_bad, lr_best, state, hist = np.inf, 0, 0, np.inf, None, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tl = []
        for b in batches(len(T['Xtr']), 128, True, True, g):
            opt.zero_grad()
            out = model(T['Xtr'][b].to(dev), T['Itr'][b].to(dev))
            loss = crit(out, T['ytr'][b].to(dev) / 125)
            loss.backward(); opt.step(); tl.append(loss.item())
        model.eval()
        vl = []
        with torch.no_grad():
            for b in batches(len(T['Xva']), 128, False, True):
                vl.append(crit(model(T['Xva'][b].to(dev), T['Iva'][b].to(dev)), T['yva'][b].to(dev) / 125).item())
        v = float(np.mean(vl)); hist.append((float(np.mean(tl)), v))
        if v < best:
            best, bad, state = v, 0, {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
        # LR adapter of the release (counter also increments on the first epoch)
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
    model.load_state_dict(state)
    p = predict(model, T['Xte'], T['Ite'], dev) * 125
    y = d['yte'].ravel()
    res = dict(sub=a.sub, seed=a.seed, tag=a.tag, rmse=rmse(p, y), score=score(p, y), epochs=ep + 1,
               best_val=best, minutes=(time.time() - t0) / 60)
    od = f'{ROOT}/results/runs/{a.tag}'
    os.makedirs(od, exist_ok=True)
    np.savez(f'{od}/{a.sub}_s{a.seed}.npz', pred=p, true=y, ids=d['test_ids'], hist=np.array(hist))
    json.dump(res, open(f'{od}/{a.sub}_s{a.seed}.json', 'w'))
    torch.save(state, f'{od}/{a.sub}_s{a.seed}.pt')
    print(json.dumps(res))


if __name__ == '__main__':
    main()
