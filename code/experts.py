"""Expert pool of TRACE-RUL: three small networks, LightGBM and TabPFN v2.

Networks take causal windows (N, L, C) whose last channel is the age; they are trained on
RUL/cap with MSE, AdamW, early stopping on an inner engine-level validation split.
"""
import os
import numpy as np
import torch
from torch import nn
from aidgn import AIDGN

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TABPFN_CKPT = os.environ.get('TRACE_TABPFN_CKPT', os.path.expanduser('~/.cache/tabpfn/tabpfn-v2-regressor.ckpt'))


class AIDGNA(nn.Module):
    """AIDGN architecture with the age channel fed unscaled (eps = 1) and drift channels."""
    def __init__(self, c_in, L, hidden=64, aate=10):
        super().__init__()
        self.net = AIDGN(feature_num=c_in - 1, sequence_len=L, hidden_dim=hidden, AATE_dim=aate)

    def forward(self, w):
        return self.net(w[..., :-1], w[..., -1:])


class CNN1D(nn.Module):
    def __init__(self, c_in, L, width=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * width + 1, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, w):
        h = self.conv(w.transpose(1, 2))
        z = torch.cat([h.mean(-1), h[..., -1], w[:, -1, -1:]], 1)
        return self.head(z)


class GRUNet(nn.Module):
    def __init__(self, c_in, L, hidden=64):
        super().__init__()
        self.rnn = nn.GRU(c_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + 1, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, w):
        h, _ = self.rnn(w)
        return self.head(torch.cat([h[:, -1], w[:, -1, -1:]], 1))


NETS = {'aidgna': AIDGNA, 'cnn': CNN1D, 'gru': GRUNet}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def fit_net(kind, Wtr, ytr, Wva, yva, cap, seed, epochs=80, bs=256, lr=1e-3, patience=10,
            soft=None, return_model=False):
    """soft: optional teacher targets for distillation (same scale as y)."""
    torch.manual_seed(seed)
    L, C = Wtr.shape[1], Wtr.shape[2]
    m = NETS[kind](C, L).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=4)
    Xt = torch.tensor(Wtr, device=DEV)
    yt = torch.tensor((ytr if soft is None else soft) / cap, device=DEV, dtype=torch.float32).view(-1, 1)
    Xv = torch.tensor(Wva, device=DEV)
    yv = torch.tensor(yva, dtype=torch.float32).view(-1)
    g = torch.Generator(device='cpu').manual_seed(seed)
    best, bad, state = np.inf, 0, None
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), generator=g).to(DEV)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = ((m(Xt[b]) - yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
        pv = predict_net(m, Xv) * cap
        v = float(torch.sqrt(((torch.tensor(pv) - yv) ** 2).mean()))
        sched.step(v)
        if v < best - 1e-4:
            best, bad, state = v, 0, {k: x.detach().clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    return (m, best) if return_model else (lambda W: predict_net(m, torch.tensor(W, device=DEV)) * cap, best)


@torch.no_grad()
def predict_net(m, X, bs=4096):
    m.eval()
    return torch.cat([m(X[i:i + bs]).float().cpu() for i in range(0, len(X), bs)]).numpy().ravel()


def fit_lgbm(Ftr, ytr, Fva, yva, seed):
    import lightgbm as lgb
    r = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=seed, n_jobs=4, verbose=-1)
    r.fit(Ftr, ytr, eval_set=[(Fva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return r.predict, float(np.sqrt(np.mean((r.predict(Fva) - yva) ** 2)))


def fit_tabpfn(Ftr, ytr, Fva, yva, seed, n_ctx=4000, n_estimators=2):
    from tabpfn import TabPFNRegressor
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Ftr), size=min(n_ctx, len(Ftr)), replace=False)
    r = TabPFNRegressor(model_path=TABPFN_CKPT, device='cuda', random_state=seed, n_estimators=n_estimators,
                        ignore_pretraining_limits=True)
    r.fit(Ftr[idx], ytr[idx])

    def pred(F, bs=5000):
        return np.concatenate([r.predict(F[i:i + bs]) for i in range(0, len(F), bs)])
    return pred, float(np.sqrt(np.mean((pred(Fva) - yva) ** 2)))


def fit_net_fixed(kind, Wtr, ytr, cap, seed, epochs=40, bs=512, lr=1e-3, soft=None, return_model=False):
    """Fixed training budget with cosine decay and no validation split (N-CMAPSS leave-one-unit-out mode,
    where every development unit is needed for training)."""
    torch.manual_seed(seed)
    L, C = Wtr.shape[1], Wtr.shape[2]
    m = NETS[kind](C, L).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(Wtr, device=DEV)
    yt = torch.tensor((ytr if soft is None else soft) / cap, device=DEV, dtype=torch.float32).view(-1, 1)
    steps = epochs * ((len(Xt) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator(device='cpu').manual_seed(seed)
    m.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g).to(DEV)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = ((m(Xt[b]) - yt[b]) ** 2).mean()
            loss.backward(); opt.step(); sched.step()
    if return_model:
        return m, float('nan')
    return lambda W: predict_net(m, torch.tensor(W, device=DEV)) * cap


def fit_lgbm_fixed(Ftr, ytr, seed, n_estimators=800):
    import lightgbm as lgb
    r = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=seed, n_jobs=4, verbose=-1).fit(Ftr, ytr)
    return r.predict
