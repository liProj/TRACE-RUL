"""Stacking and trajectory-consistent tracking (TT) of TRACE-RUL.

TT: below the cap a running engine loses exactly one cycle of RUL per cycle, so every causal
prediction p_s made at cycle c_s with p_s < tau is an end-of-life estimate c_s + p_s.  At cycle
c_T we fuse the informative estimates with exponential forgetting gamma^(c_T - c_s) and blend
with the current prediction:  r_T = beta * p_T + (1 - beta) * clip(EOL_hat - c_T, 0, cap).
The recursion only uses s <= T, so it is causal.  (gamma, beta, tau) are chosen on the
out-of-fold trajectories of the training engines, never on the test set.
"""
import itertools
import numpy as np
from scipy.optimize import nnls

GAMMAS = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]
BETAS = [round(0.1 * i, 1) for i in range(11)]
TAU_FRAC = [0.8, 0.88, 0.92, 0.96, 1.01]   # tau = frac * cap ; 1.01 keeps every prediction


def tt_filter(p, c, unit, gamma, beta, tau, cap):
    """p, c, unit: flat arrays ordered by (unit, time).  Returns filtered predictions."""
    out = np.empty_like(p, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(unit)) + 1, len(p)]
    for a, b in zip(starts[:-1], starts[1:]):
        num = den = 0.0
        prev = c[a]
        for i in range(a, b):
            dec = gamma ** (c[i] - prev) if gamma > 0 else (1.0 if c[i] == prev else 0.0)
            num *= dec; den *= dec
            prev = c[i]
            if p[i] < tau:
                num += c[i] + p[i]; den += 1.0
            if den > 0:
                out[i] = beta * p[i] + (1 - beta) * min(max(num / den - c[i], 0.0), cap)
            else:
                out[i] = p[i]
    return out


def tt_filter_fast(p, c, unit, gamma, beta, tau, cap):
    try:
        from numba import njit  # noqa
    except ImportError:
        return tt_filter(p, c, unit, gamma, beta, tau, cap)
    return _tt_numba(p.astype(np.float64), c.astype(np.float64), unit.astype(np.int64), gamma, beta, tau, cap)


try:
    from numba import njit

    @njit(cache=True)
    def _tt_numba(p, c, unit, gamma, beta, tau, cap):
        out = np.empty_like(p)
        num = 0.0; den = 0.0; prev = c[0]
        for i in range(len(p)):
            if i == 0 or unit[i] != unit[i - 1]:
                num = 0.0; den = 0.0; prev = c[i]
            if gamma > 0:
                dec = gamma ** (c[i] - prev)
            else:
                dec = 1.0 if c[i] == prev else 0.0
            num *= dec; den *= dec
            prev = c[i]
            if p[i] < tau:
                num += c[i] + p[i]; den += 1.0
            if den > 0:
                r = num / den - c[i]
                r = min(max(r, 0.0), cap)
                out[i] = beta * p[i] + (1 - beta) * r
            else:
                out[i] = p[i]
        return out
except ImportError:
    pass


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def fit_tt(p, c, unit, y, mask, cap):
    """Grid search on OOF trajectories; mask selects the evaluation cut points."""
    best = (rmse(p[mask], y[mask]), 1.0, 1.0, 1.01 * cap)  # identity
    for g, b, tf in itertools.product(GAMMAS, BETAS, TAU_FRAC):
        if b == 1.0:
            continue
        f = tt_filter_fast(p, c, unit, g, b, tf * cap, cap)
        r = rmse(f[mask], y[mask])
        if r < best[0]:
            best = (r, g, b, tf * cap)
    return dict(oof_rmse=best[0], gamma=best[1], beta=best[2], tau=best[3])


def fit_stack(P, y, mask):
    """Non-negative least squares on OOF predictions (no intercept), weights renormalised."""
    w, _ = nnls(P[mask], y[mask])
    if w.sum() <= 0:
        w = np.ones(P.shape[1])
    return w / w.sum()
