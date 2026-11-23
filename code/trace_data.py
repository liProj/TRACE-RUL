"""Engine-level data for TRACE-RUL.

Every engine is a sequence of normalised channels (same 14 sensors and the same per-condition
min-max scaling as the reference pipeline).  For every cycle t of every engine we build one
causal sample: the window of the last L cycles (front-padded with the first row, as the
reference does for short test engines), plus
  * age  a_t = cycle / mean training life (the reference uses the same quantity times 1e-3),
  * drift: channels minus the mean of the first K cycles of the same engine (self-referencing).
The target is the piecewise-linear RUL capped at 125 (identical to the reference labels).
For the C-MAPSS test set the reference evaluates only the last cycle of each engine; we keep
all cycles so that the trajectory step can use the monitored history (still causal).
"""
import numpy as np
from cmapss import load_frames, SENSORS

K_BASE = 20


class Engines:
    def __init__(self, frame, mean_life, L=50, cap=125.0, chans=SENSORS):
        self.L, self.cap = L, cap
        self.ids, self.X, self.age, self.y, self.cyc, self.drift = [], [], [], [], [], []
        for e, d in frame.groupby('engine_id', sort=True):
            x = d[chans].values.astype(np.float32)
            self.ids.append(e)
            self.X.append(x)
            self.cyc.append(d.cycle.values.astype(np.float32))
            self.age.append(d.cycle.values.astype(np.float32) / mean_life)
            self.y.append(np.minimum(d.RUL.values.astype(np.float32), cap))
            self.drift.append(x - x[:K_BASE].mean(0, keepdims=True))
        self.ids = np.array(self.ids)
        self.n = np.array([len(x) for x in self.X])
        self.offset = np.concatenate([[0], np.cumsum(self.n)])

    # flat sample index <-> (engine, t)
    def sample_engine(self):
        return np.repeat(np.arange(len(self.ids)), self.n)

    def sample_t(self):
        return np.concatenate([np.arange(n) for n in self.n])

    def targets(self):
        return np.concatenate(self.y)

    def ages(self):
        return np.concatenate(self.age)

    def windows(self, use_drift=True, engines=None):
        """(N, L, C) causal windows ending at every cycle, front padded; channels = sensors [+ drift] + age."""
        engines = range(len(self.ids)) if engines is None else engines
        out = []
        for i in engines:
            parts = [self.X[i]] + ([self.drift[i]] if use_drift else []) + [self.age[i][:, None]]
            z = np.concatenate(parts, 1)
            zp = np.concatenate([np.repeat(z[:1], self.L - 1, 0), z], 0)
            idx = np.arange(len(z))[:, None] + np.arange(self.L)[None, :]
            w = zp[idx]
            # the age channel of padded rows repeats the first cycle, exactly like the reference padding
            out.append(w)
        return np.concatenate(out, 0).astype(np.float32)

    def features(self, engines=None):
        """Tabular summary of each causal window for the tree / TabPFN experts."""
        engines = range(len(self.ids)) if engines is None else engines
        out = []
        for i in engines:
            x, dr, a = self.X[i], self.drift[i], self.age[i]
            n = len(x)
            cs = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), 0)
            f = []
            for t in range(n):
                lo10, lo30, lo50 = max(0, t - 9), max(0, t - 29), max(0, t - 49)
                m10 = (cs[t + 1] - cs[lo10]) / (t + 1 - lo10)
                m30 = (cs[t + 1] - cs[lo30]) / (t + 1 - lo30)
                m50 = (cs[t + 1] - cs[lo50]) / (t + 1 - lo50)
                seg = x[lo30:t + 1]
                if len(seg) >= 3:
                    tt = np.arange(len(seg)) - (len(seg) - 1) / 2
                    slope = (tt[:, None] * (seg - seg.mean(0))).sum(0) / (tt ** 2).sum()
                else:
                    slope = np.zeros(x.shape[1])
                f.append(np.concatenate([m10, m30, m50, slope, m10 - x[:K_BASE].mean(0), [a[t], t + 1]]))
            out.append(np.asarray(f, np.float32))
        return np.concatenate(out, 0)


def hp_noise(frame, alpha, seed, cutoff=0.5, chans=SENSORS):
    """Noise protocol of the reference (Experiment.add_sensor_noise): Gaussian noise with std
    alpha * std(feature), FFT high-pass keeping normalised frequencies >= cutoff; applied to the
    whole monitored sequence of every engine (the reference applies it to each test window)."""
    rng = np.random.default_rng(seed)
    f = frame.copy()
    sd = f[chans].values.std(0)
    for e, d in f.groupby('engine_id'):
        x = d[chans].values.astype(np.float64)
        n = rng.standard_normal(x.shape) * alpha * sd
        F = np.fft.rfft(n, axis=0)
        F[np.linspace(0, 1, F.shape[0]) < cutoff] = 0
        f.loc[d.index, chans] = x + np.fft.irfft(F, n=len(x), axis=0)
    return f


def load(sub, L=50, noise_test=0.0, noise_seed=0):
    tr, te, mean_life, _, _ = load_frames(sub)
    if noise_test > 0:
        te = hp_noise(te, noise_test, noise_seed)
    return Engines(tr, mean_life, L), Engines(te, mean_life, L), mean_life
