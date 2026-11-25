"""Alignment heads: reading per-head attention dumps, frame ranges of word spans, head mixing weights via EM,
gap audio window w_t ∝ F_prev(t) (1 - F_g(t)) (F = attention CDF)."""
import numpy as np
import torch


class Att:
    """Attention from the 02 dump (per row (step, layer x head, audio frame), summing to 1 over audio frames)."""
    def __init__(self, prefix):
        z = np.load(prefix + ".npz"); self.mm = np.memmap(prefix + ".f16", dtype=np.float16, mode="r")
        self.ix = {str(r): i for i, r in enumerate(z["rid"])}; self.aoff, self.nQ, self.nA = z["aoff"], z["nQ"], z["nA"]
        self.NH = int(z["NH"])

    def get(self, rid):
        i = self.ix.get(rid)
        if i is None: return None
        a = np.asarray(self.mm[self.aoff[i]:self.aoff[i + 1]], dtype=np.float32)
        return a.reshape(int(self.nQ[i]), -1, int(self.nA[i]))


def span_frames(sp, nA, dur):
    rr = nA / dur; f0 = min(nA - 1, int(np.floor(sp["s"] * rr))); f1 = min(nA, max(f0 + 1, int(np.ceil(sp["e"] * rr)))); return f0, f1


def em(M):
    """M (n, H) > 0: per-head attention mass for each span -> mixing weights pi (H,) (maximising sum log M pi).
    Starts from uniform and iterates until the log-likelihood stops increasing (at most 5000 iterations, as in the run that produced the calibration values)."""
    M = torch.as_tensor(M, dtype=torch.float64); n, H = M.shape; pi = torch.full((H,), 1.0 / H, dtype=torch.float64); ll = -np.inf
    for _ in range(5000):
        mix = M @ pi; new = float(torch.log(mix).sum())
        if new <= ll: break
        ll = new; pi = (M * pi / mix[:, None]).mean(0)
    return pi.numpy()


def audio_window(att, steps, j_g):
    """att (step, audio frame): mixed attention. steps: word-start steps [(slot, j)].
    Audio window F_prev (1 - F_g) (sum 1) of gap step j_g and its mass before normalisation."""
    ag = att[j_g]; Fg = np.cumsum(ag)
    prev = [j for s, j in steps if j < j_g]
    Fp = np.cumsum(att[max(prev)]) if prev else np.ones_like(Fg)
    w = Fp * np.clip(1 - Fg, 0, None); s = float(w.sum())   # the CDF never exceeds 1: only negatives from float16 attention cumsum rounding are set to 0
    return (w / s if s > 0 else ag), s
