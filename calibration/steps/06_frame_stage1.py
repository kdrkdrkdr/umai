"""06. Audio frame stage-1 LDA (CPU). Frame labels come from original AMI word times (words.jsonl): W / F / Efrag / Erep, S outside words.
A 5-class LDA (shared covariance OAS, frame-frequency priors) for each of 25 sources (24 encoder layers + projector). Uses the GPU if available (still float64).
Frame feature S1 = 4 values per source (class discriminant - S discriminant) = 100 dims. Each frame is scored with weights fitted without its own fold.
Scores are computed and stored in float64 (class discriminants are ~10^5, so in float32/16 cancellation noise from the subtraction leaves errors above 1 in S1).
  python 06_frame_stage1.py   -> $UMAI_FIT/stage1.pt (weights fitted on all data), $UMAI_FIT/frames.npz (S1, labels, folds)
"""
import glob, json, math, os, sys
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import paths as UP  # calibration/paths.py
sys.path.insert(0, os.path.join(UP.CAL, "lib"))
from stats import oas
CL = ["S", "W", "F", "Efrag", "Erep"]; NC = len(CL); PRI = {"W": 1, "Erep": 3, "Efrag": 4, "F": 5}; K = 5; NE = 24
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def frame_labels(words, nA, dur):
    """Paint word spans onto frames. On overlap, the class with the larger PRI wins. Unpainted frames are S (0)."""
    r = nA / dur; lab = np.zeros(nA, np.int8); pr = np.zeros(nA, np.int8)
    for c, s, e in words:
        f0 = min(nA - 1, int(math.floor(s * r))); f1 = min(nA, max(f0 + 1, int(math.ceil(e * r))))
        for f in range(f0, f1):
            if PRI[c] > pr[f]: pr[f] = PRI[c]; lab[f] = CL.index(c)
    return lab


def chunks(enc, emb, B=16384):
    """Frames in chunks of B: (start index, count, features of the 25 sources)."""
    base = 0
    for e in enc:
        for a in range(0, len(e), B):
            X = np.asarray(e[a:a + B], dtype=np.float32); n = len(X)
            yield base + a, n, [torch.from_numpy(X[:, s_]).to(DEV) for s_ in range(NE)] + [torch.from_numpy(np.asarray(emb[base + a: base + a + n], dtype=np.float32)).to(DEV)]
        base += len(e)


def main():
    D, O = UP.DUMP, UP.FIT
    fr = dict(np.load(f"{D}/frames.npz")); W = {json.loads(l)["id"]: json.loads(l) for l in open(f"{O}/words.jsonl")}
    fo = fr["foff"]; lab = np.full(fo[-1], -2, np.int8)          # -2: rows without word times (unused)
    for i, rid in enumerate(fr["rid"]):
        w = W.get(str(rid))
        if w is not None: lab[fo[i]:fo[i + 1]] = frame_labels(w["words"], int(fo[i + 1] - fo[i]), float(fr["dur"][i]))
    fold = np.repeat(fr["fold"], np.diff(fo))
    enc = [np.memmap(e, dtype=np.float16, mode="r").reshape(-1, NE, 1024) for e in sorted(glob.glob(f"{D}/frames/enc_frames_[0-9][0-9][0-9].f16"))]
    assert sum(len(e) for e in enc) == fo[-1], "enc frames != foff"
    emb = np.load(f"{D}/emb_frames.npy", mmap_mode="r"); assert len(emb) == fo[-1]
    # pass 1: count, sum and sum of products per fold x class
    dims = [1024] * NE + [2048]; NS = len(dims)
    N = torch.zeros(K, NC, dtype=torch.float64); Sx = [torch.zeros(K, NC, d, dtype=torch.float64, device=DEV) for d in dims]
    Sxx = [torch.zeros(K, NC, d, d, dtype=torch.float64, device=DEV) for d in dims]
    for a, n, X in chunks(enc, emb):
        lb = lab[a:a + n]; fk = fold[a:a + n]
        for k in range(K):
            for c in range(NC):
                mk = np.where((lb == c) & (fk == k))[0]
                if len(mk) == 0: continue
                mk_t = torch.from_numpy(mk).to(DEV); N[k, c] += len(mk)
                for s_ in range(NS):
                    Xc = X[s_][mk_t].double(); Sx[s_][k, c] += Xc.sum(0); Sxx[s_][k, c] += Xc.T @ Xc

    def fit_all(excl):
        ks = [k for k in range(K) if k not in excl]; n = N[ks].sum(0).to(DEV); lpi = torch.log(n / n.sum())
        Ws, Bs = [], []
        for s_ in range(NS):
            mu = Sx[s_][ks].sum(0) / n[:, None]
            Sw = sum(Sxx[s_][ks, c].sum(0) - n[c] * torch.outer(mu[c], mu[c]) for c in range(NC)) / n.sum()
            Am = torch.linalg.solve(oas(Sw, float(n.sum())), mu.T).T
            Ws.append(Am); Bs.append(-0.5 * (Am * mu).sum(1) + lpi)          # float64, kept on device for scoring
        return Ws, Bs
    WS = {k: fit_all({k}) for k in range(K)}; full = fit_all(set())
    torch.save({"classes": CL, "W": [w.cpu() for w in full[0]], "B": [b.cpu() for b in full[1]],   # float64
                "note": "per source 5-class frame LDA (encoder layers 0..23, then projector), AMI NXT word-time frame labels"}, f"{O}/stage1.pt")
    del Sxx
    # pass 2: frame features (weights without own fold)
    S1 = np.zeros((fo[-1], 4 * NS), np.float64)
    for a, n, X in chunks(enc, emb):
        fk = fold[a:a + n]; out = torch.zeros(n, 4 * NS, dtype=torch.float64, device=DEV)
        for k in range(K):
            ix = torch.from_numpy(np.where(fk == k)[0]).to(DEV)
            if len(ix) == 0: continue
            Ws, Bs = WS[k]; f = []
            for s_ in range(NS):
                g = X[s_][ix].double() @ Ws[s_].T + Bs[s_]; f.append(g[:, 1:] - g[:, :1])
            out[ix] = torch.cat(f, 1)
        S1[a:a + n] = out.cpu().numpy()
    assert np.isfinite(S1).all()
    np.savez(f"{O}/frames.npz", S1=S1, lab=lab, foff=fo, rid=fr["rid"], series=fr["series"], fold=fr["fold"], dur=fr["dur"])
    print("frame stage 1:", fo[-1], "frames, per class", N.sum(0).tolist(), flush=True)


if __name__ == "__main__":
    main()
