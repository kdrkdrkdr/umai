"""08. Fit the filler decision (CPU). Closed-form statistics only.
Gaps: gaps from 05 (h_g, h_g1, classes O/F/E/U/Q). O is subsampled, so each class is weighted by original count / sample count.
Folds: 5 folds by meeting series. All scores fed to stage 2 and isotonic come from weights fitted without their own fold; the stored weights are fitted on all data.
Stage 1: per layer (28 layers x g, g1) 5-class LDA (shared covariance OAS, original-frequency priors) -> 4 values per layer (class discriminant - O discriminant).
Audio: 10 values pooling the posterior p and log p of the frame stage-2 5-class LDA over the audio window F_prev (1 - F_g) of the mixed alignment-head attention,
  and 10 values pooled over the next-word window F_g (1 - F_after_next).
Whether the previous word is a filler Qwen wrote itself (0/1).
Stage 2: 5-class LDA over [stage1 g, stage1 g1, audio, next-word window, prev-word filler] -> logit P(F).
  End gaps (gaps with no following step: the eos step, or the last token step of a row cut at max tokens) have no g+1 step, so
  they are fitted separately on [stage1 g, audio, prev-word filler]. The end-gap model is fitted on end gaps only (class distribution and features differ from regular gaps).
  Inference (model.py) identifies end gaps with the same definition.
  Qwen output-layer distribution features (filler token probabilities, etc.) and window mass are not used: they come from the same decoder state and overlap with the stage-1 layer scores,
  and adding them did not improve the decision.
isotonic (PAV): out-of-fold logit P(F) -> P(F). The inference write rule is P(F) > write_threshold (config.json; the exported default is 0.5).
  python 08_fit_filler.py   -> $UMAI_FIT/filler_head.pt
"""
import json, os, sys
from collections import Counter
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import paths as UP  # calibration/paths.py
sys.path.insert(0, os.path.join(UP.CAL, "lib"))
from stats import oas, lda_multi, pav
from heads import Att, audio_window
from align import is_filler
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
K = 5; CLS = ["O", "F", "E", "U", "Q"]; NC = len(CLS)      # Q = Qwen wrote a filler there itself (not F)
FEATURE_ORDER = ("gap: S1g (112) | S1g1 (112) | audio (10) | next-word window audio (10) | previous word is Qwen filler (1); "
                 "end gap: S1g (112) | audio (10) | previous word is Qwen filler (1)")
NOTE = "UMAI filler gate: closed-form LDAs on greedy-condition gaps (decoder states, audio windows), natural weights, isotonic P(F); write P > write_threshold"


def wlda_multi(X, y, w):
    """Weighted 5-class LDA: class means are sample means; covariance and priors use weights w."""
    X = torch.as_tensor(X, dtype=torch.float64); y = torch.as_tensor(y); w = torch.as_tensor(w, dtype=torch.float64)
    cs = [c for c in range(NC) if (y == c).any()]
    mu = torch.zeros(NC, X.shape[1], dtype=torch.float64); ws = torch.zeros(NC, dtype=torch.float64)
    for c in cs: mu[c] = X[y == c].mean(0); ws[c] = w[y == c].sum()
    Z = X - mu[y]; S = (Z * w[:, None]).T @ Z / w.sum()
    Am = torch.linalg.solve(oas(S.to(DEV), float(len(y))).cpu(), mu.T).T
    b = -0.5 * (Am * mu).sum(1) + torch.log((ws / ws.sum()).clamp_min(1e-300))
    b[[c for c in range(NC) if c not in cs]] = -float("inf")
    return Am, b


def pf_logit(G):
    """Class discriminants -> log P(F) - log P(not F)."""
    lp = torch.log_softmax(torch.as_tensor(G), 1); oth = torch.logsumexp(lp[:, [c for c in range(lp.shape[1]) if c != 1]], 1)
    return (lp[:, 1] - oth).numpy()


def oof_and_full(X, y, w, fold):
    """Out-of-fold logit P(F) and (A, b) fitted on all data."""
    s = np.zeros(len(y))
    for k in range(K):
        tr = fold != k; A, b = wlda_multi(X[tr], y[tr], w[tr]); s[fold == k] = pf_logit(torch.from_numpy(X[fold == k]) @ A.T + b)
    return s, wlda_multi(X, y, w)


def fit_stage2(XF, XE, gi, W, fold, eos, isF):
    """Stage-2 (regular gaps XF, end gaps XE) 5-class LDA and isotonic over out-of-fold scores. Returns ((A2, b2), (Ae, be), (iso_x, iso_p)).
    The regular-gap model is fitted on regular gaps only, the end-gap model on end gaps only. A single isotonic is fitted on the pooled out-of-fold scores of both models."""
    SC, full = oof_and_full(XF[~eos], gi[~eos], W[~eos], fold[~eos])
    SE, full_e = oof_and_full(XE[eos], gi[eos], W[eos], fold[eos])
    s = np.zeros(len(gi)); s[~eos] = SC; s[eos] = SE
    return full, full_e, pav(s, isF, W)


def main():
    meta = dict(np.load(f"{UP.FIT}/gaps/meta.npz"))
    steps, eos_slot, nat, qwords = {}, {}, Counter(), {}
    for l in open(f"{UP.FIT}/greedy_q.jsonl"):
        r = json.loads(l); steps[r["id"]] = sorted({(s[0], s[1]) for s in r["steps"] if s[2]})
        qwords[r["id"]] = r["qwen"].split()
        last = len(r["toks"]) if r["ended"] else len(r["toks"]) - 1          # step with no following step (eos, or last token of a truncated row)
        for s in r["steps"]:
            if s[2] and s[1] == last: eos_slot[r["id"]] = s[0]                         # end gap (no g+1 step)
        nat.update(s[4] for s in r["steps"] if s[2] and s[4] != "X")
    n_gap = len(meta["cls"]); dc = Counter(meta["cls"].tolist())
    W = np.array([nat[c] / dc[c] for c in meta["cls"]])                     # original count / sample count
    wcls = np.array([nat[c] / max(1, dc[c]) for c in CLS])
    gi = np.array([CLS.index(c) for c in meta["cls"]])
    ser = sorted(set(meta["series"].tolist())); fm = {s: i % K for i, s in enumerate(ser)}; fold = np.array([fm[s] for s in meta["series"]])
    prevq = np.array([float(int(s) >= 1 and int(s) - 1 < len(qwords[str(i)]) and is_filler(qwords[str(i)][int(s) - 1]))
                      for i, s in zip(meta["id"], meta["slot"])])[:, None]
    # ---- stage 1: per-layer 5-class LDA
    S1M, ENGM = {}, {}
    for pt in ("g", "g1"):
        H = np.load(f"{UP.FIT}/gaps/h_{pt}.npy", mmap_mode="r"); NL, _, d = H.shape
        oof = np.zeros((n_gap, NL, NC - 1)); Wm = torch.zeros(NL, NC, d, dtype=torch.float64); Bm = torch.zeros(NL, NC, dtype=torch.float64)
        for l in range(NL):
            X = torch.from_numpy(np.asarray(H[l], dtype=np.float32)).to(DEV).double()
            n = torch.zeros(K, NC, dtype=torch.float64); sx = torch.zeros(K, NC, d, dtype=torch.float64, device=DEV); sxx = torch.zeros(K, NC, d, d, dtype=torch.float64, device=DEV)
            for k in range(K):
                for c in range(NC):
                    ix = torch.from_numpy(np.where((fold == k) & (gi == c))[0]).to(DEV)
                    if len(ix) == 0: continue
                    Xc = X[ix]; n[k, c] = len(ix); sx[k, c] = Xc.sum(0); sxx[k, c] = Xc.T @ Xc

            def fitm(excl):
                ks = [k for k in range(K) if k not in excl]; wc = torch.tensor(wcls, dtype=torch.float64)
                nc = n[ks].sum(0); mu = sx[ks].sum(0) / nc.clamp_min(1).to(DEV)[:, None]; nw = nc * wc
                Sw = sum(float(wc[c]) * (sxx[ks, c].sum(0) - float(nc[c]) * torch.outer(mu[c], mu[c])) for c in range(NC) if nc[c] > 0) / float(nw.sum())
                Am = torch.linalg.solve(oas(Sw, float(nc.sum())), mu.T).T
                return Am, -0.5 * (Am * mu).sum(1) + torch.log((nw / nw.sum()).clamp_min(1e-300)).to(DEV)
            for k in range(K):
                Am, bm = fitm({k}); ix = torch.from_numpy(fold == k).to(DEV); g_ = X[ix] @ Am.T + bm
                oof[fold == k, l] = (g_[:, 1:] - g_[:, :1]).cpu().numpy()
            Am, bm = fitm(set()); Wm[l] = Am.cpu(); Bm[l] = bm.cpu()
            del X, sx, sxx
        S1M[pt] = oof.reshape(n_gap, -1); ENGM[pt] = (Wm, Bm); del H
    # ---- audio: pool frame stage-2 posteriors over the gap audio window
    fc = dict(np.load(f"{UP.FIT}/frames.npz"))
    ffold = fc["fold"][np.repeat(np.arange(len(fc["rid"])), np.diff(fc["foff"]))]
    X = fc["S1"].astype(np.float64); y = fc["lab"].astype(int); LP = np.zeros((len(y), 5))
    for k in range(K):
        tr = (ffold != k) & (y >= 0); Am, b = lda_multi(X[tr], y[tr], 5)
        LP[ffold == k] = torch.log_softmax(torch.from_numpy(X[ffold == k]) @ Am.T + b, 1).numpy()
    Am_f, b_f = lda_multi(X[y >= 0], y[y >= 0], 5); del X
    pi = np.array(json.load(open(f"{UP.FIT}/alignment_heads.json"))["pi"]); piS = pi / pi.sum()
    at = Att(f"{UP.DUMP}/att"); fx = {str(r): i for i, r in enumerate(fc["rid"])}; fo = fc["foff"]
    AUD = np.full((n_gap, 10), np.nan); AUDN = np.full((n_gap, 10), np.nan); cache = {}
    for i, (rid, slot) in enumerate(zip(meta["id"], meta["slot"])):
        rid = str(rid)
        if rid not in cache:
            a = at.get(rid); cache = {rid: None if a is None else (a * piS[None, :, None]).sum(1).astype(np.float64)}
        am = cache[rid]; ri = fx.get(rid); st_ = steps.get(rid, []); j = dict(st_).get(int(slot), -1)
        if am is None or ri is None or not (0 <= j < am.shape[0]) or am.shape[1] != fo[ri + 1] - fo[ri]: continue
        lpr = LP[fo[ri]:fo[ri + 1]]
        wA, _ = audio_window(am, st_, j); AUD[i] = np.r_[wA @ np.exp(lpr), wA @ lpr]
        nxt = [jj for _, jj in st_ if j < jj < am.shape[0]]
        if nxt:
            w2 = np.cumsum(am[j]) * np.clip(1 - np.cumsum(am[min(nxt)]), 0, None); m2 = float(w2.sum())   # remove rounding negatives as in audio_window
            if m2 > 0: w2 = w2 / m2; AUDN[i] = np.r_[w2 @ np.exp(lpr), w2 @ lpr]
    aud_mu = np.nanmean(AUD, 0); AUD[np.isnan(AUD).any(1)] = aud_mu       # gaps with no audio found get the mean
    audn_mu = np.nanmean(AUDN, 0); AUDN[np.isnan(AUDN).any(1)] = audn_mu
    # ---- stage 2 (5-class) + end-of-sentence gap model + isotonic
    eos = np.array([eos_slot.get(i, -1) == int(sl) for i, sl in zip(meta["id"], meta["slot"])])
    (A2, b2), (Ae, be), (Xi, Pi) = fit_stage2(np.concatenate([S1M["g"], S1M["g1"], AUD, AUDN, prevq], 1),
                                              np.concatenate([S1M["g"], AUD, prevq], 1), gi, W, fold, eos, meta["cls"] == "F")
    torch.save({"heads": [(int(h // at.NH), int(h % at.NH), float(p)) for h, p in enumerate(piS)],
                "frame_A": Am_f.double(), "frame_b": b_f.double(), "aud_mu": torch.tensor(aud_mu),
                "audn_mu": torch.tensor(audn_mu),
                "W1m_g": ENGM["g"][0], "B1m_g": ENGM["g"][1], "W1m_g1": ENGM["g1"][0], "B1m_g1": ENGM["g1"][1],
                "W2m": A2.double(), "b2m": b2.double(), "W2m_eos": Ae.double(), "b2m_eos": be.double(),
                "iso_x": torch.tensor(Xi), "iso_p": torch.tensor(Pi), "classes": CLS, "nat_cal": dict(nat),
                "feature_order": FEATURE_ORDER, "note": NOTE},
               f"{UP.FIT}/filler_head.pt")
    print("filler head:", n_gap, "gaps,", len(Xi), "isotonic steps ->", f"{UP.FIT}/filler_head.pt", flush=True)


if __name__ == "__main__":
    main()
