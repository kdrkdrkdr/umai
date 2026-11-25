"""09. Fit the uh/um decision (CPU). Closed-form statistics only. Targets are F gaps whose reference form is uh or um.
 Decoder: Qwen's own um/uh probability ratio lm@g = logsumexp(log p[um tokens]) - logsumexp(log p[uh tokens]), from the last-layer h_g state.
         Only the uh/um token rows of lm_head are applied after final norm (the normaliser cancels in the subtraction, so no full-vocabulary softmax is needed).
 Audio:   log-likelihood ratio of a projector-frame uh/um LDA (NXT time labels), pooled with weights audio window x p(F|frame) (AF).
 Combine: [lm@g, AF] binary LDA (out-of-fold) -> isotonic -> P(um). The inference write rule is [um] if P(um) > 0.5, else [uh].
  python 09_fit_form.py   -> $UMAI_FIT/form_head.pt
"""
import json, math, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import paths as UP  # calibration/paths.py
sys.path.insert(0, os.path.join(UP.CAL, "lib"))
from align import forms_of
from stats import lda_multi, lda2, score, pav
from heads import Att, audio_window
K = 5


def spellings(w):
    """Token spellings of a filler word (with/without leading space x lowercase/capitalised/uppercase). Same as inference/umai/model.py."""
    return [p + v for v in (w, w.capitalize(), w.upper()) for p in ("", " ")]


def qwen_form_rows():
    """Qwen3-ASR final norm weight, eps, uh/um token ids and the lm_head rows of those ids (read only during calibration).
    Token ids are those spellings() variants that encode to a single token."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    path = UP.ASR_MODEL if os.path.isdir(UP.ASR_MODEL) else snapshot_download(UP.ASR_MODEL)
    wmap = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]; cfg = json.load(open(f"{path}/config.json"))
    tok = AutoTokenizer.from_pretrained(path)
    def ids(w):
        out = set()
        for v in spellings(w):
            e = tok.encode(v, add_special_tokens=False)
            if len(e) == 1: out.add(e[0])
        return sorted(out)
    UH, UM = ids("uh"), ids("um")
    with safe_open(f"{path}/{wmap['thinker.model.norm.weight']}", "pt") as f: g = f.get_tensor("thinker.model.norm.weight").float()
    with safe_open(f"{path}/{wmap['thinker.lm_head.weight']}", "pt") as f: rows = f.get_slice("thinker.lm_head.weight")[UH + UM].float()
    eps = float(cfg["thinker_config"]["text_config"]["rms_norm_eps"])
    return {"norm": g, "eps": eps, "uh": UH, "um": UM, "rows": rows}


def form_lm(H27, q):
    """Last-layer state (n, d) (before final norm) -> lm@g (n,) = logsumexp(logit[um]) - logsumexp(logit[uh])."""
    h = torch.as_tensor(np.asarray(H27, dtype=np.float32))
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + q["eps"]) * q["norm"]
    z = h @ q["rows"].T; nu = len(q["uh"])
    return (torch.logsumexp(z[:, nu:], -1) - torch.logsumexp(z[:, :nu], -1)).double().numpy()


def fit_comb(LMG, AF, y, fC):
    """[lm@g, AF] binary LDA: isotonic on out-of-fold scores, stored weights fitted on all data. Returns (wc, bc, ISX, ISP)."""
    XC = np.c_[LMG, AF]; co = np.zeros(len(y))
    for k in range(K): co[fC == k] = score(XC[fC == k], lda2(XC[fC != k], y[fC != k]))
    wc, bc = lda2(XC, y)
    ISX, ISP = pav(co, y.astype(float), np.ones(len(y)))
    return wc, bc, ISX, ISP


def main():
    FIT = UP.FIT
    # ---- frames: uh/um labels, uh/um log-likelihood ratio (out-of-fold), p(F|frame) (out-of-fold)
    fc = dict(np.load(f"{FIT}/frames.npz")); EM = np.load(f"{UP.DUMP}/emb_frames.npy", mmap_mode="r")
    fwords = {json.loads(l)["id"]: json.loads(l) for l in open(f"{FIT}/fwords.jsonl")}
    fo = fc["foff"]; FL = np.zeros(len(fc["S1"]), np.int8)                    # 1 = uh, 2 = um
    for i, rid in enumerate(fc["rid"]):
        w = fwords.get(str(rid))
        if w is None: continue
        nA = int(fo[i + 1] - fo[i]); rr = nA / w["dur"]
        for c, s, e in w["words"]:
            if c not in ("Fuh", "Fum"): continue
            f0 = min(nA - 1, int(math.floor(s * rr))); f1 = min(nA, max(f0 + 1, int(math.ceil(e * rr))))
            FL[fo[i] + f0: fo[i] + f1] = 1 if c == "Fuh" else 2
    ffold = fc["fold"][np.repeat(np.arange(len(fc["rid"])), np.diff(fo))]
    m = FL > 0; Xl = np.asarray(EM[np.where(m)[0]], dtype=np.float32); yl = (FL[m] == 2).astype(int); fl = ffold[m]
    LLR = np.zeros(len(EM), np.float64)
    for k in range(K):
        wb = lda2(Xl[fl != k], yl[fl != k])
        for a in range(0, len(EM), 20000):
            sel = ffold[a:a + 20000] == k
            if sel.any(): LLR[a:a + 20000][sel] = score(np.asarray(EM[a:a + 20000], dtype=np.float32), wb)[sel]
    wf, bf = lda2(Xl, yl)
    X = fc["S1"].astype(np.float64); y = fc["lab"].astype(int); PF = np.zeros(len(y))
    for k in range(K):
        tr = (ffold != k) & (y >= 0); Am, b = lda_multi(X[tr], y[tr], 5)
        PF[ffold == k] = torch.softmax(torch.from_numpy(X[ffold == k]) @ Am.T + b, 1)[:, 2].numpy()
    del X
    # ---- F gaps: reference form, audio score AF
    pi = np.array(json.load(open(f"{FIT}/alignment_heads.json"))["pi"]); piS = pi / pi.sum()
    rows, steps = {}, {}
    for l in open(f"{FIT}/greedy_q.jsonl"):
        r = json.loads(l); rows[r["id"]] = r; steps[r["id"]] = sorted({(s[0], s[1]) for s in r["steps"] if s[2]})
    meta = dict(np.load(f"{FIT}/gaps/meta.npz")); at = Att(f"{UP.DUMP}/att"); fx = {str(r): i for i, r in enumerate(fc["rid"])}
    gidx, yy, ser, AF = [], [], [], []; forms = {}; cache = {}
    for g in np.where(meta["cls"] == "F")[0]:
        rid = str(meta["id"][g]); sl = int(meta["slot"][g])
        if rid not in forms: forms[rid] = forms_of(rows[rid])
        f = forms[rid].get(sl, "?")
        if f not in ("uh", "um"): continue
        gidx.append(g); yy.append(int(f == "um")); ser.append(meta["series"][g])
        if rid not in cache:
            a = at.get(rid); cache = {rid: None if a is None else (a * piS[None, :, None]).sum(1).astype(np.float64)}
        am = cache[rid]; ri = fx.get(rid); jg = dict(steps[rid]).get(sl, -1)
        if am is None or ri is None or not (0 <= jg < am.shape[0]) or am.shape[1] != fo[ri + 1] - fo[ri]: AF.append(np.nan); continue
        wA, _ = audio_window(am, steps[rid], jg); wp = wA * PF[fo[ri]:fo[ri + 1]]; sp = float(wp.sum())
        AF.append(float(wp @ LLR[fo[ri]:fo[ri + 1]].astype(np.float64)) / sp if sp > 0 else np.nan)   # no weight means no audio evidence (same rule as inference)
    y = np.array(yy); AF = np.array(AF)
    print("AF: gaps without audio evidence", int(np.isnan(AF).sum()), "/", len(AF), "| quantiles 1/25/50/75/99",
          np.round(np.nanpercentile(AF, [1, 25, 50, 75, 99]), 3).tolist(), flush=True)
    AF[np.isnan(AF)] = np.nanmean(AF)                                                       # gaps with no audio found get the mean
    fm = {s: i % K for i, s in enumerate(sorted(set(ser)))}; fC = np.array([fm[s] for s in ser])
    # ---- decoder: Qwen's own um/uh probability ratio (h_g last layer)
    q = qwen_form_rows(); print("uh tokens", q["uh"], "um tokens", q["um"], flush=True)
    Hm = np.load(f"{FIT}/gaps/h_g.npy", mmap_mode="r"); LMG = form_lm(Hm[Hm.shape[0] - 1][gidx], q)
    # ---- combine + isotonic
    wc, bc, ISX, ISP = fit_comb(LMG, AF, y, fC)
    print("combined [lm@g, AF] weights", [round(float(v), 4) for v in wc], "intercept", round(float(bc), 4), flush=True)
    torch.save({"uh_ids": q["uh"], "um_ids": q["um"], "wc": wc.double(), "bc": float(bc),
                "frame_w": wf.double(), "frame_b": float(bf), "iso_x": torch.tensor(ISX), "iso_p": torch.tensor(ISP),
                "um_prior": float(y.mean()), "af_mu": float(np.mean(AF)),
                "note": "UMAI form: lm@g (Qwen log P(um) - log P(uh) at h_g, last layer) + AF (projector frame uh/um llr pooled by window x p(F)); P(um) = iso(comb); AMI NXT labels"},
               f"{FIT}/form_head.pt")
    print("form head:", len(y), "F gaps (um share", round(float(y.mean()), 3), ") ->", f"{FIT}/form_head.pt", flush=True)


if __name__ == "__main__":
    main()
