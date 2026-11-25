"""01. Qwen3-ASR transcription and internal-value dump (GPU). Labels are not used.
  python 01_dump_decode_states.py decode K N
    Greedy-transcribe shard K of N of $AMI_DIR/rows.jsonl (all rows by default; only the listed rows if $UMAI_IDS gives an id-list file) (official prompt, eos [151645, 151643],
    max 260 tokens, one utterance at a time = same conditions as the inference engine. Batching utterances with left padding changes the computation and alters the text in about 9% of utterances). Per row: tokens, text, step list steps_nolab = [slot, j, wordstart, boundary, content].
    -> $UMAI_DUMP/greedy/greedy_K.jsonl
  python 01_dump_decode_states.py states K
    Feed prompt + greedy tokens once more to get nearly the same states as during transcription (teacher forcing; rounding-level differences).
    For each word-start gap, 28-layer decoder outputs h_g (that step), h_g1 (next step)  -> $UMAI_DUMP/q/K.npz
    Projector outputs at audio positions (decoder input) and 24 encoder layer outputs    -> $UMAI_DUMP/frames/
    q/K.npz is written last and atomically, so its presence means the whole shard is done.
"""
import json, os, re, sys
from collections import Counter
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import paths as UP  # calibration/paths.py
sys.path.insert(0, os.path.join(UP.CAL, "lib"))
from greedy import load_model, steps_of, audio, EOS
from align import nz

B = 1                                   # no batching, to get the same text as inference (model.py)


def load_rows():
    """Select calibration rows from $AMI_DIR/rows.jsonl. Default: all (file order). If $UMAI_IDS gives an id-list file, those ids in that order."""
    rows = [json.loads(l) for l in open(f"{UP.AMI_DIR}/rows.jsonl")]
    ids = os.environ.get("UMAI_IDS")
    if not ids or ids == "all": return rows
    by = {r["id"]: r for r in rows}; want = [l.strip() for l in open(ids) if l.strip()]
    miss = [i for i in want if i not in by]; assert not miss, f"{len(miss)} ids of {ids} not in rows.jsonl, e.g. {miss[:3]}"
    return [by[i] for i in want]


def decode(k, n):
    rows = load_rows()
    per = (len(rows) + n - 1) // n; rows = rows[k * per:(k + 1) * per]
    m, prompt = load_model(B); P = m.processor; tok = P.tokenizer; tok.padding_side = "left"
    from transformers import LogitsProcessorList

    class Greedy:
        """Pick the argmax token unchanged while recording each row's tokens and eos status."""
        def __init__(self, g): self.toks = [[] for _ in range(g)]; self.ended = [False] * g
        def __call__(self, input_ids, scores):
            out = torch.full_like(scores, float("-inf")); am = scores.float().argmax(-1).tolist()
            for g, t in enumerate(am):
                if self.ended[g]: out[g, EOS[0]] = 0.0; continue
                if t in EOS: self.ended[g] = True
                else: self.toks[g].append(t)
                out[g, t] = 0.0
            return out

    os.makedirs(f"{UP.DUMP}/greedy", exist_ok=True); out_p = f"{UP.DUMP}/greedy/greedy_{k:03d}.jsonl"
    with open(out_p + ".tmp", "w") as f:
        for i in range(0, len(rows), B):
            grp = rows[i:i + B]
            inp = P(text=[prompt] * len(grp), audio=[audio(r) for r in grp], return_tensors="pt", padding=True).to("cuda:0").to(getattr(torch, UP.DTYPE))
            m.model.thinker.rope_deltas = None
            R = Greedy(len(grp))
            with torch.no_grad():
                m.model.generate(**inp, max_new_tokens=260, eos_token_id=EOS, logits_processor=LogitsProcessorList([R]))
            for g, r in enumerate(grp):
                toks = R.toks[g]; text = tok.decode(toks, skip_special_tokens=True).strip(); qw = [nz(w) for w in text.split()]
                st = [[slot, j, int(ws), int(bnd), int(slot == len(qw) or (slot < len(qw) and bool(qw[slot])))]
                      for slot, j, ws, bnd in steps_of(tok, toks, R.ended[g])]
                f.write(json.dumps(dict(r, toks=toks, ended=R.ended[g], qwen=text, steps_nolab=st), ensure_ascii=False) + "\n")
    os.replace(out_p + ".tmp", out_p)
    print("decode", k, len(rows), "rows ->", out_p, flush=True)


def states(k):
    tag = f"{k:03d}"; od = UP.DUMP
    os.makedirs(f"{od}/q", exist_ok=True); os.makedirs(f"{od}/frames", exist_ok=True)
    m, prompt = load_model(1, max_new_tokens=8); P = m.processor; th = m.model.thinker; dev = "cuda:0"
    AUD = getattr(th.config, "audio_token_id", None) or getattr(th.config, "audio_token_index", None) or 151676
    L = len(th.model.layers); cap = {}
    for li, ly in enumerate(th.model.layers):
        ly.register_forward_hook(lambda mod, inp, out, li=li: cap.__setitem__(li, (out[0] if isinstance(out, tuple) else out)[0].detach()))
    def pre(mod, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]; cap["emb"] = hs[0].detach()
    th.model.layers[0].register_forward_pre_hook(pre, with_kwargs=True)
    best = None                                                    # layer list of the audio encoder (the longest ModuleList)
    for nm, mod in th.audio_tower.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 4 and (best is None or len(mod) > len(best)): best = mod
    enc_layers = list(best); NE = len(enc_layers); ecap = {}
    for ei, ly in enumerate(enc_layers):
        ly.register_forward_hook(lambda mod, inp, out, ei=ei: ecap.__setitem__(ei, (out[0] if isinstance(out, tuple) else out).detach()))
    rows = [json.loads(l) for l in open(f"{od}/greedy/greedy_{tag}.jsonl")]
    buf = {k_: [] for k_ in ("h_g", "h_g1", "id", "slot", "j", "content", "meeting", "series")}
    EMB = []; fenc = open(f"{od}/frames/enc_frames_{tag}.f16", "wb"); foff = [0]; frid, fser, fmt, fdur, fnA = [], [], [], [], []
    skip = Counter()
    for r in rows:
        sel = [s for s in r["steps_nolab"] if s[2]]
        wav = audio(r)
        x = P(text=[prompt], audio=[wav], return_tensors="pt", padding=True); plen = x["input_ids"].shape[1]
        ids = torch.cat([x["input_ids"], torch.tensor([r["toks"]], dtype=x["input_ids"].dtype)], 1); S = ids.shape[1]
        apos = (ids[0] == AUD).nonzero().flatten()
        kw = {"input_ids": ids.to(dev), "attention_mask": torch.ones_like(ids).to(dev)}
        for kk in ("input_features", "feature_attention_mask"):
            if kk in x: kw[kk] = x[kk].to(dev)
        if "input_features" in kw: kw["input_features"] = kw["input_features"].to(getattr(torch, UP.DTYPE))
        th.rope_deltas = None; cap.clear(); ecap.clear()
        with torch.no_grad(): th(**kw)
        mt = r["meeting"].lower(); se = re.sub(r"[a-z]$", "", mt)
        H = torch.stack([cap[li] for li in range(L)], 1)
        for s in sel:
            p = plen - 1 + s[1]
            buf["h_g"].append(H[min(p, S - 1)].half().cpu().numpy()); buf["h_g1"].append(H[min(p + 1, S - 1)].half().cpu().numpy())
            buf["id"].append(r["id"]); buf["slot"].append(s[0]); buf["j"].append(s[1]); buf["content"].append(s[4]); buf["meeting"].append(mt); buf["series"].append(se)
        X = [ecap[e].reshape(-1, ecap[e].shape[-1]) for e in range(NE)]
        if any(h.shape[0] != len(apos) for h in X): skip["frame_count_mismatch"] += 1; continue
        EMB.append(cap["emb"][apos].half().cpu().numpy()); fenc.write(torch.stack(X, 1).half().cpu().numpy().tobytes())
        foff.append(foff[-1] + len(apos)); frid.append(r["id"]); fser.append(se); fmt.append(mt); fdur.append(len(wav) / 16000); fnA.append(len(apos))
    fenc.close()
    if not EMB:
        raise RuntimeError(f"states {tag}: all {len(rows)} rows were skipped ({dict(skip)}); no frames to write")
    np.save(f"{od}/frames/emb_frames_{tag}.npy", np.concatenate(EMB))
    np.savez(f"{od}/frames/frames_{tag}.npz", foff=np.array(foff), rid=np.array(frid), series=np.array(fser), meeting=np.array(fmt),
             dur=np.array(fdur), nA=np.array(fnA), enc_shape=np.array([foff[-1], NE, X[0].shape[-1]]))
    q_p = f"{od}/q/{tag}.npz"                                      # written last, via a temp file in the same directory
    with open(q_p + ".tmp", "wb") as fq:
        np.savez(fq, **{k_: (np.stack(v) if k_.startswith("h_") else np.array(v)) for k_, v in buf.items()})
    os.replace(q_p + ".tmp", q_p)
    print("states", tag, len(rows), "rows", len(buf["id"]), "gaps", foff[-1], "frames", dict(skip), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "decode": decode(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "states": states(int(sys.argv[2]))
    else: sys.exit(__doc__)
