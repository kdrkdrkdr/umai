"""02. Per-head attention dump (GPU). Each row is teacher-forced once (prompt + greedy tokens, eager attention), and
for each step the attention of all layers/heads is restricted to audio positions, renormalised and saved. Used for alignment-head weights (07) and the audio window.
  python 02_dump_attention.py K     $UMAI_DUMP/greedy/greedy_K.jsonl -> $UMAI_DUMP/att/att_K.f16 + att_K.npz (aoff, nQ, nA, rid, NL, NH)
"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import paths as UP  # calibration/paths.py
sys.path.insert(0, os.path.join(UP.CAL, "lib"))
from greedy import load_model, audio


def main():
    tag = f"{int(sys.argv[1]):03d}"; out_p = f"{UP.DUMP}/att/att_{tag}"; os.makedirs(f"{UP.DUMP}/att", exist_ok=True)
    rows = [json.loads(l) for l in open(f"{UP.DUMP}/greedy/greedy_{tag}.jsonl")]
    m, prompt = load_model(1, max_new_tokens=8); P = m.processor; th = m.model.thinker; dev = "cuda:0"
    th.model.config._attn_implementation = "eager"
    for lay in th.model.layers:
        if hasattr(lay.self_attn, "config"): lay.self_attn.config._attn_implementation = "eager"
    AUD = getattr(th.config, "audio_token_id", None) or getattr(th.config, "audio_token_index", None) or 151676
    L = len(th.model.layers); NH = th.config.text_config.num_attention_heads; st = {"Q": None, "A": None, "att": {}}
    for li, ly in enumerate(th.model.layers):
        def ah(mod, inp, out, li=li):
            w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            if w is None: raise RuntimeError("no attention weights: eager not active")
            st["att"][li] = w[0][:, st["Q"]][:, :, st["A"]].float()          # (H, Q, A)
        ly.self_attn.register_forward_hook(ah)
    aoff, nQ, nA_, rid = [0], [], [], []
    with open(out_p + ".f16", "wb") as fo:
        for r in rows:
            x = P(text=[prompt], audio=[audio(r)], return_tensors="pt", padding=True); plen = x["input_ids"].shape[1]
            ids = torch.cat([x["input_ids"], torch.tensor([r["toks"]], dtype=x["input_ids"].dtype)], 1); S = ids.shape[1]
            apos = (ids[0] == AUD).nonzero().flatten()
            st["Q"] = torch.arange(plen - 1, S, device=dev); st["A"] = apos.to(dev); st["att"] = {}
            kw = {"input_ids": ids.to(dev), "attention_mask": torch.ones_like(ids).to(dev)}
            for kk in ("input_features", "feature_attention_mask"):
                if kk in x: kw[kk] = x[kk].to(dev)
            if "input_features" in kw: kw["input_features"] = kw["input_features"].to(getattr(torch, UP.DTYPE))
            th.rope_deltas = None
            with torch.no_grad(): th(**kw)
            a = torch.stack([st["att"][l] for l in range(L)])                      # (L, H, Q, A)
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
            a = a.reshape(L * a.shape[1], a.shape[2], a.shape[3]).permute(1, 0, 2).contiguous()   # (Q, L*H, A)
            fo.write(a.half().cpu().numpy().tobytes()); aoff.append(aoff[-1] + a.numel()); nQ.append(a.shape[0]); nA_.append(a.shape[2]); rid.append(r["id"])
    with open(out_p + ".npz.tmp", "wb") as fh:                         # written last and atomically: att_K.npz is the done-marker
        np.savez(fh, aoff=np.array(aoff), nQ=np.array(nQ), nA=np.array(nA_), rid=np.array(rid), NL=L, NH=NH)
    os.replace(out_p + ".npz.tmp", out_p + ".npz")
    print("attention", tag, len(rid), "rows ->", out_p, flush=True)


if __name__ == "__main__":
    main()
