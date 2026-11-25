"""10. Package the calibration results in the UMAI calibration-value format. No computation; values are only copied over.
  python 10_export_head.py OUT_DIR      (reads stage1.pt, filler_head.pt, form_head.pt from $UMAI_FIT)
Output: OUT_DIR/umai_head.safetensors (numeric arrays), OUT_DIR/config.json (class names, decision threshold 0.5, prompt settings, max tokens, Qwen model name,
  Qwen token ids used for the uh/um form decision — inference recomputes them the same way and checks that they match).
Alignment heads with zero weight are not stored (they only add 0 to the mixture sum, so results are identical).
LDA weights are stored as the float64 values computed in calibration (so inference uses the same values as calibration).
"""
import json, os, sys
import torch
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import paths as UP  # calibration/paths.py


def f64(x):
    return torch.tensor(float(x), dtype=torch.float64)


def main():
    out = sys.argv[1]
    S1 = torch.load(f"{UP.FIT}/stage1.pt", map_location="cpu", weights_only=False)
    V = torch.load(f"{UP.FIT}/filler_head.pt", map_location="cpu", weights_only=False)
    F = torch.load(f"{UP.FIT}/form_head.pt", map_location="cpu", weights_only=False)
    T = {}
    for s, (W, B) in enumerate(zip(S1["W"], S1["B"])):                     # frame stage 1 (per source)
        T[f"frame_stage1.W.{s}"] = W.double(); T[f"frame_stage1.B.{s}"] = B.double()
    T["frame_stage2.A"] = V["frame_A"].double(); T["frame_stage2.b"] = V["frame_b"].double()
    heads = [(l, h, w) for l, h, w in V["heads"] if w != 0.0]               # alignment heads (order preserved)
    T["heads.layer"] = torch.tensor([l for l, _, _ in heads], dtype=torch.int64)
    T["heads.head"] = torch.tensor([h for _, h, _ in heads], dtype=torch.int64)
    T["heads.weight"] = torch.tensor([w for _, _, w in heads], dtype=torch.float64)
    T["gate.stage1_g.W"] = V["W1m_g"].double();     T["gate.stage1_g.B"] = V["B1m_g"].double()      # filler decision
    T["gate.stage1_g1.W"] = V["W1m_g1"].double();   T["gate.stage1_g1.B"] = V["B1m_g1"].double()
    T["gate.stage2.W"] = V["W2m"].double();         T["gate.stage2.b"] = V["b2m"].double()
    T["gate.stage2_end.W"] = V["W2m_eos"].double(); T["gate.stage2_end.b"] = V["b2m_eos"].double()
    T["gate.iso_x"] = V["iso_x"].double();         T["gate.iso_p"] = V["iso_p"].double()
    T["gate.aud_mu"] = V["aud_mu"].double();       T["gate.audn_mu"] = V["audn_mu"].double()
    T["form.comb_w"] = F["wc"].double();            T["form.comb_b"] = f64(F["bc"])                # uh / um decision
    T["form.frame_w"] = F["frame_w"].double();      T["form.frame_b"] = f64(F["frame_b"])
    T["form.iso_x"] = F["iso_x"].double();         T["form.iso_p"] = F["iso_p"].double()
    T["form.af_mu"] = f64(F["af_mu"])
    T = {k: v.contiguous() for k, v in T.items()}
    cfg = {
        "name": "umai",
        "asr_model": "Qwen/Qwen3-ASR-1.7B",
        "language": "English",
        "context": "",
        "max_new_tokens": 260,
        "eos_token_ids": [151645, 151643],
        "gap_classes": list(V["classes"]),
        "frame_classes": list(S1["classes"]),
        "filler_class": "F",
        "forms": ["uh", "um"],
        "form_token_ids": {"uh": [int(t) for t in F["uh_ids"]], "um": [int(t) for t in F["um_ids"]]},
        "n_frame_sources": len(S1["W"]),
        "write_threshold": 0.5,
        "um_threshold": 0.5,
        "um_prior_cal": float(F["um_prior"]),
        "filler_prior_cal": {k: int(v) for k, v in V["nat_cal"].items()},
        "notes": {"gate": V["note"], "form": F["note"], "stage1": S1["note"], "feature_order": V["feature_order"]},
    }
    os.makedirs(out, exist_ok=True)
    save_file(T, os.path.join(out, "umai_head.safetensors"), metadata={"format": "pt", "name": cfg["name"]})
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)
    print("head:", len(T), "tensors ->", out, flush=True)


if __name__ == "__main__":
    main()
