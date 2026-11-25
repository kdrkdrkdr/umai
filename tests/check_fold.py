"""Check: does the inference 'folded matrix' give the same values as the two-stage calibration computation (stage-1 per-layer/per-source LDA scores -> stage-2 LDA)? (CPU, float64)
For filler scores, the full stage-2 input order (stage-1 scores + audio + next-word window + previous-word filler) is also checked.
The original computation is done here directly from the stage-1/stage-2 buffers in the head file. Inputs are random (a linear identity, so results must match for any input).
  python tests/check_fold.py [HEAD_DIR]      (default heads/default)  -> OK if all relative errors are below 1e-9
"""
import os, sys
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "inference"))
from umai.head import VerbatimHead


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def main():
    head = VerbatimHead.from_file(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "heads", "default")).double()
    b = head._buf; g = torch.Generator().manual_seed(0); out = {}

    def stage1(state, which):                 # original stage 1: per-layer 5-class score differences vs O (112,)
        s = torch.einsum("ld,lcd->lc", state, b(f"gate.stage1_{which}.W")) + b(f"gate.stage1_{which}.B")
        return (s[:, 1:] - s[:, :1]).flatten()

    L, _, d = b("gate.stage1_g.W").shape
    st, st1 = torch.randn(L, d, generator=g, dtype=torch.float64), torch.randn(L, d, generator=g, dtype=torch.float64)
    n1 = 4 * L
    W2, W2e = b("gate.stage2.W"), b("gate.stage2_end.W")
    out["filler (regular gap) decoder part"] = rel(head._decoder_logits("filler_dec_g", st) + head._decoder_logits("filler_dec_g1", st1),
                                     W2[:, :2 * n1] @ torch.cat([stage1(st, "g"), stage1(st1, "g1")]))
    out["filler (end gap) decoder part"] = rel(head._decoder_logits("filler_end_dec_g", st), W2e[:, :n1] @ stage1(st, "g"))

    # full filler decision: stage-2 input [stage1 g | stage1 g1 | audio 10 | next-word window 10 | prev-word filler 1] (end gap: [stage1 g | audio | prev-word filler])
    def pf_score(z):
        lp = torch.log_softmax(z, 0); k = head.i_filler
        return lp[k] - torch.logsumexp(torch.cat([lp[:k], lp[k + 1:]]), 0)
    aud, audn = torch.rand(10, generator=g, dtype=torch.float64), torch.rand(10, generator=g, dtype=torch.float64)
    for pq in (0.0, 1.0):
        ref = pf_score(W2 @ torch.cat([stage1(st, "g"), stage1(st1, "g1"), aud, audn, torch.tensor([pq], dtype=torch.float64)]) + b("gate.stage2.b"))
        _, new = head.filler_probability(st, st1, aud, False, audn=audn, prevq=pq)
        out[f"filler score (regular gap, prev filler {int(pq)})"] = rel(torch.tensor([new], dtype=torch.float64), ref.reshape(1))
        ref = pf_score(W2e @ torch.cat([stage1(st, "g"), aud, torch.tensor([pq], dtype=torch.float64)]) + b("gate.stage2_end.b"))
        _, new = head.filler_probability(st, st1, aud, True, audn=None, prevq=pq)
        out[f"filler score (end gap, prev filler {int(pq)})"] = rel(torch.tensor([new], dtype=torch.float64), ref.reshape(1))

    X = [torch.randn(7, b(f"frame_stage1.W.{s}").shape[1], generator=g, dtype=torch.float64) for s in range(head.n_frame_sources)]
    feats = []
    for s, x in enumerate(X):
        z = x @ b(f"frame_stage1.W.{s}").T + b(f"frame_stage1.B.{s}"); feats.append(z[:, 1:] - z[:, :1])
    ref = torch.cat(feats, 1) @ b("frame_stage2.A").T + b("frame_stage2.b")
    new = b("frame_b") + sum(x @ b(f"frame_W.{s}").T for s, x in enumerate(X))
    out["frame 5-class scores"] = rel(new, ref)

    ok = all(v < 1e-9 for v in out.values())
    for k, v in out.items(): print(f"  {k:30s} relative error {v:.1e}")
    print("OK" if ok else "FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
