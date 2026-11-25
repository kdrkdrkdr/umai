"""Alignment-head attention (without modifying model code or using output_attentions).

Why recompute: transformers' output_attentions may force eager attention instead of sdpa,
which risks changing Qwen's computation (and thus its text). Instead, rebuild exactly the q, k used by that layer's attention:
    x = input_layernorm(h_l)
    q = rope(q_norm(q_proj(x)))   k = rope(k_norm(k_proj(x)))
    a_h = softmax over audio keys of (q_h . k_h) * scaling
= the attention the model actually used, renormalised over the audio keys.

How computation is reduced while giving the same values as the reference computation:
  - k: audio-position k is built right where the alignment layer's input_layernorm output is received during prefill (same op and input as the reference computation).
  - q: decode steps take the model's own q_proj output (the reference computation recomputes q_proj on the same (1,1,d) input — bit-identical).
       Step 0 computes q_proj on the last row (1,1,d) as in the reference computation (the prefill (1,S,d) matmul uses a different kernel and may differ).
       Head selection, q_norm and rope are then row-independent, so steps are batched into one pass.
  - Scores q_h·k_h use the same (1,hd)x(hd,A) matmul as the reference computation per (step, head) — a step-batched matmul switches cuBLAS kernels
    and was observed to change float32 values. softmax and the head mixture sum_h pi_h a_h (elementwise sums in the same order) are batched over steps.
"""
import torch


def _pe_rows(pe, rows):
    cos, sin = pe
    return cos.index_select(-2, rows), sin.index_select(-2, rows)


def project(attn, rotary_apply, x, pe, rows, which, heads=None):
    """x (B,S,d) normalised layer input, rows (P,) positions -> (B, n_heads, P, head_dim) q or k (after norm + rope)."""
    hd = attn.head_dim
    x = x.index_select(1, rows)
    lin, nrm = (attn.q_proj, attn.q_norm) if which == "q" else (attn.k_proj, attn.k_norm)
    v = lin(x).view(x.shape[0], x.shape[1], -1, hd)
    if heads is not None:
        v = v[:, :, heads]
    v = nrm(v).transpose(1, 2)
    cos, sin = _pe_rows(pe, rows)
    v, _ = rotary_apply(v, v, cos, sin)
    return v


class AlignmentHeads:
    """Alignment-head attention calculator for one utterance. Stores k and per-step q_proj outputs via Recorder callbacks;
    mixtures(steps) returns the EM-mixed attention a (A,) float32 for each step."""

    def __init__(self, thinker, heads, rotary_apply):
        self.th = thinker
        self.heads = heads                                   # [(layer, head, weight)]
        self.rotary_apply = rotary_apply
        self.layers = sorted({l for l, _, _ in heads})
        self.heads_of = {l: [h for ll, h, _ in heads if ll == l] for l in self.layers}
        cfg = thinker.config.text_config
        self.group = cfg.num_attention_heads // cfg.num_key_value_heads
        self.weights = [w for _, _, w in heads]
        # head list order i -> (index within layer j, kv head)
        self.order = [(l, self.heads_of[l].index(h), h // self.group) for l, h, _ in heads]

    def begin(self, rec, audio_pos, S):
        """Call before generate. Attaches callbacks to rec."""
        self.audio_pos = audio_pos
        self.last = torch.tensor([S - 1], device=audio_pos.device)
        self.Kf, self.x0, self.q = {}, {}, {l: [] for l in self.layers}
        rec.layer_cb, rec.q_cb = self._on_prefill_norm, self._on_q
        self.rec = rec

    def release(self):
        self.Kf = self.x0 = self.q = None; self.rec = None

    def _on_prefill_norm(self, l, xn, pe0):
        att = self.th.model.layers[l].self_attn
        self.Kf[l] = project(att, self.rotary_apply, xn, pe0, self.audio_pos, "k")[0].float()   # (n_kv, A, hd)
        self.x0[l] = xn.index_select(1, self.last)                                                # (1, 1, d)

    def _on_q(self, l, q):
        self.q[l].append(q)                                                                       # (1, 1, n_heads*hd)

    @torch.no_grad()
    def mixtures(self, steps):
        """steps: list of step indices -> {step: a (A,) float32} (a = sum_h pi_h a_h / sum, same op order as the reference computation, head.attention_mixture)."""
        th, rec = self.th, self.rec
        G = len(steps)
        pe0 = rec.pe[0]
        cos = torch.cat([(pe0[0].index_select(-2, self.last) if k == 0 else rec.pe[k][0]) for k in steps], 1)   # (1, G, hd)
        sin = torch.cat([(pe0[1].index_select(-2, self.last) if k == 0 else rec.pe[k][1]) for k in steps], 1)
        Q = {}
        for l in self.layers:
            att = th.model.layers[l].self_attn
            raw = torch.cat([(att.q_proj(self.x0[l]) if k == 0 else self.q[l][k - 1]) for k in steps], 1)     # (1, G, n_heads*hd)
            v = raw.view(1, G, -1, att.head_dim)[:, :, self.heads_of[l]]
            v = att.q_norm(v).transpose(1, 2)
            v, _ = self.rotary_apply(v, v, cos, sin)                                                          # (1, n_l, G, hd)
            Q[l] = v[0].float()
        acc = 0
        for (l, j, kv), w in zip(self.order, self.weights):
            q, KT = Q[l][j], self.Kf[l][kv].transpose(-1, -2)                                                 # (G, hd), (hd, A)
            sc = torch.empty(G, KT.shape[1], dtype=torch.float32, device=q.device)
            for g in range(G):
                torch.mm(q[g][None], KT, out=sc[g:g + 1])                                                    # same (1,hd)x(hd,A) as the reference computation
            acc = acc + w * torch.softmax(sc * th.model.layers[l].self_attn.scaling, -1)
        out = {}
        for g, k in enumerate(steps):
            a = acc[g].clone()
            out[k] = a / a.sum()
        return out
