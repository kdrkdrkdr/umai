"""UMAI verbatim head — a closed-form classifier holding all statistics computed from calibration data as buffers.

No Qwen3-ASR weights are changed. This module reads the intermediate values produced during a single Qwen greedy transcription
(decoder hidden states, alignment-head attention, audio encoder outputs) and, for each gap between words,
  1) was there a filler  P(F)       -> filler_probability()
  2) if so, uh or um  P(um)        -> form_probability()
computes these. All numbers come from umai_head.safetensors + config.json (files produced by calibration/steps/10_export_head.py).

Computation order (as called by model.py):
  FrameAccumulator   per-audio-frame 5-class (S W F Efrag Erep) log posteriors + uh/um frame llr, accumulated as encoder layers run (once per utterance)
  (attention.py)     EM-weighted mixture of alignment-head attention a = sum_h pi_h a_h / sum
  pool_audio         10-dim audio evidence pooling frame posteriors over the audio window (window from model.py)
  filler_probability decoder states (h_g, h_g1) + audio + next-word window + previous-word Qwen filler
                     -> 5-class scores -> logit P(F) -> isotonic calibration P(F)
  form_probability   Qwen's own um/uh probability ratio lm@g (computed in model.py) + audio form score AF -> isotonic P(um)
Reference helpers not used by the engine: frame_scores (all frame sources at once, same result as FrameAccumulator) and
attention_mixture (the plain head mixture that attention.AlignmentHeads.mixtures reproduces).

Folding two linear stages into one (_fold, once at load time):
  Calibration fits two stages — stage 1: 5-class LDA scores per layer (or frame source), stage 2: an LDA over those concatenated scores.
  Both stages are linear, so 'stage-1 matrix then stage-2 matrix' is mathematically equal to a single pre-multiplied matrix.
  So inference uses the folded matrix (no intermediate scores, float64), and the original stage-1/stage-2 matrices are kept in the file
  (for per-layer contribution analysis and checking; the check is tests/check_fold.py).
    filler  stage1_{g,g1} (28 layers x 5 classes) + stage-1 columns of stage2  -> filler_dec_{g,g1} (5 classes, 28 layers, d) + bias
    frame  frame_stage1 (25 sources x 5 classes) + frame_stage2  -> frame_W.{s} (5 classes, d_s) + bias

dtype rules: hidden states and frame inputs come in as float32; products with calibration values (float64), frame posteriors, audio-window pooling, stage-2 combination and isotonic calibration are float64.
"""
import json

import torch
from torch import nn

N_SCORE_DIFFS = 4           # stage-1 scores: 4 differences of the 5 classes from the reference class (O or S)
N_AUDIO = 10                # number of audio-window features (5 p, 5 log p)


class VerbatimHead(nn.Module):
    def __init__(self, tensors: dict, config: dict):
        super().__init__()
        self.config = config
        for name, value in tensors.items():
            self.register_buffer(name.replace(".", "__"), value.clone(), persistent=True)
        self.n_frame_sources = int(config["n_frame_sources"])
        self.gap_classes = list(config["gap_classes"])          # ['O', 'F', 'E', 'U', 'Q']
        self.frame_classes = list(config["frame_classes"])      # ['S', 'W', 'F', 'Efrag', 'Erep']
        self.i_filler = self.gap_classes.index(config["filler_class"])
        self.i_frame_filler = self.frame_classes.index(config["filler_class"])
        self.write_threshold = float(config["write_threshold"])
        self.um_threshold = float(config["um_threshold"])
        self.form_token_ids = {k: [int(t) for t in v] for k, v in config["form_token_ids"].items()}
        self._fold()

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_file(cls, path: str, device="cpu"):
        """Read the calibration files. path = directory (umai_head.safetensors + config.json) or a .safetensors file
        (the config is then config.json in the same directory)."""
        import os
        from safetensors import safe_open
        st_path = os.path.join(path, "umai_head.safetensors") if os.path.isdir(path) else path
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(st_path)), "config.json")
        if not os.path.exists(st_path) or not os.path.exists(cfg_path):
            raise FileNotFoundError(f"UMAI head not found: need {st_path} and {cfg_path}")
        tensors = {}
        with safe_open(st_path, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        with open(cfg_path) as fh:
            config = json.load(fh)
        return cls(tensors, config).to(device)

    def _buf(self, name):
        return getattr(self, name.replace(".", "__"))

    def _scalar(self, name) -> float:
        """0-dim float64 buffer -> Python float."""
        return float(self._buf(name))

    def _add(self, name, value):
        """Folded value (buffer not saved to file)."""
        self.register_buffer(name.replace(".", "__"), value, persistent=False)

    @staticmethod
    def _isotonic(x: float, xs: torch.Tensor, ps: torch.Tensor) -> float:
        """Step-function isotonic calibration: ps[min(searchsorted(xs, x), len-1)] (float64). Uses the CPU copy of the tables (_tables) —
        searchsorted only compares, so it picks the same bin on any device, with no GPU round trip."""
        i = int(torch.searchsorted(xs, torch.tensor([x], dtype=torch.float64, device=xs.device)))
        return float(ps[min(i, len(ps) - 1)])

    def _tables(self):
        """CPU float64 copies of the isotonic tables and form combination weights (rebuilt when the device changes)."""
        key = self._buf("gate.iso_x").device
        if getattr(self, "_tab_key", None) != key:
            c = lambda n: self._buf(n).double().cpu()
            self._tab = {n: c(n) for n in ("gate.iso_x", "gate.iso_p", "form.iso_x", "form.iso_p", "form.comb_w")}
            self._tab_key = key
        return self._tab

    # ------------------------------------------------------------------ 0. fold the two linear stages (once at load time)
    def _fold(self):
        b = lambda n: self._buf(n).double()
        k = N_SCORE_DIFFS

        def score_diff(W, B):
            """Matrix and bias (linear) giving the 'difference from the reference class (index 0)' of stage-1 5-class scores g = W x + B."""
            return W[..., 1:, :] - W[..., :1, :], B[..., 1:] - B[..., :1]

        # filler: weights the leading columns of stage-2 W apply to stage-1 scores (layer l, difference c) -> matrix applied directly to per-layer states
        def fold_decoder(which, cols):
            D, DB = score_diff(b(f"gate.stage1_{which}.W"), b(f"gate.stage1_{which}.B"))   # (28,4,d), (28,4)
            C = cols.reshape(cols.shape[0], -1, k)                                          # (5,28,4)
            return torch.einsum("jlc,lcd->jld", C, D), torch.einsum("jlc,lc->j", C, DB)     # (5,28,d), (5,)
        L = self._buf("gate.stage1_g.W").shape[0]; n1 = L * k                               # number of stage-1 score columns (112)
        W2, W2e = b("gate.stage2.W"), b("gate.stage2_end.W")
        for name, (M, bias) in {"filler_dec_g": fold_decoder("g", W2[:, :n1]),
                                "filler_dec_g1": fold_decoder("g1", W2[:, n1:2 * n1]),
                                "filler_end_dec_g": fold_decoder("g", W2e[:, :n1])}.items():
            self._add(f"{name}.W", M); self._add(f"{name}.b", bias)
        self._add("filler_rest.W", W2[:, 2 * n1:]); self._add("filler_end_rest.W", W2e[:, n1:])   # remaining feature columns
        # remaining features: regular gap audio | next-word window | prev-word filler; end gap audio | prev-word filler
        assert W2.shape[1] - 2 * n1 == 2 * N_AUDIO + 1 and W2e.shape[1] - n1 == N_AUDIO + 1, (W2.shape, W2e.shape)

        # frame: source-s columns (4) of stage-2 A x stage-1 score differences of source s -> per-source matrix
        A2, fb = b("frame_stage2.A"), b("frame_stage2.b").clone()
        for s in range(self.n_frame_sources):
            D, DB = score_diff(b(f"frame_stage1.W.{s}"), b(f"frame_stage1.B.{s}"))          # (4,d_s), (4,)
            C = A2[:, k * s:k * s + k]                                                       # (5,4)
            self._add(f"frame_W.{s}", C @ D); fb = fb + C @ DB
        self._add("frame_b", fb)

    # ------------------------------------------------------------------ 1. audio frames
    def frame_scores(self, frame_sources):
        """Reference helper, not used by the engine (model.py uses FrameAccumulator directly).
        frame_sources: [encoder layer 0..23 outputs (A, 1024) ..., projector output (A, 2048)] (float32).

        Returns (log_post (A, 5), form_llr (A,)):
          log_post  per-frame 5-class log posterior = log_softmax(sum_s x_s frame_W_s^T + frame_b)
          form_llr  um-vs-uh log-likelihood ratio of projector frames (input to the form audio evidence AF)
        """
        acc = FrameAccumulator(self, frame_sources[0].shape[0])
        for s, x in enumerate(frame_sources):
            acc.add(s, x)
        return acc.finish()

    # ------------------------------------------------------------------ 2. alignment heads / audio window
    def head_list(self):
        """[(layer, head, weight)] — alignment heads chosen by EM (heads with weight 0 were dropped at the export step (calibration/steps/10_export_head.py); no effect on the sum)."""
        L, H, W = self._buf("heads.layer"), self._buf("heads.head"), self._buf("heads.weight")
        return [(int(l), int(h), float(w)) for l, h, w in zip(L.tolist(), H.tolist(), W.tolist())]

    @staticmethod
    def attention_mixture(head_weights, head_attn):
        """Reference helper, not used by the engine (attention.AlignmentHeads.mixtures computes the same mixture).
        a = sum_h pi_h a_h / sum.  head_attn: (A,) float32 attention per head, in head order (renormalised over audio keys)."""
        a = sum(w * att.float() for w, att in zip(head_weights, head_attn))
        return a / a.sum()

    def pool_audio(self, w, log_post, post=None):
        """10-dim audio evidence [sum_t w p(c|t) (5) | sum_t w log p(c|t) (5)]. Calibration mean if there are no frames.
        post: log_post.exp() precomputed once per utterance (same op, same values; computed here if absent)."""
        if log_post is None:
            return self._buf("gate.aud_mu").double()
        w = w.double()
        return torch.cat([w @ (post if post is not None else log_post.exp()), w @ log_post])

    # ------------------------------------------------------------------ 3. filler presence
    def _decoder_logits(self, name, state):
        """Contribution of decoder state (28, d) to the 5-class scores = sum_l W[:, l] . state_l + b  (folded matrix)."""
        return torch.einsum("jld,ld->j", self._buf(f"{name}.W"), state.double()) + self._buf(f"{name}.b")

    def decoder_logits_batch(self, name, states):
        """Decoder contributions for several gaps at once: states (G, 28, d) -> (G, 5) float64.
        Same formula as _decoder_logits (only the float64 summation order may differ; differences are at float64 rounding level)."""
        return torch.einsum("jld,gld->gj", self._buf(f"{name}.W"), states.double()) + self._buf(f"{name}.b")

    def filler_probability(self, st, st1, aud, end, audn, prevq):
        """5-class scores -> score = log P(F) - log P(not F) -> P(F) = isotonic(score). Returns (P, score).
          st, st1  h_g (state just before word start), h_g1 (one step later) — each (28, d)
          remaining feature order — regular gap: audio (10) | next-word window audio (10) | prev-word Qwen filler (1)
                             end gap (EOS, no g+1 step): audio (10) | prev-word Qwen filler (1)"""
        if end:
            zdec = self._decoder_logits("filler_end_dec_g", st)
        else:
            zdec = self._decoder_logits("filler_dec_g", st) + self._decoder_logits("filler_dec_g1", st1)
        return self.filler_probability_z(zdec, aud, end, audn, prevq)

    def filler_probability_z(self, zdec, aud, end, audn, prevq):
        """Part of filler_probability after the decoder contribution. zdec = dec_g + dec_g1 for a regular gap, end_dec_g for an end gap (float64, (5,))."""
        dev = zdec.device
        if audn is None: audn = self._buf("gate.audn_mu")
        pq = torch.tensor([prevq], dtype=torch.float64, device=dev)
        if end:
            rest = torch.cat([aud.double(), pq])
            z = zdec + self._buf("filler_end_rest.W") @ rest + self._buf("gate.stage2_end.b").double()
        else:
            rest = torch.cat([aud.double(), audn.double(), pq])
            z = zdec + self._buf("filler_rest.W") @ rest + self._buf("gate.stage2.b").double()
        lp = torch.log_softmax(z, 0)
        k = self.i_filler
        score = float(lp[k] - torch.logsumexp(torch.cat([lp[:k], lp[k + 1:]]), 0))
        t = self._tables()
        return self._isotonic(score, t["gate.iso_x"], t["gate.iso_p"]), score

    # ------------------------------------------------------------------ 4. uh / um
    def form_probability(self, lmg, w, log_post, form_llr, pf=None):
        """Form score fsc = [lm@g, AF] . wc + bc,  P(um) = isotonic(fsc).  Returns (P(um), fsc).
          lm@g  Qwen's own log P(um) - log P(uh) (h_g last layer, _form_lm in model.py)
          AF    mean of projector-frame um/uh llr weighted by audio window w times frame P(F)
          pf    log_post[:, F].exp() precomputed once per utterance (same op; computed here if absent)"""
        if log_post is None:
            af = self._scalar("form.af_mu")
        else:
            wp = w.double() * (pf if pf is not None else log_post[:, self.i_frame_filler].exp())
            af = float((wp @ form_llr) / wp.sum()) if float(wp.sum()) > 0 else self._scalar("form.af_mu")
        t = self._tables()
        fsc = float(torch.tensor([float(lmg), af], dtype=torch.float64) @ t["form.comb_w"] + self._scalar("form.comb_b"))
        pum = self._isotonic(fsc, t["form.iso_x"], t["form.iso_p"])
        return pum, fsc


class FrameAccumulator:
    """Accumulate 5-class frame scores as sources (encoder layers 0..23, projector) arrive — same ops and order as frame_scores:
        logits = frame_b;  logits = logits + x_s.double() @ frame_W_s^T  (s = 0, 1, ..., 24)
    so all 25 layer outputs need not be kept. finish() returns (None, None) if the row count differs from n_rows or sources are missing."""

    def __init__(self, head, n_rows):
        self.head, self.n_rows, self.next, self.ok = head, n_rows, 0, True
        self.logits = head._buf("frame_b").expand(n_rows, -1).clone()

    def add(self, s, x):
        x = x.reshape(-1, x.shape[-1])
        if not self.ok or s != self.next or x.shape[0] != self.n_rows:
            self.ok = False; return
        self.logits = self.logits + x.double() @ self.head._buf(f"frame_W.{s}").T
        self.next += 1
        if s == self.head.n_frame_sources - 1:
            self.last = x.float()

    def finish(self):
        if not self.ok or self.next != self.head.n_frame_sources:
            return None, None
        log_post = torch.log_softmax(self.logits, 1)                  # float64 (same precision as the frame posteriors in calibration 08/09)
        form_llr = self.last.double() @ self.head._buf("form.frame_w").double() + self.head._scalar("form.frame_b")   # float64, as in calibration 09
        return log_post, form_llr
