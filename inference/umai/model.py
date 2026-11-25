"""UMAI: transcription with a verbatim head (VerbatimHead) on top of a frozen Qwen3-ASR-1.7B.

Per-utterance processing order
  1. Qwen runs greedy transcription 'once' with the official prompt (empty context, language English).
     The text is Qwen's unchanged (token selection untouched).
  2. Within the same call, only the intermediate values used for decisions are captured on the spot (recording.py, subclassed modules, no hooks, no output_hidden_states).
       last-position row of decoder layer 0..27 outputs (layer 27 output = value just before final norm)
       alignment layers: prefill input_layernorm output -> audio-position k (computed directly), decode-step q_proj output
       24 audio encoder layer outputs and projector frames -> added directly to the 5-class frame scores (layer outputs not kept)
  3. Find gaps between words: boundaries are the first step of each word unit (whitespace piece with ASCII letters/digits; 'uh huh' counts as one) and the EOS step;
     boundaries other than where Qwen itself wrote uh/um are gaps.
  4. For each gap, the head receives h_g (state at that step), h_g1 (state one step later), the audio window and next-word window from the mixed alignment-head attention,
     and whether the previous word is a Qwen filler. For the form (uh/um), the um/uh probability ratio Qwen assigns at h_g is also used
     (only the 10 uh/um token rows of lm_head are applied, no full-vocabulary softmax).
  5. VerbatimHead: write a filler if P(F) > write_threshold (config.json); [um] if P(um) > 0.5, else [uh].
  6. Qwen's text is left unchanged; only tags are inserted between words.

No batched inference yet (one utterance at a time).
Runs on a CUDA GPU in bfloat16 only: the head was calibrated on bfloat16 states, and other precisions or CPU are not supported.
"""
import logging
import os
import re
import sys

import numpy as np
import torch

from .attention import AlignmentHeads
from .head import FrameAccumulator, VerbatimHead
from .recording import install_recorders

log = logging.getLogger("umai")

DTYPE = torch.bfloat16                 # precision of Qwen3-ASR during calibration; fixed
_WORD_CHARS = re.compile(r"[^a-z0-9'-]")
FILLERS = ("uh", "um")


# ---------------------------------------------------------------------- word units (same definition as calibration/lib/align.py)
def _form(w):
    """Word with only punctuation removed (Unicode letters kept): 'Uh,' -> 'uh', 'uhん' -> 'uhん'."""
    return re.sub(r"[^\w'-]|_", "", w.lower())


def is_filler(w):
    """Filler written by Qwen itself: the whole word without punctuation is uh or um (forms like 'uhん' are regular words)."""
    return _form(w) in FILLERS


def word_units(words):
    """Whitespace pieces -> (word-unit start indices, unit spellings). Pieces without ASCII letters/digits are dropped; a 'huh' right after 'uh' is attached to it."""
    starts, toks = [], []
    for j, w in enumerate(words):
        if not re.search(r"[a-z0-9]", w.lower()): continue      # punctuation only, '--', Han only: not a unit
        if _form(w) == "huh" and starts and starts[-1] == j - 1 and _form(words[j - 1]) == "uh":
            toks[-1] = "uh-huh"; continue
        starts.append(j); toks.append("#fp" if is_filler(w) else _WORD_CHARS.sub("", w.lower()))
    return starts, toks


def spellings(w):
    """Token spellings of a filler word (with/without leading space x lowercase/capitalised/uppercase)."""
    return [p + v for v in (w, w.capitalize(), w.upper()) for p in ("", " ")]


# ---------------------------------------------------------------------- audio
def _to_float(a):
    """Array -> float32; integer PCM is scaled to [-1, 1) by its type's range."""
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.unsignedinteger):
        raise ValueError("unsigned integer audio is not supported; pass float or signed integer PCM")
    if np.issubdtype(a.dtype, np.integer):
        if a.dtype.itemsize not in (2, 4):
            raise ValueError("integer audio must be int16 or int32 PCM; pass float otherwise")
        return (a.astype(np.float32) / float(-np.iinfo(a.dtype).min)).astype(np.float32)
    return a.astype(np.float32)


def load_audio(audio, target_sr: int):
    """Path / (array, sr) / array already at target_sr -> float32 mono target_sr.
    Multi-channel input is averaged over channels: a file read by soundfile is (frames, channels);
    a 2-D array is taken as channels-first when shape[0] < shape[1], else as (frames, channels).
    If not at target_sr, resample with librosa (same method as the official qwen_asr code) and log it."""
    from_file = False
    if isinstance(audio, (str, os.PathLike)):
        import soundfile as sf
        try:
            x, sr = sf.read(str(audio), dtype="float32")
        except Exception as e:                               # missing file, corrupt file, unsupported format
            raise ValueError(f"cannot read audio {audio}: {e}") from e
        from_file = True
    elif isinstance(audio, tuple):
        x, sr = audio
        x = _to_float(x)
    else:
        x, sr = _to_float(audio), target_sr
    if x.size == 0:
        raise ValueError("empty audio")
    if x.ndim > 1:
        if from_file:
            x = x.mean(1)                                    # soundfile: (frames, channels)
        elif x.ndim == 2:
            x = x.mean(0 if x.shape[0] < x.shape[1] else 1)  # channels-first if fewer rows than columns
        else:
            raise ValueError(f"audio array must be 1-D or 2-D, got shape {x.shape}")
    if sr != target_sr:
        import librosa
        log.warning("resampling audio %s Hz -> %s Hz", sr, target_sr)
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    if x.size == 0:
        raise ValueError("empty audio")
    return x


class UMAI:
    """Usage:  m = UMAI.from_pretrained(head_path); m.transcribe("a.wav")      (CUDA GPU, bfloat16)"""

    def __init__(self, asr, head: VerbatimHead, device):
        self.asr, self.head, self.dtype, self.device = asr, head, DTYPE, device
        cfg = head.config
        self.processor = asr.processor
        self.tok = self.processor.tokenizer
        self.th = asr.model.thinker
        self.max_new_tokens = int(cfg["max_new_tokens"])
        self.eos = [int(t) for t in cfg["eos_token_ids"]]
        self.prompt = asr._build_text_prompt(context=cfg["context"], force_language=cfg["language"])
        self.audio_token_id = getattr(self.th.config, "audio_token_id", None) or getattr(self.th.config, "audio_token_index")
        self.sampling_rate = int(self.processor.feature_extractor.sampling_rate)
        self.n_layers = len(self.th.model.layers)
        att0 = self.th.model.layers[0].self_attn
        rotary = sys.modules[type(att0).__module__].apply_rotary_pos_emb
        self.heads = head.head_list()
        self.align = AlignmentHeads(self.th, self.heads, rotary)
        self.rec = install_recorders(self.th, set(self.align.layers))
        # Qwen output-layer slice for the form decision: only the final norm weight and lm_head rows of uh/um tokens (float32 copy, 10 x d)
        ids = {f: self._single_token_ids(f) for f in FILLERS}
        if ids != head.form_token_ids:
            raise ValueError(f"uh/um token ids {ids} differ from the head's {head.form_token_ids} (different tokenizer?)")
        self._n_uh = len(ids["uh"])
        self._form_rows = self.th.lm_head.weight[ids["uh"] + ids["um"]].float().clone()
        self._norm_w = self.th.model.norm.weight.float()
        self._norm_eps = float(getattr(self.th.model.norm, "variance_epsilon", 1e-6))

    @classmethod
    def from_pretrained(cls, head_path=None, device="cuda:0", asr_model=None):
        """head_path: directory containing umai_head.safetensors + config.json, or that .safetensors file.
          Relative paths are resolved against the current directory. If omitted, the env var UMAI_HEAD is used;
          if that is unset too, the repository's heads/default is used when present.
        device: a CUDA device. Qwen3-ASR always runs in bfloat16 (the precision the head was calibrated in).
        asr_model: Qwen3-ASR name/path (if omitted, UMAI_ASR_MODEL, then asr_model from the config)."""
        if not str(device).startswith("cuda"):
            raise ValueError(f"device {device!r} is not supported: UMAI runs on a CUDA GPU in bfloat16, the setting the head was calibrated in")
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU available: UMAI runs on a CUDA GPU in bfloat16, the setting the head was calibrated in")
        head_path = head_path or os.environ.get("UMAI_HEAD")
        if not head_path:
            repo_default = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "heads", "default"))
            if not os.path.isdir(repo_default):
                raise ValueError("head_path not given, UMAI_HEAD not set, and no heads/default next to the package")
            head_path = repo_default
        head = VerbatimHead.from_file(head_path, device=device)
        from qwen_asr import Qwen3ASRModel
        name = asr_model or os.environ.get("UMAI_ASR_MODEL") or head.config["asr_model"]
        asr = Qwen3ASRModel.from_pretrained(name, dtype=DTYPE, device_map=device, max_inference_batch_size=1,
                                            max_new_tokens=int(head.config["max_new_tokens"]))
        asr.model.eval()
        return cls(asr, head, device)

    # ------------------------------------------------------------------ one utterance
    @torch.no_grad()
    def transcribe(self, audio):
        """audio: wav path, (array, sr), or 16 kHz array.
        Returns {'text': Qwen text, 'verbatim': text with tags, 'gaps': [...], 'truncated': bool}."""
        x = load_audio(audio, self.sampling_rate)
        inp = self.processor(text=[self.prompt], audio=[x], return_tensors="pt", padding=True).to(self.device).to(self.dtype)
        audio_pos = (inp["input_ids"][0] == self.audio_token_id).nonzero().flatten()
        n_audio, S = len(audio_pos), inp["input_ids"].shape[1]
        self.th.rope_deltas = None
        rec = self.rec
        frames = FrameAccumulator(self.head, n_audio)
        rec.begin()
        self.align.begin(rec, audio_pos, S)
        rec.enc_cb = frames.add                                   # encoder layer i output -> frame source i
        rec.embed_cb = lambda h: frames.add(len(self.th.audio_tower.layers), h[0, audio_pos])   # projector frames (audio rows of the input embeddings)
        try:
            out = self.asr.model.generate(**inp, max_new_tokens=self.max_new_tokens, eos_token_id=self.eos, do_sample=False)
        finally:
            rec.end()
        tokens = out.sequences[0, S:].tolist()
        if not (rec.step + 1 == len(rec.rows) == len(tokens)):
            raise RuntimeError(f"recorded steps do not match generated tokens: step={rec.step}, rows={len(rec.rows)}, tokens={len(tokens)}")
        log_post, form_llr = frames.finish()
        if log_post is None:
            log.warning("audio encoder rows != audio tokens; audio evidence = calibration mean")
        try:
            gaps, toks = self._decide(tokens, log_post, form_llr)
        finally:
            rec.rows = rec.pe = None; self.align.release()         # release recordings when the utterance is done
        text = self.tok.decode(toks, skip_special_tokens=True).strip()
        truncated = not any(t in self.eos for t in tokens)
        if truncated:
            log.warning("output cut at max_new_tokens=%d with no EOS: there is no end-of-utterance gap "
                        "(a word's gap is an end gap only if the word starts at the final step)", self.max_new_tokens)
        if not text:
            log.warning("empty transcription")
        return {"text": text, "verbatim": self.tag_text(text, gaps), "gaps": gaps, "truncated": truncated}


    # ------------------------------------------------------------------ internals
    def _decide(self, tokens, log_post, form_llr):
        """Gather decision features for each gap and produce P(filler) and the form. Computed from recorded values after generation ends,
        so next-word information comes from the same single inference pass.
          Word units, gaps and boundaries use the same definitions as calibration (calibration/lib/align.py word_units, 04/08):
            boundary = first step of a word unit with ASCII letters/digits (+ eos step). Punctuation-only pieces are not boundaries.
            'uh huh' is one word, so the position before huh is neither a gap nor a boundary.
            gap = a boundary that is not where Qwen itself wrote a filler (uh/um).
            end gap = gap with no following step (eos step, or last token step of a row cut at max tokens) -> end-gap model.
          filler: decoder states h_g, h_g1 (per-layer stage-1 LDA), audio window (previous boundary ~ this step) and next-word window (this step ~ next boundary),
                whether the previous piece is a filler written by Qwen.
          form: Qwen's um/uh probability ratio at h_g + uh/um score over the audio window."""
        head = self.head
        # 1) word count up to each step: a step where the count increases is the first step of that word (slot = count before the increase)
        toks, first, nw, eos_n = [], {}, 0, None
        for n, t in enumerate(tokens):
            if t in self.eos:
                eos_n = n; break
            na = len(self.tok.decode(toks + [t], skip_special_tokens=True).split())
            if na > nw: first[nw] = n
            toks.append(t); nw = na
        words = self.tok.decode(toks, skip_special_tokens=True).split()
        us, ut = word_units(words); unit = dict(zip(us, ut))
        bounds = [(first[s], s) for s in us if s in first]
        if eos_n is not None: bounds.append((eos_n, len(words)))
        last_n = len(tokens) - 1                                 # no step follows this one
        # 2) list of gaps to decide
        todo = [(i, n, slot, n >= last_n) for i, (n, slot) in enumerate(bounds) if unit.get(slot) != "#fp"]
        if not todo:
            return [], toks
        # 3) what can be computed at once: mixed alignment attention (all boundary steps), folded 5-class decoder contributions
        mix = self.align.mixtures(sorted({n for n, _ in bounds}))
        rows = self.rec.rows
        need = sorted({n for _, n, _, _ in todo} | {n + 1 for _, n, _, end in todo if not end})
        pos = {k: j for j, k in enumerate(need)}
        states = torch.cat([r for k in need for r in rows[k]]).view(len(need), self.n_layers, -1).float()   # (K, 28, d)
        g_idx = [pos[n] for _, n, _, _ in todo]
        mid = [t for t in range(len(todo)) if not todo[t][3]]
        Zg = head.decoder_logits_batch("filler_dec_g", states[g_idx])
        Zend = head.decoder_logits_batch("filler_end_dec_g", states[g_idx])
        Zg1 = head.decoder_logits_batch("filler_dec_g1", states[[pos[todo[t][1] + 1] for t in mid]]) if mid else None
        z_mid = {t: j for j, t in enumerate(mid)}
        post = log_post.exp() if log_post is not None else None
        pf = post[:, head.i_frame_filler].contiguous() if post is not None else None
        cum = {}
        def F(n):                                                # CDF (each step computed once — same op, same values)
            if n not in cum: cum[n] = torch.cumsum(mix[n], 0)
            return cum[n]
        gaps = []
        for t, (i, n, slot, end) in enumerate(todo):
            a = mix[n]
            Fg = F(n); Fp = F(bounds[i - 1][0]) if i > 0 else torch.ones_like(Fg)
            w = Fp * (1 - Fg).clamp_min(0); m1 = float(w.sum())  # only negatives from CDF rounding are set to 0 (same as calibration)
            w = w / w.sum() if m1 > 0 else a
            aud = head.pool_audio(w, log_post, post)
            pum, fsc = head.form_probability(self._form_lm(states[g_idx[t], -1]), w, log_post, form_llr, pf)
            form = "um" if pum > head.um_threshold else "uh"
            audn = None                                          # next-word window: F_g (1 - F_next_boundary)
            if i + 1 < len(bounds) and not end:
                w2 = Fg * (1 - F(bounds[i + 1][0])).clamp_min(0); m2 = float(w2.sum())
                if m2 > 0 and log_post is not None: audn = head.pool_audio(w2 / m2, log_post, post)
            prevq = float(slot >= 1 and slot - 1 < len(words) and is_filler(words[slot - 1]))
            zdec = Zend[t] if end else Zg[t] + Zg1[z_mid[t]]
            P, score = head.filler_probability_z(zdec, aud, end, audn, prevq)
            gaps.append({"slot": slot, "n": n, "P": round(P, 4), "score": round(score, 3),
                         "form": form, "pum": round(pum, 4), "fsc": round(fsc, 3)})
        return gaps, toks

    def _single_token_ids(self, w):
        """ids of the spellings of filler word w that encode to a single token (same method as calibration 09_fit_form.py)."""
        out = set()
        for v in spellings(w):
            e = self.tok.encode(v, add_special_tokens=False)
            if len(e) == 1: out.add(e[0])
        return sorted(out)

    def _form_lm(self, h):
        """lm@g = log P(um) - log P(uh): apply final norm to the last-layer state h (before norm) and multiply only the uh/um token rows.
        The log_softmax normaliser cancels in the subtraction, so this equals the logsumexp difference of the logits (same definition as calibration 09_fit_form.py)."""
        h = h.float(); h = h * torch.rsqrt(h.pow(2).mean() + self._norm_eps) * self._norm_w
        z = self._form_rows @ h
        return float(torch.logsumexp(z[self._n_uh:], 0) - torch.logsumexp(z[:self._n_uh], 0))

    def tag_text(self, text, gaps):
        """Insert [uh]/[um] before gaps with P(F) > write_threshold."""
        ws = text.split()
        ins = {}
        for g in gaps:
            if g["P"] > self.head.write_threshold and g["slot"] not in ins:
                ins[g["slot"]] = g
        out = []
        for i in range(len(ws) + 1):
            if i in ins:
                out.append(f"[{ins[i]['form']}]")
            if i < len(ws):
                out.append(ws[i])
        return " ".join(out)

