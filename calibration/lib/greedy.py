"""Shared helpers for Qwen3-ASR greedy transcription: audio loading, model loading, finding word-start steps."""
import os
import sys
import unicodedata
import soundfile as sf
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import paths as UP  # calibration/paths.py
EOS = [151645, 151643]


def audio(r):
    """Segment wav of a row (16 kHz) -> float32 mono."""
    x, sr = sf.read(r["audio"], dtype="float32")
    assert sr == 16000, (r["audio"], sr)
    return x.mean(1) if x.ndim > 1 else x


def load_model(bs, max_new_tokens=260):
    """Load Qwen3-ASR with the official prompt (empty context, language English)."""
    from qwen_asr import Qwen3ASRModel
    m = Qwen3ASRModel.from_pretrained(UP.ASR_MODEL, dtype=getattr(torch, UP.DTYPE), device_map="cuda:0",
                                      max_inference_batch_size=bs, max_new_tokens=max_new_tokens)
    m.model.eval()
    return m, m._build_text_prompt(context="", force_language="English")


def steps_of(tok, toks, ended):
    """Decision steps of a greedy token list: [(slot, j, wordstart, boundary)]. j = index of the token emitted at that step
    (j == len(toks) is the eos step), slot = number of words written so far."""
    cache = {}
    def dec(n):
        if n not in cache: cache[n] = len(tok.decode(toks[:n], skip_special_tokens=True).split())
        return cache[n]
    out = []
    for j, t in enumerate(toks):
        nb, na = dec(j), dec(j + 1)
        s = tok.convert_ids_to_tokens(t) or ""; d = tok.decode([t]).strip()
        bnd = j == 0 or s.startswith("Ġ") or (bool(d) and all(unicodedata.category(ch)[0] in "PS" and ch not in "'-" for ch in d))
        if na > nb or bnd: out.append((nb, j, na > nb, bnd))
    if ended: out.append((dec(len(toks)), len(toks), True, True))
    return out
