"""Capture intermediate values that transformers does not expose, via subclassed modules (no hooks; Qwen's computation is unchanged).

With output_hidden_states on, every step keeps 29 layer tensors (29 of (1, S, d) for prefill) until generation ends.
Decisions use only part of them, so only the needed values are captured on the spot (Recorder):

1. Decoder layers (RecordingDecoderLayer)  last-position row of layer l output (h_g state, 28 layers; layer 27 output = final norm input).
                                      Audio-position rows of layer 0 input (= input embeddings with audio inserted) go to the frame source (projector).
                                      Also captures the position embeddings (cos, sin) used at each step (for rope of the alignment-head q).
2. Alignment-layer input_layernorm (RecordingRMSNorm)  builds audio-position k directly from the prefill output (same op as the reference computation),
                                      and keeps only the last-position row (input for step-0 q). The rest of (1, S, d) is discarded.
3. Alignment-layer q_proj (RecordingLinear)  q_proj output at decode steps (1, 1, n_heads*hd). The reference computation recomputes q_proj on the same (1, 1, d) input
                                      — same input, same op with the same shape, so values are bit-identical.
4. Audio encoder (RecordingAudioEncoder) forward copied from the original (qwen-asr 0.0.6), calling a callback for each layer output
                                      inside the layer loop (added straight into the 5-class frame scores and discarded, head.FrameAccumulator).

install_recorders(thinker, align_layers) only swaps the classes of an already loaded model instance (weights and buffers unchanged).
Nothing is recorded when Recorder.on is False (every subclass forward calls super().forward unchanged).
"""
import sys

import torch
import torch.nn.functional as F
from torch import nn


class Recorder:
    """Records for one utterance. model.transcribe initialises it with begin()."""

    def __init__(self):
        self.on = False
        self.layer_cb = None              # (l, xn, pe0) prefill input_layernorm output -> alignment k
        self.q_cb = None                  # (l, q) decode-step q_proj output
        self.embed_cb = None              # (h) prefill layer 0 input (1, S, d)
        self.enc_cb = None                # (i, h) encoder layer i output

    def begin(self):
        self.on = True
        self.step = -1
        self.rows = []                    # rows[k][l] = last-position row of layer l output at step k (1, d) bf16
        self.pe = []                      # pe[k] = (cos, sin) at step k ((1, S, hd) for prefill)

    def end(self):
        self.on = False
        self.layer_cb = self.q_cb = self.embed_cb = self.enc_cb = None


def _recording_layer_class(base):
    class RecordingDecoderLayer(base):
        """Same computation as base. Appends the last-position row of the layer output to rec.rows."""
        _umai_rec = None
        _umai_idx = -1

        def forward(self, hidden_states, *args, **kwargs):
            rec = self._umai_rec
            if rec is None or not rec.on:
                return super().forward(hidden_states, *args, **kwargs)
            if self._umai_idx == 0:
                rec.step += 1
                rec.rows.append([])
                rec.pe.append(kwargs.get("position_embeddings"))
                if rec.step == 0 and rec.embed_cb is not None:
                    rec.embed_cb(hidden_states)
            out = super().forward(hidden_states, *args, **kwargs)
            # prefill (1, S, d): copy and keep one row only; decode-step (1, 1, d) is kept as is (no copy)
            rec.rows[-1].append(out[0, -1:].clone() if out.shape[1] > 1 else out[0, -1:])
            return out

    return RecordingDecoderLayer


def _recording_norm_class(base):
    class RecordingRMSNorm(base):
        """Alignment-layer input_layernorm. Passes the prefill output to rec.layer_cb (only k and the last row are kept)."""
        _umai_rec = None
        _umai_idx = -1

        def forward(self, hidden_states):
            out = super().forward(hidden_states)
            rec = self._umai_rec
            if rec is not None and rec.on and rec.step == 0 and rec.layer_cb is not None:
                rec.layer_cb(self._umai_idx, out, rec.pe[0])
            return out

    return RecordingRMSNorm


def _recording_linear_class(base):
    class RecordingLinear(base):
        """Alignment-layer q_proj. Passes decode-step (step >= 1) output to rec.q_cb."""
        _umai_rec = None
        _umai_idx = -1

        def forward(self, x):
            out = super().forward(x)
            rec = self._umai_rec
            if rec is not None and rec.on and rec.step >= 1 and rec.q_cb is not None:
                rec.q_cb(self._umai_idx, out)
            return out

    return RecordingLinear


def _recording_encoder_class(base):
    mod = sys.modules[base.__module__]
    _get_feat_extract_output_lengths = mod._get_feat_extract_output_lengths
    BaseModelOutput = mod.BaseModelOutput

    class RecordingAudioEncoder(base):
        """Same computation as base.forward + rec.enc_cb(i, layer i output) for each layer."""
        _umai_rec = None

        def forward(self, input_features, feature_lens=None, aftercnn_lens=None):
            # ---- from here identical to qwen_asr 0.0.6 Qwen3ASRAudioEncoder.forward ----
            aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
            chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

            chunk_lengths = torch.tensor([self.n_window * 2] * chunk_num.sum(), dtype=torch.long, device=feature_lens.device)
            tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
            chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
            chunk_lengths[chunk_lengths == 0] = self.n_window * 2

            chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
            padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
            feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
            padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
                [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
                batch_first=True,
            )
            padded_feature = padded_feature.unsqueeze(1)
            padded_embeds = []
            for chunk in padded_feature.split(self.conv_chunksize, dim=0):
                padded_embed = F.gelu(self.conv2d1(chunk))
                padded_embed = F.gelu(self.conv2d2(padded_embed))
                padded_embed = F.gelu(self.conv2d3(padded_embed))
                padded_embeds.append(padded_embed)
            padded_embed = torch.cat(padded_embeds, dim=0)
            b, c, f, t = padded_embed.size()
            padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

            positional_embedding = (
                self.positional_embedding.positional_embedding[: padded_embed.shape[1], :].unsqueeze(0).to(padded_embed.dtype)
            )
            padded_embed = padded_embed + positional_embedding
            hidden_states = padded_embed[padded_mask_after_cnn]
            cu_chunk_lens = [0]
            window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
            for cnn_len in aftercnn_lens:
                cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
                remainder = cnn_len % window_aftercnn
                if remainder != 0:
                    cu_chunk_lens += [remainder]
            cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

            rec = self._umai_rec
            cb = rec.enc_cb if rec is not None and rec.on else None   # (added)
            for i, encoder_layer in enumerate(self.layers):
                layer_out = encoder_layer(hidden_states, cu_seqlens)
                hidden_states = layer_out[0]
                if cb is not None:                                    # (added) pass the layer output straight on, do not keep it
                    cb(i, hidden_states.detach())

            hidden_states = self.ln_post(hidden_states)
            hidden_states = self.proj1(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.proj2(hidden_states)
            return BaseModelOutput(last_hidden_state=hidden_states)

    return RecordingAudioEncoder


def _swap(module, factory, rec, idx=None):
    if not getattr(type(module), "_umai_recording", False):
        cls = factory(type(module)); cls._umai_recording = True
        module.__class__ = cls
    module._umai_rec = rec
    if idx is not None:
        module._umai_idx = idx


def install_recorders(thinker, align_layers):
    """Swap the classes of all decoder layers, the alignment layers' input_layernorm and q_proj, and the audio encoder for recording subclasses.
    A subclass is created only once per base class. Returns: Recorder."""
    rec = Recorder()
    made = {}

    def factory_cached(f):
        def g(base):
            if (f, base) not in made: made[(f, base)] = f(base)
            return made[(f, base)]
        return g
    for l, layer in enumerate(thinker.model.layers):
        _swap(layer, factory_cached(_recording_layer_class), rec, l)
        if l in align_layers:
            _swap(layer.input_layernorm, factory_cached(_recording_norm_class), rec, l)
            _swap(layer.self_attn.q_proj, factory_cached(_recording_linear_class), rec, l)
    _swap(thinker.audio_tower, factory_cached(_recording_encoder_class), rec)
    return rec
