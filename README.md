# UMAI: UM Aware Inference with a Frozen Qwen3-ASR

Qwen3-ASR-1.7B rarely writes "uh" and "um", but its decoder states and audio attention still carry evidence of them.
UMAI runs Qwen3-ASR **once**, **only reads** those traces, and inserts `[uh]` / `[um]` tags between words.
It changes neither the model's weights nor the words it wrote.

```
Qwen3-ASR as is:  okay so we should go with the yellow one
UMAI:             okay so [uh] we should go with the [um] yellow one
```

The method, the experimental setup and all results are in the paper: [`paper/README.md`](paper/README.md)
(ICASSP-format source in [`paper/icassp2027/`](paper/icassp2027/), Korean version in [`paper/README.ko.md`](paper/README.ko.md)).

## Install and use

```bash
pip install -e .            # torch 2.11.0, transformers 4.57.6, qwen-asr 0.0.6 (pinned in pyproject)
```

```python
from umai import UMAI

m = UMAI.from_pretrained("heads/default")    # CUDA GPU, bfloat16; device="cuda:1" picks another GPU
r = m.transcribe("a.wav")
r["text"]       # Qwen3-ASR's text as is
r["verbatim"]   # text with tags inserted
r["gaps"]       # per gap: {"slot", "P" (filler probability), "form", "pum" (probability of um), ...}
r["truncated"]  # whether decoding stopped at the token limit (260)
```

- **GPU only**: UMAI runs on a CUDA GPU with Qwen3-ASR in bfloat16, the precision in which `heads/default` was calibrated. CPU and other precisions are not supported, because the calibration values assume bfloat16 states.
- **Threshold**: a tag is written if P > 0.75 (`write_threshold` in `heads/default/config.json`). Change only this value to move the operating point.
- **Head path**: a relative path such as `"heads/default"` is resolved against the current directory. If no path is given, the `UMAI_HEAD` environment variable is used, and then the repository's `heads/default`. A path that is given but does not exist raises an error.
- **Model path**: change it with the `asr_model=` argument or the `UMAI_ASR_MODEL` environment variable (default `Qwen/Qwen3-ASR-1.7B`).
- **Audio**: audio that is not 16 kHz is resampled to 16 kHz, and multi-channel audio is averaged over channels. Unreadable or empty audio raises `ValueError`.

## How it works

1. Qwen3-ASR transcribes once with greedy decoding. The position before each word (and the end of the utterance) is a gap to decide.
2. For each gap, UMAI reads the outputs of all 28 decoder layers at two steps, and pools audio-frame class posteriors over a window that decoder attention locates in the audio.
3. Two stacked LDAs and an isotonic table turn these into P(filler). A second small model chooses `[uh]` or `[um]`.
4. Qwen3-ASR's text is kept and only the tags are inserted. Only uh and um are handled as fillers.

Nothing is trained by gradient descent. The values in `heads/default` (LDA weights, attention-head mixture, isotonic tables, uh/um token ids) are statistics computed on AMI, and no Qwen3-ASR weights are included.
See Section 3 of the paper for the full description.

## Results

Filler tagging at the default threshold (0.75). Only uh and um count as fillers. Original Qwen3-ASR and UMAI output the same words; UMAI only adds tags.

| Set | System | TP | FP | FN | Filler F1 |
|---|---|---|---|---|---|
| AMI eval (3,067 reference fillers) | Original Qwen3-ASR-1.7B | 201 | 15 | 2,866 | 12.2 |
| | UMAI | 2,270 | 162 | 797 | 82.6 |
| Nyra (3,072 reference fillers) | Original Qwen3-ASR-1.7B | 704 | 9 | 2,368 | 37.2 |
| | UMAI | 2,920 | 61 | 152 | 96.5 |

AMI eval (16 official evaluation meetings, evaluated once) is the main estimate; the default threshold was chosen by looking at the Nyra results.
Inference time increases by about 14% over original Qwen3-ASR (mean per utterance 0.465 s → 0.529 s, batch size 1), and peak GPU memory goes from 4.08 GiB to 4.19 GiB.
Results at other thresholds, the relabeled AMI reference, and the scoring rules are in Sections 4 and 5 of the paper.

## Calibration

`calibration/` contains the code that builds the calibration values (`heads/`) from AMI: it dumps Qwen3-ASR's internal values on a GPU, then computes the statistics and packages them.
Data-preparation code, evaluation code and the LLM-corrected gap labels are not included, so recalibrating does not reproduce `heads/default` exactly. See [`calibration/README.md`](calibration/README.md) for the steps.

## Implementation notes

- The inference code is in `inference/umai`. It replaces three kinds of Qwen3-ASR modules with subclasses that keep intermediate values: the decoder layers, the input normalization and query projection of the layers that hold alignment heads, and the audio encoder. The audio encoder forward is copied from qwen-asr 0.0.6, so that version is pinned.
- The two stacked LDAs are linear, so the head is folded into a single matrix at load time. `python tests/check_fold.py` checks that this equals the two-stage computation (OK if the relative error is below 1e-9).
- The uh/um token ids are recomputed with the tokenizer at load time and checked against the ids stored in the calibration values.
- Values read from Qwen3-ASR's output distribution (filler token probabilities, the chosen token's probability, entropy, etc.) are not used for the filler decision: they come from the same decoder state and overlap with the layer scores.
- The exact word-unit rules (punctuation-only pieces, `uh huh`, fillers Qwen3-ASR wrote itself) are in the docstrings of `inference/umai/model.py`.

## License

- Code in this repository and `heads/`: Apache-2.0 (`LICENSE`).
- Qwen3-ASR-1.7B and `qwen-asr`: follow the Qwen team's release terms. UMAI neither contains nor modifies Qwen3-ASR weights.
- The calibration values (`heads/`) were computed from the original transcripts of the AMI Meeting Corpus (CC BY 4.0). Credit it when redistributing:
  *"Calibration statistics derived from the AMI Meeting Corpus (CC BY 4.0), https://groups.inf.ed.ac.uk/ami/corpus/ ."*
- The Nyra dataset (`nyralabs/disfluency_speech_english`) is Apache-2.0, and its official scorer is MIT.
