# Calibration

This code builds the decision values in `heads/` from scratch from the original AMI data. Qwen is not trained.
A GPU is required only to run Qwen and dump its internal values; everything else computes statistics without gradient training and runs on the CPU (steps 06 and 08 use a GPU if one is available). All steps are Python scripts.

## Running

```bash
source calibration/env.sh                  # from bash; defaults for UMAI_WORK, UMAI_DUMP, UMAI_FIT, AMI_DIR (preset values are kept)
# prepare $AMI_DIR/rows.jsonl yourself (see 'Input' below)

# GPU: split the rows into N shards and run each shard (one GPU, one shard at a time)
N=28; for K in $(seq 0 $((N-1))); do bash calibration/run_dump.sh $K $N; done

# Several GPUs: the steps always use cuda:0, so pick the GPU of each shard with CUDA_VISIBLE_DEVICES
# (here 4 GPUs, each running every 4th shard)
N=28; G=4
for g in $(seq 0 $((G-1))); do
  ( for K in $(seq $g $G $((N-1))); do CUDA_VISIBLE_DEVICES=$g bash calibration/run_dump.sh $K $N; done ) &
done; wait

# CPU (GPU if available for steps 06, 08): merge -> labels -> statistics -> package in the heads format
bash calibration/run_fit.sh heads/new
```

`run_fit.sh` skips steps whose result file exists; result files are not written atomically, so if a step is interrupted while writing, delete its partial result file before rerunning. `env.sh` must be sourced from bash (it uses `BASH_SOURCE`). `run_dump.sh` skips outputs that already exist, so an interrupted shard can be rerun;
make sure every shard has finished before running `run_fit.sh`: step 03 stops if a decoded shard lacks its attention or frame dump (it does not know N, so check that greedy_000 … greedy_{N-1} all exist), and after the merge, `run_dump.sh` skips the merged shards; a shard that was not dumped before the merge cannot be added afterwards. The dumps are always made in bfloat16, the precision inference runs in; other precisions are not supported.

## Input

- `$AMI_DIR/rows.jsonl`: one AMI headset (IHM) utterance per line.
  `{"id", "meeting", "start", "end", "audio": path to a 16 kHz mono utterance wav, "nxt_words": [{"w", "start", "end", "type"}]}`.
  `type` is `word`, `filler` (only words spelled uh or um), `fragment` (cut-off word) or `punct`. One NXT segment is used as one utterance.
  `start`/`end` are the utterance span in seconds on the meeting recording's clock, and the `nxt_words` times use the same clock (step 04 subtracts the utterance `start`; words without times are skipped); the wav holds exactly that span.
  Data-preparation code is not included.
- Calibration rows: all of rows.jsonl (`heads/default` used the 66,900 utterances of the AMI Full-corpus-ASR training meetings plus IB4005, excluding IS1003b and three utterances that start after the end of their audio file). If `UMAI_IDS` points to an id-list file, only those rows are used.
- Reader-corrected gap labels (optional): if `UMAI_LABEL_READ` points to a file of `{id, qwen_sha1, slot, cls}` lines (`cls` is O, F, E, U, or X for excluded), step 04 overwrites the automatic labels with those values.
  `heads/default` used 4,129 gaps corrected by an LLM reader; that file is not included.
- The `write_threshold` (0.75) in `heads/default` is the operating threshold chosen after calibration (`run_fit.sh` outputs 0.5).

## What each step does

**Dump (GPU, `steps/01`, `02`)**
- Qwen transcribes each row with greedy decoding: the same official prompt as inference, at most 260 tokens, one utterance at a time. For each token, the number of words written so far is counted
  to find the steps where a new word starts (the gaps).
- The prompt and the transcribed tokens are fed once more (teacher forcing) to get nearly the same internal values as during transcription
  (rounding-level differences: a single prefill pass and float16 storage). For each gap, the outputs of all 28 decoder layers
  are stored at two steps (h_g, h_g1), together with the outputs of the 24 encoder layers and of the projector at the audio positions.
- One more pass with eager attention stores, for every step, the attention that every layer and head gives to the audio positions.

**Labels (CPU, `steps/03` to `05`)**
- The shards are merged, and every frame gets a fold number (5 folds by meeting series). Every "out-of-fold" score later on comes from a model fitted without that fold.
- The NXT reference text and Qwen's text are aligned at the word-unit level (difflib), and each gap gets a class:
  O (empty gap), F (the reference has uh/um but Qwen did not write it), E (a word Qwen left out: cut-off word, repetition, other), U (misrecognized region), Q (Qwen wrote a filler itself).
- Frame labels are painted from the NXT word times: word (W), filler (F), cut-off word (Efrag), repetition (Erep), and silence (S) for the rest.
- The definitions of word units, Qwen fillers and 'uh huh' are in the header of `lib/align.py` (the same as inference model.py). Reader-corrected labels, if given, overwrite the automatic ones, and X (excluded) gaps are dropped.
- All non-O gaps and the same number of sampled O gaps (seed 0) are used. Because O is sampled, statistics weight each class by its original count over its sampled count.

**Statistics (CPU, `steps/06` to `09`)**
- Frame stage 1: a 5-class frame LDA for each of the 25 sources (24 encoder layers and the projector), with a shared covariance shrunk by OAS.
  The frame features are, per source, the 4 values (class discriminant − silence discriminant), 100 dimensions in total.
- Alignment head: non-repeated reference words of two or more characters are sampled to match the durations of filler, cut-off and repetition spans, and the sampled words that Qwen wrote correctly are kept. For each of them, the attention mass that each layer and head puts on that word's time span at the step that emitted the word's first token is measured.
  EM finds the head weights that maximize the mixture likelihood of these masses (in the original run, head 14 of layer 20, counting from 0, got 0.91).
- Filler decision: a 5-class LDA per layer (28 layers × h_g, h_g1) → the 224 layer scores plus the features below go into a second 5-class LDA → logit P(F) → isotonic regression (PAV).
  - 10 audio values: the frame stage-2 LDA posteriors p and log p, averaged with the audio-window weights F_prev(1 − F_g)
  - 10 audio values over the next-word window F_g(1 − F_next)
  - whether the previous word is an uh/um written by Qwen itself (0/1)
  The model for end gaps (gaps with no later step) drops the h_g1 features and the next-word window and is fitted on end gaps only.
  Stage 2 and the isotonic map are fitted on out-of-fold scores, so the calibration scores are not produced by models fitted on the same gaps.
- uh/um decision: only F gaps whose reference form is uh or um are used. On the decoder side there is a single value, log P(um) − log P(uh) assigned by Qwen at the last layer of h_g.
  Of Qwen's weights, only the final norm and the uh/um token rows of lm_head are read (Hugging Face cache). On the audio side, the log-likelihood ratio of a projector-frame uh/um LDA
  is pooled with weights "audio window × frame filler probability". The two are combined by an LDA, and isotonic regression gives P(um).
- `steps/10` packages the results in the `heads/` format (safetensors + config.json). It does no computation.

## Resources (estimates for the full AMI training set)

| Step | Device | Time | Disk |
|---|---|---|---|
| Dump 01, 02 | GPU | about 10 min per 1,000 rows, about 11 GPU hours in total | about 500 GB (`$UMAI_DUMP`) |
| 06 frame stage 1 | CPU (GPU if available) | 1 to 2 hours | |
| 08 filler decision | CPU (GPU if available; reads one layer at a time) | 1 to 3 hours | |
| 09 uh/um decision | CPU | about 30 minutes | |

In the actual run (66,900 utterances), the 08 filler decision used 16 CPUs and peaked at about 120 GB of memory, and steps 03 to 10 took about 4.5 hours in total.
