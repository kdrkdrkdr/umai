#!/usr/bin/env bash
# GPU steps (01, 02): transcribe shard K of N of the AMI rows and dump internal values.  bash calibration/run_dump.sh K N
# With several GPUs, run each K separately. Finished outputs are skipped, also after run_fit.sh step 03 has merged them
# (03 keeps greedy/greedy_K.jsonl and q/K.npz, and replaces att/att_K.* with the merged att.npz).
set -euo pipefail
C=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$C/env.sh"
K=${1:?usage: run_dump.sh K N}; N=${2:?usage: run_dump.sh K N}; T=$(printf %03d "$K"); S=$C/steps
if [ -s "$UMAI_DUMP/att.npz" ] && [ ! -e "$UMAI_DUMP/greedy/greedy_$T.jsonl" ]; then echo "dumps already merged; shard $K was not part of the merge" >&2; exit 1; fi
[ -s "$UMAI_DUMP/greedy/greedy_$T.jsonl" ]                                 || $PY "$S/01_dump_decode_states.py" decode "$K" "$N"
[ -s "$UMAI_DUMP/greedy/greedy_$T.jsonl" ] || { echo "shard $K has no rows"; exit 0; }
[ -s "$UMAI_DUMP/q/$T.npz" ]                                               || $PY "$S/01_dump_decode_states.py" states "$K"
[ -s "$UMAI_DUMP/att/att_$T.npz" ] || [ -s "$UMAI_DUMP/att.npz" ]          || $PY "$S/02_dump_attention.py" "$K"
