#!/usr/bin/env bash
# CPU/GPU steps (03 ~ 10): merge dumps, fit decision statistics and package them as calibration values.  bash calibration/run_fit.sh OUT_DIR
# Steps whose result file already exists are skipped (so an interrupted run does not start over).
# Result files are not written atomically: if a step is interrupted while writing, delete its partial result file before rerunning.
set -euo pipefail
OUT=${1:?usage: run_fit.sh OUT_DIR}
C=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$C/env.sh"
S=$C/steps; D=$UMAI_DUMP; F=$UMAI_FIT
run() { local done_file=$1; shift; if [ -s "$done_file" ]; then echo "skip $1 ($done_file exists)"; else $PY "$S/$1.py" "${@:2}"; fi; }
run "$D/frames.npz"             03_merge_dumps
run "$F/gapsel.json"            04_labels
run "$F/gaps/meta.npz"          05_select_gaps
run "$F/frames.npz"             06_frame_stage1
run "$F/alignment_heads.json"   07_alignment_heads
run "$F/filler_head.pt"         08_fit_filler
run "$F/form_head.pt"           09_fit_form
$PY "$S/10_export_head.py" "$OUT"
