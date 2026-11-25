# UMAI calibration paths. `source calibration/env.sh` (bash). Preset values are kept; defaults are used only when unset.
# The Python side (calibration/paths.py) uses the same names and defaults.
: "${UMAI_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${UMAI_WORK:=$(dirname "$UMAI_ROOT")/umai_work}"   # large intermediate outputs: outside the repo
: "${UMAI_DUMP:=$UMAI_WORK/dump}"
: "${UMAI_FIT:=$UMAI_WORK/fit}"
: "${AMI_DIR:=$UMAI_WORK/data/ami}"
: "${UMAI_ASR_MODEL:=Qwen/Qwen3-ASR-1.7B}"
: "${PY:=python3}"
export UMAI_ROOT UMAI_WORK UMAI_DUMP UMAI_FIT AMI_DIR UMAI_ASR_MODEL PY
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
