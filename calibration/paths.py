"""Paths for the calibration code. Values come from environment variables, falling back to defaults relative to the umai_repo folder (same defaults as calibration/env.sh).
  UMAI_WORK       large intermediate outputs (outside repo)  default: <umai_repo>/../umai_work
  UMAI_DUMP       GPU dumps                        default: $UMAI_WORK/dump
  UMAI_FIT        labels, gaps, fitted statistics  default: $UMAI_WORK/fit
  AMI_DIR         AMI row list rows.jsonl (prepare yourself)  default: $UMAI_WORK/data/ami
  UMAI_ASR_MODEL  Qwen3-ASR name or path           default: Qwen/Qwen3-ASR-1.7B
"""
import os

CAL = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CAL)
WORK = os.environ.get("UMAI_WORK") or os.path.join(os.path.dirname(ROOT), "umai_work")
DUMP = os.environ.get("UMAI_DUMP") or os.path.join(WORK, "dump")
FIT = os.environ.get("UMAI_FIT") or os.path.join(WORK, "fit")
AMI_DIR = os.environ.get("AMI_DIR") or os.path.join(WORK, "data", "ami")
ASR_MODEL = os.environ.get("UMAI_ASR_MODEL") or "Qwen/Qwen3-ASR-1.7B"
DTYPE = "bfloat16"                 # dump precision; fixed, because inference (inference/umai) runs in bfloat16 only
