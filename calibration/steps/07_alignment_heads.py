"""07. Alignment-head weights (CPU). For each span of a word Qwen wrote correctly (W), measure the mass each layer/head's attention puts on the span's frames at the step that emitted the word,
and find the head mixing weights pi by maximum likelihood (EM).
  python 07_alignment_heads.py   -> $UMAI_FIT/alignment_heads.json {"pi": [layer x head]}
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import paths as UP  # calibration/paths.py
sys.path.insert(0, os.path.join(UP.CAL, "lib"))
from heads import Att, span_frames, em


def main():
    at = Att(f"{UP.DUMP}/att"); M = []
    for l in open(f"{UP.FIT}/spans.jsonl"):
        r = json.loads(l); a = None
        for sp in r["spans"]:
            if sp["cls"] != "W" or sp["skipped"] or sp.get("own", -1) < 0: continue
            if a is None:
                a = at.get(r["id"])
                if a is None: break
            o = sp["own"]
            if o >= a.shape[0]: continue
            f0, f1 = span_frames(sp, a.shape[2], r["dur"])
            M.append(a[o, :, f0:f1].sum(1))
    M = np.maximum(np.array(M), float(np.finfo(np.float16).smallest_subnormal))    # so heads with zero mass can still take the log
    pi = em(M)
    json.dump({"pi": pi.tolist()}, open(f"{UP.FIT}/alignment_heads.json", "w"))
    print("alignment heads:", len(M), "spans ->", f"{UP.FIT}/alignment_heads.json", flush=True)


if __name__ == "__main__":
    main()
