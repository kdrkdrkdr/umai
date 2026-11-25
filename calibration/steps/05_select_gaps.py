"""05. Select gaps for calibration (CPU). Keep only the (id, slot) pairs in gapsel.json from the decoder state shards q/K.npz and attach classes.
States are stored in layer order (layer, gap, dim) so they can be read one layer at a time via memory mapping.
  python 05_select_gaps.py   -> $UMAI_FIT/gaps/{meta.npz (id, slot, cls, sub, series), h_g.npy, h_g1.npy}
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import paths as UP  # calibration/paths.py


def main():
    O = f"{UP.FIT}/gaps"; os.makedirs(O, exist_ok=True)
    sel = {(i, int(s)) for i, s in json.load(open(f"{UP.FIT}/gapsel.json"))}
    lab = {}
    for l in open(f"{UP.FIT}/greedy_q.jsonl"):
        r = json.loads(l)
        for s in r["steps"]:
            if s[2]: lab[(r["id"], s[0])] = (s[4], s[5])
    meta = {k: [] for k in ("id", "slot", "cls", "sub", "series")}; H = {}; n = 0
    for f in sorted(glob.glob(f"{UP.DUMP}/q/[0-9][0-9][0-9].npz")):
        z = np.load(f); ids = z["id"]; sl = z["slot"]
        ix = np.array([k for k, (i, s) in enumerate(zip(ids.tolist(), sl.tolist())) if (i, int(s)) in sel], dtype=np.int64)
        assert all(int(c) == 1 for c in z["content"][ix])
        cs = [lab[(ids[k], int(sl[k]))] for k in ix]
        meta["id"].append(ids[ix]); meta["slot"].append(sl[ix]); meta["series"].append(z["series"][ix])
        meta["cls"].append(np.array([c for c, _ in cs])); meta["sub"].append(np.array([b for _, b in cs]))
        for pt in ("g", "g1"):
            h = z[f"h_{pt}"]
            if pt not in H:
                H[pt] = np.lib.format.open_memmap(f"{O}/h_{pt}.npy", mode="w+", dtype=np.float16, shape=(h.shape[1], len(sel), h.shape[2]))
            H[pt][:, n:n + len(ix)] = h[ix].transpose(1, 0, 2)
        n += len(ix); del z
    assert n == len(sel), (n, len(sel))
    for h in H.values(): h.flush()
    np.savez(f"{O}/meta.npz", **{k: np.concatenate(v) for k, v in meta.items()})
    print("gaps:", n, "selected ->", O, flush=True)


if __name__ == "__main__":
    main()
