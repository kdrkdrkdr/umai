"""03. Merge sharded dumps (CPU). Merge greedy text, attention, frame index and projector outputs, and check that they agree.
Frame rows get a 5-fold index (fold) by meeting series (0..4 assigned round-robin in series-name order).
Decoder states q/K.npz and encoder frames frames/enc_frames_K.f16 are large, so they stay sharded and readers read them in shard order.
For attention and projector outputs, the original shards are deleted after checking the merged file size (no duplicate copies). Re-running reuses the merged file.
  python 03_merge_dumps.py   -> $UMAI_DUMP/{greedy.jsonl, att.f16, att.npz, frames.npz, emb_frames.npy}
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import paths as UP  # calibration/paths.py
K = 5


def main():
    D = UP.DUMP
    gs = [g for g in sorted(glob.glob(f"{D}/greedy/greedy_[0-9][0-9][0-9].jsonl")) if os.path.getsize(g) > 0]   # empty slices have no dumps
    assert gs, "no greedy shards found"
    with open(f"{D}/greedy.jsonl", "w") as fo:
        for g in gs:
            for l in open(g): fo.write(l)
    # attention
    ats = sorted(glob.glob(f"{D}/att/att_[0-9][0-9][0-9].npz"))
    if not ats and os.path.exists(f"{D}/att.npz"):                       # already merged
        z = np.load(f"{D}/att.npz"); rid, nA = z["rid"].tolist(), z["nA"].tolist()
    else:
        assert len(ats) == len(gs), f"attention shards missing: {len(ats)} of {len(gs)}"
        rid, nA = merge_attention(D, ats)
    # frames
    frs = sorted(glob.glob(f"{D}/frames/frames_[0-9][0-9][0-9].npz")); assert len(frs) == len(gs), f"frame shards missing: {len(frs)} of {len(gs)}"; foff = [0]; keys = {k: [] for k in ("rid", "series", "meeting", "dur", "nA")}; emb = []
    for f in frs:
        z = np.load(f); base = foff[-1]; foff += (z["foff"][1:] + base).tolist()
        for k in keys: keys[k] += z[k].tolist()
        ep = f.replace("frames_", "emb_frames_").replace(".npz", ".npy")
        if os.path.exists(ep): e = np.load(ep); assert len(e) == z["foff"][-1]; emb.append(e)
        es = os.path.getsize(f.replace("frames_", "enc_frames_").replace(".npz", ".f16")); assert es == 2 * int(np.prod(z["enc_shape"])), f"enc size {f}"
    if emb:
        assert sum(len(e) for e in emb) == foff[-1], "emb shards missing"
        np.save(f"{D}/emb_frames.npy", np.concatenate(emb)); del emb
        assert len(np.load(f"{D}/emb_frames.npy", mmap_mode="r")) == foff[-1]
        for f in frs: os.remove(f.replace("frames_", "emb_frames_").replace(".npz", ".npy"))
    else:
        assert len(np.load(f"{D}/emb_frames.npy", mmap_mode="r")) == foff[-1], "no emb shards and merged file does not match"
    # for the same audio, attention and frames must have the same number of audio positions (checked before frames.npz, the done-file, is written)
    fA = dict(zip(keys["rid"], keys["nA"]))
    assert all(fA[r] == n for r, n in zip(rid, nA) if r in fA), "attention / frame audio length mismatch"
    fm = {s: i % K for i, s in enumerate(sorted(set(keys["series"])))}
    np.savez(f"{D}/frames.npz", foff=np.array(foff), fold=np.array([fm[s] for s in keys["series"]]), **{k: np.array(v) for k, v in keys.items()})
    print("merged", len(gs), "shards:", len(rid), "attention rows,", len(keys["rid"]), "frame rows,", foff[-1], "frames", flush=True)


def merge_attention(D, ats):
    """Concatenate attention shards into one, check the size, then delete the shards."""
    aoff, nQ, nA, rid = [0], [], [], []
    with open(f"{D}/att.f16", "wb") as fo:
        for a in ats:
            z = np.load(a); NL, NH = int(z["NL"]), int(z["NH"]); base = aoff[-1]
            aoff += (z["aoff"][1:] + base).tolist(); nQ += z["nQ"].tolist(); nA += z["nA"].tolist(); rid += z["rid"].tolist()
            with open(a[:-4] + ".f16", "rb") as fi:
                while True:
                    b = fi.read(1 << 28)
                    if not b: break
                    fo.write(b)
    assert os.path.getsize(f"{D}/att.f16") == 2 * aoff[-1], "att size mismatch"
    np.savez(f"{D}/att.npz", aoff=np.array(aoff), nQ=np.array(nQ), nA=np.array(nA), rid=np.array(rid), NL=NL, NH=NH)
    for a in ats: os.remove(a[:-4] + ".f16"); os.remove(a)
    return rid, nA


if __name__ == "__main__":
    main()
