"""04. Labels (CPU). Assign a class to each gap in Qwen's text using the original AMI NXT transcripts, and build word spans for frames and alignment heads.
Fillers are only the spellings uh/um (written [UH]/[UM]). Cut-off words are NXT trunc marks ('xx-'). Times are relative to utterance start.
Outputs ($UMAI_FIT):
  greedy_q.jsonl  dump transcription + reference text (target) + gap classes steps = [slot, j, wordstart, boundary, cls, sub]
  gapsel.json     gaps used for calibration: all non-O gaps + an equal number of O samples (numpy seed 0)
  words.jsonl     word spans for frame labels (W / F / Efrag / Erep; other frames are S)
  fwords.jsonl    spans for uh/um frame labels (Fuh / Fum)
  spans.jsonl     word spans for alignment heads (F / Efrag / Erep + W samples matched in length distribution, random seed 0)
Read labels (optional): $UMAI_LABEL_READ (not used if unset).
  Gap classes decided by an LLM grader (not a human) reading the reference and Qwen's text. One {id, qwen_sha1, slot, cls} per line.
  qwen_sha1 is the sha1 of that row's Qwen text; stops if it differs from the dump text (so labels made on a different transcription are not used).
  cls: O/F/E/U replace the class; X means 'don't care' (Qwen wrote the reference filler sound as another word, FW) and is excluded from calibration.
"""
import collections, difflib, hashlib, json, os, random, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import paths as UP  # calibration/paths.py
sys.path.insert(0, os.path.join(UP.CAL, "lib"))
import align as AL
FILL = set(AL.SOUND)
nz = AL.nz


def target_of(r):
    """Reference text: fillers as [UH]/[UM], cut-off words as 'xx-', punctuation attached to the previous word."""
    out = []
    for w in r["nxt_words"]:
        t = w["w"]
        if w["type"] == "punct":
            if out: out[-1] += t
            continue
        if w["type"] == "filler": t = "[" + t.upper() + "]"
        elif w["type"] == "fragment" and not t.endswith("-"): t = t + "-"
        out.append(t)
    return " ".join(out)


def items_timed(r):
    """Per reference word (x, kind, non-filler word index li, start, end). Same order as label_seq_q."""
    out = []; li = 0
    for w in r["nxt_words"]:
        if w["type"] == "punct": continue
        t = w["w"]; fp = w["type"] == "filler"
        if w["type"] == "fragment" and not t.endswith("-"): t = t + "-"
        x = nz(t)
        k = "fp" if fp else ("frag" if x.endswith("-") and x.strip("-'") else "w")
        if not x or not x.strip("-'"): continue
        if k == "fp" or x in FILL: out.append((x, "fp", None, w["start"], w["end"])); continue
        out.append((x.rstrip("-") if k != "frag" else x, k, li, w["start"], w["end"])); li += 1
    return out


def read_labels():
    """Read labels {(id, slot): cls} and per-row text sha1."""
    p = os.environ.get("UMAI_LABEL_READ")
    if not p: return {}, {}, None
    if not os.path.exists(p): raise FileNotFoundError(f"UMAI_LABEL_READ={p} does not exist")
    lab, sha = {}, {}
    for l in open(p):
        x = json.loads(l); lab[(x["id"], int(x["slot"]))] = x["cls"]; sha[x["id"]] = x["qwen_sha1"]
    return lab, sha, p


def main():
    O = UP.FIT; os.makedirs(O, exist_ok=True)
    RL, RSHA, rl_path = read_labels(); n_over = collections.Counter()
    FR = dict(np.load(f"{UP.DUMP}/frames.npz"))
    dur = {str(a): float(b) for a, b in zip(FR["rid"], FR["dur"])}
    rows = [json.loads(l) for l in open(f"{UP.DUMP}/greedy.jsonl")]
    fw = open(f"{O}/words.jsonl", "w"); ff = open(f"{O}/fwords.jsonl", "w")
    spans_all, Wcand = [], []
    with open(f"{O}/greedy_q.jsonl", "w") as fo:
        for ri, r in enumerate(rows):
            r["target"] = target_of(r)
            cl = {g["slot"]: g for g in AL.classify_q(r, r["qwen"])}
            if r["id"] in RSHA:
                assert hashlib.sha1(r["qwen"].encode()).hexdigest() == RSHA[r["id"]], ("read labels were made on another transcript", r["id"])
                for sl, g in cl.items():
                    c = RL.get((r["id"], sl))
                    if c is None or c == g["cls"]: continue
                    assert g["cls"] != "Q" and c in ("O", "F", "E", "U", "X"), (r["id"], sl, g["cls"], c)
                    n_over[(g["cls"], c)] += 1; g["cls"] = c; g["sub"] = "read_" + c
            r["steps"] = [[slot, j, ws, bnd, cl[slot]["cls"], cl[slot]["sub"]] for slot, j, ws, bnd, content in r["steps_nolab"] if slot in cl]
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            it = items_timed(r); d_ = dur.get(r["id"])
            T, Kd = AL.label_seq_q(r); assert [x if k != "fp" else "#fp" for x, k, *_ in it] == T, r["id"]
            if d_ is None: continue
            # ---- word spans for frame labels
            w3, wf = [], []
            for t, (x, k, li, s0, e0) in enumerate(it):
                nxt_c = next((y for y, kk, *_ in it[t + 1:] if kk != "fp"), None)
                if k == "fp": c3 = "F"; cf = "Fuh" if x == "uh" else "Fum"
                elif k == "frag": c3 = cf = "Efrag"
                elif nxt_c is not None and nxt_c == x: c3 = cf = "Erep"
                else: c3 = cf = "W"
                if s0 is None or e0 is None: continue
                s, e = max(0.0, s0 - r["start"]), min(d_, e0 - r["start"])
                if e <= s: continue
                w3.append([c3, round(s, 3), round(e, 3)]); wf.append([cf, round(s, 3), round(e, 3)])
            mt = r["meeting"].lower(); se = re.sub(r"[a-z]$", "", mt)
            fw.write(json.dumps({"id": r["id"], "dur": d_, "meeting": mt, "series": se, "words": w3}) + "\n")
            ff.write(json.dumps({"id": r["id"], "dur": d_, "meeting": mt, "series": se, "words": wf}) + "\n")
            # ---- spans for alignment heads: match reference words to Qwen words (fillers excluded) and attach the step that emitted the word (own)
            L = [x for x, k, li, *_ in it if li is not None]
            us, ut = AL.word_units(r["qwen"].split()); H, hs = [], []
            for j, x in zip(us, ut):
                if x != "#fp": H.append(x); hs.append(j)
            m = {}
            for a, b, n in difflib.SequenceMatcher(None, L, H, autojunk=False).get_matching_blocks():
                for k in range(n): m[a + k] = b + k
            wstep = {s[0]: s[1] for s in r["steps"] if s[2]}
            rs = []
            for t, (x, k, li, s0, e0) in enumerate(it):
                if s0 is None or e0 is None: continue
                s, e = max(0.0, s0 - r["start"]), min(d_, e0 - r["start"])
                if e <= s: continue
                nxt_c = next((y for y, kk, *_ in it[t + 1:] if kk != "fp"), None)
                if k == "fp": cls = "F"
                elif k == "frag": cls = "Efrag"
                elif nxt_c is not None and nxt_c == x: cls = "Erep"
                else: cls = None
                skipped = True if k == "fp" else (li not in m)
                if cls: rs.append(dict(cls=cls, s=round(s, 3), e=round(e, 3), w=x, skipped=bool(skipped)))
                elif k == "w" and len(x) >= 2 and not (t > 0 and it[t - 1][0] == x):
                    Wcand.append((ri, dict(cls="W", s=round(s, 3), e=round(e, 3), w=x, skipped=bool(skipped),
                                           own=wstep.get(hs[m[li]], -1) if li in m else -1)))
            spans_all.append((ri, mt, d_, rs))
    fw.close(); ff.close()
    # W samples: drawn from W candidates to match the F/Efrag/Erep span length distribution (40 ms bins)
    bins = collections.Counter(int((sp["e"] - sp["s"]) / 0.04) for _, _, _, rs in spans_all for sp in rs)
    byb = collections.defaultdict(list)
    for ri, sp in Wcand: byb[int((sp["e"] - sp["s"]) / 0.04)].append((ri, sp))
    rnd = random.Random(0); addW = collections.defaultdict(list)
    for b, n in bins.items():
        pool = byb.get(b, []); rnd.shuffle(pool)
        for ri, sp in pool[:n]: addW[ri].append(sp)
    with open(f"{O}/spans.jsonl", "w") as f:
        for ri, mt, d_, rs in spans_all:
            rs = rs + addW.get(ri, [])
            if not rs: continue
            f.write(json.dumps({"id": rows[ri]["id"], "dur": d_, "spans": rs}) + "\n")
    # gaps for calibration: all non-O word-start gaps + an equal number of O samples
    # X (read label 'don't care') gaps are not selected (excluded from calibration)
    keep, Og = {}, []
    for ri, r in enumerate(rows):
        for s in r["steps"]:
            if not s[2] or s[4] == "X": continue
            (Og.append((ri, s[0])) if s[4] == "O" else keep.__setitem__((ri, s[0]), 1))
    nk = len(keep)
    for i in np.random.default_rng(0).choice(len(Og), size=min(nk, len(Og)), replace=False): keep[Og[int(i)]] = 1
    json.dump([[rows[ri]["id"], sl] for ri, sl in sorted(keep)], open(f"{O}/gapsel.json", "w"))
    print("labels:", len(rows), "rows,", nk, "non-O gaps,", len(Og), "O gaps,", len(keep), "selected", flush=True)
    print("read labels:", rl_path, "| changed (alignment -> read):", dict(n_over), flush=True)


if __name__ == "__main__":
    main()
