"""Label each gap in Qwen's text against the reference transcript (used only for calibration, not inference).
Gap jj = before Qwen word jj (jj == number of words: end of sentence). Fillers on both sides are matched as a single '#fp'.
  Q  Qwen wrote a filler there itself            F  reference has a filler and Qwen has none ('E_*+F' when combined with e.g. a cut-off word)
  O  neighbouring words match, nothing between   E  reference word Qwen did not write (E_frag cut-off word / E_rep repetition / E_miss other)
  U  a neighbouring word does not match the reference (Qwen misrecognition region)
Fillers are only the spellings uh / um. In the reference text they are written [UH] / [UM].

Qwen word units (same definition in calibration and inference, same as word_units in inference/umai/model.py):
  - word = whitespace-split piece. Pieces without ASCII letters/digits (punctuation only, '--', Han characters only, etc.) are not word units (neither gap nor boundary).
    This keeps silent punctuation steps from being used as audio-window boundaries
  - Qwen filler = a word that is exactly uh or um once punctuation is removed. 'uhん', with extra non-punctuation characters, is not a filler but
    a regular word (not a cleanly written standalone uh/um; in AMI the reference is almost always Mm-hmm).
  - The backchannel 'uh huh' (two whitespace-separated pieces uh, huh) is one word 'uh-huh' (same as the hyphenated 'Uh-huh').
    There is no gap inside it (before huh); the gap before it is a regular gap, and the 'previous word' of the gap after it is huh, which is not a Qwen filler.
"""
import difflib
import re

SOUND = {"uh": "uh", "um": "um"}
FILLER_TAG = re.compile(r"^\[(UH|UM)\]", re.I)      # filler notation in the reference text (may be followed by punctuation)


def nz(w):
    return re.sub(r"[^a-z0-9'-]", "", w.lower())


def form(w):
    """Word with only punctuation removed (Unicode letters kept): 'Uh,' -> 'uh', 'uhん' -> 'uhん'."""
    return re.sub(r"[^\w'-]|_", "", w.lower())


def is_filler(w):
    """Is it a filler Qwen wrote itself: the whole word without punctuation is uh or um."""
    return form(w) in SOUND


def word_units(words):
    """List of whitespace pieces -> list of word-unit start indices and unit spellings. Pieces without letters/digits are dropped.
    A 'huh' right after 'uh' is attached to the previous piece as one unit ('uh-huh')."""
    starts, toks = [], []
    for j, w in enumerate(words):
        x = nz(w)
        if not re.search(r"[a-z0-9]", x): continue                 # piece without letters/digits (punctuation only, '--', Han only): not a unit
        if form(w) == "huh" and starts and starts[-1] == j - 1 and form(words[j - 1]) == "uh":
            toks[-1] = "uh-huh"; continue
        starts.append(j); toks.append("#fp" if is_filler(w) else x)
    return starts, toks


def label_seq_q(row):
    """Reference text row['target'] -> (token list, kind list). Fillers are '#fp'/'fp', cut-off words ('xx-') are 'frag', the rest 'w'."""
    T, K = [], []
    for w in row["target"].split():
        if FILLER_TAG.match(w):
            T.append("#fp"); K.append("fp"); continue
        x = nz(w)
        if not x or not x.strip("-'"): continue
        if x in SOUND:
            T.append("#fp"); K.append("fp"); continue
        if x.endswith("-"):
            T.append(x); K.append("frag")
        else:
            T.append(x.rstrip("-")); K.append("w")
    return T, K


def _qwen_words(text):
    ws = text.split(); qw = [nz(w) for w in ws]
    hs, Hq = word_units(ws)
    return qw, Hq, hs


def _match(T, Hq):
    m = {}
    for a, b, n in difflib.SequenceMatcher(None, T, Hq, autojunk=False).get_matching_blocks():
        for k in range(n): m[b + k] = a + k
    return m


def _between(m, jj, nH, nT):
    """Reference positions (a, b) matching the words on both sides of Qwen gap jj. None if there is no matching pair."""
    a = m.get(jj - 1, -1 if jj == 0 else None); b = m.get(jj, nT if jj == nH else None)
    if jj == 0 and 0 not in m: a = None
    return a, b


def classify_q(row, text):
    """-> [dict(slot, jj, cls, sub)], one per Qwen word gap (jj = 0..len(Hq)). Fillers written by Qwen count as words."""
    T, Kd = label_seq_q(row)
    qw, Hq, hs = _qwen_words(text)
    m = _match(T, Hq)
    out = []
    for jj in range(len(Hq) + 1):
        slot = hs[jj] if jj < len(Hq) else len(qw)
        if jj < len(Hq) and Hq[jj] == "#fp":
            out.append(dict(slot=slot, jj=jj, cls="Q", sub="Q")); continue
        a, b = _between(m, jj, len(Hq), len(T))
        if a is None or b is None or b <= a:
            sub = "U"
        else:
            ext = list(range(a + 1, b)); f = any(Kd[i] == "fp" for i in ext); ce = [i for i in ext if Kd[i] != "fp"]
            if not ce: sub = "F" if f else "O"
            else:
                kinds = [Kd[i] for i in ce]
                ci = [i for i in range(len(T)) if Kd[i] != "fp"]; pos = {i: p for p, i in enumerate(ci)}; Lc = [T[i] for i in ci]
                ext_c = [pos[i] for i in ce]; bc = next((pos[i] for i in range(b, len(T)) if Kd[i] != "fp"), len(Lc))
                if "frag" in kinds: sub = "E_frag"
                elif bc < len(Lc) and all(Lc[i] == Lc[i + len(ext_c)] for i in ext_c if i + len(ext_c) < len(Lc)): sub = "E_rep"
                else: sub = "E_miss"
                if f: sub += "+F"
        cls = "F" if (sub == "F" or sub.endswith("+F")) else sub[0]
        out.append(dict(slot=slot, jj=jj, cls=cls, sub=sub))
    return out


def forms_of(row):
    """Reference filler form for each F gap -> {slot: 'uh' | 'um'} (the first filler if a gap has several)."""
    T, Kd = label_seq_q(row)
    qw, Hq, hs = _qwen_words(row["qwen"])
    m = _match(T, Hq)
    raw = [mt.group(1).lower() for mt in map(FILLER_TAG.match, row["target"].split()) if mt]
    fpidx = [i for i in range(len(T)) if Kd[i] == "fp"]; name = {i: (raw[k] if k < len(raw) else "?") for k, i in enumerate(fpidx)}
    out = {}
    for jj in range(len(Hq) + 1):
        slot = hs[jj] if jj < len(Hq) else len(qw)
        a, b = _between(m, jj, len(Hq), len(T))
        if a is None or b is None or b <= a: continue
        f = [name[i] for i in range(a + 1, b) if Kd[i] == "fp"]
        if f: out[slot] = SOUND.get(f[0], f[0])
    return out
