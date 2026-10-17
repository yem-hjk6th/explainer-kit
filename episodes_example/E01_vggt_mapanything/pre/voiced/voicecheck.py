"""Comparison tool between the voice-over text and the whisper transcript (shared by pick.py / check_voiced.py).
Normalization: lower-case, strip punctuation, spell numbers as words (9.5% -> nine point five percent), merge consecutive single letters into one word (V, G, G, T -> vggt),
then unify spellings of terms (map anything -> mapanything, dino v2 -> dinov2, em -> m).
Keywords = number words + terms + single-letter symbols; these must appear in the transcript (at least as many times as in the text); everything else is scored by word-level edit distance (WER).
"""
import re

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def int_words(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + int_words(n % 100))
    return int_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + int_words(n % 1000))


def _num(m):
    a, b = m.group(1), m.group(2)
    s = int_words(int(a))
    if b:
        s += " point " + " ".join(_ONES[int(c)] for c in b)
    return " " + s + " "


ALIASES = [  # (pattern on the space-joined token string, replacement)
    (r"\bmap anything(s?)\b", r"mapanything\1"), (r"\bmap any thing\b", "mapanything"),
    (r"\bmethod a\b", "methoda"), (r"\bmethod b\b", "methodb"),     # A / B are symbols: A must not pass as an article
    (r"\bvi ggt\b", "vggt"), (r"\bv ggt\b", "vggt"), (r"\bvgg t\b", "vggt"), (r"\bvg gt\b", "vggt"),
    (r"\bdino v two\b", "dinov2"), (r"\bdino v2\b", "dinov2"), (r"\bdinov two\b", "dinov2"),
    (r"\bem\b", "m"), (r"\bper cent\b", "percent"), (r"\bmlps?\b", "mlp"), (r"\bdpts?\b", "dpt"),
    # homophones / liaison, not misreadings: out come <-> outcome; the two s in "meters still" merge and are transcribed as "meter still"
    (r"\boutcome\b", "out come"), (r"\bmeter still\b", "meters still"),
    (r"\bun projects\b", "unprojects"), (r"\btable too\b", "table two"),   # whisper writes un-projects / Table too
    # the respelled tts "In-trin-sics" is split into "in trin sics"; once the 's of "V G G T's" merges into t, the single letters leave only "v g g" -> "vgg ts"
    (r"\bin trin sics\b", "intrinsics"), (r"\bvgg ts\b", "vggts"),
]
TERMS = {"vggt", "vggts", "mapanything", "mapanythings", "dinov2", "dpt", "mlp", "m", "methoda", "methodb",
         "depth", "intrinsics", "pose", "poses", "camera", "meters", "token", "tokens", "attention", "unprojects"}
NUMWORDS = set(_ONES) | set(_TENS[2:]) | {"hundred", "thousand", "point", "percent"}


def tokens(s):
    s = s.lower().replace("’", "'").replace("%", " percent ").replace("-", " ")
    s = re.sub(r"(\d+)\s+\.(\d+)", r"\1.\2", s)                # whisper sometimes writes "1 .4"
    s = re.sub(r"(\d+)(?:\.(\d+))?", _num, s)
    s = re.sub(r"\s*'s\b", "s", s)                             # that's -> thats, 1's -> ones (keep s from merging with a following single letter)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    w = s.split()
    out, run = [], []
    for x in w + ["__end__"]:                       # merge consecutive single letters (except a) into one word: v g g t -> vggt
        if len(x) == 1 and x != "a":
            run.append(x)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(x)
    s = " ".join(out[:-1])
    for p, r in ALIASES:
        s = re.sub(p, r, s)
    return s.split()


def keys(tok):
    return [x for x in tok if x in TERMS or x in NUMWORDS or (len(x) == 1 and x != "a")]


def wer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(1, len(ref))


def compare(ref_text, hyp_text):
    """{'wer', 'missing_keys'}: missing_keys = keywords of the text that are missing from the transcript (by count)."""
    r, h = tokens(ref_text), tokens(hyp_text)
    miss = []
    hk = keys(h)
    for k in keys(r):
        if k in hk:
            hk.remove(k)
        else:
            miss.append(k)
    return {"wer": round(wer(r, h), 3), "missing_keys": miss, "ref_tok": r, "hyp_tok": h}
