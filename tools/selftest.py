"""Library self-tests that are not tied to a shot (exit code 1 on failure).
  python -I -X utf8 tools/selftest.py
1. point_cloud z-buffer: two points on one viewer ray, the near one must win whatever the input order.
2. OrbitCam(world="cv") == flip y + OrbitCam(world="yup") (the E01 disp() route), max difference in px.
3. Lone operators in split formulas sit on the math axis (~0.25 em above the baseline in Computer Modern).
4. Language hooks are a no-op in the source language (en): p1_demo frames with the hooks == frames with the hooks
   replaced by identity.
5. Translated render (es): lookup, untranslated fallback, math-only strings skipped, T() templates (LaTeX braces kept),
   word count, caption word wrap.
6. OrbitCam az / el convention: az=0, el=0 looks along +z (+x to the right); az=90, el=0 is a side view with +z to
   the right; screen-down is +y in world="cv" and -y in world="yup".
"""
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parents[1]          # repo root: animkit/, examples/
sys.path.insert(0, str(LIB))
import numpy as np  # noqa: E402

from animkit import Formula, OrbitCam, Shot, T, i18n, point_cloud, richtext, style  # noqa: E402
from animkit.draw import Canvas, background  # noqa: E402
from animkit.text import caption_len, font  # noqa: E402
from animkit.timeline import load_shot  # noqa: E402

fails = []

P = OrbitCam(np.array([0, 0, 3.0]), 3.0, az=10, el=20, world="cv").at()
near, far = P.E + 2.0 * P.f, P.E + 4.0 * P.f
for order in ((near, far), (far, near)):
    cv = Canvas(1080, 1440, 0.0, background(1080, 1440))
    cols = [(255, 0, 0) if X is near else (0, 0, 255) for X in order]
    point_cloud(cv, P, np.array(order), np.array(cols, float), rad=2)
    q = np.round(P(near)).astype(int)
    px = cv.s.img[q[1], q[0]].tolist()
    print("1. z-buffer pixel:", px)
    if px != [255, 0, 0]:
        fails.append("z-buffer")

X = np.random.default_rng(0).normal(size=(50, 3)) + [0, 0, 3]
flip = np.array([1.0, -1.0, 1.0])
a = OrbitCam(np.array([0, 0, 3.0]), 4.0, az=25, el=15, world="cv").at()(X)
b = OrbitCam(np.array([0, 0, 3.0]) * flip, 4.0, az=25, el=15, world="yup").at()(X * flip)
d = float(np.abs(a - b).max())
print(f"2. world=cv vs flip+yup: {d:.2e} px")
if d > 1e-9:
    fails.append("world=cv")

S = 72
for op in ["=", r"\cdot", "-", "+", r"\times"]:
    boxes, w, h, asc = Formula(["a", op, "b"], size=S)._layout()
    bx, by, mask, mw, _ = boxes[1]
    rows = np.nonzero(mask.max(1) > 60)[0]
    c = (asc - (by + (rows.min() + rows.max()) / 2)) / S
    print(f"3. {op:7s} ink centre {c:+.3f} em above baseline")
    if not 0.2 < c < 0.3:
        fails.append(f"operator {op}")

shot = load_shot(LIB / "examples" / "p1_demo.py")
ts = [(k + 0.5) * shot.dur / 8 for k in range(8)]
A = [shot.render_frame(t) for t in ts]
hooks = i18n.tr, i18n.kind_for
i18n.tr, i18n.kind_for = (lambda s: s), (lambda k, s: k)
richtext._rich_cache.clear()
shot._cap_cache.clear()
B = [shot.render_frame(t) for t in ts]
i18n.tr, i18n.kind_for = hooks
d = max(int(np.abs(a.astype(int) - b).max()) for a, b in zip(A, B))
print(f"4. source language with hooks vs without, {len(ts)} p1_demo frames: max pixel difference {d}")
if d != 0:
    fails.append("source-language hooks not a no-op")

i18n.LANG = "es"
i18n._table = {"Three images": "Tres imágenes", "round {k}": "ronda {k}",
               "long caption": "un subtítulo deliberadamente largo que no cabe de ninguna manera en una sola línea"}
r = [i18n.tr("Three images"), i18n.tr("Not translated"), i18n.tr("$x_1$ = 3"), T("round {k}", k=3),
     T("$x^{2}$ of {v}", v=1), i18n.kind_for("bold", "Tres"), caption_len("It's 3+4+2 = 9, $q$ first")]
ok = r == ["Tres imágenes", "Not translated", "$x_1$ = 3", "ronda 3", "$x^{2}$ of 1", "bold", 7] \
    and "Not translated" in i18n.report()["untranslated"] and "$x_1$ = 3" not in i18n.report()["untranslated"]
s = Shot("t", 3.0)
c = s._cap_text("long caption")
f = font("text", style.CAPTION_SIZE)
ok = ok and c.count("\n") == 1 and max(f.getlength(x) for x in c.split("\n")) <= s.W - 120
print(f"5. es: {r}; wrapped caption {c!r}: {'OK' if ok else 'WRONG'}")
i18n.LANG, i18n._table = i18n.SOURCE, {}
if not ok:
    fails.append("translated lookup")

T0 = np.array([0, 0, 3.0])
moves = {}
for world, az, el in (("cv", 0, 0), ("cv", 90, 0), ("yup", 90, 0)):
    V = OrbitCam(T0, 4.0, az=az, el=el, world=world).at()
    moves[(world, az)] = [np.sign(np.round(V(T0 + 0.5 * e) - V(T0), 3)).tolist() for e in np.eye(3)]
want = {("cv", 0): [[1, 0], [0, 1], [0, 0]], ("cv", 90): [[0, 0], [0, 1], [1, 0]], ("yup", 90): [[0, 0], [0, -1], [1, 0]]}
print(f"6. OrbitCam screen direction of +x / +y / +z: {moves}")
if moves != want:
    fails.append("OrbitCam convention")

print("FAILED: " + ", ".join(fails) if fails else "all OK")
sys.exit(1 if fails else 0)
