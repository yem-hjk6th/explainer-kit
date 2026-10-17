"""Smoke test of the 3D elements used by E01, four 5 s sections:
A  frame-wise vs global attention block pattern + one-row softmax from q and keys, drawn as bars
B  bar chart (lower is better, source note) + multi-scale grid pyramid with fusion arrows
C  text paper card + two-column comparison table revealed row by row
D  optional / required input slots flickering on and off (fixed seed) + split screen
All numbers here are made up for the test ("example values")."""
import numpy as np

from animkit import (CYAN, GOLD, ORANGE, Shot, attention, attn_blocks, bar_chart, block_diag_mask, fade, flicker,
                     grid_pyramid, input_slot, link, module_box, paper_card, row_softmax, split, ss, table)

DUR = 20.0
shot = Shot("p1_demo", DUR)
shot.tag("example values")

N, K = 3, 4
rng = np.random.default_rng(7)
Q = rng.normal(size=(N * K, 4))
_, A_frame, _ = attention(Q, Q + 0.5 * rng.normal(size=Q.shape), mask=block_diag_mask(N, K))
_, A_glob, _ = attention(Q, Q + 0.5 * rng.normal(size=Q.shape))
q1 = np.array([1.0, 0.0])
KEYS2 = np.array([[1.0, 0.0], [0.0, 1.0]])
W2 = row_softmax(q1, KEYS2, scale=1.0)                    # softmax([1, 0]) = (0.731, 0.269)
KEYS4 = np.array([[0.5, 0.0], [-0.2, 0.0], [1.2, 0.0], [-0.2, 0.0]])
W4 = row_softmax(q1, KEYS4, scale=1.0)

for a, b, s in [(0.3, 5.0, "Frame attention: diagonal blocks only"), (5.0, 10.0, "Bar chart and multi-scale grid"),
                (10.0, 15.0, "Paper card and comparison table"), (15.0, 20.0, "Optional inputs switch randomly")]:
    shot.caption(a, b, s)


@shot.check
def check(ck):
    ck.close("2-key softmax = (e/(1+e), 1/(1+e))", W2, [np.e / (1 + np.e), 1 / (1 + np.e)], 1e-12)
    ck.close("4-key softmax sums to 1", W4.sum(), 1.0, 1e-12)
    ck.le("frame-wise weight outside own frame", A_frame[~block_diag_mask(N, K)].max(), 0.0)
    ck.record(W2=W2.round(3), W4=W4.round(3))


@shot.frame
def frame(cv, t):
    aA = fade(t, 0.2, 5.1)
    if aA > 0:
        attn_blocks(cv, (110, 300), N, K, 28, "frame", aA, A=A_frame, vmax=A_glob.max())
        attn_blocks(cv, (620, 300), N, K, 28, "global", aA, A=A_glob)
        cv.text((278, 680), "frame attention", 28, alpha=aA, anchor="ma")
        cv.text((788, 680), "global attention", 28, alpha=aA, anchor="ma")
        bar_chart(cv, (140, 820, 440, 1080), W2, ["$k_1$", "$k_2$"], vmax=1.0, progress=ss(t, 1.5, 2.6),
                  fmt="{:.3f}", title="2 keys", alpha=aA, highlight=0)
        bar_chart(cv, (560, 820, 980, 1080), W4, [f"$k_{j + 1}$" for j in range(4)], vmax=1.0,
                  progress=ss(t, 2.4, 3.6), fmt="{:.3f}", title="4 keys", alpha=aA, highlight=2)
    aB = fade(t, 5.0, 10.1)
    if aB > 0:
        bar_chart(cv, (200, 330, 900, 640), [0.62, 0.41, 0.28], ["DUSt3R", "MASt3R", "VGGT"], better="lower",
                  title="Camera pose error (example)", note="Source: example values, not a paper table", progress=ss(t, 5.3, 6.8),
                  highlight=2, alpha=aB)
        grid_pyramid(cv, 1110, [200, 100, 50, 26], ["$148^2$", "$74^2$", "$37^2$", "$19^2$"], x0=120, gap=70,
                     alpha=aB, progress=ss(t, 6.8, 9.0), n_lines=[16, 8, 4, 2])
    aC = fade(t, 10.0, 15.1)
    if aC > 0:
        paper_card(cv, (90, 290, 990, 600), 2025, "CVPR 2025 · arXiv:0000.00000 (example)",
                   "Example Paper Title That Is Long Enough To Wrap Onto Two Lines",
                   "A. Author, B. Author, C. Author", "One-line contribution: what this paper achieves.", alpha=aC)
        table(cv, (110, 660), [260, 300, 300], ["", "Method A", "Method B"],
              [["Output", "camera + depth", "$X^{\\mathrm{metric}}$"], ["Scale", "normalized", "metric"],
               ["Input", "images only", "images + optional"],
               ["Camera repr.", "$g=[q,t,f]$", "ray × depth"], ["Training set", "example", "example"]],
              reveal=ss(t, 10.6, 14.0) * 5, highlight=1 if t > 13.6 else None, alpha=aC, size=26)
    aD = ss(t, 15.0, 15.4)
    if aD > 0:
        L, R = split(cv, "lr", 0.5, aD, margin=(60, 300, 1020, 1200))
        cv.text(((L[0] + L[2]) / 2, L[1] + 10), "optional inputs", 28, alpha=aD, anchor="ma")
        cv.text(((R[0] + R[2]) / 2, R[1] + 10), "right half", 28, alpha=aD, anchor="ma")
        on = flicker(t, 3, p=0.5, period=0.7, seed=4, t0=15.0)
        enc = module_box(cv, (150, 900, 420, 980), "encoder", None, CYAN, aD)
        for k, (name, opt) in enumerate([("image", False), ("intr.", True), ("pose", True)]):
            box = (120 + k * 105, 520, 210 + k * 105, 600)
            s = input_slot(cv, box, name, optional=opt, on=1.0 if (not opt or on[k]) else 0.0,
                           prob=None if not opt else None, alpha=aD, size=22)
            link(cv, s["b"], enc["t"] + [(k - 1) * 60, 0], 1.0 if (not opt or on[k]) else 0.0, alpha=aD)
        cv.rich("Each optional input: $p=0.5$", ((L[0] + L[2]) / 2, 720), 24, alpha=aD, anchor="mm", fill=GOLD)
        module_box(cv, (R[0] + 60, 600, R[2] - 60, 700), "Other elements go here", None, ORANGE, aD)
