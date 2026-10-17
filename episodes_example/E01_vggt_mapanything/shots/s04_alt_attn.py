"""E01-04 Alternating attention: the shape switches back and forth (M, 34 s). Storyboard storyboard.md section E01-04.
Computed for real: the reshape index mapping (token (i,k) -> i·K+k, numpy rearrangement), white-dot token 799 -> 2173,
the softmax of the small attention example, the real-scale score counts, sequence length 4122.
Illustration: only 4 tokens are drawn per frame (illustrative scale); the matrix colors are computed from random q, k (example values).
Paper numbers: Tab. 5 (ETH3D point map Overall, lower is better).
"""
import numpy as np

import e01_common as ec
from animkit import (CYAN, GOLD, ORANGE, Formula, Shot, attention, block_diag_mask, bump, fade, heatmap,
                     reshape_check, reshape_rows, ss, style, token)

DUR = 31.4
shot = Shot("s04_alt_attn", DUR)
shot.title("Alternating attention", "Alternating Attention · VGGT §3.2", 0.2, 2.6)
shot.tag("Not to scale: 4 tokens drawn per frame", 0.3, 18.0)
shot.tag("Example values", 18.0, 28.6)
shot.tag("Paper, Tab. 5", 28.6, DUR)
CAPS = [(0.3, 2.8, "Tokens of the 3 images, one row each"), (2.8, 5.4, "Frame attention: each row on its own"),
        (5.4, 7.6, "Shape (3, 1374, 1024)"), (7.6, 10.4, "Join the three rows end to end"),
        (10.4, 13.0, "Global attention: all tokens together"), (13.0, 15.2, "Shape (1, 4122, 1024)"),
        (15.2, 18.0, "White-dot token: 799 → 2173"), (18.0, 20.6, "Frame attention: only the 3 diagonal blocks"),
        (20.6, 23.2, "Global attention fills the whole table"), (23.2, 25.9, "Same query: frame attention sees only its own frame"),
        (25.9, 28.6, "Global: 63% of the weight goes to frame 2"), (28.6, DUR, "Same parameter count: alternating has the lowest error")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

NB = ec.nb()
KT, W_IDX, W_G = NB["K_tok"], NB["tok_idx_code"], NB["tok_idx_global"]
N, KD = 3, 4                                   # number of frames; tokens drawn per frame (illustrative scale)
PITCH, ROWP = 72, 96
SRC_C, DST_C = (540, 650), (540, 650)
W_ROW, W_COL = 1, 3                            # the cell standing for the white-dot token in the sketch

# Small attention example
q = np.array([1.0, 0.0])
Kx = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 0.0]])     # a1, a2 (frame 1), b1, b2 (frame 2)
SCORES = Kx @ q / np.sqrt(2)
_, A_FRAME, _ = attention(q[None], Kx, mask=np.array([[True, True, False, False]]))
_, A_GLOB, _ = attention(q[None], Kx)
# Matrix sketch: random q, k (example values), 12×12
rng = np.random.default_rng(4)
Qm = rng.normal(size=(N * KD, 4)); Km = Qm + 0.7 * rng.normal(size=(N * KD, 4))
MASK = block_diag_mask(N, KD)
_, AM_F, _ = attention(Qm, Km, mask=MASK)
_, AM_G, _ = attention(Qm, Km)
CNT_F, CNT_G = N * KT ** 2, (N * KT) ** 2
TAB5 = [("Cross-Attention", 1.061), ("Global Self-Attn Only", 0.827), ("Alternating-Attention", 0.709)]

ATT = Formula([(r"\mathrm{Attn}(Z)=", "lhs"),
               (r"\mathrm{softmax}\!\left(\frac{(ZW^Q)(ZW^K)^{\mathsf{T}}}{\sqrt{d}}\right)", "sm"), (r"ZW^V", "v")],
              size=50)
ATT_XY = (540, 330)
SH_F = Formula([r"(N,\ \tilde K,\ C)=(3,\ 1374,\ 1024)"], size=42)
SH_G = Formula([r"(1,\ N\tilde K,\ C)=(1,\ 4122,\ 1024)"], size=42)


@shot.check
def check(ck):
    r = reshape_check((N, KD), reshape_rows(N, KD, 0.0, SRC_C, DST_C, PITCH, ROWP)[2])
    ck.true("reshape is one-to-one", r["bijective"])
    ck.eq("token (i,k) lands at i·K+k", r["dst_index"][:, 1], r["src_index"][:, 0] * KD + r["src_index"][:, 1])
    ck.eq("white dot: in-frame 799 -> global 1×1374+799", W_G, 1 * KT + W_IDX)
    ck.eq("global sequence length", N * KT, 4122)
    ck.close("attention scores", SCORES, [0.70711, 0.0, 1.41421, 0.0], 1e-5)
    ck.close("frame-attention softmax", A_FRAME[0, :2], [0.66976, 0.33024], 1e-5)
    ck.le("frame attention gives weight 0 to frame 2", A_FRAME[0, 2:].max(), 0.0)
    ck.close("global softmax", A_GLOB[0], [0.24911, 0.12283, 0.50523, 0.12283], 1e-5)
    ck.close("frame 2 total", A_GLOB[0, 2:].sum(), 0.62806, 1e-5)
    ck.eq("in-frame score count", CNT_F, 5663628)
    ck.eq("global score count", CNT_G, 16990884)
    ck.close("score-count ratio = N", CNT_G / CNT_F, 3.0, 1e-12)
    ck.le("matrix sketch: weight outside the in-frame blocks is 0", AM_F[~MASK].max(), 0.0)
    ck.record(A_frame=A_FRAME[0].round(5), A_glob=A_GLOB[0].round(5), counts=[CNT_F, CNT_G])


def kind(i, k):
    if k == 0:
        return "camera1" if i == 0 else "camera"
    return "patch"


def bars(cv, xy, labels, values, alpha, unit_w, colors, fmt="{:.3f}", hl=None, grow=1.0, size=26, h=30, gap=14):
    x0, y0 = xy
    for j, (lab, v) in enumerate(zip(labels, values)):
        y = y0 + j * (h + gap)
        w = unit_w * v * grow
        c = colors[j] if isinstance(colors, list) else colors
        cv.s.rect(x0, y, x0 + max(w, 2), y + h, c, (0.85 if hl == j else 0.55) * alpha, r=4)
        if hl == j:
            cv.g.rect(x0, y, x0 + max(w, 2), y + h, c, 0.5 * alpha, r=4)
        cv.rich(lab, (x0 - 12, y + h / 2), size, alpha=alpha, anchor="rm")
        cv.text((x0 + w + 10, y + h / 2), fmt.format(v), size, "light", alpha * ss(grow, 0.6, 1.0), style.TEXT, "lm")


@shot.frame
def frame(cv, t):
    # ---- A: token rows <-> one long sequence ----
    a_tok = ss(t, 0.3, 0.9) * (1 - ss(t, 17.8, 18.3))
    u = ss(t, 7.8, 9.8)
    if a_tok > 0:
        pos, uk, _ = reshape_rows(N, KD, u, SRC_C, DST_C, PITCH, ROWP, stagger=0.55, arc=26, group_gap=18)
        for idx in range(N * KD):
            i, k = divmod(idx, KD)
            hl = (i == W_ROW and k == W_COL)
            token(cv, pos[idx], 46, style.TOKEN[kind(i, k)], a_tok, glow=0.6 if hl and t > 15.2 else 0.0)
            if hl and t > 15.2:
                ec.star_marker(cv, pos[idx], a_tok, glow=0.5, rad=4)
        a_lab = a_tok * (1 - ss(t, 7.6, 8.0))
        for i in range(N):
            cv.text((SRC_C[0] - 2 * PITCH - 50, SRC_C[1] + (i - 1) * ROWP), f"Frame {i + 1}", 28, alpha=a_lab,
                    fill=ec.CAM_COLORS[i], anchor="rm")
        # frame attention: one box per row; global: one box around the whole sequence
        a_fw = fade(t, 3.0, 7.6, 0.4, 0.3)
        for i in range(N):
            y = SRC_C[1] + (i - 1) * ROWP
            x0, x1 = SRC_C[0] - 2 * PITCH, SRC_C[0] + 2 * PITCH
            pts = [(x0, y - 36), (x1, y - 36), (x1, y + 36), (x0, y + 36)]
            cv.s.polyline(pts, ORANGE, a_fw * a_tok, 2, closed=True)
            cv.g.polyline(pts, ORANGE, a_fw * a_tok * bump(t, 3.4 + 0.5 * i, 1.0), 4, closed=True)
        a_gl = fade(t, 10.6, 15.2, 0.4, 0.3)
        if a_gl > 0:
            xs = pos[:, 0]
            pts = [(xs.min() - 34, DST_C[1] - 36), (xs.max() + 34, DST_C[1] - 36), (xs.max() + 34, DST_C[1] + 36),
                   (xs.min() - 34, DST_C[1] + 36)]
            cv.s.polyline(pts, CYAN, a_gl * a_tok, 2, closed=True)
            cv.g.polyline(pts, CYAN, a_gl * a_tok * bump(t, 11.0, 1.2), 4, closed=True)
        SH_F.draw(cv, (540, 880), alpha=a_tok * fade(t, 5.4, 7.6, 0.4, 0.3))
        SH_G.draw(cv, (540, 880), alpha=a_tok * fade(t, 13.0, 18.0, 0.4, 0.3))
        a_def = a_tok * (fade(t, 5.4, 7.6, 0.4, 0.3) + fade(t, 13.0, 18.0, 0.4, 0.3))
        cv.rich(r"$N$=3 images   $\tilde K$=1374 tokens per image   $C$=1024 dims", (540, 930), 24, alpha=a_def,
                fill=style.TEXT_DIM, anchor="ma")
        cv.rich(r"With batch size $B$: frame $(B\cdot N,\ \tilde K,\ C)$   global $(B,\ N\tilde K,\ C)$", (540, 966), 24,
                alpha=a_tok * fade(t, 13.0, 18.0, 0.4, 0.3), fill=style.TEXT_DIM, anchor="ma")
        a_w = a_tok * ss(t, 15.4, 15.9)
        if a_w > 0:
            p = pos[W_ROW * KD + W_COL]
            cv.rich(f"No. $799$ in its frame → No. $1\\times 1374+799={W_G}$ overall", p - [0, 52], 28, alpha=a_w,
                    fill=style.LABEL_GOLD, anchor="mb")

    # formula (sentences 2 and 5)
    a_eq = ss(t, 2.9, 3.4) * (1 - ss(t, 17.8, 18.3))
    ATT.draw(cv, ATT_XY, alpha=a_eq, state={"lhs": "lit", "sm": "hl" if 10.6 < t < 12.0 else "lit", "v": "lit"},
             notes={"lhs": r"Frame: $Z$ = the $\tilde K$ tokens of one image" if t < 10.4 else
                    r"Global: $Z$ = the $N\tilde K$ tokens of all images"})

    # ---- B: attention matrix (illustrative scale, example values) ----
    a_m = fade(t, 18.1, 23.4, 0.5, 0.4)
    if a_m > 0:
        cell = 30
        heatmap(cv, AM_F, (70, 520), cell, 0, AM_G.max(), a_m)
        cv.rich("Frame (only the 3 diagonal blocks)", (70 + 6 * cell, 500), 26, alpha=a_m, anchor="mb")
        a_g = ss(t, 20.7, 21.4) * a_m
        heatmap(cv, AM_G, (570, 520), cell, 0, AM_G.max(), a_g)
        cv.rich("Global (whole table; off-diagonal = cross-frame)", (570 + 6 * cell, 500), 26, alpha=a_g, anchor="mb")
        for x0 in (70, 570):
            for j in range(N):
                cv.text((x0 - 10, 520 + (j * KD + 2) * (cell + 2)), f"Fr. {j + 1}", 20, alpha=a_m, fill=style.TEXT_DIM,
                        anchor="rm")
        cv.rich("Row = query, column = key; each row sums to 1", (540, 930), 24, alpha=a_m, fill=style.TEXT_DIM, anchor="ma")
        cv.rich("At real size ($N$=3, $\\tilde K$=1374), per head per layer:\nframe $5.7\\times10^6$ scores, global $1.7\\times10^7$, exactly 3 times",
                (540, 980), 22, alpha=a_m * ss(t, 21.4, 21.9), fill=style.TEXT_DIM, anchor="ma")

    # ---- B': weights of one query (computed for real) ----
    a_q = fade(t, 23.3, 28.6, 0.4, 0.3)
    if a_q > 0:
        cv.rich("$d$=2; token $a_1$ of frame 1: $q=[1,0]$", (540, 470), 28, alpha=a_q, anchor="ma")
        cv.rich("Keys: $a_1=[1,0]$, $a_2=[0,1]$ (frame 1); $b_1=[2,0]$, $b_2=[0,0]$ (frame 2)", (540, 515), 26, alpha=a_q,
                fill=style.TEXT_DIM, anchor="ma")
        cv.rich(r"Scores $q^{\mathsf{T}}k/\sqrt{2}$: 0.707, 0, 1.414, 0", (540, 560), 26, alpha=a_q, fill=style.TEXT_DIM,
                anchor="ma")
        labs = ["$a_1$", "$a_2$", "$b_1$", "$b_2$"]
        cols = [ec.CAM_COLORS[0]] * 2 + [ec.CAM_COLORS[1]] * 2
        cv.rich("Frame: softmax over $a_1,a_2$ only", (290, 640), 28, alpha=a_q, fill=ORANGE, anchor="ma")
        bars(cv, (180, 690), labs[:2], list(A_FRAME[0, :2]), a_q, 300, cols[:2], grow=ss(t, 23.6, 24.6), size=30,
             h=36, gap=18)
        for j in (2, 3):
            y = 690 + j * 54 + 18
            cv.rich(labs[j], (168, y), 30, alpha=a_q, anchor="rm")
            cv.text((180, y), "not computed", 26, alpha=a_q, fill=style.TEXT_DIM, anchor="lm")
        a_gq = ss(t, 26.0, 26.5) * a_q
        cv.rich("Global: softmax over all four", (790, 640), 28, alpha=a_gq, fill=CYAN, anchor="ma")
        bars(cv, (690, 690), labs, list(A_GLOB[0]), a_gq, 300, cols, hl=2, grow=ss(t, 26.1, 27.1), size=30, h=36,
             gap=18)
        cv.rich("Frame 2 total: $0.505+0.123=0.628$ (of this query's total weight)", (540, 940), 26,
                alpha=a_gq * ss(t, 27.0, 27.5), fill=style.LABEL_GOLD, anchor="ma")

    # ---- C: Tab. 5 (the original "24 rounds x 2 layers" segment was cut; 07 #5-6 covers it) ----
    a_t = ss(t, 28.7, 29.2)
    if a_t > 0:
        cv.rich("ETH3D point-map error, Overall (Chamfer distance, lower is better)\nAll three designs have the same parameter count", (540, 470), 28, alpha=a_t,
                anchor="ma", align="m")
        bars(cv, (420, 600), [n for n, _ in TAB5], [v for _, v in TAB5], a_t, 450,
             [style.GREY, style.GREY, GOLD], hl=2, grow=ss(t, 29.0, 30.2), size=26, h=40, gap=26)
