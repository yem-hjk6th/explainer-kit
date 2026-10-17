"""Smoke test of the tensor / architecture elements E01 needs (also a usage example):
tensor block split along S -> frame tokens -> reshape (S, N, C) -> (1, S*N, C) and back -> attention heatmap with a
real row softmax (frame-wise mask vs global) -> module boxes with arrows and a pulse.
Small sizes (S=3 frames, N=1 camera + 1 register + 4 patch tokens) so every token is visible; it is a schematic of
the shapes, not VGGT's real token count, hence the "schematic" tag.
"""
import numpy as np

from animkit import (CYAN, GOLD, ORANGE, VIOLET, WHITE, Formula, Shot, arrow, attention, block_diag_mask, bump, fade,
                     grid_xy, heatmap, kind_colors, module_box, move_tokens, pulse_dot, reshape_check, reshape_ids, ss,
                     style, tensor_block, tokens)

S, N_REG, N_PATCH = 3, 1, 4
N = 1 + N_REG + N_PATCH
C = 8
DUR = 24.0
shot = Shot("tensor_demo", DUR)
shot.tag("schematic: S=3, N=6", 0.0, DUR)

KINDS = np.array([["camera"] + ["register"] * N_REG + ["patch"] * N_PATCH] * S)  # (S, N)
COLS = kind_colors(KINDS)                                                       # (S, N, 3)

# frame-wise layout: one row per frame; global layout: one row of S*N tokens
PITCH = 52
SRC_XY = grid_xy(S, N, (540, 560), PITCH, 74)                                   # (S, N, 2)
DST_IDS = reshape_ids((S, N), lambda a: a.reshape(1, S * N))                    # (1, S*N)
DST_XY = grid_xy(1, S * N, (540, 560), PITCH)                                   # (1, S*N, 2)

rng = np.random.default_rng(3)
Q = rng.normal(size=(S * N, 4))
K = Q + 0.6 * rng.normal(size=(S * N, 4))
L_frame, A_frame, _ = attention(Q, K, mask=block_diag_mask(S, N))
L_glob, A_glob, _ = attention(Q, K)
ATT = Formula([r"A", "=", r"\mathrm{softmax}\!\left(\frac{QK^{\mathsf{T}}}{\sqrt{d}}\right)"], size=56)
SHAPE_F = Formula([r"(S,\,N,\,C)"], size=44)
SHAPE_G = Formula([r"(1,\,S\cdot N,\,C)"], size=44)

for a, b, s in [(0.5, 4.5, "S images, N tokens each"), (4.5, 8.0, "frame attention: within one frame"),
                (8.0, 12.0, "reshape into one long sequence"), (12.0, 17.0, "global attention: sees across frames"),
                (17.0, 20.0, "each row sums to 1 after softmax"), (20.0, 24.0, "The two attentions alternate")]:
    shot.caption(a, b, s)


@shot.check
def check(ck):
    r = reshape_check((S, N), DST_IDS)
    ck.true("reshape is a bijection", r["bijective"])
    s_idx, d_idx = r["src_index"], r["dst_index"]
    ck.eq("token (s, n) lands at global index s*N + n", d_idx[:, 1], s_idx[:, 0] * N + s_idx[:, 1])
    ck.close("frame-wise rows sum to 1", A_frame.sum(1), np.ones(S * N), 1e-12)
    ck.close("global rows sum to 1", A_glob.sum(1), np.ones(S * N), 1e-12)
    ck.le("frame-wise weight outside own frame", A_frame[~block_diag_mask(S, N)].max(), 0.0)
    ck.record(S=S, N=N, C=C, frame_shape=[S, N, C], global_shape=[1, S * N, C], A_glob_row0=A_glob[0].round(4))


def token_layer(cv, t, u, alpha):
    pos, _ = move_tokens(SRC_XY, DST_XY, DST_IDS, u, stagger=0.6, arc=30)
    tokens(cv, pos, COLS.reshape(-1, 3), 40, alpha, glow=0.25)
    return pos


@shot.frame
def frame(cv, t):
    # 0-4.5: tensor block splits into S frames, then becomes token rows
    a_blk = fade(t, 0.2, 3.4, 0.5, 0.6)
    split = ss(t, 1.2, 2.4)
    if a_blk > 0:
        tensor_block(cv, (330, 420), 300, 210, 120, CYAN, a_blk, grid=(S, 1, 1), gap=(26 * split, 0, 0),
                     colors=lambda kh, kw, kd: [GOLD, VIOLET, CYAN][kh] if split > 0.5 else CYAN,
                     labels={"h": "$S$", "w": "$N$", "d": "$C$"}, label_size=36)
    a_tok = ss(t, 3.0, 3.8)
    u = ss(t, 8.2, 10.6) * (1 - ss(t, 20.2, 22.0))          # frame-wise -> global -> back
    pos = token_layer(cv, t, u, a_tok)
    if a_tok > 0:
        for s in range(S):
            cv.text(SRC_XY[s, 0] + [-50, 0], f"Frame {s + 1}", 26, alpha=a_tok * (1 - ss(t, 8.2, 9.0)), anchor="rm")
        sh = SHAPE_F if u < 0.5 else SHAPE_G        # swap while the tokens are mid-flight (alpha dips to 0)
        sh.draw(cv, (540, 400), alpha=a_tok * (1 - bump(t, 8.6, 1.6)) * (1 - bump(t, 20.3, 1.8))
                * (1 - fade(t, 12.5, 20.4)))
    # frame-wise attention: outline each frame's row
    a_fw = fade(t, 4.6, 8.2)
    for s in range(S):
        y = SRC_XY[s, 0, 1]
        cv.s.polyline([(SRC_XY[s, 0, 0] - 30, y - 30), (SRC_XY[s, -1, 0] + 30, y - 30),
                       (SRC_XY[s, -1, 0] + 30, y + 30), (SRC_XY[s, 0, 0] - 30, y + 30)], ORANGE,
                      a_fw * (0.5 + 0.5 * bump(t, 5.0 + 0.6 * s, 1.0)), 2, closed=True)
    # 12-20: heatmaps (frame-wise masked vs global), rows revealed one by one
    a_hm = fade(t, 12.6, 20.4)
    if a_hm > 0:
        rows = 1 + (S * N - 1) * ss(t, 13.0, 16.5)
        heatmap(cv, A_frame, (100, 660), 22, 0, A_glob.max(), a_hm, rows=rows)
        heatmap(cv, A_glob, (584, 660), 22, 0, A_glob.max(), a_hm, rows=rows,
                highlight=("row", 0) if t > 17 else None)
        cv.text((100 + 9 * 22, 650), "frame attention (block diagonal)", 26, alpha=a_hm, anchor="mb")
        cv.text((584 + 9 * 22, 650), "global attention", 26, alpha=a_hm, anchor="mb")
        ATT.draw(cv, (540, 1125), alpha=a_hm, part_alpha={2: ss(t, 13.5, 14.2)}, part_glow={2: bump(t, 17.0, 1.5)})
        if t > 17:
            cv.math(r"\sum_j A_{0j} = %.3f" % A_glob[0].sum(), (540, 1222), 30, style.LABEL_CYAN,
                    fade(t, 17.0, 20.4))
    # 20-24: module boxes
    a_m = ss(t, 20.4, 21.2)
    if a_m > 0:
        b1 = module_box(cv, (120, 880, 500, 1000), "Frame Attn", "within frame", ORANGE, a_m, glow=bump(t, 21.5, 1.0))
        b2 = module_box(cv, (580, 880, 960, 1000), "Global Attn", "across frames", CYAN, a_m, glow=bump(t, 22.5, 1.0))
        p = arrow(cv, b1["r"], b2["l"], WHITE, a_m, progress=ss(t, 20.8, 21.6))
        back = arrow(cv, b2["b"] + [0, 6], b1["b"] + [0, 6], WHITE, a_m * 0.6, bend=-0.25, dashed=True,
                     progress=ss(t, 22.0, 22.8))
        cv.math(r"\times\, L", (540, 1090), 40, style.TEXT, ss(t, 22.6, 23.2))
        pulse_dot(cv, p, (t - 21.6) / 0.8, GOLD, a_m)
        pulse_dot(cv, back, (t - 22.8) / 0.9, GOLD, a_m)
