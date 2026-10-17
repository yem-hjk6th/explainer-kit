"""E01-10 Side-by-side comparison + wrap-up (S, 36.6 s). Pre copy: hides the white dot coordinates, the mean distance 1.001 and the metric scale errors 72.3% / 9.5% measured on our test scene.
Storyboard: storyboard.md section E01-10.
Besides reproducing the paper numbers, the shot gives an explanatory comparison: the same three images and the same white dot flow through VGGT and
MapAnything in two columns: what input each needs, what it outputs, how it represents things internally, units, and what each suits (the last item is our reading).
The comparison is drawn as two columns (see storyboard section 3).
Paper numbers: MapAnything Tab. 2(a) (two images, images only, points tau, mean over ETH3D / ScanNet++ v2 / TAv2; measured in the MapAnything paper).
Measured on our test scene: the two VGGT points for the white dot, mean distance 1.001, and the MapAnything metric scale error 72.3% / 9.5%.
"""
import sys
from pathlib import Path
E01_SHOTS = Path(__file__).resolve().parents[2] / "shots"
if str(E01_SHOTS) not in sys.path:
    sys.path.insert(0, str(E01_SHOTS))

import numpy as np

import e01_common as ec
from animkit import (CYAN, GOLD, GREEN, ORANGE, Formula, Shot, arrow, bar_chart, bump, fade, flicker, input_slot,
                     module_box, ss, style)

PRE_HIDE = True                                    # pre cut: hide the values measured on our test scene (white dot coordinates, mean distance, metric scale error)

DUR = 36.6
T_NUM = 31.4                                       # start of the paper-numbers segment
shot = Shot("s10_compare", DUR)
shot.title("Side by Side", "VGGT vs MapAnything", 0.2, 2.2)
shot.tag("Same 3 images; white dot and metric error measured on our test scene", 7.4, T_NUM - 0.3)
shot.tag("MapAnything paper Tab. 2(a) (VGGT via depth + camera route)", T_NUM, DUR)
shot.tag("Models: VGGT-1B (non-commercial), MapAnything (CC-BY-NC 4.0)", 34.6, DUR)
CAPS = [(0.3, 2.6, "The same three images go into both models"), (2.6, 5.0, "Input: VGGT needs only images"),
        (5.0, 7.4, "MapAnything can also take known inputs"), (7.4, 9.8, "VGGT: camera, depth, point map, one each"),
        (9.8, 12.2, "Points come twice: P directly, or from D plus g"), (12.2, 14.6, "MapAnything: factored into four parts"),
        (14.6, 17.0, "Ray times depth, to camera 1, then times m"), (17.0, 19.2, "VGGT's unit is mean distance, not meters"),
        (19.2, 21.6, "MapAnything multiplies by m; unit is meters"), (21.6, 24.2, "Our test scene, images only: metric scale off 72%"),
        (24.2, 26.6, "Our reading: images only, either model works"), (26.6, 29.0, "Need tracking? Use VGGT"),
        (29.0, T_NUM, "Calibration, pose, or meters? Use MapAnything"),
        (T_NUM, 33.6, "Paper numbers: two images, images only"), (33.6, DUR, "VGGT 43.2%, MapAnything 57.5%")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

VM = ec.vggt_metrics()
MA = ec.ma_metrics()
WD = VM["white_dot"]
P_HEAD, P_DC = np.array(WD["point_head"]), np.array(WD["depth_cam"])
MEAN_NORM = VM["points"]["point_head"]["mean_norm_pred"]
# relative metric scale error = predicted scene mean distance / GT - 1 (metric_scale_rel_err in the json keeps only 4 digits, rounding would be off by 0.1 percentage point)
ERR_A = MA["configs"]["A"]["s_pred_m"] / MA["s_gt_m"] - 1   # images only
ERR_C = MA["configs"]["C"]["s_pred_m"] / MA["s_gt_m"] - 1   # intrinsics + metric pose
TAB2A = {"VGGT": 43.2, "MapAnything": 57.5, "MapAnything all inputs": 82.0}   # paper numbers

LX, RX = 280, 800                                  # column center lines
TW = 112
TH_Y = 335
COL01 = {"g": ORANGE, "D": (240, 110, 140), "P": CYAN, "T": GOLD}        # same colors as 01
COLMA = {"R": CYAN, "D": (240, 110, 140), "OT": ORANGE, "m": GREEN}
THUMBS = [ec.view_pil(i) for i in range(3)]
BOX_L = (80, 560, 480, 640)                        # width 400 so the subtitle does not touch the edge, center line unchanged
BOX_R = (600, 560, 1000, 640)
SLOTS = ["Rays (calib.)", "Pose", "Depth"]

FMA = Formula([(r"X_i=m\,\cdot", "m"), ("(", "p1"), (r"O_i", "O"), ("(", "p2"), (r"R_i\cdot", "R"),
               (r"\tilde{D}_i", "D"), (")", "p3"), (r"+\tilde{T}_i", "T"), (")", "p4")], size=38, gap=0.08)
FMA_XY = (RX, 840)


def fv(v):
    return ec.fmt_vec(v, 3)


@shot.check
def check(ck):
    ck.true("the two white dot points are nearly identical (point head vs depth + camera, difference < 0.01)", np.linalg.norm(P_HEAD - P_DC) < 0.01,
            float(np.linalg.norm(P_HEAD - P_DC)))
    ck.close("VGGT point head mean distance (on screen 1.001)", round(MEAN_NORM, 3), 1.001, 1e-9)
    ck.close("MapAnything images only: metric scale error (on screen 72.3%)", round(100 * ERR_A, 1), 72.3, 1e-9)
    ck.close("MapAnything intrinsics + metric pose: metric scale error (on screen 9.5%)", round(100 * ERR_C, 1), 9.5, 1e-9)
    for g, e in (("A", ERR_A), ("C", ERR_C)):
        ck.close(f"config {g}: s_pred / s_gt - 1 = metric_scale_rel_err in the json", e, MA["configs"][g]["metric_scale_rel_err"],
                 1e-4)
    ck.eq("inputs of config C are intrinsics + metric pose", MA["configs"]["C"]["inputs"], "images + intrinsics + metric poses")
    ck.record(tab2a=TAB2A, white_dot_point_head=P_HEAD, white_dot_depth_cam=P_DC)


def thumbs(cv, cx, alpha, glow=0.0):
    x0 = cx - 1.5 * TW - 12
    for i in range(3):
        x = x0 + i * (TW + 12)
        ec.paste_img(cv, THUMBS[i], (x, TH_Y), TW, alpha)
        pts = [(x, TH_Y), (x + TW, TH_Y), (x + TW, TH_Y + TW), (x, TH_Y + TW)]
        cv.s.polyline(pts, ec.CAM_COLORS[i], 0.8 * alpha, 2, closed=True)
        if glow > 0.01:
            cv.g.polyline(pts, ec.CAM_COLORS[i], glow * alpha, 4, closed=True)


def chip(cv, box, text, color, alpha, size=22, glow=0.0):
    x0, y0, x1, y1 = box
    if alpha <= 0.01:
        return
    cv.s.rect(x0, y0, x1, y1, tuple(int(0.18 * c) for c in color), 0.9 * alpha, r=8)
    cv.s.polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color, 0.9 * alpha, 2, closed=True)
    if glow > 0.01:
        cv.g.polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color, glow * alpha, 4, closed=True)
    cv.rich(text, ((x0 + x1) / 2, (y0 + y1) / 2), size, alpha=alpha, anchor="mm", align="m")


def split_view(cv, t, A):
    """Explanatory comparison: VGGT on the left, MapAnything on the right, the same images flowing top to bottom. A = overall alpha."""
    cv.s.line((540, 285), (540, 1058), style.GREY, 0.5 * A, 1)
    cv.text((LX, 300), "VGGT", 34, "bold", A, style.LABEL_GOLD, "mm")
    cv.text((RX, 300), "MapAnything", 34, "bold", A, style.LABEL_GOLD, "mm")

    # 1. input
    a_in = ss(t, 0.4, 1.0) * A
    thumbs(cv, LX, a_in, glow=0.8 * bump(t, 3.0, 1.4))
    thumbs(cv, RX, a_in, glow=0.8 * bump(t, 5.2, 1.4))
    cv.rich("Only $N$ images", (LX, TH_Y + TW + 14), 22, alpha=ss(t, 2.7, 3.1) * A, fill=style.TEXT_DIM, anchor="ma")
    a_sl = ss(t, 5.1, 5.6) * A
    if a_sl > 0.01:
        on = flicker(t, 3, 0.55, 0.5, seed=4, t0=5.4) if t < 7.2 else np.ones(3, bool)
        for k, lab in enumerate(SLOTS):                # slot width 146 so the ray (calibration) label does not touch the edge
            x0 = RX - 225 + k * 152
            input_slot(cv, (x0, 462, x0 + 146, 502), lab, optional=True, on=float(on[k]), alpha=a_sl, size=19)
        cv.text((RX, 512), "Dashed = optional; give any subset, for any subset of views", 18, alpha=a_sl, fill=style.TEXT_DIM,
                anchor="ma")
    # backbone boxes
    a_bl = ss(t, 3.2, 3.7) * A
    if a_bl > 0.01:
        b = module_box(cv, BOX_L, "VGGT", "24 rounds × (frame + global), width $1024$", CYAN, a_bl, title_size=28, sub_size=20,
                       glow=0.5 * bump(t, 3.8, 0.8))
        arrow(cv, (LX, TH_Y + TW + 44), b["t"], style.WHITE, a_bl, 2, 12, progress=ss(t, 3.2, 3.6))
        cv.text((LX, 652), "Reference: frame 1 uses dedicated camera / register tokens", 19, alpha=a_bl, fill=style.TEXT_DIM,
                anchor="ma")
    a_br = ss(t, 6.4, 6.9) * A
    if a_br > 0.01:
        b = module_box(cv, BOX_R, "MapAnything", "16 layers (8 global + 8 frame), width $1536$", CYAN, a_br, title_size=28,
                       sub_size=20, glow=0.5 * bump(t, 7.0, 0.8))
        arrow(cv, (RX, 538), b["t"], style.WHITE, a_br, 2, 12, progress=ss(t, 6.4, 6.8))
        cv.text((RX, 652), "Reference: view 1 gets a fixed embedding", 19, alpha=a_br, fill=style.TEXT_DIM, anchor="ma")

    # 2. VGGT outputs: one of each of the four
    a_o = ss(t, 7.5, 8.0) * A
    if a_o > 0.01:
        # (no arrow here: it would cross the "Reference" line, and the output cells sit right under the model box)
        items = [("g", "$g_i$: camera (9 nums)"), ("D", "$D_i$: depth (z)"), ("P", "$P_i$: point map"), ("T", "$T_i$: track features")]
        for k, (key, txt) in enumerate(items):
            u = ss(t, 7.7 + 0.3 * k, 8.1 + 0.3 * k) * A
            x0 = LX - 225 + (k % 2) * 230
            y0 = 696 + (k // 2) * 54
            chip(cv, (x0, y0, x0 + 220, y0 + 44), txt, COL01[key], u,
                 glow=0.7 * (bump(t, 10.2, 1.6) if key in ("P",) else bump(t, 10.6, 1.6) if key in ("D", "g") else 0))
    # redundancy: P given directly vs D + g unprojected, the two results for the white dot are nearly identical
    a_r = ss(t, 9.9, 10.4) * A
    if a_r > 0.01:
        if not PRE_HIDE:                               # pre cut: hide the measured white dot (camera 2) and its two coordinate lines
            cv.text((LX - 225, 818), "White dot (camera 2):", 21, alpha=a_r, fill=style.TEXT_DIM, anchor="lm")
            cv.rich(f"$P$ directly: {fv(P_HEAD)}", (LX - 225, 852), 21, alpha=a_r, fill=COL01["P"], anchor="lm")
            cv.rich(f"$D+g$ unprojected: {fv(P_DC)}", (LX - 225, 886), 21, alpha=a_r * ss(t, 10.5, 10.9), fill=COL01["D"],
                    anchor="lm")
        cv.text((LX - 225, 922), "Same point computed twice → redundancy", 22, alpha=a_r * ss(t, 11.0, 11.4), fill=style.TEXT,
                anchor="lm")

    # 3. MapAnything: four factors that compose the point
    a_f = ss(t, 12.3, 12.8) * A
    if a_f > 0.01:
        pass                                    # no arrow, as in the left column (it would cross the "Reference" line)
        items = [("R", "$R_i$: ray per pixel"), ("D", r"$\tilde D_i$: ray depth"),
                 ("OT", r"$O_i,\tilde T_i$: pose"), ("m", "$m$: one scene scale")]
        for k, (key, txt) in enumerate(items):
            u = ss(t, 12.5 + 0.3 * k, 12.9 + 0.3 * k) * A
            x0 = RX - 235 + (k % 2) * 240
            y0 = 696 + (k // 2) * 54
            chip(cv, (x0, y0, x0 + 230, y0 + 44), txt, COLMA[key], u, size=20)
        pa = {"R": ss(t, 14.7, 15.1), "D": ss(t, 14.7, 15.1), "p2": ss(t, 14.7, 15.1), "p3": ss(t, 14.7, 15.1),
              "O": ss(t, 15.4, 15.8), "T": ss(t, 15.4, 15.8), "p1": ss(t, 15.4, 15.8), "p4": ss(t, 15.4, 15.8),
              "m": ss(t, 16.1, 16.5)}
        pc = {"R": COLMA["R"], "D": COLMA["D"], "O": COLMA["OT"], "T": COLMA["OT"], "m": COLMA["m"]}
        if t > 14.6:
            FMA.draw(cv, FMA_XY, alpha=A, part_alpha=pa, part_color=pc,
                     part_glow={"R": bump(t, 14.9, 0.8), "O": bump(t, 15.6, 0.8), "m": bump(t, 16.3, 0.8)})
            cv.text((RX, 880), "Each quantity once; points are built from them", 21, alpha=A * ss(t, 16.4, 16.8), fill=style.TEXT,
                    anchor="ma")
            cv.rich("$X_i$: 3D point per pixel of view $i$ (camera 1 frame, meters)", (RX, 912), 18,
                    alpha=A * ss(t, 16.4, 16.8), fill=style.TEXT_DIM, anchor="ma")

    # 4. units
    a_u = ss(t, 17.1, 17.6) * A
    if a_u > 0.01:
        cv.text((LX, 965), "Unit: mean distance = 1, not meters", 24, alpha=a_u, fill=style.LABEL_GOLD, anchor="ma")
        if not PRE_HIDE:                               # pre cut: hide the measured point head mean distance 1.001
            cv.rich(f"Our test scene, point head: mean distance to camera 1 ${MEAN_NORM:.3f}$", (LX, 1003), 20,
                    alpha=a_u * ss(t, 17.8, 18.2), fill=style.TEXT_DIM, anchor="ma")
    a_u2 = ss(t, 19.3, 19.8) * A
    if a_u2 > 0.01:
        cv.rich("Unit: meters (times $m$)", (RX, 965), 24, alpha=a_u2, fill=style.LABEL_GOLD, anchor="ma")
        a_e = ss(t, 21.7, 22.2) * A
        if not PRE_HIDE:                               # pre cut: hide the measured metric scale error 72.3% / 9.5%
            cv.rich(f"Our test scene, images only: scale off by ${100 * ERR_A:.1f}\\%$", (RX, 1003), 20, alpha=a_e, fill=style.TEXT,
                    anchor="ma")
            cv.rich(f"With intrinsics + metric pose: ${100 * ERR_C:.1f}\\%$", (RX, 1033), 20, alpha=a_e * ss(t, 22.6, 23.0),
                    fill=style.TEXT_DIM, anchor="ma")

    # 5. what each suits (our reading)
    a_s = ss(t, 24.3, 24.8) * A
    if a_s > 0.01:
        cv.s.line((40, 1068), (1040, 1068), style.GREY, 0.4 * a_s, 1)
        cv.badge((40, 1082), "Our take", 22, a_s, anchor="la", fill=style.LABEL_GOLD, border=(140, 120, 80))
        cv.text((560, 1098), "Images only: both work (MapAnything paper's own test: better than VGGT even then, Tab. 2)", 20,
                alpha=a_s, fill=style.TEXT, anchor="mm")
        a_l = ss(t, 26.7, 27.2) * A
        cv.rich("Need point tracking → VGGT (only it outputs $T_i$)", (LX, 1150), 22, alpha=a_l, fill=COL01["T"], anchor="mm")
        a_rr = ss(t, 29.1, 29.6) * A
        cv.text((RX, 1150), "Calibration (incl. fisheye / non-pinhole), depth,\nIMU / GPS pose, or metric scale → MapAnything", 20,
                alpha=a_rr, fill=COLMA["R"], anchor="mm", align="center", spacing=6)


@shot.frame
def frame(cv, t):
    A = 1 - ss(t, T_NUM - 0.3, T_NUM + 0.2)
    if A > 0.01:
        split_view(cv, t, A)
    a_b = ss(t, T_NUM + 0.1, T_NUM + 0.6)
    if a_b > 0:
        bar_chart(cv, (300, 500, 960, 780), [TAB2A["VGGT"], TAB2A["MapAnything"]], ["VGGT", "MapAnything"], vmax=70,
                  horizontal=True, highlight=1, progress=ss(t, 33.6, 34.8), fmt="{:.1f}%",
                  title="Two images, images only: share of pixels whose distance\nto camera 1 has error < 3%", better="higher", alpha=a_b,
                  note="Mean over ETH3D, ScanNet++ v2, TAv2; each side first divided by its mean distance")
        cv.rich(f"Same table, adding intrinsics + pose + depth: MapAnything reaches {TAB2A['MapAnything all inputs']:.1f}%", (540, 900), 22,
                alpha=a_b * ss(t, 34.8, 35.2), fill=style.TEXT_DIM, anchor="ma")
