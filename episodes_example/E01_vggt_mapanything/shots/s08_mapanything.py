"""E01-08 MapAnything: the same white dot, split into ray x depth x pose x scale (M, 55.6 s). Storyboard storyboard.md section E01-08.
Computed for real: unit ray R, depth along the ray, z-depth, the relation between the two (white dot + edge pixel), L = R·D, X = O L + T, global scaling 1/s and x m,
and in the pop-up the whole camera-2 D~ map (ray-cast z-depth x |K^-1[u,v,1]| / s). All computed by scene_e01 and asserted by checks.
Illustrative assumption: the scale of the network's output X~ is arbitrary; here the mean distance of X~ is set to 1, so m = s (corner tag from sentence 8).
Two pop-ups, definition first and numbers second: what D~ is (10.8-23.2 s), what m is (39.6-52.2 s);
the z-depth part gives the definition (z-depth = the coordinate along the optical axis, R_z = cos of the off-axis angle) and the product for the edge pixel.
Quantities with a tilde always come with normalized numbers (÷ s); hat = known quantity (input / training ground truth); m is learned through L_scale to be about ẑ / z~. D^ / D~ are written as symbols only; only the difference in meaning is stated (known ray depth vs the network's
up-to-scale ray depth); m is described by the shared definition sentence e01_common.M_DEF (the same sentence in 09), stating it is shared across views and one m per forward pass.
"""
import numpy as np

import e01_common as ec
import scene_e01 as sc
from animkit import (CYAN, GOLD, ORANGE, Formula, ImagePanel, Shot, arrow, bump, colorbar, colorize, fade, lerp,
                     measure, module_box, ss, style, table, vector)
from e01_common import Popup, popup_check

DUR = 55.6
shot = Shot("s08_mapanything", DUR)
shot.tag("Illustrative: mean output distance set to 1; measured metric error at end of shot 09", 18.4, DUR)
CAPS = [(0.3, 2.8, "MapAnything changes the output format"), (2.8, 5.4, "The same white dot, split into four factors"),
        (5.4, 8.0, "ray: this pixel's unit direction"), (8.0, 10.6, "The whole image's rays are the intrinsics"),
        # pop-up D~ (10.8-23.2)
        # symbols only, no talk about "tilde / hat" notation, only the difference in meaning
        (10.6, 13.2, "ray depth: how far along the ray"), (13.2, 15.8, "One number per pixel: 1×H×W"),
        (15.8, 18.4, "Up-to-scale: only relative distance, no units"), (18.4, 21.0, "The white dot's cell is 0.922"),
        (21.0, 23.6, "Known depth vs network output"),       # "known depth", not "true depth": hat = given, not necessarily metric
        # along the ray vs along the optical axis
        (23.6, 26.2, "Measured along the ray, not z-depth"), (26.2, 28.8, "They differ by R's z component"),
        (28.8, 31.4, "The nearer the image edge, the bigger the gap"),
        (31.4, 34.0, "Multiply to get the point in camera 2's frame"), (34.0, 36.8, "Pose O, T moves it into camera 1's frame"),
        (36.8, 39.4, "Only the scale is missing now"),
        # pop-up m (39.6-52.2); explains the role and scope of m (same as 09)
        (39.4, 42.0, "m: metric scale factor, scales the scene into meters"), (42.0, 44.6, "Shared across views: one m for all"),
        (44.6, 47.2, "Training pushes m ≈ GT distance ÷ output distance"), (47.2, 49.8, "Assume mean output distance 1: m = 2.153"),
        (49.8, 52.4, "Relative-scale-only data can train it too"),
        (52.4, DUR, "m = 2.153; multiply back to get meters")]
for a, b, s in CAPS:
    shot.caption(a, b, s)

NB = ec.nb()
CAM = sc.EX_CAM
R_cam, t_cam, C2 = sc.cameras()[CAM]
O, T = NB["O"], NB["T"]
RAY = NB["ray"]                                   # unit ray in the camera-2 coordinate frame
RAY_W = O @ RAY                                   # direction in the world frame
DR, DZ = NB["ray_depth"], NB["zdepth"]
S = NB["s"]
XS = NB["X_star"]
UV = NB["uv"]
L_n = RAY * DR / S
X_n = XS / S
T_n = T / S
ZAX = R_cam[2]                                    # direction of the camera-2 optical axis in the world frame
# Contrast: a pixel near the lower-left corner of the image (the ray is much more slanted, so the two depths visibly differ)
EDGE_UV = np.array([30.0, 470.0])
_, _zE, _XE = sc.render(CAM, EDGE_UV[None])
ZE, XE = float(_zE[0]), _XE[0]
RAY_E = np.linalg.solve(sc.K, [EDGE_UV[0], EDGE_UV[1], 1.0]); RAY_E /= np.linalg.norm(RAY_E)
DRE = float(np.linalg.norm(R_cam @ XE + t_cam))
TH_STAR = np.degrees(np.arccos(RAY[2]))           # angle between ray and optical axis (true value)
TH_EDGE = np.degrees(np.arccos(RAY_E[2]))

# Pop-up D~: the whole camera-2 D~ map (computed). The render has z component = 1, so ray depth = z-depth x |K^-1 [u, v, 1]|
_, _ZMAP, _ = sc.full(CAM)
_uu, _vv = np.meshgrid(np.arange(sc.W, dtype=float), np.arange(sc.H, dtype=float))
_NRM = np.sqrt(((_uu - sc.CX) / sc.FX) ** 2 + ((_vv - sc.CY) / sc.FY) ** 2 + 1.0)
DT_MAP = _ZMAP * _NRM / S                         # background = inf
DT_OK = np.isfinite(DT_MAP)
DT_LO, DT_HI = 0.6, 1.9
DT_PANEL = ImagePanel(colorize(DT_MAP, DT_LO, DT_HI), (40, 322), w=200)
_, _zS, _ = sc.render(CAM, UV[None])
DT_STAR = float(_zS[0] * np.sqrt(((UV[0] - sc.CX) / sc.FX) ** 2 + ((UV[1] - sc.CY) / sc.FY) ** 2 + 1.0) / S)


def v3(v):
    return r",\ ".join(f"{x:.3f}" for x in v)


# A lone \cdot, = or + gets typeset on the baseline (library v0 issue, reported), so each operator is merged into the adjacent term
MAIN = Formula([r"X_i^{\mathrm{metric}}", r"=m\,\cdot", "(", r"O_i", "(", r"R_i\cdot", r"\tilde{D}_i", ")",
                r"+\tilde{T}_i", ")"], size=56, gap=0.1)
MAIN_XY = (540, 215)
LIT = {5: 5.6, 6: 10.8, 4: 31.6, 7: 31.6, 2: 34.2, 3: 34.2, 8: 34.2, 9: 34.2, 1: 39.6, 0: 52.6}
LEGEND = [(5.4, 10.6, r"R_i", ": one unit direction per pixel (3×H×W)"),
          (10.6, 31.4, r"\tilde{D}_i", ": distance from each pixel along its ray to the point, up-to-scale"),
          (31.4, 34.0, r"\tilde{L}_i=R_i\cdot\tilde{D}_i", ": the point in camera 2's own frame"),
          (34.0, 39.4, r"O_i,\ \tilde{T}_i", ": camera 2's rotation and translation in camera 1's frame"),
          (39.4, DUR, "m", ": metric scale factor, shared by all views; multiply to get meters")]
LEG_F = [Formula([sym], 34) for _, _, sym, _ in LEGEND]
SLOT_A, SLOT_B = (540, 1085), (540, 1165)          # two readout rows at the bottom (the caption area starts at 1265)

F_R = Formula([r"R=(" + v3(RAY) + ")"], 40)
# Quantities with a tilde come with normalized numbers (÷ s), consistent across the shot; 3 decimals for the white dot and 4 for the edge pixel so each is self-consistent (the check compares strings)
F_ZR = Formula([f"{DR / S:.3f}", r"\times" + f"{RAY[2]:.3f}", "=" + f"{DZ / S:.3f}"], 42)
F_ZE = Formula([f"{DRE / S:.4f}", r"\times" + f"{RAY_E[2]:.4f}", "=" + f"{ZE / S:.4f}"], 42)
NOTE_ZR = {0: r"$\tilde D$", 1: f"$R_z=\\cos{TH_STAR:.1f}^\\circ$", 2: "z-depth"}
NOTE_ZE = {0: r"$\tilde D$", 1: f"$R_z=\\cos{TH_EDGE:.1f}^\\circ$", 2: "z-depth"}
F_DT = Formula([r"\tilde{D}=" + f"{DR:.3f}" + r"\,/\," + f"{S:.3f}" + "=" + f"{DR / S:.3f}"], 40)
F_L = Formula([r"\tilde{L}=R\cdot\tilde{D}=(" + v3(L_n) + ")"], 40)
F_X = Formula([r"\tilde{X}=O\,\tilde{L}+\tilde{T}=(" + v3(X_n) + ")"], 40)
F_M = Formula([r"X^{\mathrm{metric}}=" + f"{S:.3f}" + r"\,\tilde{X}=(" + v3(XS) + r")\ \mathrm{m}"], 40)
F_T = Formula([r"\tilde{T}=(" + v3(T_n) + ")"], 30)
F_m = Formula([r"m=" + f"{S:.3f}"], 48)
F_MDEF = Formula([r"X_i^{\mathrm{metric}}", r"=m\,\cdot", r"\tilde{X}_i", r",\quad i=1,\dots,N"], 46, gap=0.1)

CLOUD, CRGB, _, _ = ec.cloud(5)
FIT = np.r_[np.zeros((1, 3)), [C2, XS], ec.plane_corners(0, 0.45), ec.plane_corners(CAM, 0.45), CLOUD[::7]]
AZ0, AZ1, EL = -8.0, 4.0, 52.0
OC = ec.viewer(az=0.5 * (AZ0 + AZ1), el=EL, r=7.0, fit_pts=FIT, box=(880, 620), center=(540, 690),
               target=np.mean([np.zeros(3), C2, XS], 0))


def az_at(t):
    return AZ0 + (AZ1 - AZ0) * ss(t, 3.0, DUR)


XS_PT = tuple(OC.at(az=az_at(17.0))(ec.disp(XS)))          # the pop-up D~ leader points at the white dot

class _PopupLowTitle(Popup):
    """The tilde of $\\tilde D$ would be squeezed against the window top border in the default title (size 28, centered in the title bar), so this window draws its own title,
    size 26, moved down 5 px. e01_common.Popup (shared) is not changed; title bar, divider and border are still drawn by Popup.draw."""

    def draw(self, cv, t, body=None):
        title = self.title

        def body2(sub, lt, box):
            if body is not None:
                body(sub, lt, box)
            sub.rich(title, (ec.POPUP_PAD, ec.POPUP_TITLE_H / 2 + 5), 26, "bold", 1.0, anchor="lm", shadow=False)

        self.title = ""
        try:
            return super().draw(cv, t, body2)
        finally:
            self.title = title


POP_D = _PopupLowTitle((60, 690, 960, 550), t_in=10.8, t_out=23.2, title=r"$\tilde D_i$: depth along the ray (ray depth)",
                       corner="bl", accent=GOLD, point_to=XS_PT)
POP_M = Popup((60, 330, 960, 720), t_in=39.6, t_out=52.2, title="$m$: the metric scale factor shared by all views", corner="tl",
              accent=GOLD)                  # no leader: it would cover the legend at line 300; the m in the formula is glowing at the same time


@shot.check
def check(ck):
    ck.close("|R| = 1", np.linalg.norm(RAY), 1.0, 1e-12)
    ck.close("ray depth × R_z = z-depth", DR * RAY[2], DZ, 1e-12)
    ck.eq("on-screen white-dot product is self-consistent: normalized, 3 decimals (compares the formatted strings)", f"{round(DR / S, 3) * round(RAY[2], 3):.3f}",
          f"{DZ / S:.3f}")
    ck.eq("white-dot z-depth (÷ s) = ground truth of shot 05, 0.918", f"{DZ / S:.3f}", "0.918")
    ck.close("O = R_vggt^T", O, R_cam.T, 1e-12)
    ck.close("T = camera 2 center = -R^T t", T, -R_cam.T @ t_cam, 1e-12)
    ck.close("O (R·D) + T = white dot", O @ (RAY * DR) + T, XS, 1e-9)
    ck.close("still holds after normalization: O L~ + T~ = X~", O @ L_n + T_n, X_n, 1e-12)
    ck.close("m · X~ = metric point (m = s)", S * X_n, XS, 1e-12)
    ck.true("edge pixel hits the scene", np.isfinite(ZE))
    ck.close("edge pixel: ray depth × R_z = z-depth", DRE * RAY_E[2], ZE, 1e-9)
    ck.eq("on-screen edge-pixel product is self-consistent: normalized, 4 decimals", f"{round(DRE / S, 4) * round(RAY_E[2], 4):.4f}", f"{ZE / S:.4f}")
    ck.close("R_z = cos(off-axis angle): white dot", np.cos(np.radians(TH_STAR)), RAY[2], 1e-12)
    ck.eq("on-screen off-axis angles 34.8° / 5.4°", (f"{TH_EDGE:.1f}", f"{TH_STAR:.1f}"), ("34.8", "5.4"))
    ck.eq("edge pixel in meters 1.572 / 1.292", (f"{DRE:.3f}", f"{ZE:.3f}"), ("1.572", "1.292"))
    ck.eq("on-screen edge pixel normalized 0.730 / 0.600", (f"{DRE / S:.3f}", f"{ZE / S:.3f}"), ("0.730", "0.600"))
    # pop-up D~
    ck.close("D~ map at the white-dot pixel = ray depth / s (recomputed along the same ray)", DT_STAR, DR / S, 1e-9)
    ck.close("D~ map: value at the white-dot pixel ≈ 0.922", DT_MAP[int(round(UV[1])), int(round(UV[0]))], DR / S, 5e-3)
    ck.true("all valid D~ map values lie inside the colorbar range", DT_MAP[DT_OK].min() >= DT_LO and DT_MAP[DT_OK].max() <= DT_HI,
            f"{DT_MAP[DT_OK].min():.3f}–{DT_MAP[DT_OK].max():.3f}")
    ck.eq("on-screen 1.984 ÷ 2.153 = 0.922 (self-consistent at 3 decimals)", f"{round(DR, 3) / round(S, 3):.3f}", f"{DR / S:.3f}")
    ck.eq("D~ map shape 1×518×518", (1,) + DT_MAP.shape, (1, 518, 518))
    popup_check(ck, [POP_D, POP_M], DUR)
    ck.record(ray=RAY.round(4), ray_depth=round(DR, 4), z_depth=round(DZ, 4), s=round(S, 4), L_n=L_n.round(4),
              X_n=X_n.round(4), T_n=T_n.round(4), X_metric=XS.round(4), edge_uv=EDGE_UV, edge_ray_depth=round(DRE, 4),
              edge_z_depth=round(ZE, 4), edge_Rz=round(float(RAY_E[2]), 4), edge_deg=round(TH_EDGE, 2),
              star_deg=round(TH_STAR, 2), dtilde_star=round(DR / S, 4),
              dtilde_range=[round(float(DT_MAP[DT_OK].min()), 3), round(float(DT_MAP[DT_OK].max()), 3)],
              dtilde_valid_px=int(DT_OK.sum()))


def inset(cv, a, t):
    """Cross-section inset on the right: the cross-sections of the two rays through the optical axis, rotated into one plane; lengths drawn to a px/m scale."""
    if a <= 0.01:
        return
    x0, y0, x1, y1 = 600, 350, 1050, 720
    cv.s.rect(x0, y0, x1, y1, (20, 24, 32), 0.85 * a, r=12)
    cv.text((x0 + 14, y0 + 10), "Each ray with the axis, rotated into one plane\nTrue angles, lengths; numbers = along ray / axis", 20,
            alpha=a, fill=style.TEXT_DIM, spacing=6)
    o = np.array([x0 + 40, y0 + 250.0])
    ppm = 170.0 * S                                 # pixels per normalized unit (= 170 px per meter)
    ax_end = o + [ppm * 2.15 / S, 0]
    cv.s.line(o, ax_end, CYAN, 0.6 * a, 1)
    cv.text((ax_end[0] - 4, ax_end[1] + 6), "optical axis", 22, alpha=a, fill=style.LABEL_CYAN, anchor="ra")
    hl_w = bump(t, 26.4, 2.2)                        # sentence 11: the white-dot ray flashes
    hl_e = ss(t, 28.9, 29.4)                         # sentence 12: the edge-pixel ray lights up
    for th, dr, dz, col, hl in [(TH_EDGE, DRE / S, ZE / S, GOLD, hl_e), (TH_STAR, DR / S, DZ / S, (235, 235, 235), hl_w)]:
        d = np.array([np.cos(np.radians(th)), -np.sin(np.radians(th))])
        p = o + ppm * dr * d
        f = o + [ppm * dz, 0]
        cv.s.line(o, p, col, a, 2)
        if hl > 0.01:
            cv.g.line(o, p, col, 0.7 * hl * a, 3)
            cv.s.line(f, o, CYAN, hl * a, 2)        # the segment along the optical axis (= z-depth)
        cv.s.dashed(p, f, style.GREY, 0.8 * a, 1, 8)
        cv.s.dot(p, col, 4, a)
    cv.s.dot(o, (235, 235, 235), 4, a)
    cv.text((o[0], o[1] + 8), "Camera 2", 22, alpha=a, fill=ec.CAM_COLORS[CAM], anchor="ma")
    cv.text((x0 + 14, y1 - 72), f"Gold: edge pixel, {TH_EDGE:.1f}° off: {DRE / S:.3f} / {ZE / S:.3f}", 22,
            alpha=a * (0.6 + 0.4 * hl_e), fill=style.LABEL_GOLD)
    cv.text((x0 + 14, y1 - 40), f"White: white dot, {TH_STAR:.1f}° off: {DR / S:.3f} / {DZ / S:.3f}", 22, alpha=a,
            fill=style.TEXT)


def scale_k(t):
    return lerp(1.0, 1.0 / S, ss(t, 37.0, 38.2)) * lerp(1.0, S, ss(t, 52.6, 54.0))


# ---------------------------------------------------------------- pop-up D~
def body_d(sub, lt, box):
    """Input / output (definition) -> shape (the whole map) -> what the tilde means -> white-dot value -> the hat difference."""
    # top left: schematic. Optical center -> unit ray R (length 1) -> walk D~ along it to the 3D point; in part 3 both rays are scaled together and the pixels do not move
    a1 = ss(lt, 0.3, 0.7)
    o = np.array([66.0, 214.0])
    dA = np.array([np.cos(np.radians(20)), -np.sin(np.radians(20))])
    dB = np.array([np.cos(np.radians(36)), -np.sin(np.radians(36))])
    env = ss(lt, 5.5, 5.8) * (1 - ss(lt, 7.4, 7.6))
    k = 1.0 + 0.3 * np.sin(2 * np.pi * (lt - 5.6) / 1.8) * env
    LA, LB = 300.0 * k, 170.0 * k
    xp = o[0] + 40                                  # image plane
    sub.s.line((xp, 150), (xp, 230), style.GREY, 0.8 * a1, 2)
    sub.badge((24, 64), "Illustration", 18, a1, anchor="la")
    pa = o + LA * dA * ss(lt, 0.9, 1.9)
    sub.s.dashed(o, pa, GOLD, 0.8 * a1, 2, 30)
    arrow(sub, o, o + 90 * dA, GOLD, a1, th=3, head=12, progress=ss(lt, 0.4, 0.9))
    a_pt = ss(lt, 1.7, 2.0)
    if a_pt > 0.01:
        sub.s.dot(pa, GOLD, 6, a_pt)
        sub.g.dot(pa, GOLD, 6, 0.6 * a_pt)
        sub.s.dot(o + dA * (xp - o[0]) / dA[0], GOLD, 3, a_pt)
    a_b = ss(lt, 5.2, 5.6)
    if a_b > 0.01:                                  # ray of a second pixel: both segments lengthen and shorten together, the pixel position stays
        pb = o + LB * dB
        sub.s.dashed(o, pb, (235, 235, 235), 0.7 * a_b, 2, 20)
        sub.s.dot(pb, (235, 235, 235), 5, a_b)
        sub.s.dot(o + dB * (xp - o[0]) / dB[0], (235, 235, 235), 3, a_b)
    sub.s.dot(o, (235, 235, 235), 4, a1)
    sub.text((o[0] - 8, o[1]), "center", 18, alpha=a1, fill=style.TEXT_DIM, anchor="rm")
    sub.text((xp + 6, 232), "image plane", 18, alpha=a1, fill=style.TEXT_DIM, anchor="lb")
    sub.rich("Gold arrow = $R_i$: unit direction, length 1", (24, 252), 18, alpha=a1 * ss(lt, 0.6, 1.0), fill=style.LABEL_GOLD,
             anchor="lm")
    sub.rich(r"Gold dashes: to the 3D point; length = $\tilde D_i$", (24, 278), 18, alpha=a_pt, fill=style.LABEL_GOLD,
             anchor="lm")

    # bottom left: the whole camera-2 D~ map (computed), swept in from top to bottom
    a2 = ss(lt, 2.6, 3.0)
    if a2 > 0.01:
        sub.rich(r"Camera 2's $\tilde D_2$ map (computed)", (40, 306), 18, alpha=a2, fill=style.TEXT_DIM, anchor="lm")
        DT_PANEL.draw(sub, a2, border=style.GREY, border_alpha=0.6)
        bx0, by0, bx1, by1 = DT_PANEL.box
        cut = by0 + (by1 - by0) * ss(lt, 2.7, 4.0)
        if cut < by1 - 0.5:
            sub.s.rect(bx0 - 1, cut, bx1 + 2, by1 + 2, (0, 0, 0), 1.0)
        colorbar(sub, (bx1 + 12, by0, bx1 + 26, by1), DT_LO, DT_HI, fmt="{:.1f}", alpha=a2, size=18)
        a4 = ss(lt, 7.8, 8.2)
        if a4 > 0.01:
            DT_PANEL.mark(sub, UV[0], UV[1], (255, 255, 255), rad=4 + 2 * bump(lt, 8.2, 1.0), alpha=a4,
                          label=f"{DR / S:.3f}", label_offset=(10, -12), size=22)

    # right column: text, appearing paragraph by paragraph
    X = 460
    rows = [(0.4, 80, "Definition: one number per pixel of view $i$,", 25, style.TEXT),
            (0.8, 114, "distance along unit ray $R_i$ to the 3D point", 25, style.TEXT),
            (2.6, 162, r"Shape: $1\times H\times W$ = $1\times518\times518$ (left)", 24, style.TEXT),
            (5.2, 210, "What up-to-scale means:", 25, style.LABEL_GOLD),            (5.6, 244, "Rescale the whole scene: pixels stay the same;", 23, style.TEXT),
            (6.2, 274, "so only relative distance is known, no units", 23, style.TEXT),
            (7.8, 322, r"White-dot cell: $\tilde D$ = " + f"{DR / S:.3f}", 28, style.LABEL_GOLD),
            (8.4, 358, f"= true {DR:.3f} m ÷ scene mean distance {S:.3f} m", 21, style.TEXT_DIM),
            (8.8, 386, "(illustration: mean output distance set to 1)", 21, style.TEXT_DIM),
            # symbols only, only the difference in meaning
            (10.4, 428, r"$\hat D_i$: known depth along the ray", 23, style.LABEL_ORANGE),            (10.6, 461, "(can be an input at inference; ground truth in training)", 20, style.TEXT_DIM),
            (11.0, 498, r"$\tilde D_i$: the network's up-to-scale ray depth", 23, style.LABEL_GOLD)]
    for t0, y, s, size, col in rows:
        a = ss(lt, t0, t0 + 0.4)
        if a > 0.01:
            sub.rich(s, (X, y), size, alpha=a, fill=col, anchor="lm")


# ---------------------------------------------------------------- pop-up m
SHAPES = [[r"$R_i$", r"$3\times H\times W$", "Per view, per pixel"],
          [r"$\tilde D_i$", r"$1\times H\times W$", "Per view, per pixel"],
          # O_i and T~_i get separate rows; the network outputs a quaternion Q_i, O_i is computed from it
          [r"$O_i$", "quat. $Q_i$: 4", "Per view"],
          [r"$\tilde T_i$", "translation: 3", "Per view"],
          [r"$m$", "scalar", "Shared by all views"]]
M_ROW = len(SHAPES) - 1
DY = 24                                             # the definition sentence takes two lines, so everything below sits 24 px lower


def body_m(sub, lt, box):
    """Definition + formula -> shapes (compared with the other quantities) -> shared by all views -> numeric example (illustrative assumption) -> why it is separate."""
    a1 = ss(lt, 0.3, 0.7)
    F_MDEF.draw(sub, (480, 102), alpha=a1, part_glow={1: bump(lt, 0.9, 1.0)}, shadow=False)
    # shared definition sentence (the same in 09, e01_common.M_DEF)
    sub.rich(ec.M_DEF_1, (480, 150), 24, alpha=a1, fill=style.LABEL_GOLD, anchor="mm")
    sub.rich(ec.M_DEF_2, (480, 180), 24, alpha=a1, anchor="mm")
    # left: shape comparison table (row by row, the m row is boxed)
    table(sub, (24, 196 + DY), [100, 170, 210], ["Quantity", "Shape", "Scope"], SHAPES, row_h=40, size=22,
          reveal=len(SHAPES) * ss(lt, 0.8, 2.2), highlight=M_ROW if lt > 2.0 else None, alpha=a1)
    # right: the three views share one m (sentence 2)
    a2 = ss(lt, 2.6, 3.0)
    if a2 > 0.01:
        mc = np.array([862.0, 312.0 + DY])
        for j in range(3):
            y = 236 + DY + 78 * j
            b = module_box(sub, (540, y - 26, 740, y + 26), f"View {j + 1}: " + r"$m\,\tilde X_" + f"{j + 1}$",
                           color=ec.CAM_COLORS[j], alpha=a2 * ss(lt, 2.7 + 0.3 * j, 3.0 + 0.3 * j), title_size=22)
            arrow(sub, mc + [-34, 0], b["r"], GOLD, a2, th=2, head=10, progress=ss(lt, 3.0 + 0.3 * j, 3.6 + 0.3 * j),
                  glow=0.4 * bump(lt, 3.6 + 0.3 * j, 0.6))
        sub.s.dot(mc, GOLD, 30, 0.25 * a2)
        sub.g.dot(mc, GOLD, 22, 0.5 * a2 * (0.5 + 0.5 * bump(lt, 4.2, 1.0)))
        sub.math("m", tuple(mc), 46, style.LABEL_GOLD, a2, shadow=False)
        # shared across views; one forward pass gives only one m (Eq. (1) outputs {m, (...)_{i=1}^N})
        sub.text((mc[0], mc[1] + 38), "Shared by all views", 20, alpha=a2, fill=style.TEXT_DIM, anchor="ma")
        sub.rich("One $m$ per pass", (mc[0], mc[1] + 66), 18, alpha=a2 * ss(lt, 3.4, 3.8), fill=style.TEXT_DIM,
                 anchor="ma")
    # sentence 3: how m is learned (scale loss L_scale)
    a3 = ss(lt, 5.2, 5.6)
    if a3 > 0.01:
        sub.rich(r"How it is learned: the scale loss $L_{\mathrm{scale}}$ makes $m\cdot\tilde z\approx\hat z$ in training, so $m\approx\hat z\,/\,\tilde z$",
                 (480, 490), 24, alpha=a3, fill=style.TEXT, anchor="mm")
        sub.rich(r"$\tilde z$ = mean distance of network output points (up-to-scale); $\hat z$ = mean distance of GT points (meters)", (480, 526), 21,
                 alpha=a3 * ss(lt, 5.8, 6.2), fill=style.TEXT_DIM, anchor="mm")
    # sentence 4: numeric example (illustrative assumption)
    a4 = ss(lt, 7.8, 8.2)
    if a4 > 0.01:
        sub.rich(r"Illustrative assumption: set $\tilde z$ = 1, then $m=\hat z$ = true mean scene distance = " + f"{S:.3f}", (480, 568), 23,
                 alpha=a4, fill=style.LABEL_GOLD, anchor="mm")
        u = ss(lt, 8.6, 9.6)
        sub.rich(r"White dot $\tilde X$" if u < 0.5 else r"White dot $X^{\mathrm{metric}}$", (270, 612), 24, alpha=a4,
                 fill=style.LABEL_GOLD, anchor="rm")
        vector(sub, (490, 612), lerp(X_n, XS, u), fmt="{:.3f}", size=34, color=style.MATH, alpha=a4)
        if u > 0.98:
            sub.text((700, 612), "m", 30, "light", a4, style.MATH, "lm")
        sub.badge((760, 612), r"$\times\,$" + f"{S:.3f}", 22, a4 * fade(lt, 8.2, 10.0, 0.3, 0.3), anchor="lm",
                  fill=style.LABEL_GOLD, border=GOLD)
    # sentence 5: why it is separate
    a5 = ss(lt, 10.4, 10.8)
    if a5 > 0.01:
        sub.rich(r"Why separate: datasets with only relative scale can still train $R,\ \tilde D,\ O,\ \tilde T$ (paper §1)", (480, 654), 22,
                 alpha=a5, fill=style.TEXT, anchor="mm")
        sub.rich("Such data just doesn't train $m$ (this half-sentence is inferred)", (480, 684), 22, alpha=a5 * ss(lt, 11.0, 11.4),
                 fill=style.TEXT_DIM, anchor="mm")


def slot(F, xy, a, **kw):
    if a > 0.01:
        F.draw(cv_ref[0], xy, alpha=a, **kw)


cv_ref = [None]


@shot.frame
def frame(cv, t):
    cv_ref[0] = cv
    ec.paper_card(cv, (540, 700), fade(t, 0.3, 2.9, 0.4, 0.4), "2025", "arXiv 2509.13414",
                  "MapAnything: Universal Feed-Forward\nMetric 3D Reconstruction",
                  "Keetha et al. · Meta Reality Labs, CMU",
                  "One network: images + optional intrinsics, poses, depth,\noutputs metric 3D directly")
    dm = 1.0 - 0.5 * max(POP_D.on(t), POP_M.on(t))          # main view dimmed while a pop-up is shown

    # formula + definition of the current term
    a_eq = ss(t, 2.9, 3.4)
    if a_eq > 0:
        pa = {k: 0.3 + 0.7 * ss(t, t0, t0 + 0.4) for k, t0 in LIT.items()}
        pg = {k: bump(t, t0 + 0.2, 0.8) for k, t0 in LIT.items()}
        MAIN.draw(cv, MAIN_XY, alpha=a_eq, part_alpha=pa, part_glow=pg)
        for (a, b, _, txt), F in zip(LEGEND, LEG_F):
            al = a_eq * fade(t, a, b, 0.3, 0.3)
            if al > 0.01:
                _, _, x1, _ = F.draw(cv, (80, 300), style.LABEL_GOLD, al, "lm")
                cv.text((x1 + 4, 300), txt, 28, alpha=al, fill=style.TEXT_DIM, anchor="lm")

    a3 = ss(t, 2.9, 3.6)
    if a3 <= 0:
        return
    k = scale_k(t)
    P = OC.at(az=az_at(t))
    ec.splat(cv, P, k * CLOUD, CRGB, 0.5 * a3 * dm, size=2)
    a_c1 = a3 * (0.35 + 0.65 * ss(t, 34.0, 34.6)) * dm
    ec.frustum(cv, P, 0, 0.45, a_c1, label="Camera 1", scale=k)
    ec.frustum(cv, P, CAM, 0.45, a3 * dm, tex_alpha=0.5, label="Camera 2", scale=k)
    c2 = P(ec.disp(k * C2))
    xs = P(ec.disp(k * XS))

    # c3: unit ray
    a_ray = ss(t, 5.6, 6.2) * (1 - ss(t, 10.6, 11.0))
    if a_ray > 0:
        tip = P(ec.disp(k * (C2 + RAY_W)))
        arrow(cv, c2, tip, GOLD, a_ray, th=3, progress=ss(t, 5.6, 6.4), glow=0.6)
        cv.text((tip[0] + 12, tip[1]), "length 1", 26, alpha=a_ray, fill=style.LABEL_GOLD, anchor="lm")
    slot(F_R, SLOT_A, ss(t, 6.2, 6.6) * (1 - ss(t, 10.4, 10.8)), color=style.LABEL_GOLD)
    a_cs = fade(t, 6.4, 8.0, 0.4, 0.3)
    if a_cs > 0:
        cv.text(SLOT_B, "Camera 2's own coordinate frame", 26, alpha=a_cs, fill=style.TEXT_DIM, anchor="mm")

    # c4: ray fan of the whole image (thinned to 10x10)
    a_fan = fade(t, 8.1, 10.8, 0.5, 0.4)
    if a_fan > 0:
        Kinv = np.linalg.inv(sc.K)
        for v in np.linspace(0, sc.H, 10):
            for u in np.linspace(0, sc.W, 10):
                d = Kinv @ [u, v, 1.0]
                d = O @ (d / np.linalg.norm(d))
                cv.s.line(c2, P(ec.disp(k * (C2 + 0.8 * d))), CYAN, 0.45 * a_fan, 1)
        cv.text(SLOT_B, "ray map: 3 × 518 × 518, equivalent to intrinsics", 28, alpha=a_fan, fill=style.LABEL_CYAN, anchor="mm")

    # c5 (during pop-up D~): walk along the ray to the white dot; c9-c11: along the ray vs along the optical axis
    a_gold = ss(t, 10.8, 11.6) * (1 - ss(t, 31.4, 32.0))
    if a_gold > 0:
        g_end = c2 + (np.asarray(xs) - c2) * ss(t, 10.8, 12.0)
        cv.s.line(c2, g_end, GOLD, a_gold * dm, 3)
        cv.g.line(c2, g_end, GOLD, 0.5 * a_gold * dm, 3)
    a_m = ss(t, 23.6, 24.2) * (1 - ss(t, 31.4, 32.0))
    if a_m > 0:
        f2 = P(ec.disp(k * (C2 + DZ * ZAX)))
        cv.s.dashed(c2, f2, CYAN, a_m, 2, 34)
        cv.s.dashed(xs, f2, style.GREY, 0.8 * a_m, 1, 8)
        inset(cv, a_m * ss(t, 24.0, 24.6), t)
    a_zdef = fade(t, 23.8, 26.2, 0.4, 0.3)
    if a_zdef > 0:
        cv.text(SLOT_B, "z-depth = the point's coordinate along the optical axis (VGGT's D)", 26, alpha=a_zdef, fill=style.LABEL_CYAN,
                anchor="mm")
    a_l1 = fade(t, 24.4, 28.8, 0.4, 0.3)
    if a_l1 > 0:
        cv.rich(r"Along ray $\tilde D$: " + f"{DR / S:.3f}", (80, SLOT_A[1]), 30, alpha=a_l1, fill=style.LABEL_GOLD,
                anchor="lm")
        cv.rich(f"Along axis (z-depth): {DZ / S:.3f} (VGGT's D)", (470, SLOT_A[1]), 30, alpha=a_l1,
                fill=style.LABEL_CYAN, anchor="lm")
    a_zr = ss(t, 26.4, 26.8) * (1 - ss(t, 28.6, 29.0))
    slot(F_ZR, SLOT_B, a_zr, part_alpha={1: ss(t, 26.7, 27.1), 2: ss(t, 27.1, 27.5)},
         part_glow={2: bump(t, 27.4, 0.8)}, notes=NOTE_ZR, note_size=22)
    a_e = fade(t, 28.9, 31.4, 0.4, 0.4)
    if a_e > 0:
        cv.text(SLOT_A, f"Edge pixel (30, 470): ray {TH_EDGE:.1f}° off the axis", 30, alpha=a_e, fill=style.LABEL_GOLD,
                anchor="mm")
    slot(F_ZE, SLOT_B, a_e, part_alpha={1: ss(t, 29.2, 29.6), 2: ss(t, 29.6, 30.0)},
         part_glow={2: bump(t, 29.9, 0.8)}, notes=NOTE_ZE, note_size=22)

    # c13: L = R·D (camera-2 coordinate axes)
    a_l = ss(t, 31.4, 31.9) * (1 - 0.6 * ss(t, 36.8, 37.2))
    if a_l > 0:
        ec.axes(cv, P, CAM, 0.4, a_l * dm, scale=k)
        cv.s.line(c2, xs, GOLD, 0.6 * a_l * dm, 2)
    slot(F_DT, SLOT_A, ss(t, 31.4, 31.8) * (1 - ss(t, 33.8, 34.2)))
    slot(F_L, SLOT_B, ss(t, 32.0, 32.4) * (1 - ss(t, 33.8, 34.2)))

    # c14: the pose moves the point into the camera-1 frame
    a_p = ss(t, 34.0, 34.5)
    if a_p * dm > 0.01:                     # ec.axes() draws nothing and returns None at alpha <= 0.01
        o1 = ec.axes(cv, P, 0, 0.4, a_p * dm, scale=k)
        arrow(cv, o1, c2, ORANGE, a_p * dm, th=3, progress=ss(t, 34.2, 35.0), glow=0.4 * bump(t, 35.0, 1.0))
        mid = 0.5 * (np.asarray(o1) + np.asarray(c2))
        F_T.draw(cv, (mid[0], mid[1] - 18), style.LABEL_ORANGE, a_p * (1 - ss(t, 36.8, 37.2)), "mb")
    slot(F_X, SLOT_A, ss(t, 35.0, 35.4) * (1 - ss(t, 52.4, 52.8)))

    # c15-c21: scale by 1/s; pop-up m; then x m
    a_ns = fade(t, 37.2, 52.4, 0.4, 0.4)
    if a_ns > 0:
        cv.text(SLOT_B, "Coordinates have no units: only the shape is known", 30, alpha=a_ns, fill=style.TEXT_DIM, anchor="mm")
    F_m.draw(cv, (880, 420), style.LABEL_GOLD, ss(t, 52.5, 52.9), glow=bump(t, 52.8, 1.2))
    slot(F_M, SLOT_A, ss(t, 53.2, 53.8))
    ec.star_marker(cv, xs, a3 * dm, glow=0.7 + 0.3 * bump(t, 3.4, 1.0) + 0.5 * bump(t, 53.8, 1.0), rad=6)

    POP_D.draw(cv, t, body_d)
    POP_M.draw(cv, t, body_m)
