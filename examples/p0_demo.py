"""Smoke test of the basic E01 elements on a synthetic scene, three 6-7 s sections:
A  formula states + notes, counter / vector / matrix readouts, badges, tensor shape morph
B  folded token strip (n = 1374, token 799 forced), (N, K) -> (1, N K) reshape, single-token fly, branching arrows
C  3D in an OpenCV world: K,R,t frustums with thumbnails, 60k-point cloud from unprojected depth, marker, measure
   lines, ray fan, axes; image panel with patch grid / cell / ruler / marker; depth map + colorbar
Synthetic scene = one sphere on a ground plane, ray-cast analytically here (not E01's scene)."""
import numpy as np

from animkit import (CYAN, GOLD, ORANGE, VIOLET, WHITE, Formula, ImagePanel, OrbitCam, PinholeCam, Shot, TokenStrip,
                     arrow_path, branches, camera_axes, bump, colorbar, colorize, counter, draw_camera, fade, fly_token,
                     fold, log_px, marker3d, matrix, measure, module_box, point_cloud, pulse_dot, ray_fan,
                     reshape_rows, ss, style, tensor_morph, token, vector)

DUR = 20.0
shot = Shot("p0_demo", DUR)
shot.tag("schematic", 0, 13.0)
shot.tag("example values", 0, 6.5)
shot.tag("synthetic scene, not model output", 13.0, DUR)

# ---------------------------------------------------------------- synthetic scene (OpenCV world = camera 1 frame)
IMG = 336                                     # ~6e4 of the 336^2 rays hit the scene
FOV = 60.0
F_PX = (IMG / 2) / np.tan(np.radians(FOV) / 2)
K = np.array([[F_PX, 0, IMG / 2], [0, F_PX, IMG / 2], [0, 0, 1.0]])
SPH_C, SPH_R, GROUND_Y = np.array([0.0, 0.05, 2.6]), 0.45, 0.5


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def cam_pose(yaw):
    """Camera orbiting the sphere centre at the same height: R world->camera, t."""
    C = SPH_C + rot_y(yaw) @ np.array([0, -0.35, -2.6])
    f = (SPH_C - C) / np.linalg.norm(SPH_C - C)
    r = np.cross(f, [0, -1, 0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    return R, -R @ C


def raycast(R, t):
    """z-depth (inf = miss) and shaded RGB for every pixel of a camera."""
    u, v = np.meshgrid(np.arange(IMG) + 0.5, np.arange(IMG) + 0.5)
    dc = np.stack([(u - K[0, 2]) / F_PX, (v - K[1, 2]) / F_PX, np.ones_like(u)], -1)   # z = 1
    C = -R.T @ t
    d = dc @ R                                                                       # world dirs = R^T dc
    oc = C - SPH_C
    b = d @ oc
    a = np.einsum("...i,...i", d, d)
    disc = b * b - a * (oc @ oc - SPH_R ** 2)
    ts = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
    tg = np.where(d[..., 1] > 1e-9, (GROUND_Y - C[1]) / d[..., 1], np.inf)
    Xg = C + np.where(np.isfinite(tg), tg, 0)[..., None] * d
    tg = np.where((np.abs(Xg[..., 0]) < 2.2) & (np.abs(Xg[..., 2] - 3.0) < 2.2), tg, np.inf)
    z = np.minimum(ts, tg)
    rgb = np.zeros(z.shape + (3,)) + [28, 30, 38]
    sph = ts <= tg
    X = C + np.where(np.isfinite(z), z, 0)[..., None] * d
    n = (X - SPH_C) / SPH_R
    lam = np.clip(n @ np.array([0.3, -0.8, -0.5]) / np.linalg.norm([0.3, -0.8, -0.5]), 0, 1) * 0.7 + 0.3
    rgb[sph & np.isfinite(z)] = (np.array([70, 120, 220]) * lam[..., None])[sph & np.isfinite(z)]
    chk = (np.floor(X[..., 0] / 0.3) + np.floor(X[..., 2] / 0.3)) % 2 == 0
    g = (~sph) & np.isfinite(z)
    rgb[g] = np.where(chk[g][:, None], [165, 165, 160], [105, 105, 102])
    return z, np.clip(rgb, 0, 255).astype(np.uint8)


POSES = [cam_pose(y) for y in (0.0, 30.0, -30.0)]
CAMS = [PinholeCam.from_KRt(K, R, t, IMG, IMG, plane_d=0.35) for R, t in POSES]
DEPTH, RGB = raycast(*POSES[0])
uu, vv = np.meshgrid(np.arange(IMG) + 0.5, np.arange(IMG) + 0.5)
hit = np.isfinite(DEPTH)
PTS = CAMS[0].unproject(np.c_[uu[hit], vv[hit]], DEPTH[hit])
PCOL = RGB[hit].astype(float)
THUMBS = [RGB] + [raycast(*p)[1] for p in POSES[1:]]
X_STAR = PTS[np.argmin(np.linalg.norm(PTS - (SPH_C + [0.0, -0.2, -0.38]), axis=1))]
view = OrbitCam(SPH_C, 4.2, el=28, world="cv")
view.fit(np.vstack([PTS[::50], *[c.C for c in CAMS], *[p for c in CAMS for p in c.corners()]]), 760, 520,
         (540, 520), az=-25)
DIRS = np.stack([CAMS[0].ray_dir((u, v)) for v in np.linspace(10, IMG - 10, 8) for u in np.linspace(10, IMG - 10, 8)])

# ---------------------------------------------------------------- A
EQ = Formula([("P", "P"), "=", (r"R^{\mathsf{T}}", "R"), "(", ("D", "D"), (r"K^{-1}", "K"),
              r"[u,v,1]^{\mathsf{T}}", "-", ("t", "t"), ")"], size=58)
KM = [["f_x", 0, "c_x"], [0, "f_y", "c_y"], [0, 0, 1]]
N_IMG, N_TOK, C_DIM = 3, 1374, 1024
# ---------------------------------------------------------------- B
STRIP = TokenStrip(N_TOK, fold(N_TOK, head=5, tail=2, force=[799]), rows=3, center=(540, 470), size=30, append=1)


def kinds(r, i):
    if i == N_TOK:
        return "scale"
    base = "camera" if i == 0 else "register" if i <= 4 else "patch"
    return base + "1" if r == 0 and base != "patch" else base


for a, b, s in [(0.3, 3.2, "Formula terms light up, notes below"), (3.2, 6.5, "Vector, matrix, shape change together"),
                (6.5, 9.6, "1374 tokens, middle folded"), (9.6, 13.0, "Rearranged by index into one row"),
                (13.0, 16.5, "Depth unprojected to a point cloud"), (16.5, 20.0, "Each cell is a 14-pixel patch")]:
    shot.caption(a, b, s)


@shot.check
def check(ck):
    px, z = CAMS[0].pix_many(PTS)
    ck.le("unproject -> project pixel residual (px)", np.abs(px - np.c_[uu[hit], vv[hit]]).max(), 1e-9)
    ck.le("unproject -> project depth residual", np.abs(z - DEPTH[hit]).max(), 1e-9)
    ck.ge("point count", len(PTS), 50000)
    ck.eq("token 799 shown in the folded strip", 799 in STRIP.shown, True)
    _, _, ids = reshape_rows(N_IMG, 4, 1.0, (540, 0), (540, 0), 30, 50)
    ck.eq("(N, K) -> (1, N K): slot of token (i, k) = i K + k", ids.ravel(), np.arange(N_IMG * 4))
    ck.eq("global index of frame 1, token 799 (K = 1374)", 1 * N_TOK + 799, 2173)
    ck.record(points=len(PTS), x_star=X_STAR.round(3), depth_range=[float(DEPTH[hit].min()), float(DEPTH[hit].max())])


@shot.frame
def frame(cv, t):
    # ---- A: formula + readouts + tensor morph
    aA = fade(t, 0.2, 6.6)
    if aA > 0:
        lit = ["P", "R", "D", "K", "t"]
        k = int(np.clip((t - 0.6) / 0.5, -1, 4))
        st = {sid: ("hl" if j == k else "lit" if j < k else "dim") for j, sid in enumerate(lit)}
        EQ.draw(cv, (540, 190), alpha=aA, state=st, notes={"D": "$D$: depth map", "K": "$K$: intrinsics"} if t > 2.4 else None)
        counter(cv, (210, 360), ss(t, 1.0, 3.0), 1, 24, "{:d}", size=56, alpha=aA)
        cv.rich("layers $L$", (210, 410), 26, alpha=aA, anchor="ma")
        v = np.array([-0.300, 0.139, 2.075]) * (1 - ss(t, 3.4, 4.6)) + np.array([-0.145, 0.067, 1.000]) * ss(t, 3.4,
                                                                                                             4.6)
        vector(cv, (700, 360), v, size=40, alpha=aA, part_color={2: style.LABEL_CYAN})
        cv.rich("$X^{\\star}$ (meters → normalized)", (700, 410), 26, alpha=aA, anchor="ma")
        matrix(cv, (210, 560), KM, size=36, alpha=aA, box={(0, 0): bump(t, 4.0, 1.2), (1, 1): bump(t, 4.0, 1.2)})
        cv.rich("$K$", (210, 470), 30, alpha=aA, anchor="mb")
        u = ss(t, 4.4, 5.8)
        pa = (log_px(N_TOK, 20, 70), log_px(N_IMG, 30, 60), log_px(C_DIM, 0, 30))
        pb = (log_px(N_IMG * N_TOK, 20, 70), log_px(1, 30, 60), log_px(C_DIM, 0, 30))
        tensor_morph(cv, (520, 520), pa, pb, u, {"h": "$N=3$", "w": "1374 token", "d": "$C=1024$"},
                     {"h": "$1$", "w": "4122 token", "d": "$C=1024$"}, label_size=26, alpha=aA, color=CYAN)
    # ---- B: token strip + reshape + fly + branches
    aB = fade(t, 6.4, 13.1)
    if aB > 0:
        lab = {(1, 799): "799"} if t > 7.4 else None
        STRIP.draw(cv, kinds, aB * (1 - ss(t, 9.4, 9.8)), highlight={(1, 799): ORANGE} if t > 7.4 else None,
                   labels=lab, band={2: VIOLET} if t > 8.2 else None)
        r = ss(t, 9.6, 11.2)
        if t > 9.4:
            pos, uk, _ = reshape_rows(3, 4, r, (540, 470), (540, 470), 40, 57, stagger=0.5, arc=20, group_gap=18)
            cols = [style.TOKEN[kinds(i, k if k < 2 else 900)] for i in range(3) for k in range(4)]
            for p, c in zip(pos, cols):
                token(cv, p, 30, c, aB, 0.2)
        if t > 11.2:
            a = np.array([540, 520])
            fly_token(cv, a, (600, 620), ss(t, 11.3, 12.4), ORANGE, 30, alpha=aB,
                      label="frame 799 → global $1\\times1374+799=2173$" if t > 11.6 else None)
        top = module_box(cv, (390, 700, 690, 780), "Aggregator", "$(N, 1374, 1024)$", CYAN, aB, state="lit",
                         glow=bump(t, 7.0, 1.0))
        heads = [(165 + 250 * k, 950) for k in range(4)]
        for k, (pth, name) in enumerate(zip(branches(top["b"], [(h[0], 900) for h in heads], 840),
                                            ["Camera", "Depth", "Point", "Track"])):
            arrow_path(cv, pth, WHITE, aB, 2, 14, progress=ss(t, 7.2 + 0.2 * k, 8.2 + 0.2 * k))
            module_box(cv, (heads[k][0] - 105, 900, heads[k][0] + 105, 980), name, None, GOLD, aB,
                       state="lit" if k == int(t * 2) % 4 else "dim")
            pulse_dot(cv, pth, (t - 8.4 - 0.2 * k) % 2.0, GOLD, aB)
    # ---- C: 3D + panels
    aC = ss(t, 13.0, 13.6)
    if aC > 0:
        P = view.at(az=-25 + 20 * ss(t, 13.0, DUR))
        grow = ss(t, 13.4, 15.0)
        n_show = int(len(PTS) * grow)
        point_cloud(cv, P, PTS[:n_show], PCOL[:n_show], rad=1, alpha=aC)
        for k, cam in enumerate(CAMS):
            draw_camera(cv, cam, P, aC, label=f"camera {k + 1}", image=THUMBS[k], label_size=24)
        ray_fan(cv, P, CAMS[0].C, DIRS, 1.6, GOLD, 0.35 * aC, progress=ss(t, 14.0, 15.0))
        camera_axes(cv, P, CAMS[0], 0.35, alpha=aC, label_size=24)
        xs = marker3d(cv, P, X_STAR, "$X^{\\star}$", alpha=aC * ss(t, 15.0, 15.4))
        if t > 15.4:
            measure(cv, P(CAMS[0].C), xs, f"ray depth ${np.linalg.norm(X_STAR - CAMS[0].C):.3f}$", style.LABEL_GOLD,
                    aC * ss(t, 15.4, 15.8), offset=-20, label_size=24)
        panel = ImagePanel(RGB, (90, 870), w=280)
        panel.draw(cv, aC, title="Image seen by camera 1")
        panel.grid(cv, 14, alpha=0.4 * aC, progress=ss(t, 16.5, 17.5))
        if t > 17.5:
            panel.scan(cv, ss(t, 17.5, 19.5), 14, alpha=aC)
        panel.ruler(cv, "left", f"{IMG} px", alpha=aC)
        pu, pv = CAMS[0].pix(X_STAR)[0]
        panel.mark(cv, pu, pv, label="$x^{\\star}$", alpha=aC)
        dpanel = ImagePanel(colorize(DEPTH, 1.6, 3.6), (420, 870), w=280)
        dpanel.draw(cv, aC, title="z-depth (GT)")
        dpanel.mark(cv, pu, pv, label=f"${DEPTH[int(pv), int(pu)]:.3f}$ m", alpha=aC, size=22)
        colorbar(cv, (730, 870, 752, 1150), 1.6, 3.6, "m", alpha=aC, size=22)
