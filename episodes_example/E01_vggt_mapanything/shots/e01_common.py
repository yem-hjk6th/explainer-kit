"""Shared by all E01 shots: scene data, display coordinates, textured camera frusta, z-buffered point-cloud splats, paper cards.

The scene world frame is OpenCV (y down, = camera 1's camera frame); animkit's OrbitCam looks with y up,
so 3D drawing always goes through disp() first: (x, y, z) -> (x, -y, z). That way an observer camera at
az=0, el=0 sees the same picture as camera 1 (no mirroring).
The frustum texturing and point splats in this file are temporary local implementations, to be replaced
by the library's equivalents once they exist.
"""
import json
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

import scene_e01 as sc
from animkit import GOLD, WHITE, Canvas, OrbitCam, ss, style

FLIP = np.array([1.0, -1.0, 1.0])
CAM_COLORS = [(235, 235, 235), (120, 230, 150), (175, 140, 255)]     # cameras 1 / 2 / 3 (white, green, purple)
CAM_NAMES = ["Camera 1", "Camera 2", "Camera 3"]

# "Our test scene" = the test we built ourselves (defined at the start of shot 01): the whole episode uses only this phrase, and the corner tag uses only SCENE_TAG
SCENE = "Our test scene"
SCENE_TAG = "Measured on our test scene (our own test, not a paper experiment)"


# Unified definition sentence for m (identical in 08 and 09, same symbol style; the paper calls m the "global metric scaling factor" with X_i^metric = m·X~_i, i∈[1,N])
M_DEF_1 = "$m$: the metric scale factor shared by all views in one forward pass:"
M_DEF_2 = "scales the whole up-to-scale scene from the network into meters at once"
M_DEF = M_DEF_1 + M_DEF_2


def scene_tag(shot, t0, t1=None, detail=None):
    """Unified corner tag: the first line is always SCENE_TAG; detail (units, license, ...) goes on the second line."""
    shot.tag(SCENE_TAG, t0, t1)
    if detail:
        shot.tag(detail, t0, t1)


@lru_cache(maxsize=None)
def nb():
    return sc.numbers()


RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_runs")


@lru_cache(maxsize=None)
def vggt_out():
    """Real VGGT-1B output on the 3 images (model_runs/vggt/vggt_out.npz)."""
    p = os.path.join(RUNS, "vggt", "vggt_out.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{os.path.normpath(p)} missing: this shot draws real VGGT output. Run model_runs/vggt/run_vggt.py first (GPU; code from github.com/facebookresearch/vggt, weights download from the official Hugging Face repo facebook/VGGT-1B, non-commercial license). See episodes_example/README.md")
    d = np.load(p)
    return {k: d[k] for k in d.files}


@lru_cache(maxsize=None)
def ma_metrics():
    """MapAnything metrics for the three input types (model_runs/mapanything/ma_metrics.json)."""
    with open(os.path.join(RUNS, "mapanything", "ma_metrics.json"), encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def vggt_metrics():
    with open(os.path.join(RUNS, "vggt", "vggt_metrics.json"), encoding="utf-8") as f:
        return json.load(f)


def disp(X):
    return np.asarray(X, float) * FLIP


def view_rgb(i):
    return sc.full(i)[0]


def view_pil(i, mark=False):
    rgb = view_rgb(i)
    return Image.fromarray(rgb)


@lru_cache(maxsize=None)
def cloud(stride=3):
    """World points unprojected from all valid pixels of the three images (one per stride). Returns X (n,3), rgb (n,3), source camera (n,), pixel (n,2)."""
    Xs, cs, ids, uvs = [], [], [], []
    for i in range(3):
        rgb, tz, X = sc.full(i)
        v, u = np.mgrid[0:sc.H:stride, 0:sc.W:stride]
        ok = np.isfinite(tz[v, u])
        Xs.append(X[v, u][ok]); cs.append(rgb[v, u][ok].astype(float)); ids.append(np.full(ok.sum(), i))
        uvs.append(np.stack([u[ok], v[ok]], -1).astype(float))
    return np.concatenate(Xs), np.concatenate(cs), np.concatenate(ids), np.concatenate(uvs)


def scene_center():
    return sc.to_world(sc.PIVOT)


def viewer(az=0.0, el=12.0, r=4.5, fit_pts=None, box=(900, 640), center=(540, 820), target=None):
    oc = OrbitCam(disp(scene_center() if target is None else target), r, az, el)
    if fit_pts is not None:
        oc.fit(disp(fit_pts), box[0], box[1], center)
    return oc


def plane_corners(i, depth):
    """World coordinates of the four corners of camera i's image plane (at z = depth)."""
    R, t, _ = sc.cameras()[i]
    Kinv = np.linalg.inv(sc.K)
    px = np.array([[0, 0, 1], [sc.W, 0, 1], [sc.W, sc.H, 1], [0, sc.H, 1]], float)
    Xc = depth * (px @ Kinv.T)
    return (Xc - t) @ R


def plane_point(i, uv, depth):
    R, t, _ = sc.cameras()[i]
    Xc = depth * np.linalg.solve(sc.K, [uv[0], uv[1], 1.0])
    return R.T @ (Xc - t)


def frustum(cv, P, i, depth=0.45, alpha=1.0, color=None, tex_alpha=0.0, glow=0.0, label=None, scale=1.0):
    """Frustum of camera i: center -> four image-plane corners; with tex_alpha>0 the image is pasted onto the image plane (perspective-correct homography).
    scale: factor by which the whole scene is scaled about the world origin (camera 1); used when explaining "only a scale differs"."""
    if alpha <= 0.01:
        return
    color = CAM_COLORS[i] if color is None else color
    c = sc.cameras()[i][2]
    C = P(disp(scale * c))
    cs = P(disp(scale * plane_corners(i, depth)))
    if tex_alpha > 0.01:
        _paste_quad(cv.s.img, view_rgb(i), cs, tex_alpha * alpha)
    else:
        cv.s.add_poly(cs, style.PANEL, 0.35 * alpha)
    for q in cs:
        cv.s.line(C, q, color, 0.55 * alpha, 1)
    for k in range(4):
        cv.s.line(cs[k], cs[(k + 1) % 4], color, 0.95 * alpha, 2)
    cv.s.dot(C, color, 4, alpha)
    if glow > 0:
        cv.g.dot(C, color, 7, glow * alpha)
    if label:
        cv.text((C[0], C[1] + 14), label, 26, alpha=alpha, fill=color, anchor="ma")
    return C, cs


def frustum_pose(cv, P, R, c, depth=0.45, alpha=1.0, color=WHITE, image=None, tex_alpha=0.0, label=None, K=None):
    """Frustum for an arbitrary pose (R = world->camera rotation, c = camera center in the world frame; K defaults to the scene's ground-truth intrinsics)."""
    if alpha <= 0.01:
        return None
    Kinv = np.linalg.inv(sc.K if K is None else K)
    px = np.array([[0, 0, 1], [sc.W, 0, 1], [sc.W, sc.H, 1], [0, sc.H, 1]], float)
    corners = (depth * (px @ Kinv.T)) @ R + c
    C = P(disp(c))
    cs = P(disp(corners))
    if image is not None and tex_alpha > 0.01:
        _paste_quad(cv.s.img, image, cs, tex_alpha * alpha)
    else:
        cv.s.add_poly(cs, style.PANEL, 0.35 * alpha)
    for q in cs:
        cv.s.line(C, q, color, 0.55 * alpha, 1)
    for k in range(4):
        cv.s.line(cs[k], cs[(k + 1) % 4], color, 0.95 * alpha, 2)
    cv.s.dot(C, color, 4, alpha)
    if label:
        cv.text((C[0], C[1] + 14), label, 24, alpha=alpha, fill=color, anchor="ma")
    return C


def quat_to_R(q):
    """(x, y, z, w), scalar last -> rotation matrix (same convention as VGGT rotation.py)."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _paste_quad(img, src, quad, alpha):
    """Paste the src image with perspective onto the screen quad (order: top-left, top-right, bottom-right, bottom-left)."""
    h, w = src.shape[:2]
    quad = np.asarray(quad, np.float32)
    x0, y0 = np.floor(quad.min(0)).astype(int)
    x1, y1 = np.ceil(quad.max(0)).astype(int)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
                                    np.float32(quad - np.float32([x0, y0])))
    warp = cv2.warpPerspective(src, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    m = mask.astype(np.float32)[..., None] / 255.0 * alpha
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = np.clip(roi * (1 - m) + warp.astype(np.float32) * m + 0.5, 0, 255).astype(np.uint8)


def paste_img(cv, im, xy, w, alpha=1.0, nearest=False):
    """Paste an image (RGB array or PIL) into the sharp layer cv.s (cv.image pastes onto the text layer and would cover lines and dots drawn later). Returns (x0, y0, x1, y1)."""
    arr = np.asarray(im)[..., :3]
    h0, w0 = arr.shape[:2]
    w = int(round(w)); h = int(round(h0 * w / w0))
    x, y = int(round(xy[0])), int(round(xy[1]))
    if alpha <= 0.01 or w < 2:
        return x, y, x + w, y + h
    interp = cv2.INTER_NEAREST if nearest else (cv2.INTER_AREA if w < w0 else cv2.INTER_LINEAR)
    small = cv2.resize(arr, (w, h), interpolation=interp).astype(np.float32)
    img = cv.s.img
    H, W = img.shape[:2]
    xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if xb <= xa or yb <= ya:
        return x, y, x + w, y + h
    roi = img[ya:yb, xa:xb].astype(np.float32)
    src = small[ya - y:yb - y, xa - x:xb - x]
    img[ya:yb, xa:xb] = np.clip(roi * (1 - alpha) + src * alpha + 0.5, 0, 255).astype(np.uint8)
    return x, y, x + w, y + h


def splat(cv, P, X, cols, alpha=1.0, size=2, layer="s"):
    """Point-cloud splat with a z-buffer (only the nearest point is kept per pixel). X is (n,3) in the world frame, cols is (n,3) or a single RGB."""
    if alpha <= 0.01 or len(X) == 0:
        return
    D = disp(X)
    Q = P(D)
    z = P.depth(D)
    img = (cv.s if layer == "s" else cv.g).img
    H, W = img.shape[:2]
    cols = np.broadcast_to(np.asarray(cols, np.float32), (len(X), 3))
    ok = (z > 1e-3) & np.isfinite(Q).all(1)
    Q, z, cols = Q[ok], z[ok], cols[ok]
    xs0 = np.floor(Q[:, 0]).astype(np.int64)
    ys0 = np.floor(Q[:, 1]).astype(np.int64)
    xs, ys, zs, cc = [], [], [], []
    for dx in range(size):
        for dy in range(size):
            xs.append(xs0 + dx); ys.append(ys0 + dy); zs.append(z); cc.append(cols)
    xs, ys, zs, cc = map(np.concatenate, (xs, ys, zs, cc))
    m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, zs, cc = xs[m], ys[m], zs[m], cc[m]
    lin = ys * W + xs
    order = np.lexsort((zs, lin))
    lin_s = lin[order]
    first = np.ones(len(lin_s), bool)
    first[1:] = lin_s[1:] != lin_s[:-1]
    sel = order[first]
    yy, xx = ys[sel], xs[sel]
    base = img[yy, xx].astype(np.float32)
    img[yy, xx] = np.clip(base * (1 - alpha) + cc[sel] * alpha + 0.5, 0, 255).astype(np.uint8)


AXIS_COLORS = [(255, 110, 110), (120, 230, 150), (100, 170, 255)]


def axes(cv, P, i, length=0.35, alpha=1.0, scale=1.0, labels=True, th=2):
    """Axes of camera i (x right, y down, z forward, OpenCV), drawn at the camera center. Returns the origin's screen coordinates."""
    if alpha <= 0.01:
        return None
    R, _, c = sc.cameras()[i]
    o = P(disp(scale * c))
    for k in range(3):
        e = P(disp(scale * c + length * R[k]))           # row k of R = direction of camera axis k in the world frame
        cv.s.line(o, e, AXIS_COLORS[k], alpha, th)
        if labels:
            cv.text((e[0] + 4, e[1] - 4), "xyz"[k], 22, "light", alpha, AXIS_COLORS[k], "lb")
    return o


def star_marker(cv, p, alpha=1.0, glow=0.8, rad=6, label=None, label_dy=-18, size=28):
    """The white dot (the point that runs through the whole episode) at screen position p."""
    if alpha <= 0.01:
        return
    cv.g.dot(p, (255, 255, 255), rad + 3, glow * alpha)
    cv.s.dot(p, (255, 255, 255), rad, alpha)
    if label:
        cv.text((p[0], p[1] + label_dy), label, size, alpha=alpha, anchor="mb")


def paper_card(cv, center, alpha, year, venue, title, authors, line):
    """Text-only paper card: large year + venue + title + authors + one-line contribution (front-page figure still to be decided)."""
    if alpha <= 0.01:
        return
    cx, cy = center
    n_title = title.count("\n") + 1
    w, h = 960, 380 + 44 * (n_title - 1)
    x0, y0 = cx - w / 2, cy - h / 2
    cv.s.rect(x0, y0, x0 + w, y0 + h, (22, 26, 34), 0.92 * alpha, r=18)
    cv.s.polyline([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)], (90, 100, 115), 0.8 * alpha, 2,
                  closed=True)
    cv.text((x0 + 40, y0 + 34), year, 96, "light", alpha, style.LABEL_GOLD)
    cv.text((x0 + 290, y0 + 62), venue, 34, "light", alpha, style.TEXT_DIM)
    cv.text((x0 + 290, y0 + 108), authors, 24, "light", 0.85 * alpha, style.TEXT_DIM)
    cv.text((x0 + 40, y0 + 172), title, 34, "bold", alpha, spacing=8)
    cv.text((x0 + 40, y0 + 240 + 44 * (n_title - 1)), line, 32, "text", alpha, style.TEXT, spacing=12)


def fmt_vec(v, nd=3):
    return "(" + ", ".join(f"{x:.{nd}f}".replace("-", "−") for x in v) + ")"


# ---- Popup: a small window that pops out of a corner, with its own independent small canvas (to be merged into animkit later) ----
POPUP_BG = (22, 28, 38)
POPUP_TITLE_H = 50
POPUP_PAD = 16
CAPTION_TOP = 1250          # the caption area starts here; popups must not enter it


@lru_cache(maxsize=None)
def _popup_bg(w, h):
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    base = np.asarray(POPUP_BG, np.float32)[None, None, :]
    return np.broadcast_to(base * (1.08 - 0.16 * yy), (h, w, 3)).astype(np.float32)


@lru_cache(maxsize=None)
def _popup_mask(w, h, r=16):
    m = np.zeros((h, w), np.uint8)
    from animkit import rrect_pts
    pts = np.round(np.asarray(rrect_pts(0, 0, w - 1, h - 1, r, 8))).astype(np.int32)
    cv2.fillPoly(m, [pts], 255, cv2.LINE_AA)
    return m


class Popup:
    """Popup: a small window pops out of a corner of the frame to explain a side concept quickly (DPT, quaternion, ...), then retracts without interrupting the main line.

    Popup(box, t_in, t_out, title, corner="tr", pop=0.35, accent=ORANGE, point_to=None)
      box      (x0, y0, w, h): position and size (pixels) of the fully open window on the main frame. y0 + h must stay out of the caption area (>= 1250).
      t_in     time the popup starts opening (main timeline, seconds); t_out is when it starts retracting. Opening and retracting each take pop seconds (smoothstep).
      corner   the corner the window grows from and shrinks back to: "tl" | "tr" | "bl" | "br" (this corner stays fixed during the animation).
      title    title-bar text (rich, may contain $formulas$); accent = border / title-bar color.
      point_to a point (x, y) on the main frame: if given, a dashed leader is drawn from the window corner nearest to it (showing "this is what is being explained").
    Call pop.draw(cv, t, body) at the **end** of the frame function; body(sub, lt, box) draws on the small canvas:
      sub  animkit Canvas of size w x h, origin = window top-left; sub.s / sub.g / text / rich / math / Formula / readout work as usual
      lt   window-local time (seconds, counted from t_in); use with ss(lt, a, b) for small animations inside the window
      box  content area (x0, y0, x1, y1) in small-canvas coordinates, with the title bar and margins already removed
    The window is pasted onto the main frame's text layer (covering lines, dots and text drawn earlier), below captions / titles / corner tags.
    pop.on(t) returns the open fraction 0-1 (the main frame can use it to dim the background or make room); pop.local(t) = t - t_in.
    """

    def __init__(self, box, t_in, t_out, title="", corner="tr", pop=0.35, accent=None, point_to=None, tag=None):
        x0, y0, w, h = box
        assert x0 >= 0 and x0 + w <= 1080 and y0 >= 0, f"popup {title!r} off-screen: {box}"
        assert y0 + h <= CAPTION_TOP, f"popup {title!r} overlaps the caption area: {box}"
        assert t_out > t_in + 2 * pop, f"popup {title!r} duration too short"
        self.box = (int(x0), int(y0), int(w), int(h))
        self.t_in, self.t_out, self.pop = t_in, t_out, pop
        self.title, self.corner, self.point_to, self.tag = title, corner, point_to, tag
        self.accent = style.ORANGE if accent is None else accent

    def on(self, t):
        return ss(t, self.t_in, self.t_in + self.pop) * (1.0 - ss(t, self.t_out, self.t_out + self.pop))

    def local(self, t):
        return t - self.t_in

    def content_box(self):
        w, h = self.box[2:]
        return (POPUP_PAD, POPUP_TITLE_H + POPUP_PAD // 2, w - POPUP_PAD, h - POPUP_PAD)

    def draw(self, cv, t, body=None):
        u = self.on(t)
        if u <= 0.01:
            return None
        x0, y0, w, h = self.box
        sub = Canvas(w, h, t, _popup_bg(w, h), cv.glow)
        acc = self.accent
        sub.s.rect(0, 0, w, POPUP_TITLE_H, tuple(int(0.22 * c) for c in acc), 1.0, r=0)
        if body is not None:
            body(sub, t - self.t_in, self.content_box())
        sub.s.line((0, POPUP_TITLE_H), (w, POPUP_TITLE_H), acc, 0.6, 1)
        sub.rich(self.title, (POPUP_PAD, POPUP_TITLE_H / 2), 28, "bold", 1.0, anchor="lm", shadow=False)
        if self.tag:
            sub.badge((w - POPUP_PAD, POPUP_TITLE_H / 2 - 17), self.tag, 22, 1.0, anchor="ra")
        rgb = np.ascontiguousarray(sub.composite()).copy()
        from animkit import rrect_pts
        border = np.round(np.asarray(rrect_pts(1, 1, w - 2, h - 2, 15, 8))).astype(np.int32)
        cv2.polylines(rgb, [border], True, acc, 2, cv2.LINE_AA)
        # scale open: the anchor corner stays fixed
        sc_ = 0.6 + 0.4 * u
        sw, sh = max(2, int(round(w * sc_))), max(2, int(round(h * sc_)))
        small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(_popup_mask(w, h), (sw, sh), interpolation=cv2.INTER_AREA)
        px = x0 if self.corner[1] == "l" else x0 + w - sw
        py = y0 if self.corner[0] == "t" else y0 + h - sh
        # leader: point on the main frame -> nearest window corner
        if self.point_to is not None:
            corners = [(px, py), (px + sw, py), (px, py + sh), (px + sw, py + sh)]
            q = np.asarray(self.point_to, float)
            c = min(corners, key=lambda p: (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
            n = 14
            for k in range(n):
                if k % 2 == 0:
                    a = q + (np.asarray(c) - q) * k / n
                    b = q + (np.asarray(c) - q) * (k + 1) / n
                    cv.dr.line([tuple(a), tuple(b)], fill=tuple(acc) + (int(220 * u),), width=2)
            cv.dr.ellipse([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5], fill=tuple(acc) + (int(230 * u),))
        # shadow + window
        sh_a = (mask.astype(np.float32) * 0.55 * u).astype(np.uint8)
        cv.paste_rgba(np.dstack([np.zeros((sh, sw, 3), np.uint8), sh_a]), px + 6, py + 8)
        a = (mask.astype(np.float32) * u).astype(np.uint8)
        cv.paste_rgba(np.dstack([small, a]), px, py)
        return px, py, px + sw, py + sh


def popup_check(ck, pops, shot_dur):
    """For checks: every popup lies within the shot duration and no two popups appear at overlapping positions at the same time."""
    for p in pops:
        ck.true(f"popup {p.title!r} within shot duration", p.t_out + p.pop <= shot_dur + 1e-6, f"{p.t_in}–{p.t_out}")
    for i, p in enumerate(pops):
        for q in pops[i + 1:]:
            ov_t = p.t_in < q.t_out + q.pop and q.t_in < p.t_out + p.pop
            bx, by, bw, bh = p.box
            cx, cy, cw, chh = q.box
            ov_xy = bx < cx + cw and cx < bx + bw and by < cy + chh and cy < by + bh
            ck.true(f"popup {p.title!r} / {q.title!r} do not overlap while both visible", not (ov_t and ov_xy))


@lru_cache(maxsize=None)
def s01_s02_weight_shapes():
    """(Used by 01 / 02) JSON header of the local facebook/VGGT-1B safetensors: {parameter name: shape}; reads only the header, does not load weights.
    Returns None when this machine has no weight file (checks then skip the shape assertions and record that)."""
    import glob
    import struct
    ps = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/*/"
                                      "model.safetensors"))
    if not ps:
        return None
    with open(ps[0], "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {k: v["shape"] for k, v in hdr.items() if k != "__metadata__"}
