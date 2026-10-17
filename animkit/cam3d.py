"""3D: pinhole cameras in the scene and the viewer camera that films the scene.

Scene cameras use the OpenCV convention (x right, y down, z forward), extrinsics world -> camera: Xc = R X + t.
Two world conventions for the viewer:
  world="cv"   right-handed OpenCV world (E01: world = camera 1's frame, y down). Visual up = -y.
  world="yup"  epipolar_demo's world (x right, y up, z forward; left-handed). Visual up = +y.
"""
import numpy as np

from . import style
from .arch import arrow


def look_at(C, target, up=(0, 1, 0)):
    """epipolar_demo's look-at (y-up world): forward f, right r = up x f, up u = f x r."""
    f = np.asarray(target, float) - C
    f /= np.linalg.norm(f)
    r = np.cross(np.asarray(up, float), f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return f, r, u


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class PinholeCam:
    """Camera with intrinsics K and extrinsics R, t (world -> camera). The image plane is drawn plane_d in front of
    the centre (camera z = plane_d). Build it from a pose (C, target) like epipolar_demo, or from K, R, t."""

    def __init__(self, C, target, img_w=640, img_h=480, plane_d=1.0, plane_w=1.6, up=(0, 1, 0)):
        self.C = np.asarray(C, float)
        self.IMG_W, self.IMG_H, self.D = img_w, img_h, plane_d
        f, r, u = look_at(self.C, target, up)
        fx = img_w / plane_w * plane_d
        self._set(np.array([[fx, 0, img_w / 2], [0, fx, img_h / 2], [0, 0, 1]]), np.stack([r, -u, f]))
        self.t = -self.R @ self.C

    @classmethod
    def from_KRt(cls, K, R, t, img_w, img_h, plane_d=0.3):
        cam = cls.__new__(cls)
        cam.IMG_W, cam.IMG_H, cam.D = img_w, img_h, plane_d
        cam._set(np.asarray(K, float), np.asarray(R, float))
        cam.t = np.asarray(t, float)
        cam.C = -cam.R.T @ cam.t
        return cam

    def _set(self, K, R):
        self.K, self.R = K, R
        self.Kinv = np.linalg.inv(K)

    def pix(self, X):
        """Pixel of world point X and its camera-frame depth z."""
        Xc = self.R @ X + self.t
        p = self.K @ Xc
        return p[:2] / p[2], Xc[2]

    def pix_many(self, X):
        """(n,3) world -> (n,2) pixels, (n,) depths."""
        Xc = np.asarray(X, float) @ self.R.T + self.t
        p = Xc @ self.K.T
        return p[:, :2] / p[:, 2:3], Xc[:, 2]

    def plane_pt(self, px, py, depth=None):
        """World point at camera depth z = plane_d (or `depth`) on the ray through pixel (px, py)."""
        d = self.D if depth is None else depth
        return self.C + d * (self.R.T @ (self.Kinv @ np.array([px, py, 1.0])))

    def corners(self, depth=None):
        """Image-plane corners TL, TR, BR, BL in world coordinates."""
        return [self.plane_pt(0, 0, depth), self.plane_pt(self.IMG_W, 0, depth),
                self.plane_pt(self.IMG_W, self.IMG_H, depth), self.plane_pt(0, self.IMG_H, depth)]

    def hit_plane(self, X):
        """Where the segment X -> C crosses the drawn image plane: (world point, pixel, depth)."""
        p, z = self.pix(X)
        return self.plane_pt(*p), p, z

    def ray_dir(self, pix):
        d = self.plane_pt(*pix) - self.C
        return d / np.linalg.norm(d)

    def unproject(self, px, depth):
        """Pixels (n,2) with camera-frame z-depth (n,) -> world points (n,3): X = R^T (z K^-1 [u v 1]^T - t)."""
        px = np.atleast_2d(np.asarray(px, float))
        uv1 = np.c_[px, np.ones(len(px))]
        Xc = (uv1 @ self.Kinv.T) * np.asarray(depth, float).reshape(-1, 1)
        return (Xc - self.t) @ self.R


def relative_pose(cam1, cam2):
    """R, t taking cam1's camera frame to cam2's: X2 = R X1 + t."""
    R = cam2.R @ cam1.R.T
    return R, cam2.t - R @ cam1.t


def fundamental(cam1, cam2):
    """F with x2^T F x1 = 0: F = K2^-T [t]x R K1^-1."""
    R, t = relative_pose(cam1, cam2)
    return np.linalg.inv(cam2.K).T @ skew(t) @ R @ np.linalg.inv(cam1.K)


# ---------------------------------------------------------------- viewer
class ViewCam:
    """Projector for the viewer: P(X) maps (3,) -> (2,) or (n,3) -> (n,2) screen px; P.depth(X) = view depth.
    up: visual up direction in world coords. right_handed: True for the OpenCV world, False for epipolar_demo's."""

    def __init__(self, eye, target, up=(0, -1, 0), focal=1000.0, cx=style.W / 2, cy=760.0, right_handed=True):
        self.E = np.asarray(eye, float)
        f = np.asarray(target, float) - self.E
        f /= np.linalg.norm(f)
        up = np.asarray(up, float)
        if right_handed:
            r = np.cross(f, up)
            r /= np.linalg.norm(r)
            dn = np.cross(f, r)
        else:
            r = np.cross(up, f)
            r /= np.linalg.norm(r)
            dn = -np.cross(f, r)
        self.f, self.r, self.dn = f, r, dn
        self.focal, self.cx, self.cy = focal, cx, cy

    def depth(self, X):
        return (np.asarray(X, float) - self.E) @ self.f

    def __call__(self, X):
        d = np.asarray(X, float) - self.E
        z = d @ self.f
        return np.stack([self.cx + self.focal * (d @ self.r) / z, self.cy + self.focal * (d @ self.dn) / z], -1)

    def fit(self, pts, w=900, h=760, center=(style.W / 2, 760)):
        """Scale focal and shift (cx, cy) so pts fill a w x h box centred at center."""
        Q = self(np.asarray(pts, float))
        self.focal *= min(w / np.ptp(Q[:, 0]), h / np.ptp(Q[:, 1]))
        Q = self(np.asarray(pts, float))
        self.cx += center[0] - 0.5 * (Q.min(0)[0] + Q.max(0)[0])
        self.cy += center[1] - 0.5 * (Q.min(0)[1] + Q.max(0)[1])
        return self


class OrbitCam:
    """Viewer on a sphere of radius r around target; az / el in degrees; az = 0 looks along +z from behind.
    .at(az=..., el=...) returns a ViewCam for that pose (sharing focal / cx / cy); .fit() rescales them."""

    def __init__(self, target, r, az=0.0, el=30.0, focal=1000.0, cx=style.W / 2, cy=760.0, world="yup"):
        self.target = np.asarray(target, float)
        self.r, self.az, self.el, self.focal, self.cx, self.cy = r, az, el, focal, cx, cy
        self.world = world

    def eye(self, az=None, el=None, r=None):
        az = np.radians(self.az if az is None else az)
        el = np.radians(self.el if el is None else el)
        r = self.r if r is None else r
        upv = 1.0 if self.world == "yup" else -1.0
        return self.target + r * np.array([np.sin(az) * np.cos(el), upv * np.sin(el), -np.cos(az) * np.cos(el)])

    def at(self, az=None, el=None, r=None):
        yup = self.world == "yup"
        return ViewCam(self.eye(az, el, r), self.target, (0, 1, 0) if yup else (0, -1, 0), self.focal, self.cx,
                       self.cy, right_handed=not yup)

    def fit(self, pts, w=900, h=760, center=(style.W / 2, 760), **pose):
        v = self.at(**pose).fit(pts, w, h, center)
        self.focal, self.cx, self.cy = v.focal, v.cx, v.cy
        return self


# ---------------------------------------------------------------- drawing
def draw_camera(cv, cam, P, alpha=1.0, color=style.WHITE, fill=style.PANEL, fill_alpha=0.35, label=None,
                label_size=28, image=None, image_alpha=1.0, depth=None):
    """Frustum + translucent image plane + glowing centre (epipolar_demo). image: RGB array pasted on the plane
    (thumbnail of what the camera sees). depth: draw the plane at this camera depth instead of plane_d."""
    if alpha <= 0:
        return
    cs = [P(c) for c in cam.corners(depth)]
    if image is not None:
        cv.s.warp_quad(image, cs, image_alpha * alpha)
    else:
        cv.s.add_poly(cs, fill, fill_alpha * alpha)
    C = P(cam.C)
    for c in cs:
        cv.s.line(C, c, color, 0.5 * alpha, 1)
    for i in range(4):
        cv.s.line(cs[i], cs[(i + 1) % 4], color, 0.95 * alpha, 2)
    cv.g.dot(C, color, 6, 0.6 * alpha)
    cv.s.dot(C, color, 4, alpha)
    if label:
        cv.rich(label, C + [0, 16], label_size, alpha=alpha, anchor="ma")


def floor_grid(x=(-4, 5), z=(-1, 10), y=-1.3, step=1.0):
    """Segments of a floor grid at height y."""
    segs = []
    for xv in np.arange(x[0], x[1] + 0.01, step):
        segs.append((np.array([xv, y, z[0]]), np.array([xv, y, z[1]])))
    for zv in np.arange(z[0], z[1] + 0.01, step):
        segs.append((np.array([x[0], y, zv]), np.array([x[1], y, zv])))
    return segs


_disks = {}


def _disk(r):
    if r not in _disks:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        m = xx ** 2 + yy ** 2 <= r * r + r * 0.6
        _disks[r] = (yy[m], xx[m])
    return _disks[r]


def point_cloud(cv, P, X, colors, rad=2.0, alpha=1.0, ref_depth=None, max_rad=6, glow=0.0, near=1e-3):
    """Splat (n,3) points with (n,3) RGB colors (or one color) through a z-buffer (nearest point wins per pixel).
    rad: dot radius in px (0 = single pixel); ref_depth: if given, radius scales as rad * ref_depth / depth.
    Handles ~60k points per frame (vectorised)."""
    X = np.asarray(X, float).reshape(-1, 3)
    cols = np.broadcast_to(np.asarray(colors, float), X.shape).copy()
    ok = np.isfinite(X).all(1)
    X, cols = X[ok], cols[ok]
    z = P.depth(X)
    keep = z > near
    X, cols, z = X[keep], cols[keep], z[keep]
    if not len(X) or alpha <= 0:
        return
    q = P(X)
    r = np.full(len(X), rad, float) if ref_depth is None else rad * ref_depth / z
    r = np.clip(np.round(r), 0, max_rad).astype(int)
    for layer, a in ((cv.s, alpha), (cv.g, glow * alpha)):
        if a <= 0:
            continue
        img = layer.img
        H, W = img.shape[:2]
        pix, zz, ci = [], [], []
        for rr in np.unique(r):
            sel = np.nonzero(r == rr)[0]
            dy, dx = _disk(int(rr))
            xi = (np.round(q[sel, 0]).astype(int)[:, None] + dx[None]).ravel()
            yi = (np.round(q[sel, 1]).astype(int)[:, None] + dy[None]).ravel()
            idx = np.repeat(sel, len(dx))
            v = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            pix.append(yi[v] * W + xi[v])
            zz.append(z[idx[v]])
            ci.append(idx[v])
        pix, zz, ci = np.concatenate(pix), np.concatenate(zz), np.concatenate(ci)
        order = np.lexsort((zz, pix))
        _, first = np.unique(pix[order], return_index=True)
        win = order[first]
        flat = img.reshape(-1, 3)
        p, c = pix[win], cols[ci[win]]
        if a >= 1:
            flat[p] = np.clip(c + 0.5, 0, 255).astype(np.uint8)
        else:
            flat[p] = np.clip(flat[p] * (1 - a) + c * a + 0.5, 0, 255).astype(np.uint8)


def marker3d(cv, P, X, label=None, color=style.WHITE, rad=6, alpha=1.0, label_offset=(16, -16), label_size=28,
             label_color=style.TEXT):
    """Glowing dot at world point X with a label that follows it on screen. Returns the screen point."""
    p = P(X)
    if alpha > 0.01:
        cv.g.dot(p, color, rad * 1.6, alpha)
        cv.s.dot(p, (255, 255, 255), rad * 0.6, alpha)
        if label:
            cv.rich(label, p + np.asarray(label_offset, float), label_size, alpha=alpha, fill=label_color,
                    anchor="lb")
    return p


def ray_fan(cv, P, C, dirs, length, color=style.GOLD, alpha=0.6, th=1, progress=1.0, glow=0.0):
    """Lines from centre C along unit directions dirs (..., 3) (e.g. a 12x12 ray map), length in world units."""
    if alpha <= 0 or progress <= 0:
        return
    D = np.asarray(dirs, float).reshape(-1, 3)
    c = P(C)
    ends = P(np.asarray(C, float) + D * length * progress)
    for e in ends:
        cv.s.line(c, e, color, alpha, th)
        if glow:
            cv.g.line(c, e, color, glow * alpha, th + 2)


def camera_axes(cv, P, cam, length=0.3, **kw):
    """A scene camera's own x / y / z axes (OpenCV: right, down, forward) at its centre. With world -> camera R,
    the camera axes in world coordinates are the ROWS of R, i.e. the columns of R^T."""
    return axes3d(cv, P, cam.C, cam.R.T, length, **kw)


def axes3d(cv, P, origin, R=np.eye(3), length=0.3, labels=("$x$", "$y$", "$z$"), colors=style.AXIS, alpha=1.0, th=3,
           label_size=30, progress=1.0):
    """Coordinate triad: COLUMNS of R are the axis directions (world coords), drawn as arrows from origin.
    For a scene camera use camera_axes() (it passes R^T)."""
    if alpha <= 0:
        return
    o = P(origin)
    for k in range(3):
        e = P(np.asarray(origin, float) + np.asarray(R, float)[:, k] * length)
        arrow(cv, o, e, colors[k], alpha, th, head=14, progress=progress)
        if labels and labels[k] and progress >= 1:
            d = (e - o) / max(np.linalg.norm(e - o), 1e-9)
            cv.rich(labels[k], e + d * 18, label_size, alpha=alpha, fill=colors[k], anchor="mm")
