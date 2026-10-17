"""E01 toy scene used throughout the episode: ground + red box + blue ball + yellow ball, 3 cameras, 518x518.

Coordinate conventions match the VGGT code (OpenCV: x right, y down, z forward; extrinsics world->camera,
world frame = camera 1's camera frame; pixel coordinates on an integer grid, principal point W/2, H/2). Units: meters.
The scene is first laid out in a "scene frame" (ground y = const, box axis-aligned, cameras looking down),
and the cameras and 3D points exposed to the outside are all converted to the world frame (camera 1).
All shots import from here; every number is computed here, none is typed in by hand.
"""
from functools import lru_cache

import numpy as np

H = W = 518
PATCH = 14
FOV_DEG = 60.0                       # same in width and height
FX = FY = (W / 2) / np.tan(np.radians(FOV_DEG) / 2)
CX, CY = W / 2, H / 2
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])

# ---- Scene frame ----
GROUND_Y = 0.55
GROUND_X = (0.0, 1.4)                # a finite patch of ground: center, half-width
GROUND_Z = (2.6, 1.2)
PIVOT = np.array([0.0, 0.25, 2.6])   # cameras orbit it and look at it
DIST = 2.4
PITCH_DEG = 15.0                     # looking down
YAWS_DEG = [0.0, 30.0, -30.0]        # cameras 1/2/3

BOX = dict(lo=np.array([-0.55, 0.05, 2.25]), hi=np.array([-0.05, 0.55, 2.75]), rgb=(205, 65, 55))
SPHERES = [dict(c=np.array([0.42, 0.20, 2.95]), r=0.35, rgb=(70, 120, 220)),
           dict(c=np.array([0.10, 0.40, 2.10]), r=0.15, rgb=(230, 190, 55))]
LIGHT = np.array([0.35, 0.8, 0.5]); LIGHT /= np.linalg.norm(LIGHT)   # direction of light propagation
X_STAR_S = np.array([-0.30, 0.30, 2.25])   # the point used throughout the episode: on the front face of the red box (scene frame)
EX_CAM = 1                                 # the example uses camera 2 (0-based index = 1)


def rot_x(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _cams_scene():
    out = []
    for yaw in YAWS_DEG:
        Rc2w = rot_y(yaw) @ rot_x(-PITCH_DEG)
        c = PIVOT + Rc2w @ np.array([0, 0, -DIST])
        out.append((Rc2w.T, c))          # (world->camera rotation, center), both in the scene frame
    return out


_CS = _cams_scene()
_R1S, _C1S = _CS[0]


def to_world(Xs):
    """Scene frame -> world frame (camera 1's camera frame)."""
    return (np.asarray(Xs) - _C1S) @ _R1S.T


def cameras():
    """[(R, t, c)] in the world frame: R, t are world->camera (the VGGT extrinsics), c is the camera center."""
    out = []
    for Rs, cs in _CS:
        R = Rs @ _R1S.T
        c = to_world(cs)
        out.append((R, -R @ c, c))
    return out


X_STAR = to_world(X_STAR_S)


def mat_to_quat_xyzw(R):
    """Scalar last (x,y,z,w), same as VGGT rotation.py; only handles trace>0 (enough for our test scene)."""
    tr = np.trace(R)
    assert tr > 0
    w = np.sqrt(1 + tr) / 2
    return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w), w])


def project(R, t, X):
    Xc = R @ X + t
    uvw = K @ Xc
    return uvw[:2] / uvw[2], Xc


def _hit_sphere(o, d, c, r):
    oc = o - c
    a = np.einsum('...i,...i', d, d)
    b = np.einsum('...i,...i', d, oc)
    cc = np.einsum('...i,...i', oc, oc) - r * r
    disc = b * b - a * cc
    tt = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
    return np.where(tt > 1e-6, tt, np.inf)


def _hit_box(o, d, lo, hi):
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / d
        t0 = (lo - o) * inv
        t1 = (hi - o) * inv
    tmin = np.nanmax(np.minimum(t0, t1), axis=-1)
    tmax = np.nanmin(np.maximum(t0, t1), axis=-1)
    return np.where((tmax >= tmin) & (tmin > 1e-6), tmin, np.inf)


def render(cam_i, uv=None):
    """Ray-cast camera cam_i (0-based). uv=None renders the whole HxW image, otherwise only the (n,2) pixels.
    Returns rgb (uint8 or None), z-depth (inf on a miss), world-frame 3D points (nan on a miss)."""
    Rs, cs = _CS[cam_i]
    if uv is None:
        u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    else:
        uv = np.atleast_2d(np.asarray(uv, float)); u, v = uv[:, 0], uv[:, 1]
    dcam = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], -1)   # z component = 1
    d = dcam @ Rs                         # scene-frame direction = Rs^T dcam
    o = np.broadcast_to(cs, d.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(d[..., 1] > 1e-9, (GROUND_Y - o[..., 1]) / d[..., 1], np.inf)
        Xg = o + np.where(np.isfinite(tg), tg, 0)[..., None] * d
    tile = (np.abs(Xg[..., 0] - GROUND_X[0]) <= GROUND_X[1]) & (np.abs(Xg[..., 2] - GROUND_Z[0]) <= GROUND_Z[1])
    tg = np.where((tg > 1e-6) & tile, tg, np.inf)
    allt = np.stack([tg, _hit_box(o, d, BOX['lo'], BOX['hi'])] +
                    [_hit_sphere(o, d, s['c'], s['r']) for s in SPHERES], -1)
    idx = np.argmin(allt, -1)
    tz = np.min(allt, -1)                 # dcam has z=1, so the parameter t is the z-depth
    hit = np.isfinite(tz)
    Xs = o + np.where(hit, tz, np.nan)[..., None] * d
    Xw = to_world(Xs)
    if uv is not None:
        return None, tz, Xw
    n = np.zeros_like(Xs)
    n[idx == 0] = [0, -1, 0]
    bx = (idx == 1) & hit
    Xb = Xs[bx]; nb = np.zeros_like(Xb)
    for ax in range(3):
        nb[np.abs(Xb[:, ax] - BOX['lo'][ax]) < 1e-4, ax] = -1
        nb[np.abs(Xb[:, ax] - BOX['hi'][ax]) < 1e-4, ax] = 1
    n[bx] = nb
    for k, s in enumerate(SPHERES):
        m = (idx == 2 + k) & hit
        n[m] = (Xs[m] - s['c']) / s['r']
    lam = np.clip(-(n @ LIGHT), 0, 1) * 0.7 + 0.3
    base = np.zeros_like(Xs)
    with np.errstate(invalid='ignore'):
        chk = (np.floor(Xs[..., 0] / 0.25) + np.floor(Xs[..., 2] / 0.25)) % 2 == 0
    g = (idx == 0) & hit
    base[g] = np.where(chk[g][:, None], [165, 165, 160], [105, 105, 102])
    base[bx] = BOX['rgb']
    for k, s in enumerate(SPHERES):
        base[(idx == 2 + k) & hit] = s['rgb']
    rgb = np.zeros(d.shape, float) + np.array([28, 30, 38])
    rgb[hit] = base[hit] * lam[hit][:, None]
    return np.clip(rgb, 0, 255).astype(np.uint8), tz, Xw


@lru_cache(maxsize=None)
def full(cam_i):
    """(rgb, z-depth, world points) of the whole image, computed only once per process."""
    return render(cam_i)


def unproject(R, t, uv, zdepth):
    """VGGT form: P = R^T (D K^{-1}[u,v,1]^T - t)."""
    return R.T @ (zdepth * np.linalg.solve(K, [uv[0], uv[1], 1.0]) - t)


def numbers():
    """All numbers that appear on screen in the storyboard are computed here and self-checked."""
    cams = cameras()
    R, t, c = cams[EX_CAM]
    assert np.allclose(cams[0][0], np.eye(3)) and np.allclose(cams[0][1], 0)   # camera 1 = world frame
    uv, Xc = project(R, t, X_STAR)
    _, zd, Xhit = render(EX_CAM, uv[None])
    assert np.allclose(Xhit[0], X_STAR, atol=1e-9), 'the point used throughout is occluded in camera 2'
    P = unproject(R, t, uv, zd[0])
    assert np.allclose(P, X_STAR, atol=1e-9)
    gh, gw = H // PATCH, W // PATCH
    col, row = int(uv[0] // PATCH), int(uv[1] // PATCH)
    pidx = row * gw + col
    Kp = gh * gw
    Kt = Kp + 5
    ray = Xc / np.linalg.norm(Xc)
    O, T = R.T, c
    assert np.allclose(O @ (ray * np.linalg.norm(Xc)) + T, X_STAR, atol=1e-9)  # MapAnything form
    out = dict(grid=(gh, gw), K_patch=Kp, K_tok=Kt, uv=uv, patch_rc=(row, col), patch_idx=pidx,
               tok_idx_code=5 + pidx, tok_idx_global=EX_CAM * Kt + 5 + pidx,
               R=R, t=t, c=c, q_xyzw=mat_to_quat_xyzw(R), fx=FX,
               Kinv_p=np.linalg.solve(K, [uv[0], uv[1], 1.0]), zdepth=zd[0], Xcam=Xc,
               X_star=X_STAR, ray=ray, ray_depth=np.linalg.norm(Xc), O=O, T=T)
    vis = []
    for i, (Ri, ti, ci) in enumerate(cams):
        uvi, _ = project(Ri, ti, X_STAR)
        _, _, Xi = render(i, uvi[None])
        vis.append((uvi, bool(np.allclose(Xi[0], X_STAR, atol=1e-9))))
    out['vis'] = vis
    # A-18: scaling scene points and camera centers together by k>0 leaves pixels unchanged (07 #21 "the three images are identical")
    for k in (0.5, 2.0, 1.0 / 2.1525):
        for (Ri, ti, ci) in cams:
            uv0, _ = project(Ri, ti, X_STAR)
            uvk, _ = project(Ri, -Ri @ (k * ci), k * X_STAR)
            assert np.allclose(uv0, uvk, atol=1e-9), 'A-18 scale invariance does not hold'
    norms = []
    for i in range(3):
        _, tz, X = full(i)
        norms.append(np.linalg.norm(X[np.isfinite(tz)], axis=-1))
    out['s'] = float(np.concatenate(norms).mean())        # mean distance used by VGGT / MapAnything
    out['n_valid'] = int(sum(len(n) for n in norms))
    out['X_star_norm'] = X_STAR / out['s']
    out['T_norm'] = T / out['s']
    out['ray_depth_norm'] = out['ray_depth'] / out['s']
    return out


if __name__ == '__main__':
    import sys
    np.set_printoptions(precision=4, suppress=True)
    nb = numbers()
    for k, v in nb.items():
        print(f'{k}: {v}')
    if len(sys.argv) > 1:            # preview: 3 images + 3 z-depth maps
        from PIL import Image
        tiles = []
        for i, (uvi, _) in enumerate(nb['vis']):
            rgb, tz, _ = full(i)
            rgb = rgb.copy()
            u, v = np.round(uvi).astype(int)
            rgb[max(v - 4, 0):v + 5, max(u - 4, 0):u + 5] = (255, 255, 255)
            dz = np.where(np.isfinite(tz), tz, np.nan)
            lo, hi = np.nanmin(dz), np.nanmax(dz)
            g = np.where(np.isfinite(dz), 255 * (1 - (dz - lo) / (hi - lo)), 0).astype(np.uint8)
            tiles.append(np.concatenate([rgb, np.repeat(g[..., None], 3, -1)], 0))
        Image.fromarray(np.concatenate(tiles, 1)).save(sys.argv[1])
