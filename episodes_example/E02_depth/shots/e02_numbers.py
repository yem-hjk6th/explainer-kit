"""Every number shown on screen in the E02 storyboard is computed and self-checked here.

The E01 toy scene `scene_e01.py` is imported read-only (never edited); a new camera (the imaginary stereo pair)
is computed separately with the same ray-intersection functions, leaving E01's files untouched. Coordinate
convention as in E01: OpenCV, world frame = camera 1.
Run: python -I -X utf8 e02_numbers.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E01_SHOTS = os.path.normpath(os.path.join(HERE, "..", "..", "E01_vggt_mapanything", "shots"))
sys.path.insert(0, E01_SHOTS)
import scene_e01 as sc  # noqa: E402

EX = sc.EX_CAM                      # camera 2 (0-based index 1)


def render_cam(Rs, cs, uv):
    """Intersect a scene-frame camera (world->camera rotation Rs, center cs) at pixel uv and return the z-depth.
    Same intersection as scene_e01.render; only the camera is given as arguments."""
    uv = np.atleast_2d(np.asarray(uv, float))
    dcam = np.stack([(uv[:, 0] - sc.CX) / sc.FX, (uv[:, 1] - sc.CY) / sc.FY, np.ones(len(uv))], -1)
    d = dcam @ Rs
    o = np.broadcast_to(cs, d.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(d[..., 1] > 1e-9, (sc.GROUND_Y - o[..., 1]) / d[..., 1], np.inf)
        Xg = o + np.where(np.isfinite(tg), tg, 0)[..., None] * d
    tile = (np.abs(Xg[..., 0] - sc.GROUND_X[0]) <= sc.GROUND_X[1]) & (np.abs(Xg[..., 2] - sc.GROUND_Z[0]) <= sc.GROUND_Z[1])
    tg = np.where((tg > 1e-6) & tile, tg, np.inf)
    allt = np.stack([tg, sc._hit_box(o, d, sc.BOX['lo'], sc.BOX['hi'])] +
                    [sc._hit_sphere(o, d, s['c'], s['r']) for s in sc.SPHERES], -1)
    return np.min(allt, -1)


def dlt_rays(pix, dirs):
    """DA3 Eq. 2: find a 3x3 matrix A with A p ∥ d (zero cross product), ||A||=1, via the smallest singular vector of an SVD."""
    rows = []
    for p, d in zip(pix, dirs):
        x, y, z = d
        rows.append(np.concatenate([np.zeros(3), -z * p, y * p]))
        rows.append(np.concatenate([z * p, np.zeros(3), -x * p]))
    _, _, Vt = np.linalg.svd(np.array(rows))
    return Vt[-1].reshape(3, 3)


def rq(M):
    """M = U Q, U upper triangular (positive diagonal), Q orthogonal."""
    P = np.flipud(np.eye(3))
    Q_, R_ = np.linalg.qr((P @ M).T)
    U = P @ R_.T @ P
    Q = P @ Q_.T
    S = np.diag(np.sign(np.diag(U)))
    return U @ S, S @ Q


def numbers():
    nb = sc.numbers()
    out = {}
    R, t, c = nb['R'], nb['t'], nb['c']
    uv, D = nb['uv'], nb['zdepth']
    f, W = sc.FX, sc.W
    kp = nb['Kinv_p']

    # ---- 02 depth / inverse depth / disparity ----
    rng = D * np.linalg.norm(kp)
    assert np.isclose(rng, nb['ray_depth'])
    rho = 1.0 / D
    B = 0.12                                    # imaginary stereo baseline (m), illustrative value
    Rs2, cs2 = sc._CS[EX]
    cs_r = cs2 + Rs2.T @ np.array([B, 0, 0])    # shift by B along camera 2's own x axis: a rectified parallel stereo pair
    Xs = sc.X_STAR_S
    Xc_r = Rs2 @ (Xs - cs_r)
    uv_r = (sc.K @ Xc_r)[:2] / Xc_r[2]
    disp = uv[0] - uv_r[0]
    assert np.isclose(disp, f * B / D, atol=1e-9) and np.isclose(uv_r[1], uv[1], atol=1e-9)
    zr = render_cam(Rs2, cs_r, uv_r[None])[0]
    assert np.isclose(zr, D, atol=1e-9), 'the white dot is occluded in the imaginary right camera'
    _, tz, _ = sc.full(EX)
    valid = np.isfinite(tz)
    out.update(D=D, range=rng, rho=rho, B=B, disp=disp, uv_right=uv_r,
               D_min=float(tz[valid].min()), D_max=float(tz[valid].max()),
               rho_min=float(1 / tz[valid].max()), rho_max=float(1 / tz[valid].min()))
    # δZ ≈ Z² δρ
    out['dZ'] = {Z: Z * Z * 0.01 for Z in (1.0, D, 10.0)}

    # ---- 03 scale ambiguity + least-squares alignment (affine: s, b) ----
    # 5 pixels: white dot on the red box, blue ball, yellow ball, near ground, far ground (picked on the camera-2 image)
    picks = {}
    rgb, tz2, X2 = sc.full(EX)
    picks['white dot (red box)'] = (uv[0], uv[1])
    for name, Xc_s in [('blue ball', sc.SPHERES[0]['c'] - np.array([0, 0, sc.SPHERES[0]['r']])),
                       ('yellow ball', sc.SPHERES[1]['c'] - np.array([0, 0, sc.SPHERES[1]['r']]))]:
        p, _ = sc.project(R, t, sc.to_world(Xc_s))
        picks[name] = (p[0], p[1])
    picks['near ground'] = (259.0, 500.0)
    picks['far ground'] = (60.0, 330.0)
    pu = np.array(list(picks.values()))
    _, Dk, _ = sc.render(EX, pu)
    assert np.all(np.isfinite(Dk))
    s0, b0 = 1.5, 1.2                       # illustrative: unknown scale and shift of a "relative-depth model"
    eps = np.array([0.010, -0.015, 0.005, 0.012, -0.008])   # fixed small noise (illustrative)
    Dt = (Dk - b0) / s0 + eps                # illustrative relative depth
    A = np.stack([Dt, np.ones_like(Dt)], 1)
    N_ = A.T @ A
    rhs = A.T @ Dk
    s_hat, b_hat = np.linalg.solve(N_, rhs)
    res = s_hat * Dt + b_hat - Dk
    assert np.allclose(np.linalg.lstsq(A, Dk, rcond=None)[0], [s_hat, b_hat])
    out.update(lsq_names=list(picks), lsq_D=Dk, lsq_Dt=Dt, lsq_N=N_, lsq_rhs=rhs,
               s_hat=s_hat, b_hat=b_hat, lsq_rms=float(np.sqrt(np.mean(res ** 2))))
    # scene + camera scaled ×2 together: pixels unchanged (asserted as A-18 in scene_e01), depth ×2
    out['scale2_D'] = 2 * D

    # ---- 04 Depth Pro multi-scale tiling (our 518 image resized to 1536) ----
    S1 = 1536
    k = S1 / W
    rs = lambda n: (uv + 0.5) * n / W - 0.5     # pixel-center mapping = torch bilinear resize with align_corners=False
    p1 = rs(S1)
    def tiles(side, n):
        stride = (side - 384) / (n - 1)
        starts = np.arange(n) * stride
        return stride, starts
    st1, s1 = tiles(1536, 5)
    st2, s2 = tiles(768, 3)                  # reading A: scale 2 = 768 (not stated in the paper; to be checked against the official code)
    assert st1 == 288 and st2 == 192
    def covering(x, starts):
        return [i for i, a in enumerate(starts) if a <= x < a + 384]
    def voronoi(x, starts):
        cen = starts + 192
        return int(np.argmin(np.abs(cen - x)))
    p2 = rs(768)
    p3 = rs(384)
    out.update(k1536=k, p1536=p1, p768=p2, p384=p3, stride1=st1, overlap1=384 - st1,
               stride2=st2, overlap2=384 - st2,
               cover1=(covering(p1[1], s1), covering(p1[0], s1)), keep1=(voronoi(p1[1], s1), voronoi(p1[0], s1)),
               cover2=(covering(p2[1], s2), covering(p2[0], s2)), keep2=(voronoi(p2[1], s2), voronoi(p2[0], s2)))
    r1, c1 = out['keep1']
    loc = p1 - np.array([s1[c1], s1[r1]])
    tok = (int(loc[1] // 16), int(loc[0] // 16))
    glob = (int(p1[1] // 16), int(p1[0] // 16))
    assert glob == (int(s1[r1] // 16) + tok[0], int(s1[c1] // 16) + tok[1])
    out.update(tok_in_patch=tok, tok_in_patch_idx=tok[0] * 24 + tok[1], tok_merged=glob,
               tok_merged_idx=glob[0] * 96 + glob[1], patch_idx1=r1 * 5 + c1)
    # Voronoi boundaries: midpoints between neighboring tile centers
    out['voronoi_lines1'] = [(s1[i] + s1[i + 1]) / 2 + 192 for i in range(4)]
    n_plain = (1536 // 16) ** 2
    out.update(n_tok_plain=n_plain, attn_plain=n_plain ** 2, attn_dp=36 * 576 ** 2,
               attn_ratio=n_plain ** 2 / (36 * 576 ** 2), tok_dp=36 * 576, tok_ratio=36 * 576 / n_plain)

    # ---- 05 canonical inverse depth C -> meters ----
    fw = f / W
    assert np.isclose(fw, 1 / (2 * np.tan(np.radians(sc.FOV_DEG) / 2)))
    C = fw / D
    px_1m = f * 1.0 / D
    assert np.isclose(px_1m / W, C)
    fw70 = 1 / (2 * np.tan(np.radians(70) / 2))
    D70 = fw70 / C
    out.update(fw=fw, C=C, px_1m=px_1m, fw70=fw70, D70=D70, D70_rel=D70 / D - 1,
               C_min=fw / out['D_max'], C_max=fw / out['D_min'])

    # ---- 06 derivative loss: C along one row of camera 2 (v = 297), 41 pixels around the largest object edge in the row (yellow ball right edge -> ground behind it) ----
    # Two illustrative predictions with the same MAE: A = shifted as a whole (sharp edge), B = Gaussian blur (soft edge).
    row = int(round(uv[1]))
    Crow = np.where(np.isfinite(tz2[row]), fw / tz2[row], 0.0)
    dC = np.abs(np.diff(Crow))
    inner = np.where((Crow[:-1] > 0) & (Crow[1:] > 0))[0]
    jump = int(inner[np.argmax(dC[inner])])            # largest object edge in the row (surface on both sides)
    win = slice(jump - 20, jump + 21)
    seg = Crow[win]
    g = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2); g /= g.sum()
    segB = np.convolve(np.pad(seg, 9, mode='edge'), g, mode='valid')
    maeB = float(np.mean(np.abs(segB - seg)))
    segA = seg + maeB                                   # shift chosen so its MAE equals that of B
    maeA = float(np.mean(np.abs(segA - seg)))
    gA = float(np.mean(np.abs(np.diff(segA) - np.diff(seg))))
    gB = float(np.mean(np.abs(np.diff(segB) - np.diff(seg))))
    assert np.isclose(maeA, maeB) and gA < 1e-12 < gB
    out.update(row=row, edge_u=jump, edge_C=(float(Crow[jump]), float(Crow[jump + 1])),
               edge_D=(float(tz2[row, jump]), float(tz2[row, jump + 1])),
               mae_A=maeA, mae_B=maeB, grad_A=gA, grad_B=gB)
    # boundary metric (Eq. 3): depth ratio of two adjacent pixels on the same edge; is it an occlusion contour at t = 15?
    out['edge_ratio'] = float(tz2[row, jump + 1] / tz2[row, jump])
    assert out['edge_ratio'] > 1.15

    # ---- 07 DA3 depth-ray: the same white dot ----
    R_c2w = R.T
    d = R_c2w @ kp                      # not normalized: z component in the camera frame = 1
    assert np.isclose((R @ d)[2], 1.0)
    P = c + D * d
    assert np.allclose(P, sc.X_STAR, atol=1e-9)
    out.update(da3_kp=kp, da3_d=d, da3_dnorm=float(np.linalg.norm(d)), da3_t=c, da3_P=P,
               da3_range=float(D * np.linalg.norm(d)), ma_unit=nb['ray'])
    # full ray map of camera 2 -> solve the camera with Eq. 1 / Eq. 2
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(W, dtype=float))
    pix = np.stack([uu, vv, np.ones_like(uu)], -1).reshape(-1, 3)
    dirs = pix @ np.linalg.inv(sc.K).T @ R_c2w.T            # per-pixel d = R_c2w K^-1 p
    origins = np.broadcast_to(c, dirs.shape)
    tc = origins.mean(0)
    sub = np.arange(0, W, 37)                                # 14×14 pixels for the DLT (sparse, illustrative)
    sel = (sub[:, None] * W + sub[None, :]).ravel()
    A_ = dlt_rays(pix[sel], dirs[sel])
    Ki_R = np.linalg.inv(A_)                                 # ∝ K R_c2w^T
    Kh, Rh = rq(Ki_R)
    if np.linalg.det(Rh) < 0:
        Rh = -Rh; Kh = -Kh
    Kh = Kh / Kh[2, 2]
    assert np.allclose(Kh, sc.K, atol=1e-6) and np.allclose(Rh, R, atol=1e-8)
    ang = np.degrees(np.arccos((np.trace(Rh) - 1) / 2))
    rs = np.random.default_rng(0)
    noisy = dirs[sel] * rs.uniform(0.5, 2.0, (len(sel), 1)) + rs.normal(0, 0.002, (len(sel), 3))
    Kn, Rn = rq(np.linalg.inv(dlt_rays(pix[sel], noisy)))
    if np.linalg.det(Rn) < 0:
        Rn = -Rn; Kn = -Kn
    Kn = Kn / Kn[2, 2]
    angn = np.degrees(np.arccos(np.clip((np.trace(Rn.T @ R) - 1) / 2, -1, 1)))
    out.update(raymap_shape=(W, W, 6), tc=tc, dlt_n=len(sel), K_hat=Kh, rot_deg=ang,
               K_noisy=Kn, rot_err_noisy_deg=angn)

    # ---- 08 DA3 tokens: 504×504, patch 14 ----
    S = 504
    uv504 = (uv + 0.5) * S / W - 0.5            # as above, pixel-center mapping
    pr, pc = int(uv504[1] // 14), int(uv504[0] // 14)
    n_p = (S // 14) ** 2
    T = 1 + n_p
    out.update(uv504=uv504, patch504=(pr, pc), patch504_idx=pr * 36 + pc, T=T,
               tok_view=1 + pr * 36 + pc, tok_cross=EX * T + 1 + pr * 36 + pc,
               attn_within=3 * T * T, attn_cross=(3 * T) ** 2)
    assert out['attn_cross'] == 3 * out['attn_within']

    # ---- 09 RANSAC scale-shift alignment (DA3 Eq. 8), 6 sparse points, 1 outlier ----
    sp = np.array([[80, 420], [200, 480], [uv[0], uv[1]], [330, 200], [420, 380], [150, 300]], float)
    _, Ds, _ = sc.render(EX, sp)
    assert np.all(np.isfinite(Ds))
    Dteach = (Ds - 1.2) / 1.5 + np.array([0.004, -0.006, 0.003, -0.002, 0.005, -0.004])  # illustrative teacher
    Dsparse = Ds.copy()
    Dsparse[4] = Ds[4] * 1.8                       # illustrative: one bad sparse depth point
    best = None
    for i in range(6):
        for j in range(i + 1, 6):
            Aij = np.array([[Dteach[i], 1], [Dteach[j], 1]])
            s_, b_ = np.linalg.solve(Aij, Dsparse[[i, j]])
            if s_ <= 0:
                continue
            r = s_ * Dteach + b_ - Dsparse
            med = np.median(r)
            thr = np.mean(np.abs(r - med))
            inl = np.abs(r - med) < thr
            if best is None or inl.sum() > best[0]:
                best = (int(inl.sum()), (i, j), s_, b_, thr, inl)
    inl = best[5]
    sA = np.stack([Dteach[inl], np.ones(inl.sum())], 1)
    s_r, b_r = np.linalg.lstsq(sA, Dsparse[inl], rcond=None)[0]
    assert not inl[4], 'RANSAC did not reject the outlier'
    out.update(ransac_D=Ds, ransac_sparse=Dsparse, ransac_teacher=Dteach, ransac_best=best[:5],
               ransac_s=s_r, ransac_b=b_r)
    out['s_norm'] = nb['s']
    out['n_valid'] = nb['n_valid']
    return out


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    for kk, vv in numbers().items():
        print(f'{kk}: {vv}')
