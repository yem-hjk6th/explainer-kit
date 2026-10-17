"""Smoke test of ImagePanel.tiles, four 5 s sections on a 1536 x 1152 synthetic image:
A  tiles (384 px, stride 288) drop in one by one, neighbours staggered
B  one tile highlighted + the overlap band filled
C  Voronoi borders midway between tile centres, cells tinted by nearest centre
D  one Voronoi cell kept, the rest dimmed
The image is a synthetic gradient with a few shapes (schematic)."""
import cv2
import numpy as np

from animkit import GOLD, ImagePanel, Shot, fade, ss, tile_starts

DUR = 20.0
shot = Shot("tiles_demo", DUR)
shot.tag("schematic")
for a, b, s in [(0.3, 5.0, "Overlapping tiles drop in one by one"), (5.0, 10.0, "One tile highlighted, overlap filled"),
                (10.0, 15.0, "Each pixel goes to the nearest tile center"), (15.0, 20.0, "Keep only this tile's cell")]:
    shot.caption(a, b, s)

IH, IW, SIZE, STRIDE = 1152, 1536, 384, 288
img = np.zeros((IH, IW, 3), np.uint8)
yy, xx = np.mgrid[0:IH, 0:IW]
img[..., 0] = 40 + 90 * xx / IW
img[..., 1] = 50 + 80 * yy / IH
img[..., 2] = 110
cv2.circle(img, (400, 400), 180, (230, 200, 90), -1)
cv2.rectangle(img, (900, 600), (1400, 1000), (90, 210, 220), -1)
XS, YS = tile_starts(IW, SIZE, STRIDE), tile_starts(IH, SIZE, STRIDE)
panel = ImagePanel(img, (90, 330), w=900)
HL = (1, 2)


@shot.check
def check(ck):
    ck.eq("x starts", XS, [0, 288, 576, 864, 1152])
    ck.eq("y starts", YS, [0, 288, 576, 768])
    ck.eq("last tile ends at the image edge", (XS[-1] + SIZE, YS[-1] + SIZE), (IW, IH))
    o = panel.tiles(None, XS, YS, SIZE, alpha=0.0)
    cx = [x + SIZE / 2 for x in XS]
    cy = [y + SIZE / 2 for y in YS]
    ok = all(o["owner"](cx[c], cy[r]) == (r, c) for r in range(len(YS)) for c in range(len(XS)))
    ck.true("every tile centre belongs to its own Voronoi cell", ok, "")
    ck.close("Voronoi cells cover the image", float(np.sum(np.diff(o["vor_x"])) * np.sum(np.diff(o["vor_y"]))),
             IW * IH, 1e-6)
    # brute force nearest-centre assignment == searchsorted assignment
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [IW, IH], (400, 2))
    C = np.array([(x, y) for y in cy for x in cx])
    near = np.argmin(((pts[:, None] - C[None]) ** 2).sum(-1), 1)
    mine = np.array([r * len(XS) + c for r, c in (o["owner"](u, v) for u, v in pts)])
    ck.true("owner() == brute-force nearest centre (400 random pixels)", bool((near == mine).all()), "")
    ck.record(tiles=len(XS) * len(YS))


@shot.frame
def frame(cv, t):
    a = fade(t, 0.1, 20.0, 0.3, 0.0)
    panel.draw(cv, a, title="1536 × 1152 image (schematic)")
    ov = ss(t, 6.0, 7.0) * (1 - ss(t, 9.0, 9.8))
    vor = ss(t, 10.5, 11.5)
    fill = ss(t, 12.0, 13.0) * (1 - ss(t, 14.5, 15.0)) * 0.45
    keep = ss(t, 15.5, 16.5)
    panel.tiles(cv, XS, YS, SIZE, progress=ss(t, 0.6, 3.6), alpha=a,
                highlight={HL: GOLD} if 5.2 < t < 9.8 else None,
                overlap_color=GOLD, overlap_alpha=0.35 * ov,
                voronoi=vor * (1 - keep * 0), voronoi_fill=True, voronoi_fill_alpha=fill,
                keep=HL if t > 15 else None, keep_alpha=keep)
