"""Smoke test of plot2d, three 6 s sections:
A  a curve growing left to right (delta D = D^2 * delta rho) + one marked point
B  scatter points appearing one by one, a hypothesis line morphing into the least-squares line y = s x + t, residual bars
C  three curves on one axis (truth / prediction A / prediction B) + sparse stems with one bad point in red
All numbers here are made up for the test ("example values")."""
import numpy as np

from animkit import CYAN, GOLD, GREEN, GREY, ORANGE, RED, Plot2D, Shot, fade, number, ss

DUR = 18.0
shot = Shot("plot2d_demo", DUR)
shot.tag("example values")
for a, b, s in [(0.3, 6.0, "The curve is drawn left to right"), (6.0, 12.0, "Points appear one by one; the line fits"),
                (12.0, 18.0, "Several curves and sparse points")]:
    shot.caption(a, b, s)

BOX = (150, 330, 960, 900)

# A: delta D = D^2 * delta rho
D = np.linspace(0.5, 5.0, 91)
DRHO = 0.01
dD = D ** 2 * DRHO
PA = Plot2D(BOX, (0, 5), (0, 0.3), xlabel="$D$ (m)", ylabel="$\\delta D = D^{2}\\,\\delta\\rho$ (m)", fmt_y="{:.2f}")

# B: y = 0.8 x + 0.5 + noise, least-squares line
rng = np.random.default_rng(3)
xb = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
yb = 0.8 * xb + 0.5 + rng.normal(0, 0.35, len(xb))
S_LS, T_LS = Plot2D.fit_line(xb, yb)
S0, T0 = 0.2, 3.0                                     # a bad first guess
PB = Plot2D(BOX, (0, 7), (0, 7), xlabel="$x$ (example)", ylabel="$y = s\\,x + t$")

# C: three profiles
xc = np.linspace(0, 10, 120)
truth = 1.0 + 0.6 * np.sin(xc * 0.9)
pa = truth + 0.05 * np.sin(5 * xc)
pb = truth + 0.35 * np.exp(-((xc - 6) / 1.2) ** 2)
PC = Plot2D(BOX, (0, 10), (0, 2), xlabel="pixel column $u$", ylabel="$C$ (example)")
xs_sp = np.array([1.0, 2.5, 4.0, 5.5, 7.0, 8.5])
ys_sp = np.interp(xs_sp, xc, truth)
ys_sp[3] += 0.7                                       # one bad sparse point


@shot.check
def check(ck):
    ck.close("A: curve = D^2 * delta rho", dD[-1], 25.0 * DRHO, 1e-12)
    s, t = np.polyfit(xb, yb, 1)
    ck.close("B: fit_line slope = np.polyfit", S_LS, s, 1e-10)
    ck.close("B: fit_line intercept = np.polyfit", T_LS, t, 1e-10)
    r = yb - (S_LS * xb + T_LS)
    ck.close("B: residuals sum to 0", r.sum(), 0.0, 1e-10)
    ck.close("B: residuals orthogonal to x", float(r @ xb), 0.0, 1e-9)
    p = PB.to_screen([0, 7], [0, 7])
    ck.close("to_screen corners", p.ravel(), [BOX[0], BOX[3], BOX[2], BOX[1]], 1e-9)
    ck.true("nice_ticks 4-6", 4 <= len(PA.xticks) <= 7 and 4 <= len(PA.yticks) <= 7, (PA.xticks, PA.yticks))
    ck.record(slope=round(S_LS, 4), intercept=round(T_LS, 4))


@shot.frame
def frame(cv, t):
    a = fade(t, 0.1, 6.1)
    if a > 0:
        PA.axes(cv, a)
        tip = PA.curve(cv, D, dD, CYAN, ss(t, 0.8, 4.0), a)
        k = ss(t, 4.0, 4.6)
        if k > 0:
            p = PA.scatter(cv, [4.0], [16 * DRHO], GOLD, 1.0, a * k, labels=["$D=4$ m: $0.16$ m"])
            cv.s.dashed(p[0], PA.to_screen(4.0, 0), GOLD, 0.6 * a * k, 1, n=14)
    a = fade(t, 6.0, 12.1)
    if a > 0:
        PB.axes(cv, a)
        u = ss(t, 8.5, 10.5)
        s, tt = Plot2D.morph((S0, T0), (S_LS, T_LS), u)
        PB.line(cv, s, tt, ORANGE if u < 1 else GREEN, ss(t, 7.4, 8.2), a)
        PB.scatter(cv, xb, yb, CYAN, ss(t, 6.3, 7.6), a, labels=[f"$p_{k + 1}$" for k in range(len(xb))])
        PB.residuals(cv, xb, yb, s, tt, RED, ss(t, 10.3, 11.5), a)
        number(cv, (500, 1040), s, "{:.3f}", 38, GOLD, a, "rm")
        cv.rich("$s$ =", (370, 1040), 34, alpha=a, anchor="rm")
        number(cv, (800, 1040), tt, "{:.3f}", 38, GOLD, a, "rm")
        cv.rich("$t$ =", (670, 1040), 34, alpha=a, anchor="rm")
    a = fade(t, 12.0, 18.0, 0.3, 0.0)
    if a > 0:
        PC.axes(cv, a)
        PC.curve(cv, xc, truth, GREY, ss(t, 12.4, 14.0), a, th=3, dashed=True)
        PC.curve(cv, xc, pa, CYAN, ss(t, 13.0, 14.6), a)
        PC.curve(cv, xc, pb, ORANGE, ss(t, 13.6, 15.2), a)
        cv.rich("GT", (330, 1040), 26, alpha=a, fill=GREY)
        cv.rich("prediction A", (480, 1040), 26, alpha=a, fill=CYAN)
        cv.rich("prediction B", (650, 1040), 26, alpha=a, fill=ORANGE)
        cols = [GREEN] * len(xs_sp)
        cols[3] = RED
        PC.stem(cv, xs_sp, ys_sp, GREY, ss(t, 15.4, 17.0), 0.4 * a, th=2, rad=3)
        PC.scatter(cv, xs_sp, ys_sp, cols, ss(t, 15.4, 17.0), a, rad=8)
