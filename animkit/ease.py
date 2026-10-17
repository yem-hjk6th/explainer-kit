"""Scalar timing curves. All take the current time t (s) and return a float in [0, 1]."""
import numpy as np


def ss(x, a, b):
    """smoothstep from a to b."""
    if b <= a:
        return float(x >= b)
    u = np.clip((x - a) / (b - a), 0, 1)
    return float(u * u * (3 - 2 * u))


def fade(t, a, b, fin=0.4, fout=0.4):
    """0 -> 1 over [a, a+fin], held, 1 -> 0 over [b-fout, b]."""
    return ss(t, a, a + fin) * (1 - ss(t, b - fout, b))


def lin(x, a, b):
    """linear ramp from a to b, clamped."""
    if b <= a:
        return float(x >= b)
    return float(np.clip((x - a) / (b - a), 0, 1))


def bump(t, t0, dur=0.6):
    """0 -> 1 -> 0 over [t0, t0+dur]; envelope for highlight pulses."""
    u = lin(t, t0, t0 + dur)
    return float(np.sin(np.pi * u) ** 2)


def lerp(a, b, u):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * u
