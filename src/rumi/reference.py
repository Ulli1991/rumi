"""Independent NumPy pair loops for small-case validation and exact diagnostics."""

import numpy as np


def acceleration(position, mass, *, G=1.0, softening=0.0, subsystem=None):
    x, m = np.asarray(position), np.asarray(mass)
    a = np.zeros_like(x, dtype=np.float64)
    sid = np.full(len(m), -1) if subsystem is None else np.asarray(subsystem)
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            if sid[i] >= 0 and sid[i] == sid[j]:
                continue
            r = x[j] - x[i]
            r2 = float(r @ r) + softening**2
            if r2 <= 0:
                raise ValueError("coincident unsoftened particles")
            f = G * r / r2**1.5
            a[i] += m[j] * f
            a[j] -= m[i] * f
    return a


def gradient(position, mass, accel, *, G=1.0, softening=0.0, subsystem=None):
    """Force-gradient term without its dt^2/24 coefficient."""
    x, m, a = np.asarray(position), np.asarray(mass), np.asarray(accel)
    q = np.zeros_like(x, dtype=np.float64)
    sid = np.full(len(m), -1) if subsystem is None else np.asarray(subsystem)
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            if sid[i] >= 0 and sid[i] == sid[j]:
                continue
            r, da = x[j] - x[i], a[j] - a[i]
            r2 = float(r @ r) + softening**2
            f = G * (r2 * da - 3 * float(r @ da) * r) / r2**2.5
            q[i] += m[j] * f
            q[j] -= m[i] * f
    return q


def diagnostics(state, *, G=1.0, softening=0.0):
    """Physical totals including subsystem internal potentials; O(N^2), for validation only."""
    x, v, m = (np.asarray(a) for a in state[:3])
    energy = float(np.sum(0.5 * m[:, None] * v * v))
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            energy -= G * m[i] * m[j] / np.sqrt(np.sum((x[j] - x[i])**2) + softening**2)
    return {"energy": energy, "momentum": np.sum(m[:, None] * v, axis=0),
            "angular_momentum": np.sum(m[:, None] * np.cross(x, v), axis=0)}
