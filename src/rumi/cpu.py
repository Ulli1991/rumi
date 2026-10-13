"""Native Newtonian LogH tasks; ctypes releases the GIL during each native integration."""

import ctypes
from functools import lru_cache
import math
import os
from pathlib import Path
import sys

import numpy as np


@lru_cache(maxsize=4)
def _library(path):
    library = ctypes.CDLL(str(path))
    ptr = ctypes.POINTER(ctypes.c_double)
    library.rumi_logh.argtypes = [ctypes.c_int, ptr, ptr, ptr, ctypes.c_double, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_int, ptr, ptr,
                                  ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int]
    library.rumi_logh.restype = ctypes.c_int
    return library


def integrate_logh(position, velocity, mass, dt, *, G=1.0, tolerance=1e-11,
                   max_steps=100000, library=None):
    """Integrate 2..8 unsoftened bodies in a normalized COM frame; fail on incomplete evolution."""
    x, v, m = (np.ascontiguousarray(a, dtype=np.float64) for a in (position, velocity, mass))
    n = len(m)
    if n < 2 or n > 8 or x.shape != (n, 3) or v.shape != x.shape:
        raise ValueError("LogH needs 2..8 bodies with positions and velocities of shape (N,3)")
    if not all(np.isfinite(a).all() for a in (x, v, m)) or np.any(m <= 0):
        raise ValueError("LogH requires finite state and positive masses")
    if not all(math.isfinite(a) and a > 0 for a in (dt, G, tolerance)) or max_steps < 1:
        raise ValueError("dt, G and tolerance must be finite and positive; max_steps >= 1")
    total = float(m.sum())
    center, speed = np.sum(m[:, None] * x, axis=0)/total, np.sum(m[:, None] * v, axis=0)/total
    relative = x-center
    scale = float(np.max(np.linalg.norm(relative, axis=1)))
    if not scale > 0:
        raise ValueError("coincident subsystem")
    velocity_scale = np.sqrt(G*total/scale)
    time_scale = scale/velocity_scale
    xn, vn, mn = (np.ascontiguousarray(a) for a in (relative/scale, (v-speed)/velocity_scale, m/total))
    ox, ov = np.empty_like(xn), np.empty_like(vn)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    default = Path(__file__).resolve().parents[2] / "build" / ("librumi_cpu" + suffix)
    path = Path(library or os.environ.get("RUMI_CPU_LIBRARY", default))
    if not path.exists():
        raise RuntimeError("CPU library missing; run python scripts/build_cpu.py or set RUMI_CPU_LIBRARY")
    ptr = ctypes.POINTER(ctypes.c_double)
    arrays = [a.ctypes.data_as(ptr) for a in (xn, vn, mn)]
    steps, message = ctypes.c_int(), ctypes.create_string_buffer(256)
    status = _library(str(path)).rumi_logh(n, *arrays, dt/time_scale, 1.0, tolerance, max_steps,
                       ox.ctypes.data_as(ptr), ov.ctypes.data_as(ptr), ctypes.byref(steps), message, len(message))
    if status:
        raise RuntimeError(message.value.decode())
    result_x = ox*scale + center + dt*speed
    result_v = ov*velocity_scale + speed
    if not np.isfinite(result_x).all() or not np.isfinite(result_v).all():
        raise RuntimeError("native LogH returned a non-finite state")
    return result_x, result_v
