"""Fourth-order forward symplectic force-gradient integration."""

import math

import jax

from .forces import make_force_functions


def make_fsi_step(**force_options):
    """Compiled fixed-step Newtonian FSI for field particles (no subsystem integration)."""
    acceleration, gradient = make_force_functions(**force_options)

    def step(state, dt):
        x, v, m, ids, sid = state
        v = v + (dt / 6) * acceleration(x, m, sid)
        x = x + (dt / 2) * v
        a = acceleration(x, m, sid)
        a = a + (dt * dt / 24) * gradient(x, m, sid, a)
        v = v + (2 * dt / 3) * a
        x = x + (dt / 2) * v
        v = v + (dt / 6) * acceleration(x, m, sid)
        return state._replace(position=x, velocity=v)

    compiled = jax.jit(step)

    def checked(state, dt):
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        # Fixed membership is host metadata: checking once per call is acceptable for this baseline.
        import numpy as np
        if np.any(np.asarray(state.subsystem) >= 0):
            raise ValueError("use HybridFSI for regularized subsystems")
        return compiled(state, dt)

    checked.compiled = compiled
    return checked
