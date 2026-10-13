"""Particle state with stable IDs and explicit subsystem membership."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class State(NamedTuple):
    position: object       # (N, 3), FP64
    velocity: object       # (N, 3), FP64
    mass: object           # (N,), FP64
    ids: object            # (N,), unique integer IDs
    subsystem: object      # (N,), -1 for field, nonnegative fixed group ID

    @classmethod
    def create(cls, position, velocity, mass, *, ids=None, subsystem=None):
        """Validate host data before transferring it to JAX."""
        position = np.asarray(position, dtype=np.float64)
        n = len(position)
        state = cls(position, np.asarray(velocity, dtype=np.float64),
                    np.asarray(mass, dtype=np.float64),
                    np.arange(n, dtype=np.int64) if ids is None else np.asarray(ids),
                    np.full(n, -1, dtype=np.int32) if subsystem is None else np.asarray(subsystem))
        validate_state(state)
        result = cls(*(jnp.asarray(a) for a in state))
        if result.position.dtype != jnp.float64:
            raise RuntimeError("Rumi requires FP64: enable jax_enable_x64 before creating a state")
        return result


def validate_state(state):
    """Host-side validation at initialization and acceptance boundaries, not inside kernels."""
    x, v, m, ids, sid = (np.asarray(a) for a in state)
    n = len(m)
    if n < 1 or x.shape != (n, 3) or v.shape != (n, 3) or ids.shape != (n,) or sid.shape != (n,):
        raise ValueError("expected nonempty (N,3) positions/velocities and (N,) mass/IDs/subsystems")
    if not all(np.isfinite(a).all() for a in (x, v, m)) or np.any(m <= 0):
        raise ValueError("positions, velocities and positive masses must be finite")
    if not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != n:
        raise ValueError("particle IDs must be unique integers")
    if not np.issubdtype(sid.dtype, np.integer) or np.any(sid < -1):
        raise ValueError("subsystem IDs must be integers >= -1")
    for group in np.unique(sid[sid >= 0]):
        if np.count_nonzero(sid == group) < 2:
            raise ValueError("a subsystem must contain at least two particles")
