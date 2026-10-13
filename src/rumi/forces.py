"""Bounded-memory direct forces and force gradients in XLA or tiled Pallas."""

from functools import lru_cache
import math

import jax
import jax.numpy as jnp


def _check(position, mass, subsystem, G, softening):
    if position.ndim != 2 or position.shape[1] != 3 or mass.shape != position.shape[:1]:
        raise ValueError("expected (N,3) positions and (N,) masses")
    if subsystem.shape != mass.shape:
        raise ValueError("subsystem shape differs from mass")
    if position.dtype != jnp.float64 or mass.dtype != jnp.float64:
        raise TypeError("the initial Rumi force backend requires FP64")
    if not math.isfinite(G) or not math.isfinite(softening) or not G > 0 or not softening >= 0:
        raise ValueError("G must be positive and softening nonnegative")


def _xla(position, mass, subsystem, G, eps2, accel=None):
    """One source at a time: O(N) working memory; correctness backend, not the GPU speed target."""
    n = len(mass)
    indices = jnp.arange(n)

    def body(j, total):
        r = position[j] - position
        valid = (indices != j) & ~((subsystem >= 0) & (subsystem == subsystem[j]))
        r2 = jnp.where(valid, jnp.sum(r * r, axis=1) + eps2, 1.0)
        inv = jax.lax.rsqrt(r2)
        if accel is None:
            term = r * (G * mass[j] * inv**3)[:, None]
        else:
            da = accel[j] - accel
            term = (r2[:, None] * da - 3 * jnp.sum(r * da, axis=1)[:, None] * r)
            term *= (G * mass[j] * inv**5)[:, None]
        return total + jnp.where(valid[:, None], term, 0.0)

    return jax.lax.fori_loop(0, n, body, jnp.zeros_like(position))


@lru_cache(maxsize=128)
def _pallas_program(n, G, eps2, gradient, interpret, sinks, sources):
    from jax.experimental import pallas as pl

    def kernel(x_ref, m_ref, sid_ref, a_ref, out_ref):
        ii = pl.program_id(0) * sinks + jnp.arange(sinks)
        xi = tuple(pl.load(x_ref, (ii, d), mask=ii < n, other=0.0) for d in range(3))
        si = pl.load(sid_ref, (ii,), mask=ii < n, other=-1)
        ai = tuple(pl.load(a_ref, (ii, d), mask=ii < n, other=0.0) for d in range(3)) if gradient else None

        def body(tile, total):
            jj = tile * sources + jnp.arange(sources)
            xj = tuple(pl.load(x_ref, (jj, d), mask=jj < n, other=0.0) for d in range(3))
            mj = pl.load(m_ref, (jj,), mask=jj < n, other=0.0)
            sj = pl.load(sid_ref, (jj,), mask=jj < n, other=-1)
            r = tuple(xj[d][None, :] - xi[d][:, None] for d in range(3))
            valid = ((ii[:, None] < n) & (jj[None, :] < n) & (ii[:, None] != jj[None, :])
                     & ~((si[:, None] >= 0) & (si[:, None] == sj[None, :])))
            r2 = jnp.where(valid, sum(rd * rd for rd in r) + eps2, 1.0)
            inv = jax.lax.rsqrt(r2)
            if gradient:
                aj = tuple(pl.load(a_ref, (jj, d), mask=jj < n, other=0.0) for d in range(3))
                da = tuple(aj[d][None, :] - ai[d][:, None] for d in range(3))
                dot = sum(r[d] * da[d] for d in range(3))
                term = tuple((r2 * da[d] - 3 * dot * r[d]) * (G * mj[None, :] * inv**5) for d in range(3))
            else:
                term = tuple(rd * (G * mj[None, :] * inv**3) for rd in r)
            return tuple(total[d] + jnp.sum(jnp.where(valid, term[d], 0.0), axis=1) for d in range(3))

        result = jax.lax.fori_loop(0, (n + sources - 1) // sources, body,
                                   tuple(jnp.zeros(sinks, dtype=jnp.float64) for _ in range(3)))
        for d in range(3):
            pl.store(out_ref, (ii, d), result[d], mask=ii < n)

    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((n, 3), jnp.float64),
                          grid=((n + sinks - 1) // sinks,), interpret=interpret,
                          backend=None if interpret else "triton")


def make_force_functions(*, backend="xla", G=1.0, softening=0.0, interpret=False,
                         sink_tile=16, source_tile=64):
    """Return compiled (acceleration, gradient); explicit Pallas never silently falls back."""
    if backend not in ("xla", "pallas"):
        raise ValueError("backend must be xla or pallas")
    for tile in (sink_tile, source_tile):
        if tile < 1 or tile & (tile - 1):
            raise ValueError("Pallas tile sizes must be positive powers of two")
    if backend == "pallas" and not interpret and jax.default_backend() == "cpu":
        raise RuntimeError("Pallas GPU requested on CPU; use interpret=True only for validation")

    def call(x, m, sid, a=None):
        _check(x, m, sid, G, softening)
        if backend == "xla":
            return _xla(x, m, sid, G, softening**2, a)
        program = _pallas_program(len(m), G, softening**2, a is not None, interpret, sink_tile, source_tile)
        return program(x, m, sid, jnp.zeros_like(x) if a is None else a)

    return jax.jit(lambda x, m, sid: call(x, m, sid)), jax.jit(lambda x, m, sid, a: call(x, m, sid, a))
