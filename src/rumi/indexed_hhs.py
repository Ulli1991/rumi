"""Device-resident HHS with runtime subset counts and an explicit stack."""

from typing import NamedTuple
import math

import jax
import jax.numpy as jnp

from .hhs import HHSOptions
from .interactions import symmetrized_step


class Work(NamedTuple):
    calls: object
    depth: object
    pairs_by_depth: object
    status: object


def make_indexed_kernel(options, *, tile=16):
    """XLA baseline: only active sink tiles and active sources are evaluated."""
    o = options

    def evaluate(state, targets, nt, sources, ns, acceleration, mode):
        n = len(state.mass)
        width = 1 if mode == "steps" else 3
        output = jnp.zeros((n, width), dtype=state.position.dtype)

        def sink_tile(k, output):
            slots = k * tile + jnp.arange(tile)
            active = slots < nt
            ii = targets[jnp.minimum(slots, n - 1)]
            xi, vi, mi = state.position[ii], state.velocity[ii], state.mass[ii]

            def source(j, total):
                jj = sources[j]
                r = state.position[jj] - xi
                valid = active & (state.ids[ii] != state.ids[jj])
                r2 = jnp.where(valid, jnp.sum(r*r, axis=1) + o.softening**2, 1.0)
                inv = jax.lax.rsqrt(r2)
                if mode == "steps":
                    dv = state.velocity[jj] - vi
                    invv = jax.lax.rsqrt(jnp.sum(dv*dv, axis=1) + 1e-30)
                    mu = o.G * (mi + state.mass[jj])
                    distance = r2 * inv
                    ff = r2 * jax.lax.rsqrt(distance * (mu + 1e-30))
                    fb = distance * invv
                    rv = jnp.sum(r*dv, axis=1) * inv**2

                    dt = jnp.minimum(symmetrized_step(ff, 1.5*rv, o.eta_ff),
                                     symmetrized_step(fb, rv*(1+mu*inv*invv**2), o.eta_fb))
                    return jnp.minimum(total, jnp.where(valid[:, None], dt[:, None], jnp.inf))
                if mode == "gradient":
                    da = acceleration[jj] - acceleration[ii]
                    term = r2[:, None]*da - 3*jnp.sum(r*da, axis=1)[:, None]*r
                    term *= (o.G * state.mass[jj] * inv**5)[:, None]
                else:
                    term = r * (o.G * state.mass[jj] * inv**3)[:, None]
                return total + jnp.where(valid[:, None], term, 0.0)

            initial = jnp.full((tile, width), jnp.inf if mode == "steps" else 0.0)
            result = jax.lax.fori_loop(0, ns, source, initial)
            return output.at[jnp.minimum(slots, n-1)].add(jnp.where(active[:, None], result, 0.0))

        return jax.lax.fori_loop(0, (nt+tile-1)//tile, sink_tile, output)

    return evaluate


class DeviceHHS:
    """Field-only HHS experiment; CPU encounters and HHS-CC are separate work."""

    def __init__(self, options=None, *, stack_depth=32, backend="xla", interpret=False):
        self.options = options or HHSOptions()
        o = self.options
        if o.connected_components or any((o.eta_grad, o.eta_jerk, o.eta_va)):
            raise ValueError("this experiment supports pair timesteps and the S/F split only")
        if not all(math.isfinite(v) and v > 0 for v in (o.G, o.dt_min, o.eta_ff, o.eta_fb)):
            raise ValueError("positive finite HHS parameters required")
        if o.softening < 0 or not math.isfinite(o.softening) or stack_depth < 2 or o.max_calls < 1:
            raise ValueError("invalid softening, stack depth or call limit")
        if backend == "xla":
            kernel = make_indexed_kernel(o)
        elif backend == "pallas":
            from .indexed_pallas import make_indexed_kernel as pallas_kernel
            kernel = pallas_kernel(o, interpret=interpret)
        else:
            raise ValueError("backend must be xla or pallas")
        self.compiled = jax.jit(self._build(kernel, stack_depth))

    def _build(self, kernel, capacity):
        o = self.options

        def integrate(state, dt):
            n = len(state.mass)
            slots = jnp.arange(n, dtype=jnp.int32)
            indices = jnp.zeros((capacity, n), jnp.int32).at[0].set(slots)
            counts = jnp.zeros(capacity, jnp.int32).at[0].set(n)
            slow = jnp.zeros_like(indices)
            slow_counts = jnp.zeros_like(counts)
            fast = jnp.zeros_like(indices)
            fast_counts = jnp.zeros_like(counts)
            phases = jnp.zeros_like(counts)
            root_steps = kernel(state, slots, n, slots, n, state.position, "steps")[:, 0]
            work = Work(jnp.int32(0), jnp.int32(0),
                        jnp.zeros(capacity, jnp.int64).at[0].set(n*n),
                        jnp.where(jnp.any(state.subsystem >= 0), jnp.int32(4), jnp.int32(0)))

            def force(state, sinks, nsink, sources, nsource, a=None):
                return kernel(state, sinks, nsink, sources, nsource,
                              state.position if a is None else a,
                              "force" if a is None else "gradient")

            def scatter(values, subset, count):
                return jnp.zeros_like(state.position).at[subset].add(
                    jnp.where((slots < count)[:, None], values, 0.0))

            def kick(state, subset, count, a, h):
                return state._replace(velocity=state.velocity+h*scatter(a, subset, count))

            def drift(state, subset, count, h):
                movement = scatter(state.velocity[subset], subset, count)
                return state._replace(position=state.position+h*movement)

            def fsi(state, subset, count, h):
                def advance(state):
                    a = force(state, subset, count, subset, count)
                    state = kick(state, subset, count, a, h/6)
                    state = drift(state, subset, count, h/2)
                    a = force(state, subset, count, subset, count)
                    q = force(state, subset, count, subset, count, scatter(a, subset, count))
                    state = kick(state, subset, count, a+h*h*q/24, 2*h/3)
                    state = drift(state, subset, count, h/2)
                    return kick(state, subset, count, force(state, subset, count, subset, count), h/6)
                return jax.lax.cond(count > 0, advance, lambda s: s, state)

            def cross(state, si, sn, fi, fn, h, modified):
                def advance(state):
                    sa = force(state, si, sn, fi, fn)
                    fa = force(state, fi, fn, si, sn)
                    if modified:
                        a = scatter(sa, si, sn) + scatter(fa, fi, fn)
                        sa += h*h*force(state, si, sn, fi, fn, a)/24
                        fa += h*h*force(state, fi, fn, si, sn, a)/24
                    coefficient = 2*h/3 if modified else h/6
                    state = kick(state, si, sn, sa, coefficient)
                    return kick(state, fi, fn, fa, coefficient)
                return jax.lax.cond((sn > 0) & (fn > 0), advance, lambda s: s, state)

            class Frame(NamedTuple):
                state: object
                top: object
                indices: object
                counts: object
                slow: object
                slow_counts: object
                fast: object
                fast_counts: object
                phases: object
                work: object

            def add_work(f, pairs):
                w = f.work
                return f._replace(work=w._replace(pairs_by_depth=w.pairs_by_depth.at[f.top].add(pairs)))

            def push(f):
                d = f.top
                return f._replace(top=d+1, indices=f.indices.at[d+1].set(f.fast[d]),
                                  counts=f.counts.at[d+1].set(f.fast_counts[d]),
                                  phases=f.phases.at[d+1].set(0))

            def enter(f):
                d, subset, count = f.top, f.indices[f.top], f.counts[f.top]
                h = jnp.ldexp(dt, -d)
                w = f.work._replace(calls=f.work.calls+1, depth=jnp.maximum(f.work.depth, d))
                f = f._replace(work=w)

                def leaf(f):
                    return add_work(f._replace(state=fsi(f.state, subset, count, h)), 4*count.astype(jnp.int64)**2)._replace(top=d-1)

                def split(f):
                    local = jax.lax.cond(d == 0, lambda: root_steps,
                        lambda: kernel(f.state, subset, count, subset, count, f.state.position, "steps")[:, 0])
                    f = add_work(f, jnp.where(d == 0, 0, count.astype(jnp.int64)**2))
                    steps = jnp.minimum(local, root_steps[subset])
                    invalid = jnp.any((slots < count) & (jnp.isnan(steps) | (steps <= 0)))
                    f = f._replace(work=f.work._replace(status=jnp.where(invalid, jnp.int32(3), f.work.status)))
                    is_slow = (slots < count) & (steps >= h)
                    is_fast = (slots < count) & (steps < h)
                    sn, fn = jnp.sum(is_slow, dtype=jnp.int32), jnp.sum(is_fast, dtype=jnp.int32)
                    si = subset[jnp.nonzero(is_slow, size=n, fill_value=0)[0]]
                    fi = subset[jnp.nonzero(is_fast, size=n, fill_value=0)[0]]
                    f = f._replace(slow=f.slow.at[d].set(si), slow_counts=f.slow_counts.at[d].set(sn),
                                   fast=f.fast.at[d].set(fi), fast_counts=f.fast_counts.at[d].set(fn))

                    def all_slow(f):
                        s = fsi(fsi(f.state, si, sn, h/2), si, sn, h/2)
                        return add_work(f._replace(state=s), 8*sn.astype(jnp.int64)**2)._replace(top=d-1)

                    def recurse(f):
                        def overflow(f):
                            return f._replace(work=f.work._replace(status=jnp.int32(2)))
                        def descend(f):
                            s = fsi(cross(f.state, si, sn, fi, fn, h, False), si, sn, h/2)
                            f = add_work(f._replace(state=s), 2*sn.astype(jnp.int64)*fn+4*sn.astype(jnp.int64)**2)
                            return push(f._replace(phases=f.phases.at[d].set(1)))
                        return jax.lax.cond(d+1 >= capacity, overflow, descend, f)
                    return jax.lax.cond(fn == 0, all_slow, recurse, f)

                return jax.lax.cond((count <= 1) | (h/2 < o.dt_min), leaf, split, f)

            def middle(f):
                d = f.top
                si, sn, fi, fn = f.slow[d], f.slow_counts[d], f.fast[d], f.fast_counts[d]
                h = jnp.ldexp(dt, -d)
                f = add_work(f._replace(state=cross(f.state, si, sn, fi, fn, h, True)), 4*sn.astype(jnp.int64)*fn)
                return push(f._replace(phases=f.phases.at[d].set(2)))

            def finish(f):
                d = f.top
                si, sn, fi, fn = f.slow[d], f.slow_counts[d], f.fast[d], f.fast_counts[d]
                h = jnp.ldexp(dt, -d)
                s = cross(fsi(f.state, si, sn, h/2), si, sn, fi, fn, h, False)
                return add_work(f._replace(state=s), 4*sn.astype(jnp.int64)**2+2*sn.astype(jnp.int64)*fn)._replace(top=d-1)

            def body(f):
                return jax.lax.switch(f.phases[f.top], (enter, middle, finish), f)

            initial = Frame(state, jnp.int32(0), indices, counts, slow, slow_counts, fast, fast_counts, phases, work)
            result = jax.lax.while_loop(lambda f: (f.top >= 0) & (f.work.status == 0) & (f.work.calls < o.max_calls), body, initial)
            finite = jnp.all(jnp.isfinite(result.state.position)) & jnp.all(jnp.isfinite(result.state.velocity))
            status = jnp.where(result.work.status != 0, result.work.status,
                              jnp.where(~finite, 3, jnp.where(result.top >= 0, 1, 0)))
            return result.state, result.work._replace(status=status)

        return integrate

    def step(self, state, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("finite positive dt required")
        if state.position.dtype != jnp.float64:
            raise TypeError("HHS requires FP64 state")
        result, work = self.compiled(state, jnp.asarray(dt, state.position.dtype))
        status = int(work.status)
        if status:
            raise RuntimeError({1: "HHS call limit", 2: "HHS stack exhausted",
                                3: "non-finite HHS state", 4: "CPU encounter membership unsupported"}[status])
        return result, work
