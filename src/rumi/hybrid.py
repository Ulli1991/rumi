"""Fixed-membership hybrid FSI: CPU LogH overlaps the independent field force dispatch."""

from concurrent.futures import ThreadPoolExecutor
import math

import jax
import jax.numpy as jnp
import numpy as np

from .cpu import integrate_logh
from .forces import make_force_functions
from .state import validate_state


class HybridFSI:
    """Pilot scheduler with explicit joins; immutable state, fixed groups, no merger/SSE/PN/SDAR support."""

    def __init__(self, initial, *, backend="xla", interpret=False, workers=2, G=1.0,
                 tolerance=1e-11, overlap=True, library=None):
        validate_state(initial)
        self.membership = np.asarray(initial.subsystem).copy()
        self.ids = np.asarray(initial.ids).copy()
        self.mass = np.asarray(initial.mass).copy()
        self.field = np.flatnonzero(self.membership < 0)
        self.groups = [np.flatnonzero(self.membership == s) for s in np.unique(self.membership[self.membership >= 0])]
        if any(len(g) > 8 for g in self.groups):
            raise ValueError("native pilot supports at most 8 bodies per subsystem")
        self.members = np.concatenate(self.groups) if self.groups else np.empty(0, dtype=int)
        self.G, self.tolerance, self.overlap, self.library = G, tolerance, overlap, library
        self.acceleration, self.gradient = make_force_functions(backend=backend, G=G, interpret=interpret)
        self.pool = ThreadPoolExecutor(max_workers=workers)

        # Completed field-field forces are added to member-source forces. Fixed shape means no recompilation.
        field, members = jnp.asarray(self.field), jnp.asarray(self.members)
        @jax.jit
        def finish_force(x, m, sid, field_force):
            # Correctness pilot: O(N*N_member) bounded-memory XLA; port this tail to Pallas before speed claims.
            tail = jnp.zeros_like(x)
            all_i = jnp.arange(len(m))
            def body(k, result):
                j = members[k]
                r = x[j]-x
                valid = (all_i != j) & ~((sid >= 0) & (sid == sid[j]))
                r2 = jnp.where(valid, jnp.sum(r*r, axis=1), 1.0)
                return result + jnp.where(valid[:, None], self.G*m[j]*r*jax.lax.rsqrt(r2)[:, None]**3, 0.0)
            tail = jax.lax.fori_loop(0, len(self.members), body, tail)
            # Member sinks also need field sources, which were absent from the field-only pass.
            def field_source(k, result):
                j = field[k]
                r = x[j]-x[members]
                term = self.G*m[j]*r*jax.lax.rsqrt(jnp.sum(r*r, axis=1))[:, None]**3
                return result + term
            member_field = jax.lax.fori_loop(0, len(self.field), field_source, jnp.zeros((len(self.members), 3)))
            tail = tail.at[members].add(member_field)
            return tail.at[field].add(field_force)

        self.finish_force = finish_force

    def close(self):
        self.pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _drift(self, state, dt):
        x, v, m, _, sid = state
        if not self.groups:
            x = x + dt*v
            return state._replace(position=x), self.acceleration(x, m, sid)
        # Transfer only members, and complete their transfer before dispatching the independent GPU pass.
        mx, mv, mm = jax.device_get((x[self.members], v[self.members], m[self.members]))
        jobs, offset = [], 0
        for group in self.groups:
            size = len(group)
            jobs.append(self.pool.submit(integrate_logh, mx[offset:offset+size].copy(), mv[offset:offset+size].copy(),
                         mm[offset:offset+size].copy(), dt, G=self.G, tolerance=self.tolerance, library=self.library))
            offset += size
        x = x.at[self.field].add(dt*v[self.field])
        field_force = None
        if self.overlap and len(self.field):
            field_force = self.acceleration(x[self.field], m[self.field], sid[self.field])
        for group, job in zip(self.groups, jobs):
            new_x, new_v = job.result()   # native failures propagate; input state remains untouched
            x, v = x.at[group].set(new_x), v.at[group].set(new_v)
        if field_force is None:
            a = self.acceleration(x, m, sid)
        else:
            a = self.finish_force(x, m, sid, field_force)
        return state._replace(position=x, velocity=v), a

    def step(self, state, dt):
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        # Immutable topology and masses are a pilot contract, not silently cached changing physics.
        if not all(np.array_equal(np.asarray(actual), expected) for actual, expected in
                   ((state.subsystem, self.membership), (state.ids, self.ids), (state.mass, self.mass))):
            raise ValueError("HybridFSI fixed topology/IDs/masses changed; construct a new scheduler")
        a = self.acceleration(state.position, state.mass, state.subsystem)
        state = state._replace(velocity=state.velocity+(dt/6)*a)
        state, a = self._drift(state, dt/2)
        a = a + (dt*dt/24)*self.gradient(state.position, state.mass, state.subsystem, a)
        state = state._replace(velocity=state.velocity+(2*dt/3)*a)
        state, a = self._drift(state, dt/2)
        return state._replace(velocity=state.velocity+(dt/6)*a)
