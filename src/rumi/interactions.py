"""Rectangular direct interactions and symmetrized pair steps for the HHS port."""

from functools import lru_cache

import jax
import jax.numpy as jnp


def symmetrized_step(t, c, eta):
    u = c*t*eta
    root = jnp.hypot(1.0, u)
    # Rationalize the approaching-pair branch to avoid subtracting nearly equal values.
    factor = jnp.where(u < 0, (1+1/(root-u))/(root+1), 1+u/(root+1))
    return eta*t*factor


def _direct(target, source, G, eps2, ta=None, sa=None):
    nt, ns = len(target.mass), len(source.mass)
    if ns == 0 or nt == 0:
        return jnp.zeros_like(target.position)
    def body(j, total):
        r = source.position[j]-target.position
        valid = ((target.ids != source.ids[j]) & (target.mass > 0) & (source.mass[j] > 0)
                 & ~((target.subsystem>=0)&(target.subsystem==source.subsystem[j])))
        r2 = jnp.where(valid,jnp.sum(r*r,axis=1)+eps2,1.0)
        ir = jax.lax.rsqrt(r2)
        if ta is None:
            term = G*source.mass[j]*r*ir[:,None]**3
        else:
            da = sa[j]-ta
            term = G*source.mass[j]*(r2[:,None]*da-3*jnp.sum(r*da,axis=1)[:,None]*r)*ir[:,None]**5
        return total+jnp.where(valid[:,None],term,0.0)
    return jax.lax.fori_loop(0,ns,body,jnp.zeros_like(target.position))


def make_interactions(*, G=1.0, softening=0.0):
    """O(N_sink) temporary memory XLA rectangular baseline; GPU tuning follows functional parity."""
    force = jax.jit(lambda t,s: _direct(t,s,G,softening**2))
    grad = jax.jit(lambda t,s,ta,sa: _direct(t,s,G,softening**2,ta,sa))
    return force,grad


def make_pair_steps(*, G=1.0, softening=0.0, eta_ff=0.2, eta_fb=0.2):
    """FROST symmetrization including eta inside the harmonic predictor, as cpp-nbody constants.h."""
    @jax.jit
    def steps(target,source):
        if len(source.mass)==0:
            return jnp.full(len(target.mass),jnp.inf)
        def body(j,minimum):
            r, dv = source.position[j]-target.position,source.velocity[j]-target.velocity
            valid = ((target.ids!=source.ids[j]) & (target.mass > 0) & (source.mass[j] > 0)
                     & ~((target.subsystem>=0)&(target.subsystem==source.subsystem[j])))
            r2 = jnp.where(valid,jnp.sum(r*r,axis=1)+softening**2,1.0)
            ir, v2 = jax.lax.rsqrt(r2),jnp.sum(dv*dv,axis=1)
            distance = r2*ir
            mu = G*(target.mass+source.mass[j])
            tff = r2*jax.lax.rsqrt(distance*(mu+1e-30))
            invv = jax.lax.rsqrt(v2+1e-30)
            tfb = distance*invv
            rv = jnp.sum(r*dv,axis=1)
            cff = 1.5*rv*ir**2
            cfb = rv*ir**2*(1+mu*ir*invv**2)
            pair = jnp.minimum(symmetrized_step(tff,cff,eta_ff),symmetrized_step(tfb,cfb,eta_fb))
            return jnp.minimum(minimum,jnp.where(valid,pair,jnp.inf))
        return jax.lax.fori_loop(0,len(source.mass),body,jnp.full(len(target.mass),jnp.inf))
    return steps


def take(state,indices):
    return type(state)(*(a[indices] for a in state))


def put(state,indices,subset):
    # IDs/masses/membership are unchanged by a force/drift operation.
    return state._replace(position=state.position.at[indices].set(subset.position),
                          velocity=state.velocity.at[indices].set(subset.velocity))
