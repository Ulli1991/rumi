"""Indexed Pallas interactions with runtime sink/source tile loops."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .interactions import symmetrized_step


def make_indexed_kernel(options, *, interpret=False, sink_tile=16, source_tile=64, programs=32):
    o = options
    if not interpret and jax.default_backend() == "cpu":
        raise RuntimeError("Pallas GPU requested on CPU; interpreter is validation only")
    if any(t < 1 or t & (t-1) for t in (sink_tile, source_tile)) or programs < 1:
        raise ValueError("positive power-of-two tiles and positive program count required")
    cache = {}

    def evaluate(state, targets, nt, sources, ns, acceleration, mode):
        n = len(state.mass)
        width = 1 if mode == "steps" else 3
        key = (n, mode)
        if key not in cache:
            grid = min(programs, (n+sink_tile-1)//sink_tile)

            def kernel(x, v, mass, ids, target, target_count, source, source_count, a, out):
                program = pl.program_id(0)
                sink_count = pl.load(target_count, ())
                source_size = pl.load(source_count, ())
                sink_tiles = (sink_count+sink_tile-1)//sink_tile

                def sinks(k, unused):
                    slots = (program+k*grid)*sink_tile+jnp.arange(sink_tile)
                    active = slots < sink_count
                    ii = pl.load(target, (slots,), mask=active, other=0)
                    xi = tuple(pl.load(x, (ii, d), mask=active, other=0.0) for d in range(3))
                    idi = pl.load(ids, (ii,), mask=active, other=-1)
                    if mode == "steps":
                        vi = tuple(pl.load(v, (ii, d), mask=active, other=0.0) for d in range(3))
                        mi = pl.load(mass, (ii,), mask=active, other=0.0)
                    if mode == "gradient":
                        ai = tuple(pl.load(a, (ii, d), mask=active, other=0.0) for d in range(3))

                    def sources(j, total):
                        source_slots = j*source_tile+jnp.arange(source_tile)
                        live = source_slots < source_size
                        jj = pl.load(source, (source_slots,), mask=live, other=0)
                        xj = tuple(pl.load(x, (jj, d), mask=live, other=0.0) for d in range(3))
                        mj = pl.load(mass, (jj,), mask=live, other=0.0)
                        idj = pl.load(ids, (jj,), mask=live, other=-1)
                        valid = active[:, None] & live[None, :] & (idi[:, None] != idj[None, :])
                        r = tuple(xj[d][None, :]-xi[d][:, None] for d in range(3))
                        r2 = jnp.where(valid, sum(rd*rd for rd in r)+o.softening**2, 1.0)
                        inv = jax.lax.rsqrt(r2)
                        if mode == "steps":
                            vj = tuple(pl.load(v, (jj, d), mask=live, other=0.0) for d in range(3))
                            dv = tuple(vj[d][None, :]-vi[d][:, None] for d in range(3))
                            invv = jax.lax.rsqrt(sum(vd*vd for vd in dv)+1e-30)
                            mu = o.G*(mi[:, None]+mj[None, :])
                            distance = r2*inv
                            ff = r2*jax.lax.rsqrt(distance*(mu+1e-30))
                            fb = distance*invv
                            rv = sum(r[d]*dv[d] for d in range(3))*inv**2

                            pair = jnp.minimum(symmetrized_step(ff, 1.5*rv, o.eta_ff),
                                               symmetrized_step(fb, rv*(1+mu*inv*invv**2), o.eta_fb))
                            return (jnp.minimum(total[0], jnp.min(jnp.where(valid, pair, jnp.inf), axis=1)),)
                        if mode == "gradient":
                            aj = tuple(pl.load(a, (jj, d), mask=live, other=0.0) for d in range(3))
                            da = tuple(aj[d][None, :]-ai[d][:, None] for d in range(3))
                            dot = sum(r[d]*da[d] for d in range(3))
                            terms = tuple((r2*da[d]-3*dot*r[d])*(o.G*mj[None, :]*inv**5) for d in range(3))
                        else:
                            terms = tuple(rd*(o.G*mj[None, :]*inv**3) for rd in r)
                        return tuple(total[d]+jnp.sum(jnp.where(valid, terms[d], 0.0), axis=1) for d in range(3))

                    initial = tuple(jnp.full(sink_tile, jnp.inf if mode == "steps" else 0.0, dtype=jnp.float64) for _ in range(width))
                    result = jax.lax.fori_loop(0, (source_size+source_tile-1)//source_tile, sources, initial)
                    for d in range(width):
                        pl.store(out, (slots, d), result[d], mask=active)
                    return unused

                rounds = jnp.maximum(0, (sink_tiles-program+grid-1)//grid)
                jax.lax.fori_loop(0, rounds, sinks, jnp.int32(0))

            cache[key] = pl.pallas_call(kernel,
                out_shape=jax.ShapeDtypeStruct((n, width), jnp.float64), grid=(grid,),
                interpret=interpret, backend=None if interpret else "triton")
        result = cache[key](state.position, state.velocity, state.mass, state.ids, targets,
                            jnp.asarray(nt, jnp.int32), sources, jnp.asarray(ns, jnp.int32), acceleration)
        return jnp.where((jnp.arange(n) < nt)[:, None], result, 0.0)

    return evaluate
