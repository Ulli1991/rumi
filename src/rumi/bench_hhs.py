"""Complete HHS steps on one tight binary and a quiet background."""

import argparse
import hashlib
import json
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=33)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=.01)
    parser.add_argument("--eta", type=float, default=.05)
    parser.add_argument("--backend", choices=("xla", "pallas"), default="xla")
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--reference", action="store_true", help="also time the Python-recursive reference")
    parser.add_argument("--cold-background", action="store_true", help="stress nearly stationary pair timesteps")
    args = parser.parse_args()
    if args.n < 2 or args.steps < 1:
        parser.error("n >= 2 and steps >= 1 required")
    if not all(np.isfinite(v) and v > 0 for v in (args.dt, args.eta)):
        parser.error("finite positive dt and eta required")
    if args.reference and args.n > 512:
        parser.error("recursive reference comparison is limited to n <= 512")
    jax.config.update("jax_enable_x64", True)
    from . import State
    from .hhs import HHS, HHSOptions
    from .indexed_hhs import DeviceHHS
    from .reference import diagnostics

    x, v = np.zeros((args.n, 3)), np.zeros((args.n, 3))
    x[:, 0] = 10+np.arange(args.n)*3
    if not args.cold_background:
        v[:, 1] = .1+.001*np.arange(args.n)
    x[:2, 0] = [-.025, .025]
    v[:2, 1] = [-np.sqrt(20)/2, np.sqrt(20)/2]
    m = np.r_[.5, .5, np.full(args.n-2, .001)]
    initial = State.create(x, v, m)
    options = HHSOptions(eta_ff=args.eta, eta_fb=args.eta)
    solver = DeviceHHS(options, backend=args.backend, interpret=args.interpret)
    dt = jnp.asarray(args.dt, jnp.float64)
    start = time.perf_counter()
    warm, work = solver.compiled(initial, dt)
    jax.block_until_ready((warm, work))
    warm_seconds = time.perf_counter()-start
    if int(work.status):
        raise RuntimeError(f"HHS rejected warm-up: status {int(work.status)}")
    state, samples, pair_trace = initial, [], np.zeros(32, dtype=np.int64)
    for _ in range(args.steps):
        start = time.perf_counter()
        state, work = solver.compiled(state, dt)
        jax.block_until_ready((state, work))
        samples.append(time.perf_counter()-start)
        if int(work.status):
            raise RuntimeError(f"HHS rejected benchmark: status {int(work.status)}")
        pair_trace += np.asarray(work.pairs_by_depth)
    report = {
        "platform": platform.platform(), "devices": [str(d) for d in jax.devices()],
        "jax": jax.__version__, "backend": args.backend, "interpret": args.interpret,
        "n": args.n, "steps": args.steps, "dt": args.dt, "eta": args.eta,
        "cold_background": args.cold_background,
        "ic_sha256": hashlib.sha256(x.tobytes()+v.tobytes()+m.tobytes()).hexdigest(),
        "compile_and_first_step_s": warm_seconds, "median_step_s": float(np.median(samples)),
        "last_calls": int(work.calls), "last_depth": int(work.depth),
        "logical_pairs_by_depth": pair_trace.tolist(),
        "pair_count_note": "schedule accounting, not a hardware counter; tile tails add padding",
    }
    if args.n <= 512:
        before, after = diagnostics(initial), diagnostics(state)
        report["relative_energy_change"] = float(abs(after["energy"]/before["energy"]-1))
    if args.reference:
        reference = HHS(options)
        jax.block_until_ready(reference.step(initial, args.dt))
        expected, timings = initial, []
        for _ in range(args.steps):
            start = time.perf_counter()
            expected = reference.step(expected, args.dt)
            jax.block_until_ready(expected)
            timings.append(time.perf_counter()-start)
        report["recursive_median_step_s"] = float(np.median(timings))
        report["max_position_difference"] = float(np.max(np.abs(np.asarray(state.position-expected.position))))
        report["max_velocity_difference"] = float(np.max(np.abs(np.asarray(state.velocity-expected.velocity))))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
