"""Synchronized pilot benchmarks; outputs measurements without claiming GPU competitiveness."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import jax
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("force", "fsi", "hybrid"), default="force")
    parser.add_argument("--backend", choices=("xla", "pallas"), default="xla")
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--binary-fraction", type=float, default=0.1, help="fraction of particles in fixed binary groups")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--allow-slow-xla", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.n < 2 or args.steps < 1 or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error("n >= 2, steps >= 1 and finite positive dt required")
    if not 0 <= args.binary_fraction <= 1 or args.workers < 1:
        parser.error("binary-fraction in [0,1] and workers >= 1 required")
    if args.n > 4096 and args.backend == "xla" and not args.allow_slow_xla:
        parser.error("XLA is the O(N)-memory correctness path; use pallas for large N or --allow-slow-xla")
    jax.config.update("jax_enable_x64", True)
    from . import State
    from .forces import make_force_functions
    from .hybrid import HybridFSI
    from .integrator import make_fsi_step
    rng = np.random.default_rng(args.seed)
    x, v, m = rng.normal(size=(args.n, 3)), 0.1*rng.normal(size=(args.n, 3)), np.ones(args.n)/args.n
    sid = np.full(args.n, -1, dtype=np.int32)
    if args.mode == "hybrid":
        for k in range(int(args.binary_fraction*args.n)//2):
            i = 2*k
            center, speed = x[i].copy(), v[i].copy()
            x[i], x[i+1] = center+[-0.005, 0, 0], center+[0.005, 0, 0]
            internal = np.sqrt((m[i]+m[i+1])/0.01)/2
            v[i], v[i+1] = speed+[0, -internal, 0], speed+[0, internal, 0]
            sid[i:i+2] = k
    state = State.create(x, v, m, subsystem=sid)
    initial = state
    initial_hash = hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in (x, v, m, sid))).hexdigest()
    runner = None
    if args.mode == "force":
        force, _ = make_force_functions(backend=args.backend, interpret=args.interpret)
        call = lambda s: force(s.position, s.mass, s.subsystem)
    elif args.mode == "fsi":
        step = make_fsi_step(backend=args.backend, interpret=args.interpret)
        call = lambda s: step.compiled(s, args.dt)
    else:
        runner = HybridFSI(state, backend=args.backend, interpret=args.interpret, workers=args.workers,
                           overlap=not args.no_overlap)
        call = lambda s: runner.step(s, args.dt)
    try:
        t0 = time.perf_counter()
        warm = call(state)
        jax.block_until_ready(warm)
        warmup = time.perf_counter()-t0
        samples = []
        for _ in range(args.steps):
            t0 = time.perf_counter()
            result = call(state)
            jax.block_until_ready(result)
            samples.append(time.perf_counter()-t0)
            if args.mode != "force":
                state = result
        actual = result if args.mode == "force" else (state.position, state.velocity)
        if not all(np.isfinite(np.asarray(a)).all() for a in (actual if isinstance(actual, tuple) else (actual,))):
            raise RuntimeError("benchmark ended with non-finite state")
    finally:
        if runner:
            runner.close()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    errors = None
    if args.n <= 512:
        from . import reference
        if args.mode == "force":
            expected = reference.acceleration(initial.position, initial.mass, subsystem=initial.subsystem)
            errors = {"force_relative_l2": float(np.linalg.norm(np.asarray(result)-expected)/max(np.linalg.norm(expected), 1e-300))}
        else:
            before, after = reference.diagnostics(initial), reference.diagnostics(state)
            errors = {"relative_energy_error": abs(after["energy"]-before["energy"])/max(abs(before["energy"]), 1e-300),
                      "momentum_error": float(np.linalg.norm(after["momentum"]-before["momentum"])),
                      "angular_momentum_error": float(np.linalg.norm(after["angular_momentum"]-before["angular_momentum"]))}
    record = dict(mode=args.mode, backend=args.backend, interpreter=args.interpret,
                  n=args.n, steps=args.steps, seed=args.seed, dt=args.dt,
                  binary_fraction=args.binary_fraction if args.mode == "hybrid" else 0,
                  overlap=not args.no_overlap if args.mode == "hybrid" else None,
                  cpu_workers=args.workers, jax=jax.__version__, numpy=np.__version__,
                  python=platform.python_version(), host=platform.platform(),
                  devices=[str(d.device_kind) for d in jax.devices()],
                  revision=git.stdout.strip() if git.returncode == 0 else "uncommitted",
                  dirty=bool(dirty.stdout.strip()), initial_arrays_sha256=initial_hash, precision="fp64",
                  errors=errors, error_scope="independent small-N checks" if errors is not None else "not measured: run matched accuracy suite",
                  warmup_seconds=warmup, median_seconds=float(np.median(samples)),
                  min_seconds=min(samples), max_seconds=max(samples), samples_seconds=samples,
                  physical_time_reached=0 if args.mode == "force" else args.steps*args.dt,
                  scope="pilot: fixed membership; hybrid member-force tail is XLA; no HHS/SDAR/PN/mergers")
    text = json.dumps(record, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+"\n")


if __name__ == "__main__":
    main()
