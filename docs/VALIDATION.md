# Validation handoff — 2026-10-14

Latest full local run: **59 passed, 3 skipped** in 107.79 s. After adding the deferred collision
accounting regression, the simulation subset passed **6 tests**, including that new test.
The full run began before the additional test was collected. The resulting suite contains 60
locally passing tests across these runs; a single complete run of that final collection belongs
in the Raven checks.

Environment: Intel Mac CPU, JAX/jaxlib 0.4.38, Python 3.13. No GPU was used.
The three skips cover two real-GPU checks and HDF5 (h5py unavailable in the local environment).
HDF5 is now a standard installation dependency; CI installs it and runs the snapshot test.

Coverage includes forces/gradients, FSI convergence, host/indexed HHS agreement, native Kepler,
LogH/GBS, PN, SDAR and secular routines, stellar evolution, event accounts, tree exact limit,
integer positions, rollback and checkpoint continuation. The collision regression checks deferred
mass loss and global E/P/L with an external particle. The native incomplete-advance diagnostic
printed by the suite is an intentional failure-rejection case.

Source tracking check: all **74 upstream sources and 16 retained files match** the manifest.
Python compilation and CLI help also pass. This does not establish full executable parity.

Remaining external validation: matched cpp-nbody runs, repeated/native collision scenarios,
strong perturbations, real HDF5, compiled GPU kernels, GPU runtime integration and native JAX
multi-GPU sharding. See [Raven instructions](RAVEN.md). Full restart state currently uses NPZ;
HDF5 exact restart is not implemented.

The measurements below are historical CPU experiments, not current GPU performance claims.

## Indexed HHS experiment

Historical indexed-HHS suite: **44 passed, 2 skipped**; both skipped checks required actual GPUs.

Same Intel Mac CPU/JAX 0.4.38 environment as below. The device-stack S/F schedule matches the
host-recursive HHS reference; rectangular force, gradient and pair-step kernels pass XLA and Pallas
interpreter checks with empty sets, padding and shuffled indices. Complete Pallas-interpreter stepping,
floor behavior, stack/call limits and the rationalized approaching-pair timestep are tested.
With 9 versus 33 particles and only one tight binary, deeper logical pair counts are identical.

Five advancing 33-particle steps, dt=0.01, eta=0.05, one binary and a background with small velocity
differences: warmed XLA median 6.04 ms; Python-recursive median 1.108 s; initial compilation/execution
3.76 s. Maximum final position/velocity differences are 7.3e-17 / 6.4e-15; relative energy change
1.1e-12. These are small CPU measurements, taken alongside tests, not GPU performance evidence.

The cold-background stress case reaches thousands of calls and depth 15 by the third step. Its
three-step warmed CPU median was 190 ms; the longer recursive comparison was interrupted.
Reproduce with `python -m rumi.bench_hhs --n 33 --steps 3 --cold-background`.

Remaining costs include O(N) partition/scatter work, bounded-grid launch overhead and XLA control
flow. Device-resident arrays do not establish that GPU loop conditions avoid host synchronization.
Actual compiled Pallas tests and GPU profiling remain required. CPU encounters, HHS-CC and additional
timestep criteria are not integrated into this experiment.

## Original pilot

Date: 2026-10-13. Platform: Intel Mac, macOS 15.4.1, Python 3.13.13, JAX/jaxlib 0.4.38,
NumPy 2.5.3. JAX ran on CPU; cached wheels were installed in an isolated local environment.

`pytest -q`: **25 passed, 1 skipped**. The skipped test requires compiled Pallas on a real GPU.

Validated:
- XLA and Pallas interpreter forces/gradients against independent NumPy pair loops, including
  subsystem exclusion, awkward particle counts, one particle and Plummer softening.
- Force gradient against a finite-difference directional derivative of acceleration.
- Fourth-order position convergence for an eccentric orbit; energy, momentum, angular momentum,
  and fixed-step time reversibility.
- Native CPU LogH over circular/eccentric orbits (e = 0, 0.7, 0.95), COM boosts, an unequal-mass
  triple and an unbound flyby. Incomplete evolution and invalid states raise.
- Hybrid sequential versus overlapped physical updates, conservation on a short mixed case,
  and refusal of changed IDs/masses/membership.
- Smoke execution of force/Pallas-interpreter, FSI and hybrid JSON benchmark modes.

Not established: GPU compilation, GPU speed, actual CPU/GPU overlap, agreement with a frozen
cpp-nbody output corpus, dense-cluster coupling convergence, large-N behavior or multi-GPU performance.
The local smoke runs demonstrate runnable code only; they were not a controlled speed comparison.
The native CPU pilot contains Newtonian regularization only, without SDAR or PN.

Run locally:

```bash
cd ~/rumi
.venv/bin/python scripts/build_cpu.py
.venv/bin/python -m pytest -q
.venv/bin/python -m rumi.bench --mode hybrid --n 64 --steps 3
```

The implementation plan gives the GPU and whole-step acceptance gates.
