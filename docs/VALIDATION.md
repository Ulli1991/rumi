# Initial local validation

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
