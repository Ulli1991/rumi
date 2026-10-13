# Rumi

A speed-first N-body prototype with readable Python orchestration, JAX/Pallas GPU kernels and
native CPU regularization. Differentiability is not a requirement.

The target is competitive **complete-step performance at controlled physical error** against
cpp-nbody. That target is not yet established. [The implementation plan](docs/PLAN.md) sets the gates.

## Implemented pilot

- FP64 direct forces and force gradients, subsystem exclusion, bounded-memory XLA and tiled Pallas paths.
- Fourth-order fixed-step forward symplectic integration (FSI).
- Native Newtonian LogH/GBS for 2..8 bodies, with COM normalization and explicit failure checks.
- Fixed-membership hybrid integration: CPU encounters overlap an independent field-force dispatch.
- Numerical tests and synchronized JSON benchmarks.

HHS, slowdown/SDAR, PN, dynamic subsystem detection, mergers, SSE and multi-GPU evolution are planned.
The hybrid force-completion tail is currently XLA. This is a numerical/scheduling pilot, not a production
cluster solver. No GPU speed or correctness claim follows from the CPU interpreter tests.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python scripts/build_cpu.py
pytest -q
python -m rumi.bench --mode fsi --n 256 --steps 10
python -m rumi.bench --mode hybrid --n 256 --steps 10 --output results/hybrid.json
python -m rumi.bench --mode hybrid --n 256 --steps 10 --no-overlap
```

The CPU library needs a C++17 compiler; it has no Kokkos or Python-header dependency.
An installed package can point to a separately built library using `RUMI_CPU_LIBRARY`.

For an actual GPU install the JAX build appropriate to its CUDA/ROCm stack, pin that environment,
then run the real-GPU tests and kernels explicitly:

```bash
pytest -q -m gpu
python -m rumi.bench --mode force --backend pallas --n 32768 --steps 20
```

The initial Pallas implementation selects Triton explicitly. Its availability and performance must
be verified on the pinned GPU stack. `--interpret` is only for CPU validation. Compilation/warm-up
is reported separately; warmed measurements wait for device completion.

The API leaves precision configuration explicit:

```python
import jax
jax.config.update('jax_enable_x64', True)
from rumi import State
from rumi.integrator import make_fsi_step

state = State.create([[-0.5, 0, 0], [0.5, 0, 0]],
                     [[0, -0.5, 0], [0, 0.5, 0]], [0.5, 0.5])
state = make_fsi_step(backend='xla')(state, 0.01)
```

## Provenance

The architecture and numerical conventions are informed by the user's cpp-nbody at `8b7729c`.
The pilot is a separate implementation; no Kokkos runtime or source files are vendored. Native
LogH currently covers Newtonian dynamics only and is not a drop-in replacement for cpp-nbody's
full regularization module. Source licensing for any future extraction must be settled explicitly.
