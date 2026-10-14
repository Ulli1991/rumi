# Rumi

A Python/JAX/Pallas port of cpp-nbody with native CPU few-body physics. Raw speed at controlled
physical error is the goal; differentiability is not required.

## Current implementation

- FP64 direct forces, force gradients, FSI and adaptive HHS with connected components.
- Dynamic subsystem detection; retained Kepler, LogH/GBS, slowdown/SDAR, PN and secular routines.
- Stellar evolution, collisions, merger/recoil policies and conservation accounts.
- Tides, escapers, integer positions, a CPU tree reference and cluster diagnostics.
- Configuration/initial conditions, HDF5 snapshots/import and atomic NPZ checkpoints with solver carries.
- Experimental field-only indexed JAX/Pallas HHS and fixed-membership CPU/field overlap.

The complete physical runtime currently uses host orchestration. The indexed device HHS experiment
does not yet include every runtime feature. Native JAX sharding is the intended multi-GPU path;
the optional MPI reference adapter is not the target architecture.

This is a validation handoff, not a claim of numerical parity or competitive GPU speed.
See [validation status](docs/VALIDATION.md) and the [Raven handoff](docs/RAVEN.md).

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python scripts/build_cpu.py
pytest -q
python -m rumi --help
python -m rumi --steps 3 --output results/smoke
python -m rumi --restart results/smoke/restart.npz --steps 3 --output results/resumed
```

The native library needs a C++17 compiler, without Kokkos or Python headers.
`RUMI_CPU_LIBRARY` can select a separately built library. HDF5 is a standard dependency.
Snapshots contain physical state; exact continuation uses NPZ checkpoints. HDF5 snapshots alone
do not restore native carries and all integration state.

On Raven, install the JAX build appropriate to its GPU software stack before installing Rumi.
Do not replace that build with a CPU-only JAX package. CPU Pallas interpretation does not validate
compiled GPU kernels.

## Upstream tracking

The reference checkout is cpp-nbody at `8b7729c091a18caba109de4c51ac6ca9743ad388`.
Retained CPU source provenance and hashes are in `docs/upstream_manifest.json`.

```bash
python scripts/sync_cpp_nbody.py --source ../cpp-nbody --check
```

This detects upstream changes; Python orchestration changes still require review and porting.
`--apply` regenerates retained sources and the parameter table: inspect its diff and rerun validation.
Keep cpp-nbody untouched and compare frozen particle arrays, rather than different random generators.
