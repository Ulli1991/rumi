# Raven handoff

```bash
git clone https://github.com/Ulli1991/rumi.git
cd rumi
python -m venv .venv
source .venv/bin/activate
# Install the GPU JAX build matching Raven's modules/drivers, then:
pip install -e '.[test]'
python scripts/build_cpu.py
python -c 'import jax; print(jax.__version__); print(jax.devices())'
pytest -q
python -m rumi --steps 3 --output results/smoke
python -m rumi --restart results/smoke/restart.npz --steps 3 --output results/resumed
python -m rumi.bench --mode force --backend pallas --n 32768 --steps 20
python -m rumi.bench_hhs --backend pallas --n 4096 --steps 20
```

Use the same Python environment to build and run. The initial Pallas path uses Triton; compatibility
with Raven's actual JAX/GPU stack must be checked. Record installed packages, CPU/GPU models, git
SHAs, initial-array hash, effective configuration and physical end time with results.

Start with one GPU. Validate the complete host-orchestrated runtime separately from the field-only
device HHS experiment. Native JAX multi-GPU sharding is planned, not implemented by the MPI adapter.

Accuracy work: use identical frozen input arrays in Rumi and cpp-nbody. Compare trajectory convergence,
conservation including physical sinks, event identities/times/remnants and restart continuation.
Different IC generators do not produce identical seeded realizations. CPU tree approximations and
changed reduction order also need accuracy sweeps rather than bitwise comparison.

Known checks requiring attention: deferred collision finalization, chained mergers, strong external
perturbers, PN/spin evolution, dense cluster coupling, HDF5 schema and real GPU empty/padded subsets.
HDF5 snapshots are physical-state imports; NPZ checkpoints carry full restart state. HDF5 exact restart
is not implemented. Keep these limitations visible when interpreting test/benchmark results.
