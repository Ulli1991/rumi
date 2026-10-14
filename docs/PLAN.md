# Completion and Raven plan

Complete the physics/runtime port as closely as possible to cpp-nbody, then validate accuracy and
GPU behavior together on Raven. Local dependency limitations do not gate the handoff.

The physical runtime includes host HHS/CC, subsystem detection, retained native encounters, stellar
and merger events, tides, escapers, diagnostics and checkpointing. The separate device HHS path is
field-only; integrating the complete runtime with it remains GPU engineering work.

## Raven validation

1. Build retained native routines and run the complete tests, including real HDF5 and compiled Pallas.
2. Freeze cpp-nbody SHA, configuration and initial particle arrays. Compare forces, timesteps,
   eccentric binaries, PN/SDAR/secular cases, stellar evolution and encounter-rich trajectories.
3. Check contact/TDE/BH mergers, deferred mass loss/recoil, repeated mergers and E/P/L accounts.
4. Check uninterrupted versus restarted evolution and HDF5 schema/particle-property round trips.
5. Measure complete steps at matched physical accuracy; separate compilation, transfers, CPU
   encounter work, GPU work and join waits. Optimize after correctness failures are resolved.

## GPU architecture

Use native JAX sharding and collectives for multiple GPUs, with Pallas for suitable local kernels.
Python MPI is not a requirement; the existing replicated-force MPI adapter is a reference experiment.
Integrate native CPU encounters at explicit joins, preserve task ownership and reject incomplete or
non-finite advances. Avoid global N-by-N intermediates and full-N pair work at every HHS level.
Prove overlap with a device timeline rather than assuming asynchronous dispatch overlaps computation.

## Future upstream changes

Run `scripts/sync_cpp_nbody.py --check` against each new cpp-nbody revision. Review changed source
hashes, regenerate retained routines where appropriate, port corresponding orchestration changes,
and repeat matched accuracy cases. The manifest detects drift; it does not automatically port logic.

No numerical parity, GPU correctness, speed or scaling claim is established by the local CPU suite.
