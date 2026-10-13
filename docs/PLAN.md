# Rumi implementation and performance plan

Rumi is a Python-led collisional N-body project using JAX/Pallas for GPU kernels and a compiled CPU
backend for irregular encounters. The target is cpp-nbody-like physics at competitive complete-run
speed. Differentiability is not a requirement. Numerical correctness, measured error, readable
orchestration and explicit CPU/GPU ownership are requirements.

## First decision: prove speed before porting all physics

The existing cpp-nbody checkout, initially 8b7729c, remains untouched. Freeze its SHA, effective
configuration, compiler flags, kernel variants and initial particle arrays for each comparison.
Do not compare different IC generators or call a lower-accuracy run faster without identifying it.

Two comparisons are needed:

1. Same numerical workload: direct FP64 forces, FSI coefficients, timesteps, regularization tolerances,
   particle count, GPU count and output cadence.
2. Same measured physical error: time to the same physical end time at matched energy/convergence
   accuracy. This comparison matters most if reductions or scheduling change.

Success target (a proposed engineering gate, not a measured claim): within 20% of the faster validated
cpp-nbody configuration for a complete 200k-star step, across three seeds, with 10% binaries on one
A100 node. Continue to the full port if close and profiling identifies a plausible remaining gain.
A speedup is preferred. Test one A100 first and MI300A separately; one vendor's result is not portable
evidence. Report startup/compilation separately, plus amortized time for short and long runs.

## Architecture

- Python: configuration, task scheduler, acceptance boundaries, restart/I/O, profiling.
- JAX: state transformations, gathers/scatters, fixed-shape compiled stepping and collective orchestration.
- Pallas: direct forces, force gradients, pair timesteps and the rectangular source/sink kernels.
- Native CPU extension: LogH/GBS, later slowdown/PN and secular paths. Batch API, native thread pool,
  GIL released, no access to mutable GPU buffers or global cluster state.
- Task-local state: particle ID/index lists, group membership, scratch, retry state and event records.
  Commit events and changes to topology at defined synchronization points.

FP64 is the first baseline. Reductions need not reproduce cpp-nbody bit for bit, but must meet independent
accuracy and conservation gates. Benchmark compensated or deterministic reductions against their cost.
Reject non-finite states and incomplete integrations. Never replace failed physics with a fabricated state.
Custom GPU code needs no derivative rule; do not spend effort keeping an XLA adjoint path.

## Milestone 0: executable correctness pilot (implemented locally)

- Explicit FP64 particle state, IDs and fixed subsystem membership.
- O(N)-working-memory XLA direct reference, tiled Pallas force and force-gradient kernels.
- Fourth-order fixed-step FSI and independently calculated small-N diagnostics.
- Standalone native Newtonian LogH/GBS for 2..8 bodies, COM normalization, finite/error/time checks.
- Python native-worker scheduler and fixed-membership hybrid FSI with overlap on/off.
- CPU tests including Pallas interpreter; separate real-GPU gate.
- Synchronized JSON benchmark runner with compilation/warm-up time reported separately.

Limits: this is not HHS, SDAR, PN, dynamic detection, mergers, stellar evolution, distributed execution
or an established production integrator. Hybrid member-source completion is still an XLA correctness
path; it must become a tiled GPU kernel before interpreting large-N results as competitive.

## Milestone 1: competitive single-GPU building blocks

1. Validate compiled Pallas on A100, then MI300A: awkward lengths, empty/padded sources, unequal masses,
   subsystem exclusions, softened field cases, unsoftened few-body cases and boosted frames.
2. Implement rectangular forces (selected sinks and source sets) with stable particle-ID exclusion;
   replace the hybrid XLA tail. Also benchmark force-gradient and pair-step kernels independently.
3. Port symmetrized free-fall/fly-by pair steps from cpp-nbody. Add gradient/perturbation criteria only
   with independent reference checks. Keep exact FP64 first; assess conservative FP32 rejection later.
4. Sweep sink/source tiles, source chunking and reduction order. Avoid N-by-N global intermediates.
   Test FP64 register pressure and rsqrt refinement rather than assuming a bandwidth bottleneck.
5. Capture a matched cpp-nbody kernel input/output corpus on the GPU machine with provenance.

Gate: small-case accuracy passes on actual hardware; force/gradient/timestep throughput is measured
at 2^10..2^18 particles and representative rectangular subsets. Include memory, compiled variants and
launch overhead. CPU interpreter success does not satisfy this gate.

## Milestone 2: hybrid CPU/GPU overlap

Start with the current fixed-membership pilot and measured encounter batches. Native CPU work is
already outside the JAX graph; do not make a blocking Python callback part of every GPU kernel chain.

Schedule per half-drift:

1. Gather only the required subsystem members into owned host task buffers.
2. Submit native CPU integrations and enqueue the field drift and field-field GPU force calculation.
3. Wait for member results only when member-dependent forces are required.
4. Upload completed member states; finish member-source and member-sink force terms.
5. Join before the complete-force gradient and kick.

The pilot supports fixed memberships and masses only. Mergers invalidate speculative work; a future
runner must defer mutation, then discard/recompute affected tasks using generation/version IDs. Never
use old member positions to bypass the join. Avoid buffer donation until ownership is proven.

Performance work:

- Replace per-group Python future submissions with a batched native API when overhead matters.
- Use reusable host staging and bounded buffers; test pinned memory if supported by the actual stack.
- Keep CPU worker budgeting separate from JAX dispatch and MPI progress.
- Schedule long CPU encounters first; track longest task, total work and retry tail.
- Batch small GPU tasks; select CPU execution for tiny independent work where measurements favor it.
- Do not assume separate JAX dispatches use concurrent GPU streams; prove the timeline.

Gate: overlap on/off produces equivalent physical updates within a documented tolerance, including
failure propagation. GPU trace shows actual concurrency, and complete step time decreases after
including staging, dispatch, joins and native CPU work. Test 0%, 10% and 50% binaries over three seeds.

## Milestone 3: adaptive HHS without Python recursion per encounter

Represent time bins and active subsets by fixed-capacity buffers, active counts and index lists. Use
a small measured capacity ladder, not recompilation for every exact population. Masks must exclude
padding from forces, timestep minima, physical validation and event processing.

Initially reproduce the sequential HHS/FSI schedule with explicit task records. Preserve all-slow
behavior and floor tests. Then execute independent S/F branches and connected components concurrently
between parent cross-kicks. Each task owns its native state, scratch and event output; the parent joins
and commits in canonical order. Do not advance a future step of the same particles using stale forces.

Gate: eccentric Kepler/planet order and round-off tests, cluster timestep sweeps, exactly-once subsystem
completion, no lost/duplicated IDs, and no unbounded compile-count growth during adaptive runs.
Compare complete 200k steps with cpp-nbody before proceeding to every physics module.

## Milestone 4: production encounters and physical events

Keep the pilot native implementation small. Decide whether to extract/refactor cpp-nbody's full
regularization module into a standalone library or maintain a separately validated port. Confirm
source/license provenance before vendoring. Do not transplant its global scheduler state.

Add in order: native slowdown/SDAR and secular binaries; dynamic neighbour detection and membership;
collision/merger events and energy/momentum/angular-momentum accounts; PN/spins; stellar evolution.
Each module gets independent physical observables. Booked Newtonian PN energy alone is not an accuracy
test. Test strong external perturbers and cluster/subsystem coupling convergence explicitly.

Native task result contract: stable IDs, final state, actual time reached, finite/convergence status,
retry/step counts and candidate events. Python accepts or raises; no best-effort continuation by default.

## Milestone 5: multi-GPU, restart and longer runs

Start from replicated particles and sink-distributed forces to match cpp-nbody. Use explicit ordered
collectives and rank-local task work; assign CPU subsystem tasks by cost and share results at the join,
not after every individual binary call. Test one/two/four ranks with identical initial particles.

Checkpoints include IDs, time-bin/task state, regularization carries, energy accounts and RNG state.
Require deterministic continuation within one backend; document cross-hardware rounding differences.
Do not infer cluster fidelity from a few successful steps: run through encounter-rich intervals and
compare energy drift, binary distributions, relaxation and merger observables.

## Measurement contract

Record revision, dependency versions, GPU/CPU, ranks/devices, precision, kernel/tile parameters, IC
hash, physical end time reached and effective config. Separate compile time, warmed median step time,
dispersion, full wall time, peak memory, transfers, CPU work/critical path, GPU active time and join waits.
Use device completion for timing, and Nsight/rocprof for actual overlap. Report numerical errors beside
speed. Compare both the stable reference and its fastest validated switches. No extrapolated scaling
curve is a measurement.

## Status and external prerequisites

The local pilot was tested using cached JAX 0.4.38 wheels on an Intel Mac CPU. This is a compatibility
and correctness environment, not the target GPU stack. Pin and record the GPU environment before
kernel tuning. Pallas/Triton APIs and backend support are version-sensitive.

Repository publication is pending GitHub connectivity/authentication. Intended name: Ulli1991/rumi;
private initially unless the user chooses public. No hosted repository is claimed until creation succeeds.

Official implementation references:
- https://docs.jax.dev/en/latest/async_dispatch.html
- https://docs.jax.dev/en/latest/control-flow.html
- https://docs.jax.dev/en/latest/pallas/quickstart.html
- https://docs.jax.dev/en/latest/ffi.html
