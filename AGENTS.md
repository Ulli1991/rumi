# Rumi

Raw speed at controlled numerical error is the goal. Differentiability is not a requirement.

- Keep Python as the readable orchestration layer; compile GPU work and native CPU encounters.
- Do not materialize N-by-N pair arrays in production kernels.
- Do not replace a failed integration with a plausible state. Reject non-finite values and incomplete advances.
- Measure complete steps, CPU/GPU overlap, transfers, and numerical error on named hardware.
- CPU interpreter results do not establish GPU correctness or speed.
- Keep cpp-nbody untouched as the numerical and performance reference.
- No claim of parity until a matched physical run at matched accuracy supports it.
- Runtime physics state belongs to a task, not module globals.
