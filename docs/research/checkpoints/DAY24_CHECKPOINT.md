# Day 24 checkpoint: diagnostic-local unclipped two-stage allocation replay

## Boundary

- Work Order: `WO-GR-F0-003`
- Base: `bcc8b0d57c9f88dd1279ec94a05e14b45776ed0b`
- Generator/test commit: `728e8b9f89e4f92c7fa4eefee9808a82904975c2`
- Platform/case: exact accepted Day-22 `cf2x_L250` Default/Seed-05 Figure-8 replay
- Sampling: 500-Hz simulation, 100-Hz state control, five substeps per interval, 100 intervals
- Stored diagnostic sample: fifth 500-Hz substep only
- Scope: diagnostic-local two-stage capture; no controller, allocator, bound, gain or default change
- Protected Day 10 / Day 13 / Test reads: zero
- Analyzer/F0.1 CLI execution: zero

This is a worker review candidate pending independent Gradient-Lead acceptance.

## Replay identity and finiteness

The local wrapper executes the unchanged six-stage production pipeline. All nine custom trace
fields are shape/dtype/array-identical to a separate production replay for Default and Seed 05.
Production motor commands, tracking arrays, tracking metrics and Day-22 lower/upper decisions are
array-identical to accepted Day 22. The local Stage-A postclip wrench and Stage-B command satisfy
the predeclared `64*float32_eps*max(scale,abs(lhs),abs(rhs))`, `rtol=0` contract. Every array is
finite and every Stage-A preclip quadratic speed-inverse discriminant is nonnegative.

## Exact two-stage result

- Default: Stage-A torque clip `0/300`, motor clip `0/400`, Stage-B additional clip `0/400`.
- Seed 05: Stage-A torque clip `0/300`; motor clip `11/400`, only motor 0 upper, indices 48–58,
  interval `[0.48,0.59) s`; Stage-B additional clip `0/400`.
- Seed-05 maximum motor-0 request at index 54/state time 0.55 s: `74753.53125 PWM`,
  `0.13687989115715027 N`, `24716.875` in the existing Crazyflow convention.
- Corresponding upper exceedance: `9218.53125 PWM`, `0.016879891976714134 N`,
  `1501.0977783203125`.
- Maximum absolute Stage-A wrench distortion at index 54: collective
  `0.016879886388778687 N`, roll `0.0005491026677191257 N m`, pitch
  `0.0005491029005497694 N m`, yaw `0.0001240818528458476 N m`.
- Stage B creates zero additional wrench distortion.
- Hover motor speed is `18967.771484375`, fraction of configured maximum
  `0.8170207356628263`; calibrated thrust fraction is `0.6519561779191235`.

The Day-22 `1e-4` value remains a classification detection threshold only. Every exceedance is
`max(raw-upper,0)` or `max(lower-raw,0)` without threshold subtraction.

## DATA-AUDIT-01

Only `artifacts/day19-h100-analysis/` derivative files and analyzer source were read. Its checksum
index hash is `fe607e7ee7b129e6e9741d2cc62e40625080b408388fcf4d584aca47789b8b17` and 3/3 checks pass.
The derivative records ten seeds, each with 5,000 Train, 5,000 Validation and zero Test rows;
100,000 Train/Validation and zero Test rows total; every per-seed and panel
`motor_saturation_fraction` maximum is `0.0`. Source inspection confirms Test-row rejection,
complete alternating Train/Validation order, per-seed maxima across metric rows and then the panel
maximum. The analyzer was not executed. This proves no registered stored-objective saturation
only, not motor traces, reserve or allocation distortion.

## Verification

- Focused prescribed Pytest: `92 passed`.
- Ruff lint: PASS.
- Ruff format check: PASS.
- Day-19/21/22/23 checksum sets: PASS before generation.
- Day-24 checksum index: 4/4 PASS; index SHA-256
  `f75e4d17bfcef14a56235eab91ac395904e340ead24936338007ce88dbc2f51a`.
- Two fresh `/tmp` packages: 4/4 each and byte-identical to each other and final package.
- Narrow correction: machine diagnostic JSON remained byte-identical at
  `5a7900398c177ef15b533e8dd40cb36ab709b4450671db2af8c574a0b064ef52`; provenance changed with
  the amended generator commit, and PNG/handoff/checksum index changed for the figure contract.
- Figure: inspected at original 2240×1600 after correction; motor-0 unclipped/clipped demand and
  true limit, all four motors' fixed-window pre/post commands, signed wrench distortion, exact
  maximum/count/status text and the simulation-only caveat are legible.
- Source/test and evidence/documentation are split into exactly two linear worker commits.
- Live checkout and its three protected untracked roots remained unchanged; no push/integration.

The five package files total 1,943,983 bytes.

## Claim boundary

Status remains `DIAGNOSTIC_ONLY_SATURATION_PRESENT`; the unchanged zero-saturation gate remains
`FAIL_ZERO_MOTOR_SATURATION_GATE`. This is pure simulation evidence for one frozen case, not
candidate acceptance, improvement, superiority, safety, Test, firmware, hardware, flight,
`cf21B_500` or Sim2Real evidence.
