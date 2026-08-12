# Day-26 G3 closure checkpoint

- Date: 2026-08-09
- Work Order: `WO-GR-G3-004`
- Package: `artifacts/day26-g3-identifiability-freeze-v2/`
- Package status: `COMPLETE_G3_4_PARAMETER_ELIGIBILITY_AUDIT`

## Scope and completed boundary

This checkpoint records the reviewed G3.4 parameter-eligibility package and the completed G3
diagnostic chain for `cf21B_500`. The retained execution used exactly 24 ordered Train/Validation
parents, six fixed contiguous batches of four, full six-second carry, the unchanged Loss v1,
seven declared continuous parameter coordinates, and zero optimizer initializations/updates.

The six-by-four mode was selected prospectively by D-073 after the observed local primary-24 OOM.
It was not selected dynamically from available memory and was not a runtime fallback. G4 was not
executed. The prospective G4 batch contract remains `WITHHELD_PENDING_REBENCHMARK`.

## Reviewed result

| Parameter | Final G3.4 disposition | Reason or boundary |
|---|---|---|
| `kp_xy` | `GO_FOR_SINGLE_PARAMETER_OPTIMIZATION` | All objective-gradient, detectability, stability, conditioning and technical gates passed |
| `kp_z` | `GO_FOR_SINGLE_PARAMETER_OPTIMIZATION` | All objective-gradient, detectability, stability, conditioning and technical gates passed |
| `ki_z` | `GO_FOR_SINGLE_PARAMETER_OPTIMIZATION` | Individually eligible; not needed in the smallest qualifying joint subset |
| `kd_xy` | `DIAGNOSTIC_ONLY` | Not individually GO-eligible |
| `kd_z` | `WITHHELD_TECHNICAL_ROBUSTNESS` | `NONFINITE_NEAR_LIMIT_EFFECT_ROLLOUT` |
| `mass` | `WITHHELD_TECHNICAL_ROBUSTNESS` | `NONFINITE_NEAR_LIMIT_EFFECT_ROLLOUT` |
| `mass_thrust` | `WITHHELD_TECHNICAL_ROBUSTNESS` | `NONFINITE_NEAR_LIMIT_EFFECT_ROLLOUT`; simulation relaxation only |

The permitted Near-limit nonfinite records retain finite path/index/shape/count metadata and do
not attribute causality to a controller, allocator, dynamics operation, mathematical singularity,
calibration, firmware, or hardware. No finite feasible/default/objective/Reverse/FD failure was
converted into a pass.

The optimizer-objective contract contains exactly the seven scored-window Loss-v1 outputs. The
other 136 finite features retain full AD/FD comparison and disclosure as
`DIAGNOSTIC_AD_NOT_OPTIMIZER_CONTRACT`; those values are never represented as optimizer
gradients. The inactive integral candidate remains excluded, and the Loss recommendation is
exactly `RETAIN_LOSS_V1_PLUS_EXTERNAL_GATES`.

The package records `PROPOSE_G4_PARAMETER_FREEZE_FOR_HAUPTLEITUNG_REVIEW` for the joint pair
`kp_xy + kp_z`. This is a proposal for separate review only. It is not candidate selection,
optimization authority, or permission to start G4.

## Package integrity and reproduction

The final directory contains exactly 13 regular mode-`600` files totaling `31,722,155` bytes.
`SHA256SUMS` indexes and verifies the other 12 files. The canonical complete 13-file hash-list pin
is:

```text
57934a75f9fcbfec5480184405170fd87b6ff82b8c0cf942e1b55af82f87dbfe
```

The exact inventory and every file SHA-256 are recorded in
[`ARTIFACT_MANIFEST.md`](../ARTIFACT_MANIFEST.md#day-26-g34-parameter-eligibility-package).
Package A, Package B and Final are byte-identical, including the PNG payloads and strict semantic
JSON. The complete post-package regression passed `174` tests; post-Ruff lint and check-only
format gates passed. Optimizer initialization/update counts remained `0/0` throughout.

Reproduction authority is generator commit
`5c28db1aae6bee67f06a2764f535f5c38924495e` with subject
`research: support g3.4 recovery provenance`. The later compatibility commit
`5994f49a342b14e3b0fa47117f26488a0c11a0b4`, subject
`research: stabilize robust generator on extended history`, is separately identified and must
never be supplied or reported as the package generator. The exact detached-checkout CPU command,
hash-list verification and stderr boundary are in
[`RUNBOOK.md`](../RUNBOOK.md#day-26-g34-freeze-package-verification-and-reproduction).

The only accepted nonempty startup telemetry for a JAX-starting reproduction is the reviewed
111-byte JAX NVIDIA/CUDA CPU-fallback notice, SHA-256
`61d6836bb62ae5de8df26847232e5d1848fa056fa22a93506765685f242cd82d`,
and only when backend `cpu` is independently proven and the target otherwise exits zero. It is
external environment telemetry, not package or scientific data. Every other stderr byte is a
failure.

## Artifact/documentation commit boundary

The prospective artifact/documentation commit has parent
`5994f49a342b14e3b0fa47117f26488a0c11a0b4`, subject
`research: record g3.2 reverse-mode freeze evidence`, and exactly these 17 paths:

1. `artifacts/day26-g3-identifiability-freeze-v2/SHA256SUMS`
2. `artifacts/day26-g3-identifiability-freeze-v2/TECHNICAL_HANDOFF.md`
3. `artifacts/day26-g3-identifiability-freeze-v2/g3_freeze_overview.png`
4. `artifacts/day26-g3-identifiability-freeze-v2/g3_freeze_report.json`
5. `artifacts/day26-g3-identifiability-freeze-v2/g4_g5_freeze_proposal.json`
6. `artifacts/day26-g3-identifiability-freeze-v2/loss_diagnostics.json`
7. `artifacts/day26-g3-identifiability-freeze-v2/parameter_semantics.json`
8. `artifacts/day26-g3-identifiability-freeze-v2/parameter_sensitivity_matrix.png`
9. `artifacts/day26-g3-identifiability-freeze-v2/provenance.json`
10. `artifacts/day26-g3-identifiability-freeze-v2/sensitivity_conditioning.json`
11. `artifacts/day26-g3-identifiability-freeze-v2/train_manifest.json`
12. `artifacts/day26-g3-identifiability-freeze-v2/trajectory_contract.json`
13. `artifacts/day26-g3-identifiability-freeze-v2/validation_manifest.json`
14. `docs/research/ARTIFACT_MANIFEST.md`
15. `docs/research/PROJECT_STATE.md`
16. `docs/research/RUNBOOK.md`
17. `docs/research/checkpoints/DAY26_G3_2_CHECKPOINT.md`

No other product, artifact, documentation, config, dependency, lockfile, controller, loss,
trajectory, Test, protected primary-data or coordination path belongs to that commit.

## Claim boundary and next decision

This checkpoint closes only the G3 simulation diagnostic. It establishes no optimized parameter,
improvement, generalization, Test result, physical calibration, integer/firmware
differentiability, controller superiority, hardware transfer, safety, flight readiness, or
Sim2Real performance. G4 remains `WITHHELD_PENDING_REBENCHMARK` and requires a separate archived
Hauptleitung decision; it must not autostart.
