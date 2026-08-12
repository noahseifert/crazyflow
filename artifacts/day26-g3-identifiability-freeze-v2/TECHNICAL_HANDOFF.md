# Day 26 technical handoff: G3.4 parameter eligibility audit

## Status

`COMPLETE_G3_4_PARAMETER_ELIGIBILITY_AUDIT` on `cf21B_500`. Exactly 24 Train/Validation parents, full six-second carry, unchanged Loss v1, and zero optimizer initializations/updates. No G4 or G5 run occurred. Withheld parameter reasons: `kd_z` = `WITHHELD_TECHNICAL_ROBUSTNESS`; `mass` = `WITHHELD_TECHNICAL_ROBUSTNESS`; `mass_thrust` = `WITHHELD_TECHNICAL_ROBUSTNESS`.

## Required focused sequence

1. Static leaf/dtype and direct state-controller parity: PASS.
2. Two-parent full default parity across controller, allocation, rollout, loss, features, and counters: PASS with exact array equality.
3. All seven log-coordinates pass scalar/vector Reverse/FD at `h=0.01` on both frozen focus parents: finite, sign-consistent when significant, and within tolerance.
4. Separate `DISCRETE_INTEGER_RESPONSE` at ±0.01 and ±0.05: reported without an AD or GO label. It neither replaces nor validates the continuous reverse-mode relaxation.

## Continuous classifications

- `kp_xy`: `GO_FOR_SINGLE_PARAMETER_OPTIMIZATION`
- `kp_z`: `GO_FOR_SINGLE_PARAMETER_OPTIMIZATION`
- `kd_xy`: `DIAGNOSTIC_ONLY`
- `kd_z`: `WITHHELD`
- `ki_z`: `GO_FOR_SINGLE_PARAMETER_OPTIMIZATION`
- `mass`: `WITHHELD`
- `mass_thrust`: `WITHHELD`

The seven Loss-v1 objective outputs alone carry the optimizer-gradient contract. The other 136 finite outputs retain complete AD comparisons as `DIAGNOSTIC_AD_NOT_OPTIMIZER_CONTRACT`; their mismatches are disclosed without an optimizer-gradient or causal claim. Objective-gradient and technical-withholding reasons remain separate and cumulative. A technical record identifies only nonfinite full-effect-rollout leaf and array-index metadata and makes no causal, controller-operation, singularity, firmware, or hardware claim.

`mass` and `mass_thrust` remain effective controller parameters. The latter category applies only to its simulation relaxation and is not an integer, calibration, or firmware claim.

## Inactive integral candidate

Recommendation: `RETAIN_LOSS_V1_PLUS_EXTERNAL_GATES`. Its active weight is exactly zero and it was not added to the objective.

## Prospective freeze only

Status: `PROPOSE_G4_PARAMETER_FREEZE_FOR_HAUPTLEITUNG_REVIEW`; proposed parameters: `kp_xy, kp_z`. This is a Hauptleitung review proposal, not candidate selection or permission to start G4.

## Reproduction

Generator commit: `5c28db1aae6bee67f06a2764f535f5c38924495e`. D-073 prospectively selected six contiguous four-parent batches after the observed local primary-24 OOM. This is not a runtime fallback; the prospective 32-world G4 batch freeze is WITHHELD_PENDING_REBENCHMARK. Volatile time/RSS observations remain only in `/tmp` process manifests. Required command:

```bash
PYTHONDONTWRITEBYTECODE=1 SCIPY_ARRAY_API=1 JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=false JAX_ENABLE_COMPILATION_CACHE=false /home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python examples/jax/mellinger_g3_freeze_v2.py --generate-package --generator-commit 5c28db1aae6bee67f06a2764f535f5c38924495e --execution-mode six-batches-of-four --batch-timeout-seconds 900 --total-timeout-seconds 2400 --output-dir <NEW_OUTPUT_DIR>
```

## Claim boundary

Pure JAX Float32 CPU simulation evidence. No optimized parameter, improvement, Test result, physical calibration, integer/firmware differentiability, hardware transfer, safety, flight readiness, or Sim2Real claim.
