# Presentation handoff — vertical ki_z sensitivity pilot

- Status: `GO_FOR_FUTURE_KI_Z_OPTIMIZATION`
- Evidence label: `DIAGNOSTIC_ONLY_NOT_OPTIMIZATION`
- Presentation role: backup/next-step evidence only; the Friday deck is independent.

## What the figure proves

For the one frozen 5 s `cf2x_L250` simulation hold, it shows the matched-mass reference and the audited 0.029 kg controller versus 0.0319 kg dynamics mismatch, the default integral response, and the predeclared local Loss-v1 sensitivity check for `ki_z`. All six Lower/Default/Upper rollouts are finite, saturation-free, clear the motor/integrator/ground gates, and both AD/central-FD checks pass.

## Exact evidence

- Mismatch default maximum absolute integral state: `0.1340937316 m s`.
- Matched default maximum absolute integral state: `0.03675120696 m s`.
- Mismatch local `d loss_v1 / d raw_ki_z`: `-0.0007489713025`.
- Failed predeclared scientific criteria: `none`.

## What it does not prove

No optimizer step occurred. This does not prove improvement, superiority, convergence, robustness, hardware transfer, flight readiness, or that a future optimization will succeed. It changes no default, registry bound, loss, platform, firmware, or accepted Friday artifact.

## One-sentence backup-slide takeaway

`GO_FOR_FUTURE_KI_Z_OPTIMIZATION: the frozen vertical mass-bias hold meets every predeclared ki_z identifiability threshold, with zero saturation; this is a diagnostic authorization signal only, not an optimized result.`

## Exact verbal caveat

“This is one deterministic simulation sensitivity pilot under the known cf2x_L250 mass mismatch; it contains no optimization and makes no flight or hardware-transfer claim.”
