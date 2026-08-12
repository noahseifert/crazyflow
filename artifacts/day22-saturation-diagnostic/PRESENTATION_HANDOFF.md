# Saturation-explicit presentation handoff

## Fixed status and gate

Seed 05 is shown only with status `DIAGNOSTIC_ONLY_SATURATION_PRESENT`. The unchanged zero-saturation acceptance gate is
`FAIL_ZERO_MOTOR_SATURATION_GATE`. This package is neither candidate acceptance nor flight readiness.

| Figure | What it proves | What it does not prove | One-sentence slide takeaway |
|---|---|---|---|
| `trajectory_error_saturation.png` | Default and Seed 05 use the identical frozen cf2x_L250 H100 simulation contract; reference, actual trajectory and time-resolved error are shown, with every Seed-05 saturation sample marked. | Superiority, robustness, Test performance, convergence, flight readiness, `cf21B_500`, firmware or hardware transfer. | Seed 05 tracks this frozen simulation case descriptively, but `11` of `400` motor samples saturate, so it remains diagnostic only. |
| `motor_saturation_detail.png` | Commands, lower/upper thresholds and the exact per-motor saturation intervals are visible for both variants. | That brief saturation is acceptable, safe, or removable without a new scientific decision. | The unchanged gate fails because Seed-05 saturation is explicitly attributable per motor and time. |

## Exact saturation facts

- Default: `0/400` commands,
  fraction `0.0`.
- Seed 05: `11/400`
  commands, fraction `0.027499999850988388`; `M0=11/100, M1=0/100, M2=0/100, M3=0/100`.
- Lower and upper decisions are retained separately. Exact indices, state times and interval
  boundaries are in `saturation_diagnostic.json`.

## Exact verbal caveat for Noah

> Derselbe Simulationsfall wurde mit Default und Seed 05 ausgeführt. Seed 05 wird nur diagnostisch gezeigt, weil Motorsättigung das unveränderte Akzeptanzgate verletzt; der Satz ist weder finaler noch flugbereiter Kandidat und belegt nichts für `cf21B_500` oder Hardware. Gemessen wurden Default `0/400` und Seed 05 `11/400` sättigende Motor-Samples.

## Claim boundary

Simulation diagnostic only. Descriptive tracking metrics are inseparable from status `DIAGNOSTIC_ONLY_SATURATION_PRESENT`.
No Test, tuning, candidate acceptance, superiority, safety, firmware, hardware, real-flight or
Sim2Real claim is made.
