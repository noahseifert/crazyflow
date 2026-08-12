# Day 23 checkpoint: vertical ki_z excitation and sensitivity pilot

## Boundary

- Work Order: `WO-GR-S2-002`
- Base: `853b3d738233436d438d821e50198d9742965697`
- Generator/test commit: `beb59393280b6b515c328edce13c430ebfbaa826`
- Platform: `cf2x_L250`
- Scope: `ki_z` only; no optimizer, training, tuning or default change
- Horizon: 500 state-control intervals / 5.0 s
- Frequencies: 500-Hz dynamics, 100-Hz state control, 500-Hz attitude/force-torque
- Reference: constant `[0,0,0.75] m`, zero velocity/acceleration/yaw
- Cases: matched controller/dynamics mass 0.0319 kg and audited 0.029/0.0319-kg mismatch
- Day 10 / Day 13 / Test / accepted Day-20/21/22 reads: zero
- Friday presentation dependency: none; backup/next-step evidence only

This is a worker review candidate pending independent Gradient-Lead acceptance.

## Frozen gain and threshold contract

The unchanged Stage-2 Registry bounds are `[0.025,0.10] N/(m s)`. Raw Lower/Default/Upper are
`-0.9431471824645996`, `-0.6931471824645996`, and `-0.4431471824645996`, mapping to approximately
`0.0460198782`, `0.0500000007`, and `0.0543243475 N/(m s)`. Only raw `ki_z` changes. All
excitation, clip, saturation, motor-margin, roundoff, AD/FD, nontrivial-sensitivity,
bias-contrast and transform-conditioning thresholds were encoded before the first result.

## Result

All six Lower/Default/Upper rollouts are finite and pass the technical gates:

- motor saturation: exactly zero;
- zero thrust, floor/ground and nonfinite fractions: exactly zero;
- minimum normalized motor-bound margin: at least `0.24099181592464447`;
- maximum absolute integral state: at most `0.1349191665649414 m s`, below the predeclared
  `0.36 m s` clip-margin threshold.

Default mismatch integral excitation is `0.1340937316417694 m s`; matched is
`0.03675120696425438 m s`, yielding contrast `0.09734252467751503 m s`. Mismatch JAX gradient is
`-0.0007489713025279343` and central FD is `-0.0007464457303285599`. Matched values are
`-0.000056265325838467106` and `-0.00005596131086349487`. Both AD/FD gates pass. Every
predeclared scientific gate passes, with no failed criterion.

Binary worker recommendation: `GO_FOR_FUTURE_KI_Z_OPTIMIZATION`.

## Verification

- Focused Pytest: 45 tests passed.
- Ruff lint: PASS.
- Ruff format check: PASS.
- `git diff --check`: PASS before evidence commit.
- Day-23 checksum index: 4/4 PASS.
- Two fresh packages: byte-identical to each other and the final package.
- Figure: inspected at original 2160×1440 resolution; title, status, traces, axes, legends,
  excitation threshold and sensitivity points/tangents are legible.
- Source/test and evidence/documentation are split into exactly two linear worker commits.
- Live checkout and its three protected untracked roots remained unchanged; no push/integration.

The five files total 4,935,078 bytes. `SHA256SUMS` hashes to
`2898301f8668aed1cfe52ca79787aaa59957235bea6b922f5908ab7e61a40738`.

## Claim boundary

`GO_FOR_FUTURE_KI_Z_OPTIMIZATION` only supports considering a future, separately authorized
`ki_z` optimization Work Order. It does not prove optimization, improvement, superiority,
convergence, robustness, candidate validity, hardware transfer, flight readiness or safety.
