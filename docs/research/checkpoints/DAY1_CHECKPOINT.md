# Day-1 checkpoint

Date: 2026-07-24 (Europe/Berlin)  
Repository: `/home/noah3/bachelorarbeit/crazyflow-gradient-research`  
Branch: `research/differentiable-mellinger`  
HEAD: `1aa648b15e9e9ed9bada8aaa7408cdff22d417fd` (`Add differentiable Mellinger tracking rollout`)

## Repository state

Preflight `git status --short --branch`:

```text
## research/differentiable-mellinger
?? artifacts/
```

There were no tracked modifications and no pre-existing `AGENTS.md`. The untracked artifact tree
was treated as user-owned and preserved. No dependency or lockfile was installed or updated.

This checkpoint adds only root/research documentation, targeted ignore rules, and new generated
evidence. The final status is recorded after verification in the closing section.

## Actual Day-1 files

- `crazyflow/trajectory.py`
- `crazyflow/control/mellinger/tracking.py`
- `crazyflow/control/mellinger/__init__.py`
- `examples/jax/mellinger_tracking.py`
- `tests/unit/test_mellinger_tracking.py`
- `docs/user-guide/mellinger-gradient-rollout.md`

All six were read completely. None was modified during this audit.

## Static audit result

The committed path is:

```text
Trajectory(T+1)
  -> reference[1:] and state commands (T, 13)
  -> broadcast (T, N, M, 13)
  -> Mellinger state -> attitude -> force/torque -> rotor RPM
  -> first-principles dynamics, 5 ticks per command
  -> outer lax.scan with SimData carry
  -> normalized scalar tracking loss
  -> value_and_grad over raw kp_xy, kp_z, kd_xy, kd_z
```

The audited defaults are `N=M=1`, 500 Hz dynamics, 100 Hz state control, five dynamics ticks per
command, and horizons 20/200. Gains use bounded logistic transforms. Loss weights and non-smooth
branches are catalogued in [`../PROJECT_STATE.md`](../PROJECT_STATE.md).

Two pre-existing prose mismatches were found without changing source: acceleration setpoints are
actually consumed by `state2attitude`; the inspected nonpositive-thrust gate does not explicitly
reset controller integrators.

## Executed verification commands

All Python commands used the repository-local interpreter explicitly.

```bash
.venv/bin/python --version
.venv/bin/python -m pip --version
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q tests/unit/test_mellinger_tracking.py
.venv/bin/python -m ruff --version
.venv/bin/python -m ruff check crazyflow/trajectory.py crazyflow/control/mellinger/tracking.py crazyflow/control/mellinger/__init__.py examples/jax/mellinger_tracking.py tests/unit/test_mellinger_tracking.py
.venv/bin/python -m ruff format --check crazyflow/trajectory.py crazyflow/control/mellinger/tracking.py crazyflow/control/mellinger/__init__.py examples/jax/mellinger_tracking.py tests/unit/test_mellinger_tracking.py
```

The main example was executed four times with the exact commands in
[`../RUNBOOK.md`](../RUNBOOK.md): two H20 fixed-seed smoke runs in separate directories, an H200
Figure-8 run, and an H200 circle run. Plot hashes and scientific JSON payloads from the two smoke
runs were compared.

## Test and numerical results

- Focused test: `7 passed in 11.44s`.
- Ruff lint: `All checks passed!`.
- Ruff format: `5 files already formatted`.
- Environment integrity: `No broken requirements found`.

| Run | Loss | Gradient norm | Max directional relative error | All validations |
|---|---:|---:|---:|---|
| Figure-8 H20, seed 20260724 | `0.0022911166306585073` | `0.0001269973727175966` | `0.01853998377919197` | true |
| Figure-8 H200, seed 20260724 | `0.07310394197702408` | `0.05602377653121948` | `0.00015475136751774698` | true |
| Circle H200, seed 20260724 | `0.02409282885491848` | `0.014793669804930687` | `0.007185550406575203` | true |

H20 gradient leaves:

```text
kp_xy=-3.962873051932547e-06
kp_z =-7.611652108607814e-06
kd_xy=-7.964330143295228e-05
kd_z =-9.854762902250513e-05
```

H200 Figure-8 gradient leaves:

```text
kp_xy= 0.020742792636156082
kp_z =-0.010739986784756184
kd_xy=-0.050698306411504745
kd_z =-0.0047681089490652084
```

H200 circle gradient leaves:

```text
kp_xy= 0.00048197622527368367
kp_z =-0.009044521488249302
kd_xy=-0.011416671797633171
kd_z =-0.002544915769249201
```

Every loss and gradient leaf was finite. All runs passed nontrivial-gradient, deterministic replay,
JIT/eager match, three directional derivative checks, and negative-gradient descent checks. The
two independent H20 runs had equal scientific JSON payloads after excluding only timing fields and
byte-identical plots with SHA-256
`b97c113ac6102c1a2e2ce62d7afb2ddc0c56feb21dd00d4e35f69ff4ae195b8f`.

## Generated audit artifacts

- `artifacts/day1-audit/smoke-run-1/figure8_result.json`
- `artifacts/day1-audit/smoke-run-1/figure8_tracking.png`
- `artifacts/day1-audit/smoke-run-2/figure8_result.json`
- `artifacts/day1-audit/smoke-run-2/figure8_tracking.png`
- `artifacts/day1-audit/figure8-h200/figure8_result.json`
- `artifacts/day1-audit/figure8-h200/figure8_tracking.png`
- `artifacts/day1-audit/circle-h200/circle_result.json`
- `artifacts/day1-audit/circle-h200/circle_tracking.png`

All older artifact files remained in place and were not overwritten.

## Gitignore review

Existing coverage for `**/__pycache__/`, `.pytest_cache`, `.venv`, `build`, `dist`, and
`*.egg-info` was retained. The file was not replaced. The following targeted additions were made:

- `*.py[cod]` covers stray compiled Python files outside `__pycache__`.
- `.ruff_cache/` and `.mypy_cache/` cover standard local quality-tool caches.
- `.jax_cache/`, `.jax-cache/`, `jax_cache/`, and `.xla_cache/` cover project-local temporary
  JAX/XLA compilation or benchmark caches without hiding scientific outputs.
- `!artifacts/**/*.json` reverses the pre-existing broad `*.json` rule specifically for
  machine-readable scientific evidence under `artifacts/`.

No rule ignores the complete `artifacts/` directory.

## Warnings and non-blocking limitations

- Optional `warp` and `mujoco_warp` imports are unavailable; CPU runs still exited successfully.
- Pip reported its user cache as unwritable but found no broken requirements.
- `jq` was unavailable for one inspection attempt; no package was installed, and JSON evidence was
  inspected using the existing repository-local Python environment.
- An initial version probe assumed `ruff.__version__`, which that module does not expose; the
  supported `.venv/bin/python -m ruff --version` command succeeded.
- No full test suite, GPU run, hardware run, multi-world run, disturbance study, or long-horizon
  memory assessment is claimed by this checkpoint.

## Open risks

- Controller/dynamics mass mismatch (`0.029 kg` versus `0.0319 kg`) is preserved.
- Passing trajectories do not activate saturation, floor clipping, or nonpositive-thrust gates.
- Piecewise controller/dynamics branches can produce zero or discontinuous local sensitivities.
- Long-horizon reverse-mode memory has not been characterized.
- Existing historical artifacts identify an earlier commit/source state; new audit records identify
  the committed Day-1 implementation at the current HEAD.

## Prerequisites for Day 2

- Project lead confirms the Day-2 experiment matrix and acceptance criteria.
- Decide explicitly whether mass matching is an experimental factor.
- Preserve this checkpoint and use fresh artifact output directories.
- Re-run the focused test from the exact commit used for new experiments.
- Update all four research status documents and create a Day-2 checkpoint after that sprint.

No Day-2 implementation was started.

## Closing repository status

Post-documentation `git status --short --branch`:

```text
## research/differentiable-mellinger
 M .gitignore
?? AGENTS.md
?? artifacts/
?? docs/research/
```

Post-documentation `git diff --stat`:

```text
 .gitignore | 13 +++++++++++++
 1 file changed, 13 insertions(+)
```

Git does not include untracked files in `git diff --stat`; the five new research Markdown files
(including this checkpoint), root `AGENTS.md`, and artifact files are therefore visible in
`git status` but not in that statistic. No Day-1 source or test file is modified.
