# Differentiable Mellinger runbook

This is the authoritative command source for the differentiable Mellinger tracking work.
The scientific Day-1/Day-2/Day-3 commands were verified on 2026-07-24 under WSL. The Sprint-4
freeze/audit verification was executed on 2026-07-26 from the same repository root.

## Repository and environment

Required working directory for every command:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
```

Verify the existing repository-local environment without changing it:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
test -x .venv/bin/python
.venv/bin/python --version
.venv/bin/python -m pip --version
.venv/bin/python -m pip check
```

Expected environment: Python 3.12.13, JAX 0.10.1, NumPy 2.5.1, pytest 9.1.1, and Ruff 0.16.0.
`pip check` must report `No broken requirements found`.

Do not rely on shell activation. Optional interactive activation is:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
source .venv/bin/activate
```

Even after activation, automated commands must use `.venv/bin/python`,
`.venv/bin/python -m pytest`, and `.venv/bin/python -m pip`; never use unqualified `python`,
`pytest`, or `pip`.

`MPLCONFIGDIR` below avoids writes to an unavailable home-level Matplotlib cache. It changes no
dependency and no scientific input.

## Focused Day-1 test

Working directory: repository root.

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
.venv/bin/python -m pytest -q tests/unit/test_mellinger_tracking.py
```

Expected: `7 passed`. This covers trajectory derivatives and shapes, command batching, real
Crazyflow rollout shapes, finite/nontrivial gradients, presence of `scan`, JIT/eager determinism,
directional finite differences, and local descent.

## Smoke run

Working directory: repository root.

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
MPLCONFIGDIR=/tmp/crazyflow-gradient-research-matplotlib \
  .venv/bin/python examples/jax/mellinger_tracking.py \
  --trajectory figure8 \
  --horizon 20 \
  --control-freq 100 \
  --sim-freq 500 \
  --seed 20260724 \
  --output-dir artifacts/day1-audit/smoke-run-1
```

Expected outputs:

- `artifacts/day1-audit/smoke-run-1/figure8_result.json`
- `artifacts/day1-audit/smoke-run-1/figure8_tracking.png`

Expected audited values: loss `0.0022911166306585073`, gradient norm
`0.0001269973727175966`, and `all_validations_passed: true`.

The verified path already contains evidence. To preserve it during a future run, supply a new
`--output-dir`; the program intentionally writes fixed result filenames within that directory.

## Reproducibility check

Repeat the same seed in a distinct output directory:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
MPLCONFIGDIR=/tmp/crazyflow-gradient-research-matplotlib \
  .venv/bin/python examples/jax/mellinger_tracking.py \
  --trajectory figure8 \
  --horizon 20 \
  --control-freq 100 \
  --sim-freq 500 \
  --seed 20260724 \
  --output-dir artifacts/day1-audit/smoke-run-2
```

Compare plots byte-for-byte:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
sha256sum \
  artifacts/day1-audit/smoke-run-1/figure8_tracking.png \
  artifacts/day1-audit/smoke-run-2/figure8_tracking.png
cmp -s \
  artifacts/day1-audit/smoke-run-1/figure8_tracking.png \
  artifacts/day1-audit/smoke-run-2/figure8_tracking.png
```

Compare the scientific JSON payload while excluding only wall-clock timing fields:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
.venv/bin/python -c "import json; from pathlib import Path; a=json.loads(Path('artifacts/day1-audit/smoke-run-1/figure8_result.json').read_text()); b=json.loads(Path('artifacts/day1-audit/smoke-run-2/figure8_result.json').read_text()); ignored={'compile_and_first_run_seconds','cached_forward_backward_seconds'}; print('scientific_payload_equal=' + str({k:v for k,v in a.items() if k not in ignored} == {k:v for k,v in b.items() if k not in ignored}))"
```

Expected: identical PNG SHA-256 values, successful `cmp`, and
`scientific_payload_equal=True`. Timing is deliberately excluded because it is not deterministic.
The audited PNG hash is
`b97c113ac6102c1a2e2ce62d7afb2ddc0c56feb21dd00d4e35f69ff4ae195b8f`.

## Full Day-1 runs

Figure-eight, 200 control intervals:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
MPLCONFIGDIR=/tmp/crazyflow-gradient-research-matplotlib \
  .venv/bin/python examples/jax/mellinger_tracking.py \
  --trajectory figure8 \
  --horizon 200 \
  --control-freq 100 \
  --sim-freq 500 \
  --seed 20260724 \
  --output-dir artifacts/day1-audit/figure8-h200
```

Expected outputs:

- `artifacts/day1-audit/figure8-h200/figure8_result.json`
- `artifacts/day1-audit/figure8-h200/figure8_tracking.png`

Expected audited values: loss `0.07310394197702408`, gradient norm `0.05602377653121948`,
maximum directional relative error `0.00015475136751774698`, all validations true.

Circle, 200 control intervals:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
MPLCONFIGDIR=/tmp/crazyflow-gradient-research-matplotlib \
  .venv/bin/python examples/jax/mellinger_tracking.py \
  --trajectory circle \
  --horizon 200 \
  --control-freq 100 \
  --sim-freq 500 \
  --seed 20260724 \
  --output-dir artifacts/day1-audit/circle-h200
```

Expected outputs:

- `artifacts/day1-audit/circle-h200/circle_result.json`
- `artifacts/day1-audit/circle-h200/circle_tracking.png`

Expected audited values: loss `0.02409282885491848`, gradient norm `0.014793669804930687`,
maximum directional relative error `0.007185550406575203`, all validations true.

## Relevant quality checks

Working directory: repository root. These commands inspect but do not rewrite the Day-1 Python
files.

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
.venv/bin/python -m ruff check \
  crazyflow/trajectory.py \
  crazyflow/control/mellinger/tracking.py \
  crazyflow/control/mellinger/__init__.py \
  examples/jax/mellinger_tracking.py \
  tests/unit/test_mellinger_tracking.py
.venv/bin/python -m ruff format --check \
  crazyflow/trajectory.py \
  crazyflow/control/mellinger/tracking.py \
  crazyflow/control/mellinger/__init__.py \
  examples/jax/mellinger_tracking.py \
  tests/unit/test_mellinger_tracking.py
```

Expected: `All checks passed!` and `5 files already formatted`.

## Common failures and handling

- **`.venv/bin/python` is missing:** stop. Do not fall back to system Python. Confirm repository
  root and ask the project lead before rebuilding from the project-approved lock/environment.
- **Pip cache is not writable:** a cache warning is harmless when `pip check` still reports no
  broken requirements. Do not install or upgrade packages to silence it.
- **`warp` or `mujoco_warp` import warnings:** these optional packages are absent in the verified
  CPU environment. They are not required for this path; treat a nonzero process exit as the actual
  failure signal.
- **Matplotlib config is not writable:** keep the documented `MPLCONFIGDIR=/tmp/...` prefix.
- **Frequency validation fails:** `--sim-freq` must be divisible by `--control-freq`; the verified
  pair is 500/100.
- **Output already exists:** use a new `--output-dir` to preserve prior evidence.
- **A validation is false:** preserve the JSON and plot, record commit/seed/environment, rerun the
  focused test, and diagnose before changing source.
- **Directional derivative is unstable:** verify `--fd-epsilon 1e-2`, saturation/floor/gate
  diagnostics, and exact commit before considering code changes.

The user guide and this runbook use the explicit repository-local interpreter required for this
WSL checkout.

## Day-2 batch verification

Day 2 was verified from base/end HEAD
`9eac04ea28997e76b9bc8c6e97ca53ef93398d55` on
`research/differentiable-mellinger`. Day-2 source and artifacts were intentionally left
uncommitted.

Run the environment and focused checks:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research

.venv/bin/python -m pip check

.venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py

.venv/bin/python -m ruff check \
  crazyflow/control/mellinger/tracking.py \
  crazyflow/control/mellinger/__init__.py \
  examples/jax/mellinger_tracking.py \
  examples/jax/mellinger_batch_diagnostics.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py

.venv/bin/python -m ruff format --check \
  crazyflow/control/mellinger/tracking.py \
  crazyflow/control/mellinger/__init__.py \
  examples/jax/mellinger_tracking.py \
  examples/jax/mellinger_batch_diagnostics.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py
```

Actual Day-2 result: `No broken requirements found`, `21 passed in 43.74s`, Ruff lint
`All checks passed!`, and `6 files already formatted`. The unchanged Day-1 test was also run
separately before implementation (`7 passed in 33.42s`), after the loss refactor (`7 passed in
24.13s`), and during final verification.

### H20 smoke run 1

The output directory must not already exist:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
MPLCONFIGDIR=/tmp/crazyflow-gradient-research-matplotlib \
  .venv/bin/python examples/jax/mellinger_batch_diagnostics.py \
  --horizon 20 \
  --control-freq 100 \
  --sim-freq 500 \
  --seed 20260724 \
  --fd-epsilon 1e-2 \
  --gain-perturbation-fraction 0.10 \
  --profile-repeats 3 \
  --output-dir artifacts/day2-audit/smoke-run-1
```

### H20 smoke run 2

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
MPLCONFIGDIR=/tmp/crazyflow-gradient-research-matplotlib \
  .venv/bin/python examples/jax/mellinger_batch_diagnostics.py \
  --horizon 20 \
  --control-freq 100 \
  --sim-freq 500 \
  --seed 20260724 \
  --fd-epsilon 1e-2 \
  --gain-perturbation-fraction 0.10 \
  --profile-repeats 3 \
  --output-dir artifacts/day2-audit/smoke-run-2
```

Both runs produced train loss `0.0022911166306585073`, validation loss
`0.0011237590806558728`, combined loss `0.001707437913864851`, and
`all_validations_passed: true`.

### Smoke reproducibility comparison

Remove only these named timing-dependent profiling keys before comparing JSON:

- `compile_and_first_batch_forward_backward_seconds`
- `cached_batch_forward`
- `cached_batch_forward_backward`
- `cached_sequential_two_case_forward`
- `cached_sequential_two_case_forward_backward`
- `ratios`

The executed comparison was:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
.venv/bin/python -c "import json; from pathlib import Path; a=json.loads(Path('artifacts/day2-audit/smoke-run-1/batch_diagnostics.json').read_text()); b=json.loads(Path('artifacts/day2-audit/smoke-run-2/batch_diagnostics.json').read_text()); ignored=('compile_and_first_batch_forward_backward_seconds','cached_batch_forward','cached_batch_forward_backward','cached_sequential_two_case_forward','cached_sequential_two_case_forward_backward','ratios'); [a['profiling'].pop(k) for k in ignored]; [b['profiling'].pop(k) for k in ignored]; print('scientific_payload_equal=' + str(a == b))"

sha256sum \
  artifacts/day2-audit/smoke-run-1/batch_tracking.png \
  artifacts/day2-audit/smoke-run-2/batch_tracking.png \
  artifacts/day2-audit/smoke-run-1/loss_and_sensitivity.png \
  artifacts/day2-audit/smoke-run-2/loss_and_sensitivity.png \
  artifacts/day2-audit/smoke-run-1/gradient_diagnostics.png \
  artifacts/day2-audit/smoke-run-2/gradient_diagnostics.png

cmp -s artifacts/day2-audit/smoke-run-1/batch_tracking.png \
  artifacts/day2-audit/smoke-run-2/batch_tracking.png
cmp -s artifacts/day2-audit/smoke-run-1/loss_and_sensitivity.png \
  artifacts/day2-audit/smoke-run-2/loss_and_sensitivity.png
cmp -s artifacts/day2-audit/smoke-run-1/gradient_diagnostics.png \
  artifacts/day2-audit/smoke-run-2/gradient_diagnostics.png
```

Actual result: `scientific_payload_equal=True`; all `cmp` commands exited zero. Plot SHA-256:

- `batch_tracking.png`:
  `578fa448fc85862d7f9afcfac4b3d966c43716057ad589616e78d8f13752f001`
- `loss_and_sensitivity.png`:
  `00868b6aabefc1ed900737d207f46b80d3209726a70a59531bb2272e50dd9cbc`
- `gradient_diagnostics.png`:
  `d05fe557fadc5cd518779af7e0ad3c3553c015207d23399e8fa95a13d523ec2e`

### H200 full batch

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
MPLCONFIGDIR=/tmp/crazyflow-gradient-research-matplotlib \
  .venv/bin/python examples/jax/mellinger_batch_diagnostics.py \
  --horizon 200 \
  --control-freq 100 \
  --sim-freq 500 \
  --seed 20260724 \
  --fd-epsilon 1e-2 \
  --gain-perturbation-fraction 0.10 \
  --profile-repeats 5 \
  --output-dir artifacts/day2-audit/batch-h200
```

Actual result: train `0.07310396432876587`, validation `0.024092847481369972`, combined
`0.048598404973745346`, with all validation flags true. Every run directory contains exactly:

- `batch_diagnostics.json`
- `batch_tracking.png`
- `loss_and_sensitivity.png`
- `gradient_diagnostics.png`

The H200 plot SHA-256 values are:

- `batch_tracking.png`:
  `c18195f54d12d0a8fb1ce3fb7596d00d1f44e155b4dee18763b9b4a02a12c80a`
- `loss_and_sensitivity.png`:
  `9a55e7bcbf7e8202a350f54b7aadf69293fd1332f5ef80b149542f35991f5584`
- `gradient_diagnostics.png`:
  `a5f380ade5a98a78da4fa279462c195051145e413686e522bebf8146f201dc63`

## Day-2 failure handling

- **Output directory exists:** stop and choose a fresh documented directory. The CLI refuses to
  overwrite evidence.
- **Frequency validation fails:** keep `sim_freq % control_freq == 0`; the verified pair is
  500/100.
- **Batch/single or order check fails:** preserve the JSON/plots, compare command/reference order,
  initial world states, masks, and exact source state. Do not loosen `rtol=1e-5`, `atol=1e-7`.
- **Directional check fails:** preserve epsilon `1e-2`, inspect saturation/gate/floor/nonfinite
  diagnostics, and do not change the `5e-2` threshold after seeing results.
- **Local descent fails:** preserve evidence and diagnose the raw-space gradient. Do not add an
  optimizer or persistent update as a workaround.
- **Profiling varies:** variation is expected. Require only positive finite synchronized times; do
  not claim a speedup or peak-memory measurement.
- **Long-horizon memory fails:** record the failure. Day 2 does not authorize rematerialization.
- **Optional Warp warnings:** `warp` and `mujoco_warp` remain optional and absent in the verified
  CPU environment; successful exit and pure-JAX results are authoritative.

## Day-3 Optax gain optimization

Sprint 3 starts from the later clean Sprint-2 checkpoint
`32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed`. The preserved Day-2 JSONs still correctly record
their historical execution at `9eac04ea28997e76b9bc8c6e97ca53ef93398d55`, dirty source state
`a1d7f1dce6ed3b1aa9cbb7cc53abb4a3bcada379129dfbde1787c1a5d1564229`. Do not rewrite those
records.

The Sprint-3 scientific source-state is
`8559648a3c52ca262f8eb5b4160540cc8e138c42c85d748f13daefe996d268f7`. Its declared scope is
`crazyflow`, `examples`, `tests`, `pyproject.toml`, and `pixi.lock`; generated artifacts and
result-dependent documentation are excluded.

Optax preflight:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
.venv/bin/python -c "import optax; print(optax.__version__)"
```

Actual: `0.2.8`. This exact version was already resolved in `pixi.lock` through the existing Flax
dependency and importable in `.venv`. No dependency, lockfile, or environment file changed, and no
package was installed.

### Day-3 quality checks

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research

.venv/bin/python -m pip check

.venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py \
  tests/unit/test_mellinger_gain_optimization.py

.venv/bin/python -m ruff check \
  crazyflow/control/mellinger/tracking.py \
  crazyflow/control/mellinger/__init__.py \
  crazyflow/control/mellinger/optimization.py \
  examples/jax/mellinger_tracking.py \
  examples/jax/mellinger_batch_diagnostics.py \
  examples/jax/mellinger_gain_optimization.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py \
  tests/unit/test_mellinger_gain_optimization.py

.venv/bin/python -m ruff format --check \
  crazyflow/control/mellinger/tracking.py \
  crazyflow/control/mellinger/__init__.py \
  crazyflow/control/mellinger/optimization.py \
  examples/jax/mellinger_tracking.py \
  examples/jax/mellinger_batch_diagnostics.py \
  examples/jax/mellinger_gain_optimization.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py \
  tests/unit/test_mellinger_gain_optimization.py
```

Pre-artifact combined test result: `36 passed in 170.62s`. Final combined result:
`36 passed in 170.92s`, Ruff lint `All checks passed!`, and Ruff format
`9 files already formatted`. The 21 unchanged Day-1/Day-2 tests also passed independently in
`48.01s`.

### H20 smoke runs

Run 1:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
MPLCONFIGDIR=/tmp/crazyflow-gradient-research-matplotlib \
  .venv/bin/python examples/jax/mellinger_gain_optimization.py \
  --horizon 20 \
  --steps 3 \
  --learning-rate 1e-2 \
  --control-freq 100 \
  --sim-freq 500 \
  --seed 20260724 \
  --fd-epsilon 1e-2 \
  --profile-repeats 3 \
  --robustness-horizons 20 \
  --output-dir artifacts/day3-audit/smoke-run-1
```

Run 2 is identical except for:

```text
--output-dir artifacts/day3-audit/smoke-run-2
```

Both selected step 3 and changed train loss from `0.0022911166306585073` to
`0.002285428112372756`; both reported `all_validations_passed: true`.

Each Day-3 directory contains exactly:

- `optimization_result.json`
- `optimized_gains.json`
- `robustness_results.json`
- `optimization_history.png`
- `gain_history.png`
- `robustness_matrix.png`

### Smoke reproducibility comparison

Remove only these six timing-dependent keys from the `profiling` object of
`optimization_result.json`:

- `compile_and_first_optimization_step_seconds`
- `cached_optax_step`
- `total_update_loop_seconds`
- `cached_forward`
- `cached_forward_backward`
- `forward_backward_h400_over_h200_median_ratio`

The executed scientific comparison used:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/day3-audit")
a = json.loads((root / "smoke-run-1/optimization_result.json").read_text())
b = json.loads((root / "smoke-run-2/optimization_result.json").read_text())
ignored = (
    "compile_and_first_optimization_step_seconds",
    "cached_optax_step",
    "total_update_loop_seconds",
    "cached_forward",
    "cached_forward_backward",
    "forward_backward_h400_over_h200_median_ratio",
)
for key in ignored:
    a["profiling"].pop(key)
    b["profiling"].pop(key)
print("optimization_scientific_payload_equal=", a == b)
for filename in ("optimized_gains.json", "robustness_results.json"):
    print(
        filename,
        "byte_equal=",
        (root / "smoke-run-1" / filename).read_bytes()
        == (root / "smoke-run-2" / filename).read_bytes(),
    )
PY
```

Actual: optimization scientific payload equal; both other JSON pairs byte-identical. All three
`cmp` checks exited zero. Plot SHA-256 values for both runs:

- `gain_history.png`:
  `e0f8a295b5b680dfc0720355d201a4cf88170da67b4aa81b5a87cb1a770035bc`;
- `optimization_history.png`:
  `f2ae04703199f7a2c0a5ac2794509e386e034626fc6d12acf4ecdc79bff082ce`;
- `robustness_matrix.png`:
  `5f1a46358a9b78070c0224ef7018217ddf11f655f9ce6f2b299078d0b46e4539`.

### H200 optimization and full robustness run

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
MPLCONFIGDIR=/tmp/crazyflow-gradient-research-matplotlib \
  .venv/bin/python examples/jax/mellinger_gain_optimization.py \
  --horizon 200 \
  --steps 50 \
  --learning-rate 1e-2 \
  --control-freq 100 \
  --sim-freq 500 \
  --seed 20260724 \
  --fd-epsilon 1e-2 \
  --profile-repeats 3 \
  --robustness-horizons 20 100 200 400 \
  --output-dir artifacts/day3-audit/optimize-h200
```

Actual: selected step 50; train `0.07310392707586288 -> 0.04376906901597977`;
validation `0.024092912673950195 -> 0.01579861156642437`; combined
`0.04859841987490654 -> 0.029783841222524643`; all 50 updates finite;
`all_validations_passed: true`.

H400 completed. Cached forward medians H20/H100/H200/H400 were
`0.002343864/0.004290380/0.006714594/0.013119240 s`. Cached forward/backward medians were
`0.021593010/0.084925729/0.175491063/0.328635742 s`; H400/H200 was
`1.8726636923`. Compile plus first optimization step was `15.743286041 s`; cached Optax-step
minimum/median/maximum was `0.150861597/0.153715902/0.156133985 s`; the full update loop was
`32.805345890 s`. These are synchronized measurements only. No peak-memory or speedup claim is
made.

### Day-3 failure handling

- **Output directory exists:** do not overwrite evidence; use only a newly authorized destination.
- **Nonfinite loss, gradient, parameter, or Optax state:** the update loop raises immediately.
  Preserve existing files and do not change learning rate or step count after seeing the result.
- **Directional check fails:** keep epsilon `1e-2` and threshold `5e-2`; inspect derivative
  magnitudes and saturation/gate/floor/nonfinite diagnostics.
- **H400 fails:** preserve the process error and last successful configuration, record
  `robustness_matrix_complete=false`, and leave Sprint 3 incomplete. Do not add rematerialization.
- **H400/H200 backward ratio exceeds 10:** record a later-sprint scaling indicator without changing
  Sprint 3.
- **Validation or a robustness cell worsens:** report it. Validation, combined loss, matched mass,
  and robustness cells are not update, selection, or acceptance objectives.
- **Optional Warp warnings:** unchanged and nonfatal on this successful CPU path.

## Sprint-4 freeze, audit, and handover verification

Sprint 4 changes no scientific source or artifact and executes no simulation or optimization
command. It starts from the documented uncommitted Sprint-3 state at HEAD
`32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed`. The scientific source-state remains
`8559648a3c52ca262f8eb5b4160540cc8e138c42c85d748f13daefe996d268f7`.

### Read-only preflight

Executed exactly from the repository root:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
git status --short --branch
git rev-parse HEAD
git branch --show-current
git log -3 --oneline
git diff --check
test -x .venv/bin/python
.venv/bin/python --version
.venv/bin/python -m pip --version
.venv/bin/python -m pip check
```

Actual: branch `research/differentiable-mellinger`, HEAD `32cb8ffa...`, the documented
uncommitted Sprint-3 file set, Python `3.12.13`, pip `26.1.2`, and
`No broken requirements found`. `git diff --check` produced no output. The pip user-cache warning
was nonfatal and caused no environment change.

### One-time focused verification

The following block was executed exactly once:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research

.venv/bin/python -m pip check

.venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py \
  tests/unit/test_mellinger_gain_optimization.py

.venv/bin/python -m ruff check \
  crazyflow/trajectory.py \
  crazyflow/control/mellinger/tracking.py \
  crazyflow/control/mellinger/optimization.py \
  crazyflow/control/mellinger/__init__.py \
  examples/jax/mellinger_tracking.py \
  examples/jax/mellinger_batch_diagnostics.py \
  examples/jax/mellinger_gain_optimization.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py \
  tests/unit/test_mellinger_gain_optimization.py

.venv/bin/python -m ruff format --check \
  crazyflow/trajectory.py \
  crazyflow/control/mellinger/tracking.py \
  crazyflow/control/mellinger/optimization.py \
  crazyflow/control/mellinger/__init__.py \
  examples/jax/mellinger_tracking.py \
  examples/jax/mellinger_batch_diagnostics.py \
  examples/jax/mellinger_gain_optimization.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py \
  tests/unit/test_mellinger_gain_optimization.py
```

Actual Sprint-4 result:

- `pip check`: `No broken requirements found`;
- pytest: `36 passed in 118.67s (0:01:58)`;
- Ruff lint: `All checks passed!`;
- Ruff format: `10 files already formatted`.

### Read-only evidence checks

The audit parsed every JSON file below `artifacts/`, checked every result-level
`all_validations_passed` field, recomputed the Sprint-3 source-state, verified all eighteen
Tag-3 paths and checkpoint SHA-256 values, checked the exact six-file set in each Tag-3
directory, and visually inspected the unique Tag-3 PNG payloads. The smoke-run plot pairs are
byte-identical. No artifact was opened for writing. Full findings:
[`checkpoints/DAY4_CHECKPOINT.md`](checkpoints/DAY4_CHECKPOINT.md).

### Handover documents

- [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md)
- [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md)
- [`SUPERVISOR_MEETING_BRIEF.md`](SUPERVISOR_MEETING_BRIEF.md)

Do not start a later scientific sprint before the supervisor decisions listed in the meeting brief
are resolved. In particular, do not add an artificial noise/delay model without flight-log/Mocap
system identification and do not treat the selected simulation gains as a hardware release.

## Sprint-5 split/domain-randomization workflow

The user-reported oral supervisor decision recorded in D-028 permits an explicitly heuristic,
uncalibrated simulation model for infrastructure and later robustness research. This does not
turn it into an empirical model. Always use the repository-local interpreter and work from the
repository root.

### Validate configuration and fixed Validation manifest; keep Test opaque

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from crazyflow.control.mellinger.research import Split, load_config, load_manifest

root = Path("configs/research/mellinger")
config = load_config(root / "smoke.json")
validation = load_manifest(root / "validation_manifest.v1.json", Split.VALIDATION)
print(config.run_id, validation.manifest_id, config.test_manifest)
PY
```

`TrainingPipelines` has no test field. A training command may record the configured opaque Test
reference but must not call `build_test_pipeline(...)` or `build_episode_batch(...)` with
`Split.TEST`.
The configured Test path above is printed as an opaque string and must not be opened during
training, diagnostics, acceptance, or pilot work.

### Focused tests

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_research.py \
  tests/unit/test_mellinger_checkpointing.py \
  tests/integration/test_mellinger_research_runner.py \
  tests/integration/test_examples.py -k mellinger

.venv/bin/python -m pytest -q tests/unit/control/test_mellinger.py
```

The first command covers deterministic/disjoint seeds, manifests, episodic mass, exact delay
semantics, correlated wrench shapes, gain roundtrips, finite trajectory rejection, split object
separation, fixed validation, a real two-world gradient, checkpoint/optimizer roundtrip, and the
explicit research-example skips. The second includes the mixed-null/active-world regression for
both Mellinger output conversions.

For a broad headless-compatible regression in this WSL desktop environment:

```bash
.venv/bin/python -m pytest -q \
  --ignore tests/unit/test_render.py \
  --ignore tests/unit/test_visualizations.py \
  -k 'not test_render_rgb_array'
```

Do not describe this as the literally complete suite. On 2026-07-31, the unfiltered suite reached
89% and then aborted in MuJoCo viewer initialization although `DISPLAY=:0` was set. The command
above passed `523` tests with `36` skips and `2` explicitly render-related deselections.

### One-update local smoke

Purpose: verify data flow, gradient, one update, validation selection, metrics, provenance, and
checkpointing. It is not a scientific optimization result.

```bash
/usr/bin/time -v .venv/bin/python \
  examples/jax/mellinger_domain_randomized_optimization.py \
  --config configs/research/mellinger/smoke.json \
  --output-dir artifacts/day5-audit/local-smoke
```

The output directory must be absent. The command refuses to overwrite the four run records, and
the step checkpoint refuses overwrite independently. Exact expected files:

```text
resolved_config.json
provenance.json
metrics.jsonl
checkpoint-step-000001.json
run_summary.json
```

The smoke preset uses two worlds, H20, one update, ±`0.0002 kg` true-mass variation, integer delay
up to one control interval, force standard deviation `0.0001 N`, torque standard deviation
`0.000001 N m`, and correlation time `0.05 s`. The latter three values are technical heuristics,
not calibrated laboratory values.

### Resume

Checkpoints contain the current raw gain vector, every Optax leaf with dtype/shape, step and
episode counters, history, selection state, root seed, both manifest IDs, provenance, and
config/registry fingerprints. Resume into a new run directory; a mismatch is a hard error.

```bash
.venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config configs/research/mellinger/workstation.json \
  --resume /absolute/path/to/checkpoint-step-000123.json \
  --output-dir /absolute/path/to/new-resume-directory
```

An intermediate checkpoint is written after every completed update. Do not edit a config merely
to extend a run: because the complete config is fingerprinted, that is intentionally incompatible
and requires a documented new run.

### Scaling benchmark

Run each world count in a fresh process. Do not run them concurrently.

```bash
/usr/bin/time -v .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --worlds 1 --horizon 20 --output-dir artifacts/day5-audit/scaling-w1
/usr/bin/time -v .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --worlds 2 --horizon 20 --output-dir artifacts/day5-audit/scaling-w2
/usr/bin/time -v .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --worlds 4 --horizon 20 --output-dir artifacts/day5-audit/scaling-w4
```

Each process measures `lower(...).compile()`, first synchronized execution, one warm-up, and five
synchronized steady executions. `/usr/bin/time -v` remains the external authority for process
walltime and Peak-RSS.

### Prepared workstation main run — do not execute locally

```bash
.venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config configs/research/mellinger/workstation.json \
  --output-dir /absolute/workstation/path/mellinger-stage1-run
```

The prepared configuration is `16 worlds × H400 × 200 updates`, Stage 1. Only mass randomization
is enabled. It was not executed or benchmarked at this size in Sprint 5. Before authorizing it,
benchmark H100/H200/H400 and 4/8/16 worlds incrementally, define storage/runtime limits, and
confirm trajectory acceptance rates. Do not run a true test evaluation until the controller,
hyperparameters, and checkpoint have been frozen after validation.

## Sprint-6 checkpoint/resume readiness and gated scaling

This section supersedes the Sprint-5 resume and scaling commands for new work. It does not
authorize the prepared main run or Test evaluation. The current decision is **NO-GO** because the
mandatory H100 entry pilot fails deterministic trajectory construction.

### Environment and focused verification

```bash
test -x .venv/bin/python
.venv/bin/python -m pip check
.venv/bin/python -m ruff check crazyflow examples tests
.venv/bin/python -m ruff format --check crazyflow examples tests
.venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_research.py \
  tests/unit/test_mellinger_checkpointing.py \
  tests/integration/test_mellinger_research_runner.py \
  tests/integration/test_examples.py -k mellinger
```

Observed results on 2026-07-31:

- `pip check`: no broken requirements; its cache-permission warning is nonfunctional;
- Ruff: `All checks passed!`; format: `113 files already formatted`;
- focused set: `14 passed, 5 skipped, 22 deselected in 29.14s`; external 31.27 s,
  Peak-RSS 2,103,000 KiB, swaps 0;
- final focused end-to-end resume regression after source-fingerprint scoping:
  `1 passed in 17.30s`; external 18.87 s, Peak-RSS 1,259,320 KiB, swaps 0.

The five skips are deliberate research entry points. The unfiltered viewer suite was not run;
Sprint 5 already documented its EGL/X11/MuJoCo abort. No dependency, lockfile, renderer, controller,
or dynamics default changed.

### Exact fresh-process Resume acceptance

All output directories below must be absent or empty. Timing files deliberately live outside the
runner output directories because a nonempty target is rejected before JAX work.

```bash
mkdir -p artifacts/day6-readiness/timing

/usr/bin/time -v \
  -o artifacts/day6-readiness/timing/resume-final-uninterrupted-h20.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config configs/research/mellinger/sprint6_resume.json \
  --output-dir artifacts/day6-readiness/resume-final-uninterrupted

/usr/bin/time -v \
  -o artifacts/day6-readiness/timing/resume-final-paused-h20.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config configs/research/mellinger/sprint6_resume.json \
  --max-updates-this-process 1 \
  --output-dir artifacts/day6-readiness/resume-final-paused

/usr/bin/time -v \
  -o artifacts/day6-readiness/timing/resume-final-resumed-h20.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config configs/research/mellinger/sprint6_resume.json \
  --resume artifacts/day6-readiness/resume-final-paused/checkpoint-step-000001.json \
  --equivalence-reference-checkpoint \
    artifacts/day6-readiness/resume-final-uninterrupted/checkpoint-step-000002.json \
  --output-dir artifacts/day6-readiness/resume-final-resumed
```

The final command must print `resume_equivalence=pass`, `status=success`, and
`test_metrics_present=False`. Exact comparison uses `rtol=0`, `atol=0`; it includes final and
selected raw gains with dtype/shape, transformed gains, complete Optax tree/leaves, step and
episode counters, selection, full Train/Validation history and metrics, root seed, episode IDs,
config/registry/source/runtime fingerprints, and Validation manifest ID/content fingerprint.
Timing, paths, commands, and UTC fields differ by construction and are excluded.

The technical Resume fixture reuses the existing Day-5 H20 technical trajectory settings, but is
mass-only: true dynamics mass ±0.0002 kg; controller mass unchanged; Delay, Wrench, sensor noise,
and UKF off; Stage 1 only. It is not the Workstation trajectory distribution and is not a
scientific optimization result.

### Checkpoint, monitoring, overwrite, and abort rules

- A target directory must be absent or completely empty. Resume must use a different directory
  from the source checkpoint. Never edit a config to extend a checkpointed run.
- Every completed update writes an atomic, checksummed checkpoint and atomically rewrites the full
  `metrics.jsonl`. Files are flushed and same-directory replaced; no existing checkpoint or final
  record is overwritten.
- Resume hard-rejects config, gain-registry, executable-source, runtime/backend/x64, root-seed,
  Validation ID/content, Test-reference, optimizer-structure, dtype, shape, counter, history, or
  payload-checksum mismatch.
- Nonfinite gains, Optax state, loss, gradient, or metrics abort before checkpoint commit.
- Use `--max-updates-this-process 1` for a planned stop. The last completed checkpoint is the only
  restart authority.
- No signal/emergency checkpoint is implemented: a JAX call may be inside compiled work and an
  incomplete optimizer update must never be serialized. `SIGTERM`/timeout may discard only the
  current uncommitted step; resume from the previous atomic checkpoint in a fresh process.
- Missing `run_summary.json` plus a valid last checkpoint means interrupted/resumable, not success.
  A benchmark exception writes `benchmark.json` where Python control is regained; a hard kill may
  leave only the external `/usr/bin/time -v` record.
- The runner records only an opaque frozen-Test-manifest reference and does not open the file.
  Never call `build_test_pipeline(...)`, build Test episodes, or generate Test metrics here.

### Exact local scaling gate and observed result

The only authorized first pilot was:

```bash
/usr/bin/time -v \
  -o artifacts/day6-readiness/timing/pilot-final-h100-w1.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --config configs/research/mellinger/workstation.json \
  --worlds 1 --horizon 100 --steady-repeats 5 \
  --output-dir artifacts/day6-readiness/pilot-final-h100-w1
```

Observed: exit 1 after 4.41 s, Peak-RSS 591,776 KiB, swaps 0. The machine-readable artifact says
`status=failed`, `failure_phase=deterministic_episode_construction_before_jit`, and
`no valid trajectory found in 16 deterministic attempts`. Compile, first-execution, and steady
times do not exist because JIT was never reached. This is a failed scientific/configuration gate,
not a memory or runtime gate.

Therefore these conditionally allowed local commands were deliberately **not run**:

```text
H100 × 4 worlds: skipped because H100 × 1 failed
H200 × 4 worlds: skipped because the preceding H100 gate failed
H400, 8 worlds, 16 worlds, >2 updates, seeds, ablations, Test: forbidden locally
```

Do not repair this by silently changing bounds, loss, manifest, Workstation trajectory
distribution, or generator acceptance limits. A separate scientific/configuration decision must
define a horizon-consistent pilot before repeating H100.

### Workstation inventory placeholders

Before any Workstation command, fill and preserve these non-secret fields in the run log:

| Field | Required value |
|---|---|
| OS | `<OS_AND_VERSION>` |
| CPU | `<CPU_MODEL_AND_LOGICAL_CORES>` |
| RAM | `<RAM_GIB>` |
| GPU/VRAM | `<GPU_MODEL_OR_NONE>` / `<VRAM_GIB_OR_NA>` |
| JAX backend | `cpu` with current code; any other value is unsupported |
| CUDA/driver | `<CUDA_AND_DRIVER_OR_NA>` |
| Repository path | `<REPOSITORY_PATH>` |
| Artifact path | `<LOCAL_SAME_FILESYSTEM_ARTIFACT_PATH>` |
| Max runtime per pilot | `<MAX_RUNTIME_SECONDS>` |
| Max memory | `<MAX_RSS_GIB>`; must remain well below physical RAM |

Atomic rename is only assumed on the same local filesystem. Do not place active checkpoints on an
unverified NFS/network mount. The current Crazyflow pipeline constructs CPU devices explicitly;
GPU/CUDA portability is not claimed.

### Exact staged Workstation pilot commands — blocked until H100 distribution gate is resolved

After replacing every placeholder above, use a fresh process and a new directory for each stage.
These commands are the reproducible protocol, but **must not be executed under the current
NO-GO**.

Stage W1, H100 × 4:

```bash
cd <REPOSITORY_PATH>
mkdir -p <LOCAL_SAME_FILESYSTEM_ARTIFACT_PATH>/timing
/usr/bin/time -v \
  -o <LOCAL_SAME_FILESYSTEM_ARTIFACT_PATH>/timing/h100-w4.time.txt \
  timeout --signal=TERM --kill-after=30s <MAX_RUNTIME_SECONDS>s \
  .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --config configs/research/mellinger/workstation.json \
  --worlds 4 --horizon 100 --steady-repeats 5 \
  --output-dir <LOCAL_SAME_FILESYSTEM_ARTIFACT_PATH>/h100-w4
```

Stage W2, H200 × 8, only after W1 passes every gate:

```bash
cd <REPOSITORY_PATH>
/usr/bin/time -v \
  -o <LOCAL_SAME_FILESYSTEM_ARTIFACT_PATH>/timing/h200-w8.time.txt \
  timeout --signal=TERM --kill-after=30s <MAX_RUNTIME_SECONDS>s \
  .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --config configs/research/mellinger/workstation.json \
  --worlds 8 --horizon 200 --steady-repeats 5 \
  --output-dir <LOCAL_SAME_FILESYSTEM_ARTIFACT_PATH>/h200-w8
```

Stage W3, H400 × 16, only after W2 passes every gate:

```bash
cd <REPOSITORY_PATH>
/usr/bin/time -v \
  -o <LOCAL_SAME_FILESYSTEM_ARTIFACT_PATH>/timing/h400-w16.time.txt \
  timeout --signal=TERM --kill-after=30s <MAX_RUNTIME_SECONDS>s \
  .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --config configs/research/mellinger/workstation.json \
  --worlds 16 --horizon 400 --steady-repeats 5 \
  --output-dir <LOCAL_SAME_FILESYSTEM_ARTIFACT_PATH>/h400-w16
```

Monitor from a second terminal at least every five seconds:

```bash
watch -n 5 'free -h; ps -C python -o pid,etimes,rss,vsz,pcpu,pmem,args'
```

Advance only if exit status is 0; `benchmark.json` has `status=success` and
`technical_gate.passed=true`; nonfinite, zero-thrust-gate, floor-clip, and failure fractions are
zero; saturation is below its technical threshold; no swap occurred; Peak-RSS is below
`<MAX_RSS_GIB>` and well below RAM; compile/first/steady times are plausible; the exact Resume
acceptance passes on that machine; expected files exist and hash cleanly. Stop on timeout, swap,
critical RAM pressure, artifact collision, nonfinite output, technical gate/floor failure,
trajectory rejection, checksum/fingerprint mismatch, or implausible nonlinear growth.

Do not infer W2/W3 runtime or RAM by linear scaling: JAX compilation, horizon length, world count,
backend, and persistent-cache state interact. After any stage, record `/usr/bin/time -v`, config,
root seed, Git HEAD, source/config/runtime fingerprints, command, environment, and SHA-256 before
considering the next stage.

### Artifact verification

```bash
cd artifacts/day6-readiness
sha256sum -c SHA256SUMS
```

All 46 listed files passed. `SHA256SUMS` itself is 4,848 bytes with SHA-256
`7580bd41d41314b9d13a9fa9ea052b67fcb715930d2ddbb53a9759023c85316d`.

## Sprint-7 horizon-consistent trajectory diagnosis and gated CPU pilots

This section supersedes the Sprint-6 trajectory NO-GO only for the explicit
`fixed_support_prefix_v2` pilot config. It does not change or authorize legacy
`workstation.json`, a Workstation job, H400 simulation, a main optimization, or Test access.

### Construction-only diagnostics

The diagnostic mode constructs Fourier references and host-validates them. It does not build a
`Sim`, call `jax.jit`, train, validate, or open Test.

Legacy horizon audit:

```bash
mkdir -p artifacts/day7-trajectory/phase-b/timing
/usr/bin/time -v \
  -o artifacts/day7-trajectory/phase-b/timing/legacy-horizons-32-seeds.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --config configs/research/mellinger/workstation.json \
  --trajectory-diagnostics-only \
  --diagnostic-horizons 20 50 100 200 400 \
  --diagnostic-seed-count 32 --diagnostic-max-attempts 16 \
  --output-dir artifacts/day7-trajectory/phase-b/legacy-horizons-32-seeds
```

Actual success: H20/H50/H100 `0/32`, H200 `12/32`, H400 `32/32`. The fixed suite is Train
episodes 0..31, world 0, trajectory component under root seed `20260731`; its SHA-256 is
`6e9f4661e908bf055841185f53547e3f4a999250a89b3ddf06d52b1b0eb9a898`.

The estimate-only H100 command changes only the diagnostic cap and is never a pilot input:

```bash
/usr/bin/time -v \
  -o artifacts/day7-trajectory/phase-b/timing/legacy-h100-16-seeds-128-attempt-estimate.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --config configs/research/mellinger/workstation.json \
  --trajectory-diagnostics-only --diagnostic-horizons 100 \
  --diagnostic-seed-count 16 --diagnostic-max-attempts 128 \
  --output-dir artifacts/day7-trajectory/phase-b/legacy-h100-16-seeds-128-attempt-estimate
```

Actual: zero accepted among 2,048 candidates. Do not interpret a larger rejection cap as a fix.

### Distribution versions

- Missing `trajectory.distribution_version` means exactly `legacy_normalized_time_v1`.
- `fixed_support_prefix_v2` requires explicit positive `support_duration_s` and
  `minimum_duration_s`. It validates the full parent before returning a prefix.
- The Sprint-7 config uses support 4.0 s and minimum 1.0 s. H100/H200 prefixes therefore do not
  have the legacy forced-hover terminal endpoint; H400 remains exact paired legacy geometry.
- Unknown versions and durations outside the explicit range are hard errors. There is no fallback.

### Repeated 128-seed construction gate

Run twice in fresh processes and distinct absent output directories:

```bash
mkdir -p artifacts/day7-trajectory/gate128/timing
/usr/bin/time -v \
  -o artifacts/day7-trajectory/gate128/timing/fixed-prefix-run1.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --config configs/research/mellinger/sprint7_pilot.json \
  --trajectory-diagnostics-only --diagnostic-horizons 100 200 400 \
  --diagnostic-seed-count 128 --diagnostic-max-attempts 16 \
  --output-dir artifacts/day7-trajectory/gate128/fixed-prefix-run1

/usr/bin/time -v \
  -o artifacts/day7-trajectory/gate128/timing/fixed-prefix-run2.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --config configs/research/mellinger/sprint7_pilot.json \
  --trajectory-diagnostics-only --diagnostic-horizons 100 200 400 \
  --diagnostic-seed-count 128 --diagnostic-max-attempts 16 \
  --output-dir artifacts/day7-trajectory/gate128/fixed-prefix-run2
```

Require 128/128 success at each horizon, no rejection/nonfinite, paired attempt equality, exact
H100/H200 prefixes of H400, exact H400 legacy-v1 equality, different-seed diversity, and equal
timing-excluded scientific hashes. Actual hash for both processes:
`4118a23a1f562fc27fbe99cefc6dfd8c6a02966f973929d68ce57d32217c8069`.

### Local scaling sequence

Every stage uses a fresh process and absent output directory. Stop after the first failure.

```bash
mkdir -p artifacts/day7-trajectory/pilots/timing
/usr/bin/time -v -o artifacts/day7-trajectory/pilots/timing/h100-w1.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --config configs/research/mellinger/sprint7_pilot.json \
  --worlds 1 --horizon 100 --steady-repeats 5 \
  --output-dir artifacts/day7-trajectory/pilots/h100-w1

/usr/bin/time -v -o artifacts/day7-trajectory/pilots/timing/h100-w4.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --config configs/research/mellinger/sprint7_pilot.json \
  --worlds 4 --horizon 100 --steady-repeats 5 \
  --output-dir artifacts/day7-trajectory/pilots/h100-w4

/usr/bin/time -v -o artifacts/day7-trajectory/pilots/timing/h200-w4.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --config configs/research/mellinger/sprint7_pilot.json \
  --worlds 4 --horizon 200 --steady-repeats 5 \
  --output-dir artifacts/day7-trajectory/pilots/h200-w4
```

Actual external wall/Peak-RSS: H100×1 `14.13 s / 1,006,864 KiB`; H100×4
`14.13 s / 1,033,476 KiB`; H200×4 `15.23 s / 1,038,940 KiB`. All exited 0 without swap and passed
every technical gate. Local H400, 8/16 worlds, more than two updates, Test, and main training remain
forbidden.

### Exact H100 Resume repetition

```bash
mkdir -p artifacts/day7-trajectory/resume/timing
/usr/bin/time -v \
  -o artifacts/day7-trajectory/resume/timing/uninterrupted-h100.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config configs/research/mellinger/sprint7_pilot.json \
  --output-dir artifacts/day7-trajectory/resume/uninterrupted

/usr/bin/time -v -o artifacts/day7-trajectory/resume/timing/paused-h100.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config configs/research/mellinger/sprint7_pilot.json \
  --max-updates-this-process 1 \
  --output-dir artifacts/day7-trajectory/resume/paused

/usr/bin/time -v -o artifacts/day7-trajectory/resume/timing/resumed-h100.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config configs/research/mellinger/sprint7_pilot.json \
  --resume artifacts/day7-trajectory/resume/paused/checkpoint-step-000001.json \
  --equivalence-reference-checkpoint \
    artifacts/day7-trajectory/resume/uninterrupted/checkpoint-step-000002.json \
  --output-dir artifacts/day7-trajectory/resume/resumed
```

Actual: exact `rtol=0`, `atol=0` Resume passed; external walls 17.03/13.43/13.72 s, Peak-RSS
1,159,432/1,077,380/1,082,724 KiB, zero swaps. An old Sprint-6 checkpoint was separately rejected
with `checkpoint config fingerprint mismatch`.

### Sprint-7 checks and artifact verification

```bash
.venv/bin/python -m pip check
.venv/bin/python -m ruff check crazyflow examples tests
.venv/bin/python -m ruff format --check crazyflow examples tests
.venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_research.py tests/unit/test_mellinger_checkpointing.py \
  tests/integration/test_mellinger_research_runner.py tests/integration/test_examples.py \
  -k mellinger
.venv/bin/python -m pytest -q tests/unit/control/test_mellinger.py
.venv/bin/python -m pytest -q \
  tests/integration/test_mellinger_research_runner.py::test_runner_resume_is_exact_and_never_opens_test_manifest

cd artifacts/day7-trajectory
sha256sum -c SHA256SUMS
```

Actual: pip/Ruff pass; focused tests `21 passed, 5 skipped, 22 deselected`; controller tests
`52 passed`; exact Resume regression `1 passed`; all 46 artifact hashes pass. `SHA256SUMS` hashes to
`2340d25908c737f05581dd1d9766f3ff8c1f190301008059e828da48b0730bd8`.

The broad headless-compatible suite reached 88% with no displayed failure but timed out at the
mandatory 300 s cap (exit 124); do not report it as passed. Do not rerun the known viewer gap.

The next authorized step is not a main run. Gather Workstation inventory/limits, create a distinct
v2 Workstation config, repeat exact Resume on that CPU environment, and begin again with an
incremental H100×4 gate. Test remains frozen.

## Sprint-8 bounded manual CPU night pilot

This section releases only one user-started H100×4/50-update technical pilot. Sprint 8 itself did
not start it. The full operating handover is
[`artifacts/day8-night-pilot/README.md`](../../artifacts/day8-night-pilot/README.md).

### Sole preparation measurement

Exactly one fresh H100×4 Train/Validation process and at most two updates were run:

```bash
mkdir -p artifacts/day8-night-pilot/timing
/usr/bin/time -v \
  -o artifacts/day8-night-pilot/timing/preflight-h100-w4.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config artifacts/day8-night-pilot/preflight_config.json \
  --output-dir artifacts/day8-night-pilot/preflight-h100-w4
```

Actual: exit 0; 19.24 s external wall; Peak-RSS 1,198,476 KiB (1.143 GiB); swaps 0;
Train/Validation compile 7.433/0.725 s; first complete update 13.098 s from runner start; second
complete-update interval 4.096 s. Both checkpoints pass their internal payload hashes. All four
Train/Validation rows, losses, gradient norms, and technical gates are finite/pass. No Test file
was opened and no Test episode or metric exists.

Estimated walltime using the operational complete-update interval is 52.0/113.5/215.9 s for
10/25/50 updates. The released config uses 50 for the largest bounded repeated-checkpoint sample;
a factor-two allowance is about 7.2 minutes. The launcher retains the hard 21,600-second cap and
does not extend automatically.

### Launcher checks performed during preparation

```bash
bash -n artifacts/day8-night-pilot/launch_night_pilot.sh
artifacts/day8-night-pilot/launch_night_pilot.sh --dry-run
artifacts/day8-night-pilot/launch_night_pilot.sh --dry-run \
  --resume artifacts/day8-night-pilot/preflight-h100-w4/checkpoint-step-000001.json
.venv/bin/python artifacts/day8-night-pilot/inspect_pilot.py \
  artifacts/day8-night-pilot/preflight-h100-w4
```

Both launcher dry-runs validated config/arguments and printed the exact runner command without
starting training. The inspector reported `evaluation_status=PASS` on the preparation evidence.
`tmux` is installed at `/usr/bin/tmux`.

### Exact user start commands

Preferred detached tmux start:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
tmux new-session -d -s crazyflow-s8-night \
  'cd /home/noah3/bachelorarbeit/crazyflow-gradient-research && exec artifacts/day8-night-pilot/launch_night_pilot.sh'
```

Foreground alternative:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
artifacts/day8-night-pilot/launch_night_pilot.sh
```

The launcher refuses an existing run root, tracks the config and clean executable-source scope,
records Git/config/environment/command provenance, logs both streams, applies the six-hour
timeout, and dispatches the unchanged runner. The runner atomically writes one checkpoint after
each completed Train update plus Validation/selection. Nonfinite state aborts before a checkpoint.

Status, controlled SIGTERM, real-runner Resume, and morning evaluation:

```bash
tmux capture-pane -pt crazyflow-s8-night -S -30
RUN_ROOT=/absolute/path/printed/by/the/launcher
tail -f "$RUN_ROOT/stdout.log" "$RUN_ROOT/stderr.log"
kill -TERM -- "-$(cat "$RUN_ROOT/process_group.pid")"
LAST_CHECKPOINT="$(ls -1 "$RUN_ROOT/runner-output"/checkpoint-step-*.json | sort | tail -n 1)"
artifacts/day8-night-pilot/launch_night_pilot.sh --resume "$LAST_CHECKPOINT"
.venv/bin/python artifacts/day8-night-pilot/inspect_pilot.py "$RUN_ROOT"
```

SIGTERM discards only the current incomplete update; Resume writes another new run root and keeps
the parent. Do not Resume a completed step-50 checkpoint. Do not run a second config, H200/H400,
more than 50 updates, a seed loop, Test, or a follow-on main job.

Before starting, keep the laptop on mains power, prevent Windows/WSL sleep, provide cooling, do
not run `wsl --shutdown`, and do not close the foreground terminal or tmux session. Verify the
frozen preparation payload first:

```bash
cd artifacts/day8-night-pilot
sha256sum -c SHA256SUMS
```

All 13 entries passed; `SHA256SUMS` hashes to
`7b50c9bb1461cc4ce4b24e8ed9a3e11f682c061c50e12629360d769ffa88ab93`.

## Sprint-9 completed-run analysis and blocked 5,000-update pilot

The operating handover and NO-GO rationale are in
[`artifacts/day9-long-pilot/README.md`](../../artifacts/day9-long-pilot/README.md). Reproduce the
Sprint-8 analysis from immutable content plus the separately captured and hashed operational
checkpoint timings:

```bash
.venv/bin/python artifacts/day9-long-pilot/analyze_sprint8_pilot.py \
  --run-dir artifacts/day8-night-pilot/runs/20260731T213826Z-2e16f365ecfd \
  --released-config artifacts/day8-night-pilot/pilot_config.json \
  --run-sha-index artifacts/day8-night-pilot/SPRINT8_RUN_SHA256SUMS \
  --checkpoint-timing-capture \
    artifacts/day9-long-pilot/sprint8_checkpoint_timing_capture.json \
  --output /tmp/sprint9-sprint8-analysis-reproduction.json
cmp artifacts/day9-long-pilot/sprint8_analysis.json \
  /tmp/sprint9-sprint8-analysis-reproduction.json
```

Preparation verification used:

```bash
bash -n artifacts/day9-long-pilot/launch_long_pilot.sh
artifacts/day9-long-pilot/launch_long_pilot.sh --dry-run
.venv/bin/python -m pip check
.venv/bin/python -m ruff check crazyflow examples tests artifacts/day9-long-pilot
.venv/bin/python -m ruff format --check crazyflow examples tests artifacts/day9-long-pilot
.venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_research.py tests/unit/test_mellinger_checkpointing.py \
  tests/integration/test_mellinger_research_runner.py tests/integration/test_examples.py \
  -k mellinger
cd artifacts/day9-long-pilot
sha256sum -c SHA256SUMS
```

Actual: launcher syntax passed and its dry-run exited 3 at the deliberate NO-GO before creating a
run root; pip and both Ruff checks passed; focused Pytest reported `21 passed, 5 skipped, 22
deselected`; all nine payload hashes passed. The focused tests cover atomic checkpoint roundtrip,
overwrite refusal, exact new-directory Resume, config/gain/source/runtime mismatch rejection,
checksum corruption, nonfinite serialization, provenance, and an intentionally nonexistent Test
manifest that the runner never opens.

The valid config is `artifacts/day9-long-pilot/pilot_config.json` (SHA-256
`265e5ee3a0669d464eb07fe26e7ec147ef4c26ee0d6ba2a095e309e8362aad1e`, canonical fingerprint
`23ee0f46f581b6453ec51915a8cd5410e7cd77d48c9d6a07caadee74a98c5fb5`). The launcher contains
the required two-hour timeout and logging but is intentionally disabled by `release_gate.json`.
Do not edit that gate, start tmux, create a substitute shorter run, Resume into a second run, or
enter Test/H200/H400. Confirm that no job exists with:

```bash
tmux has-session -t crazyflow-s9-long 2>/dev/null; echo "tmux_status=$? (1 means absent)"
pgrep -af 'mellinger_domain_randomized_optimization.py.*day9-long-pilot' || true
find artifacts/day9-long-pilot/runs -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null
```

## Sprint-10 sparse 5,000-update CPU pilot

The complete release and operating handover is
[`artifacts/day10-sparse-long-pilot/README.md`](../../artifacts/day10-sparse-long-pilot/README.md).
The checkpoint interval is part of `optimizer`; absence means the historical value 1. The released
config uses exactly 100, so an uninterrupted 5,000-update process writes steps 100, 200, …, 5,000
and writes step 5,000 only once. A nondivisible configured total additionally writes its final
step. A planned `--max-updates-this-process` stop is also a complete restart boundary.

### Verification and measured sparse smoke

```bash
.venv/bin/python -m pip check
.venv/bin/python -m ruff check crazyflow examples tests
.venv/bin/python -m ruff format --check crazyflow examples tests
.venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_research.py tests/unit/test_mellinger_checkpointing.py \
  tests/integration/test_mellinger_research_runner.py tests/integration/test_examples.py \
  -k mellinger

/usr/bin/time -v \
  -o artifacts/day10-sparse-long-pilot/timing/sparse-smoke-210.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config artifacts/day10-sparse-long-pilot/smoke_config.json \
  --output-dir artifacts/day10-sparse-long-pilot/smoke-210

/usr/bin/time -v \
  -o artifacts/day10-sparse-long-pilot/timing/sparse-resume-200-to-210.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config artifacts/day10-sparse-long-pilot/smoke_config.json \
  --resume artifacts/day10-sparse-long-pilot/smoke-210/checkpoint-step-000200.json \
  --equivalence-reference-checkpoint \
    artifacts/day10-sparse-long-pilot/smoke-210/checkpoint-step-000210.json \
  --output-dir artifacts/day10-sparse-long-pilot/smoke-resumed-200-to-210
```

Actual: pip/Ruff pass; focused tests `26 passed, 5 skipped, 22 deselected`; smoke exit 0,
55.99 s, 1,309,284 KiB Peak-RSS, zero swap; Checkpoints 100/200/210; 420 complete finite metric
rows; exact Resume 200→210 passed in a fresh process. The byte-reproducible analysis gives a
2,289.55-second conservative 5,000-update projection and 875,333,570-byte doubled storage bound.
All 24 preparation payload hashes pass.

### Exact single authorized start

Before start, require a clean worktree and verify the release payload:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
git status --short --branch
(cd artifacts/day10-sparse-long-pilot && sha256sum -c SHA256SUMS)
artifacts/day10-sparse-long-pilot/launch_long_pilot.sh --dry-run
tmux new-session -d -s crazyflow-s10-sparse-long \
  'cd /home/noah3/bachelorarbeit/crazyflow-gradient-research && exec artifacts/day10-sparse-long-pilot/launch_long_pilot.sh'
```

The launcher requires exact GO/config/source fingerprints, a completely clean worktree, an absent
UTC/HEAD run root, `/usr/bin/time -v`, and
`timeout --signal=TERM --kill-after=30s 7200s`. It starts no Resume or follow-on process.

Status, final evaluation, and controlled stop:

```bash
tmux capture-pane -pt crazyflow-s10-sparse-long -S -40
RUN_ROOT="$(find artifacts/day10-sparse-long-pilot/runs -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
cat "$RUN_ROOT/launch_manifest.txt"
ps -o pid,ppid,pgid,etimes,rss,pcpu,pmem,args -p "$(cat "$RUN_ROOT/process_group.pid")"
tail -n 30 "$RUN_ROOT/stdout.log" "$RUN_ROOT/stderr.log"
.venv/bin/python artifacts/day10-sparse-long-pilot/inspect_long_pilot.py "$RUN_ROOT"
kill -TERM -- "-$(cat "$RUN_ROOT/process_group.pid")"
```

Do not execute the final line unless a controlled stop is actually required. `ACTIVE` is healthy
while running. Final `PASS` requires 5,000 updates, exactly 50 valid checkpoint payloads, all
10,000 ordered Train/Validation rows, finite and passing gates, exit 0, no relevant swap, and no
Test access. An interrupted valid checkpoint is technically resumable but this sprint authorizes
no automatic continuation, second run, H200/H400, seed loop, or extension past 5,000.

## Sprint-11 completed long-pilot evaluation

The completed raw run is immutable input. Do not Resume, extend, normalize, or rerun it. Reproduce
the complete analysis and independent raw-file hash index from the repository root:

```bash
.venv/bin/python artifacts/day11-long-pilot-evaluation/analyze_sprint10_long_pilot.py \
  --run-dir artifacts/day10-sparse-long-pilot/runs/20260801T081232Z-1345032536fd \
  --released-config artifacts/day10-sparse-long-pilot/pilot_config.json \
  --sprint8-run-dir artifacts/day8-night-pilot/runs/20260731T213826Z-2e16f365ecfd \
  --smoke-dir artifacts/day10-sparse-long-pilot/smoke-210 \
  --projection artifacts/day10-sparse-long-pilot/sparse_analysis.json \
  --output /tmp/sprint11-long-pilot-analysis.json \
  --run-sha-index-output /tmp/sprint11-sprint10-run-sha256sums
cmp artifacts/day11-long-pilot-evaluation/long_pilot_analysis.json \
  /tmp/sprint11-long-pilot-analysis.json
cmp artifacts/day11-long-pilot-evaluation/SPRINT10_RUN_SHA256SUMS \
  /tmp/sprint11-sprint10-run-sha256sums
sha256sum -c artifacts/day11-long-pilot-evaluation/SPRINT10_RUN_SHA256SUMS
(cd artifacts/day11-long-pilot-evaluation && sha256sum -c SHA256SUMS)
```

The generator opens no Test manifest and fails before writing if any launcher, update-count,
file-set, checkpoint-step, payload checksum, metric order/duplicate, finite-value, technical-gate,
fingerprint, selection, resource, or Test-boundary gate fails. Expected result is byte-identical
JSON and hash index, with all 61 raw-file checks reporting `OK`.

Recorded PASS: exit 0, 5,000 updates, selected/best Validation step 5,000, best Train step 4,931,
925.96 seconds external wall, 2,201,956 KiB Peak-RSS, zero swap, 61 files and 437,505,563 bytes.
Only gain vectors at the 50 interval checkpoints are persisted; loss and gradient metrics remain
complete at all 5,000 updates.

The raw directory is deliberately not staged into ordinary Git. Preserve it in place and mirror
it byte-for-byte, together with `SPRINT10_RUN_SHA256SUMS`, to backed-up institutional research
storage; rerun `sha256sum -c` at the destination. Do not use `git add .`, and do not delete,
compress, move, partially stage, or add ignore rules for the run as part of this evaluation.

The next scientific work is a separate, explicitly authorized protocol-freeze and independent-
seed H100 replication. This section authorizes no new run, seed loop, H200/H400 action, Test
access, automatic continuation, or Resume.

## Sprint-12 frozen H100 replication protocol

Historical base-freeze record: its external-storage instructions are superseded by the
Sprint-12A section below and must not be used as current execution gates. All non-persistence
parts remain active.

Sprint 12 is protocol-only. The release gate forbids all research runs. Verify the complete
machine-readable freeze without opening Test or invoking the research runner:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
git branch --show-current
git rev-parse HEAD
git status --short --branch
(cd artifacts/day12-h100-replication-protocol && sha256sum -c SHA256SUMS)
.venv/bin/python artifacts/day12-h100-replication-protocol/verify_protocol.py
bash -n artifacts/day12-h100-replication-protocol/sprint13_runbook_template.sh
artifacts/day12-h100-replication-protocol/sprint13_runbook_template.sh --dry-run 1
```

Reproduce the generated JSON byte-for-byte without modifying the freeze directory:

```bash
repro_dir="$(mktemp -d /tmp/day12-protocol-reproduction.XXXXXX)"
.venv/bin/python artifacts/day12-h100-replication-protocol/generate_protocol.py \
  --output-dir "$repro_dir"
diff -qr --no-dereference \
  --exclude=README.md --exclude=SHA256SUMS --exclude=generate_protocol.py \
  --exclude=sprint13_runbook_template.sh --exclude=verify_protocol.py \
  artifacts/day12-h100-replication-protocol "$repro_dir"
```

Expected protocol verification:

```text
protocol_verification=PASS
seed_count=10
seeds=1432116264,366692846,235438753,1369406745,1081462774,1276309202,987511476,2125653507,31735934,986925065
config_count=10
source_fingerprint=6b3fed4a445b82b2a1354115edc982a15e0d93cd4aff91fc8f51475e60f01c5e
runtime_fingerprint=e3c408d547653ebbb87107ca3c31b844c12acc43448378a5a48266306ee3d2a4
test_manifest_opened=False
research_runs_started=0
```

The ten configs are `artifacts/day12-h100-replication-protocol/configs/seed-01.json` through
`seed-10.json`. The verifier checks their schema and fingerprints, exact seed derivation, only
`run_id`/`root_seed` differences from Sprint 10, source/runtime/gain/Validation fingerprints,
loss and gain definitions, fixed selection rule, emitted metric names, sequential inventory,
failure rules, GO/NO-GO list, opaque Test policy, and every protocol SHA-256. It loads the
Validation manifest but never opens the Test manifest.

The Sprint-13 script is deliberately a nonexecuting template. It accepts only
`--dry-run SEED_INDEX`; any other mode exits 3. A later explicitly authorized launcher must use a
fresh unique root and the displayed bounded command shape, but may start only one seed. There is
no automatic loop or Resume.

Before any future first seed, Sprint 13 must reconfirm:

1. exact branch and tracked protocol commit;
2. pinned source/runtime/config/gain/Validation fingerprints;
3. CPU backend, no competing research process, and at least 10 GB approved external capacity;
4. `/usr/bin/time -v`, `timeout --signal=TERM --kill-after=30s 7200s`, full logs, atomic
   checkpoints, and a unique absent run root;
5. a tested raw-run index/copy/destination-verification workflow and storage receipt.

After each attempt, do not start the next seed until integrity has passed and run plus complete
SHA-256 index have been copied byte-for-byte to external research storage and verified there.
Expected panel compute time is 2:34:19.60 at the pilot rate or 6:21:35.45 conservatively; expected
raw data is 4,375,055,630 bytes (about 4.4 GB), with an 8,753,335,700-byte factor-two plan.

This section does not authorize a research command, H200/H400, Test, replacement seed, automatic
continuation, parallel execution, or ordinary-Git storage of raw output.

## Sprint-12A local-SSD persistence amendment

This section supersedes only the external-storage parts of the historical Sprint-12 section.
The dated amendment is
`artifacts/day12-h100-replication-protocol/amendments/2026-08-01-local-ssd-persistence-v1.md`;
commit `73868cfd84d25bec0c6612b31ee34f4e67acb307` remains the authority for the original rule.
Sprint 12A is still protocol-only and starts no runner.

Verify the amended protocol and the Seed-01 local-capacity gate:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
(cd artifacts/day12-h100-replication-protocol && sha256sum -c SHA256SUMS)
.venv/bin/python artifacts/day12-h100-replication-protocol/verify_protocol.py
bash -n artifacts/day12-h100-replication-protocol/sprint13_runbook_template.sh
artifacts/day12-h100-replication-protocol/sprint13_runbook_template.sh --dry-run 1
```

Expected additions to verifier/template output are:

```text
base_seed_manifest_unchanged=True
base_seed_configs_unchanged=True
seed01_minimum_local_free_bytes=20000000000
seed01_local_capacity_gate_passed=True
external_storage_required=False
accepted_single_drive_risk=True
local_capacity_gate=PASS
```

The template computes the exact per-seed minimum. Seed 01 requires 20,000,000,000 bytes. Before
Seed `N` for `N=2..10`, the requirement is
`10,000,000,000 + 437,505,563 * (11-N)` bytes; `(11-N)` counts every remaining seed including the
one about to start. Use `df -B1 --output=avail .` on the repository filesystem. Insufficient
space blocks launch and does not count as an attempt.

After each completed successful or failed attempt, and only after all writers have stopped,
create and immediately verify the local immutable index from inside its unique run root:

```bash
cd <RUN_ROOT>
find . -type f ! -name RUN_SHA256SUMS -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 -r sha256sum > RUN_SHA256SUMS
sha256sum -c RUN_SHA256SUMS
```

Before any later seed, rerun the full previous index. The dry-run template performs this gate when
given the previous absolute or repository-relative run root:

```bash
artifacts/day12-h100-replication-protocol/sprint13_runbook_template.sh \
  --dry-run 2 artifacts/day13-h100-replication/runs/seed-01/<UTC>-<GIT_HEAD>
```

A missing index, missing file, or mismatch blocks this and all later seeds. Runs remain under
`artifacts/day13-h100-replication/runs/`, outside ordinary Git, and may not be deleted, moved,
renamed, compressed, modified, or overwritten during the series. The Sprint-10 pilot remains
unchanged. No external storage capacity, copy, destination-hash verification, or receipt is
required.

Reproduce all generator-owned JSON, including the Amendment JSON, without modifying the freeze:

```bash
repro_dir="$(mktemp -d /tmp/day12a-protocol-reproduction.XXXXXX)"
.venv/bin/python artifacts/day12-h100-replication-protocol/generate_protocol.py \
  --output-dir "$repro_dir"
diff -qr --no-dereference \
  --exclude=README.md --exclude=SHA256SUMS --exclude=generate_protocol.py \
  --exclude=2026-08-01-local-ssd-persistence-v1.md \
  --exclude=sprint13_runbook_template.sh --exclude=verify_protocol.py \
  artifacts/day12-h100-replication-protocol "$repro_dir"
```

The user accepts that a total failure or loss of the single internal SSD can destroy every raw
run. SHA-256 provides integrity detection, not an independent backup. All seeds, configs,
scientific/statistical rules, fingerprints, Test lock, and sequential execution remain frozen.
A separate Sprint-13 authorization may cover exactly Seed 01 after a fresh exact preflight; this
section itself authorizes no research run.

## Gradient-repository closure reproduction

This section is the current command authority for the 2 August 2026 repository closure. It
analyzes only the ten existing H100 Train/Validation runs. Do **not** invoke the research runner,
generate a seed, open Test, start H200/H400, or overwrite a raw run.

### Deterministic ten-seed synthesis

The versioned result was created once, after all input gates passed, with:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
.venv/bin/python -B \
  artifacts/day19-h100-analysis/analyze_h100_replication.py --write
```

`--write` refuses to overwrite any existing synthesis output. Normal reproduction is therefore
read-only and uses `--check`:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
.venv/bin/python -B \
  artifacts/day19-h100-analysis/analyze_h100_replication.py --check
(cd artifacts/day19-h100-analysis && sha256sum -c SHA256SUMS)
```

Expected compact output:

```text
h100_replication_synthesis=CHECK_PASS
seed_count=10
selected_steps=5000,5000,5000,5000,5000,5000,5000,5000,5000,5000
mean_relative_validation_improvement=0.747788986450
relative_validation_improvement_ci95=[0.747757723574,0.747820249326]
test_manifest_opened=False
research_runs_started=0
```

The check recalculates the complete synthesis in memory and compares byte-for-byte with
`h100_replication_analysis.json`, `SOURCE_RUN_INDEXES.sha256`, and `SHA256SUMS`. It verifies all
ten source indexes and their 600 payload files, all 500 internal checkpoint payload hashes,
5,000 ordered Train plus 5,000 ordered Validation rows per seed, released/resolved configs,
source/runtime/gain/Validation fingerprints, run completion, selection, technical gates,
resources, and the opaque Test boundary. It never resolves or opens the Test-manifest path.

Run the command twice when explicitly checking repeat determinism; both executions must print the
same compact values and `CHECK_PASS` without changing Git status.

### Frozen protocol and primary-data integrity

Verify the committed pre-run protocol without a research action:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
(cd artifacts/day12-h100-replication-protocol && sha256sum -c SHA256SUMS)
.venv/bin/python -B artifacts/day12-h100-replication-protocol/verify_protocol.py
```

Verify the protected Day-10 primary data from the repository root:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
sha256sum --quiet -c artifacts/day11-long-pilot-evaluation/SPRINT10_RUN_SHA256SUMS
```

Verify the ten protected H100 indexes without modifying their directories:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
for index_path in artifacts/day13-h100-replication/runs/seed-*/*/RUN_SHA256SUMS; do
  run_dir="${index_path%/RUN_SHA256SUMS}"
  (cd "$run_dir" && sha256sum --quiet -c RUN_SHA256SUMS) || exit 1
done
```

Expected totals are Day 10: 61 files / 437,505,563 bytes; H100: 610 files /
4,375,130,360 bytes. The H100 indexes cover 600 files because each `RUN_SHA256SUMS` is itself the
unindexed sixty-first file. The synthesis additionally validates all 500 checkpoint payload
hashes.

The independent TAR receipt is documentation, not a command to recreate or upload an archive.
Its exact 2 August 2026 evidence and limits are in
[`PRIMARY_DATA_ARCHIVE.md`](PRIMARY_DATA_ARCHIVE.md). Do not delete, rename, normalize, compress,
stage, or rewrite either raw-data tree.

### Closure regression tests

The focused analysis tests are:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
.venv/bin/python -m pytest -q tests/unit/test_h100_replication_analysis.py
.venv/bin/python -m ruff check \
  artifacts/day19-h100-analysis/analyze_h100_replication.py \
  tests/unit/test_h100_replication_analysis.py
.venv/bin/python -m ruff format --check \
  artifacts/day19-h100-analysis/analyze_h100_replication.py \
  tests/unit/test_h100_replication_analysis.py
```

Relevant higher-level tests may be added only if they do not start a scientific run or open Test.
The original H100 runner and any H200/H400/Test command are out of scope for closure verification.

### Pre-commit protection checks

Before requesting review:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
git diff --check
git status --short --branch
git diff --cached --name-only
git diff --name-only
git ls-files artifacts/day10-sparse-long-pilot/runs \
  artifacts/day13-h100-replication/runs
```

The index must be empty, the two raw-run commands must list no tracked path, and the only new or
modified closure paths must be the approved analysis/test/documentation allowlist. No commit,
push, tag, merge, rebase, Test action, later horizon, or new experiment belongs to this step.

## Stage-2 H20 loss-term diagnostics

This command evaluates the fixed `cf2x_L250` Figure-8/train and circle/validation infrastructure
smoke with default gains and zero optimizer updates. It does not access Test or protected H100/
Day-10 primary data. `PYTHONPATH` pins imports to the current checkout even when the shared
repository-local environment was created from another worktree.

Create the four final artifacts only when the target directory does not yet exist:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
PYTHONPATH="$PWD" MPLCONFIGDIR=/tmp/crazyflow-gradient-stage2-matplotlib \
  .venv/bin/python examples/jax/mellinger_loss_diagnostics.py \
  --horizon 20 \
  --seed 20260724 \
  --output-dir artifacts/day20-stage2-loss-diagnostics
```

The CLI refuses to overwrite an existing output directory. For an independent reproduction,
write to a fresh temporary child and compare all four files:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
repro_parent="$(mktemp -d /tmp/crazyflow-stage2-reproduction.XXXXXX)"
PYTHONPATH="$PWD" MPLCONFIGDIR=/tmp/crazyflow-gradient-stage2-matplotlib \
  .venv/bin/python examples/jax/mellinger_loss_diagnostics.py \
  --horizon 20 \
  --seed 20260724 \
  --output-dir "$repro_parent/day20-stage2-loss-diagnostics"
diff -qr artifacts/day20-stage2-loss-diagnostics \
  "$repro_parent/day20-stage2-loss-diagnostics"
(cd artifacts/day20-stage2-loss-diagnostics && sha256sum -c SHA256SUMS)
```

Expected losses are `0.0022911166` for Train and `0.0011237586` for Validation. The JSON must
report all validations true, exactly six terms, exactly four bound-variable columns, finite
values, contribution/total agreement, the unchanged H20 totals, clear technical smoke gates,
complete plot data, and zero optimizer updates.

Focused verification:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
.venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_loss_diagnostics.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_gain_optimization.py
.venv/bin/python -m ruff check \
  crazyflow/control/mellinger/tracking.py \
  crazyflow/control/mellinger/loss_diagnostics.py \
  crazyflow/control/mellinger/__init__.py \
  examples/jax/mellinger_loss_diagnostics.py \
  tests/unit/test_mellinger_loss_diagnostics.py
.venv/bin/python -m ruff format --check \
  crazyflow/control/mellinger/tracking.py \
  crazyflow/control/mellinger/loss_diagnostics.py \
  crazyflow/control/mellinger/__init__.py \
  examples/jax/mellinger_loss_diagnostics.py \
  tests/unit/test_mellinger_loss_diagnostics.py
```

The generated plots are presentation-ready infrastructure diagnostics, not optimization,
convergence, cross-platform, firmware, or hardware evidence.

## Friday existing-evidence package

The Day-21 review candidate uses the explicit fallback because the narrow default/candidate
rollout did not meet the unchanged zero-motor-saturation gate. The fallback must not produce
rollout arrays, rollout metrics, replacement values, or the two rollout PNGs. It reads Test
neither directly nor indirectly and starts no optimizer or training run.

Create the package only when the target directory does not exist:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
PYTHONPATH="$PWD" \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-matplotlib \
.venv/bin/python examples/jax/mellinger_friday_evidence.py \
  --h100-root /home/noah3/bachelorarbeit/crazyflow-gradient-research/artifacts/day13-h100-replication/runs \
  --fallback-existing-evidence \
  --output-dir artifacts/day21-friday-evidence
```

Expected compact fields include:

```text
friday_evidence=WRITE_PASS
seed_count=10
candidate_label=provisional visualization candidate
candidate_seed=05
candidate_update=5000
rollout_status=WITHHELD_TECHNICAL_GATE
test_opened=False
training_runs_started=0
```

Verify the fallback inventory and its nine indexed payloads:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
test ! -e artifacts/day21-friday-evidence/rollout_trajectory.png
test ! -e artifacts/day21-friday-evidence/rollout_tracking_error.png
test "$(wc -l < artifacts/day21-friday-evidence/SHA256SUMS)" -eq 9
(cd artifacts/day21-friday-evidence && sha256sum --quiet -c SHA256SUMS)
```

For byte-level reproduction, use two fresh targets and compare them:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
repro_a="$(mktemp -d /tmp/wo-gr-f0-repro-a.XXXXXX)"
repro_b="$(mktemp -d /tmp/wo-gr-f0-repro-b.XXXXXX)"
for target in "$repro_a/package" "$repro_b/package"; do
  PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-matplotlib \
  .venv/bin/python examples/jax/mellinger_friday_evidence.py \
    --h100-root /home/noah3/bachelorarbeit/crazyflow-gradient-research/artifacts/day13-h100-replication/runs \
    --fallback-existing-evidence --output-dir "$target"
done
diff -qr "$repro_a/package" "$repro_b/package"
```

The normal mode retains the strict rollout gate and is not the accepted package command. It must
stop before output if either controller fails the zero-saturation contract. Do not weaken the
gate, choose another seed, change the trajectory/platform, or fill the missing rollout with
unreviewed values.

Focused verification:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-matplotlib \
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_mellinger_friday_evidence.py \
  tests/unit/test_h100_replication_analysis.py \
  tests/unit/test_mellinger_batch_diagnostics.py \
  tests/unit/test_mellinger_loss_diagnostics.py
.venv/bin/python -m ruff check \
  examples/jax/mellinger_friday_evidence.py \
  tests/unit/test_mellinger_friday_evidence.py
.venv/bin/python -m ruff format --check \
  examples/jax/mellinger_friday_evidence.py \
  tests/unit/test_mellinger_friday_evidence.py
```

## Saturation-explicit rollout diagnostic

Day 22 is a separate diagnostic-only path. It reuses the frozen F0.1 rollout builder and reads
only checksum-verified accepted Day-21 inputs. It must not read Day 10, Day 13, or Test. It does
not relax the normal F0.1 gate and does not rewrite `artifacts/day21-friday-evidence/`.

Do not run either full F0.1 CLI as part of this diagnostic verification: both modes would reread
protected Day-13 inputs. Instead, the focused test and review method loads the checksum-verified
accepted Day-21 candidate, invokes the real frozen trace builder, verifies fractions `0.0` and
`0.027499999850988388`, and requires the unchanged `build_rollout` gate to raise without an
output path. Source diff and existing F0.1 tests cover unchanged CLI/fallback behavior.

Create the six-file package only when the target directory does not exist:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
PYTHONPATH="$PWD" \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-002-matplotlib \
.venv/bin/python examples/jax/mellinger_saturation_diagnostic.py \
  --diagnostic-only-saturation-present \
  --output-dir artifacts/day22-saturation-diagnostic
```

Expected compact output:

```text
diagnostic_status=DIAGNOSTIC_ONLY_SATURATION_PRESENT
normal_acceptance_gate=FAIL_ZERO_MOTOR_SATURATION_GATE
seed05_saturated_count=11
seed05_motor_saturation_fraction=0.027499999850988388
test_opened=False
training_runs_started=0
```

Verify the five indexed payloads:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
(cd artifacts/day22-saturation-diagnostic && sha256sum --quiet -c SHA256SUMS)
(cd artifacts/day21-friday-evidence && sha256sum --quiet -c SHA256SUMS)
```

For a deterministic replay, use two fresh directories. Both commands must exit zero, both
checksum checks must pass, and both diffs must be empty:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
repro_a="$(mktemp -d /tmp/wo-gr-f0-002-repro-a.XXXXXX)"
repro_b="$(mktemp -d /tmp/wo-gr-f0-002-repro-b.XXXXXX)"
for target in "$repro_a/package" "$repro_b/package"; do
  PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-002-matplotlib \
  .venv/bin/python examples/jax/mellinger_saturation_diagnostic.py \
    --diagnostic-only-saturation-present --output-dir "$target"
  (cd "$target" && sha256sum --quiet -c SHA256SUMS)
done
diff -qr "$repro_a/package" "$repro_b/package"
diff -qr "$repro_a/package" artifacts/day22-saturation-diagnostic
```

Focused verification:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-002-matplotlib \
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_mellinger_saturation_diagnostic.py \
  tests/unit/test_mellinger_friday_evidence.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py
.venv/bin/python -m ruff check \
  examples/jax/mellinger_friday_evidence.py \
  examples/jax/mellinger_saturation_diagnostic.py \
  tests/unit/test_mellinger_saturation_diagnostic.py
.venv/bin/python -m ruff format --check \
  examples/jax/mellinger_friday_evidence.py \
  examples/jax/mellinger_saturation_diagnostic.py \
  tests/unit/test_mellinger_saturation_diagnostic.py
```

Required review facts are Default `0/400`; Seed 05 `11/400`, all upper-bound on motor 0 at
indices 48–58, one interval `0.48–0.59 s`. Every state, command, metric and plot array must be
finite. The Seed-05 label must remain exactly `DIAGNOSTIC_ONLY_SATURATION_PRESENT`. Any use of
these values without the failed-gate status, any new candidate/case/platform, or any acceptance,
flight, safety, hardware or superiority claim is outside this command.

## Vertical ki_z excitation and local-sensitivity pilot

Day 23 is a bounded diagnostic with no optimizer. It uses the fixed 5.0-s, 0.75-m vertical hold,
the Registry Lower/Default/Upper `ki_z` values and exactly two only-controller-mass-different
cases. It neither reads protected primary data nor depends on the Friday deck.

Create the five-file package only when the target directory does not exist:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
PYTHONPATH="$PWD" \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-s2-002-matplotlib \
.venv/bin/python examples/jax/mellinger_ki_z_sensitivity.py \
  --run-ki-z-identifiability-pilot \
  --output-dir artifacts/day23-ki-z-sensitivity
```

Expected compact output:

```text
recommendation=GO_FOR_FUTURE_KI_Z_OPTIMIZATION
failed_scientific_criteria=[]
output_dir=artifacts/day23-ki-z-sensitivity
```

Verify the four indexed payloads and focused regressions:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
(cd artifacts/day23-ki-z-sensitivity && sha256sum --quiet -c SHA256SUMS)
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-s2-002-matplotlib \
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_mellinger_ki_z_sensitivity.py \
  tests/unit/test_mellinger_research.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_gain_optimization.py
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
RUFF_CACHE_DIR=/tmp/crazyflow-gradient-s2-002-ruff \
.venv/bin/python -m ruff check \
  examples/jax/mellinger_ki_z_sensitivity.py \
  tests/unit/test_mellinger_ki_z_sensitivity.py
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -m ruff format --check \
  examples/jax/mellinger_ki_z_sensitivity.py \
  tests/unit/test_mellinger_ki_z_sensitivity.py
```

For byte-level reproduction, generate two fresh packages from the committed generator and require
both checksum checks and both directory comparisons to be empty:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
repro_a="$(mktemp -d /tmp/wo-gr-s2-002-repro-a.XXXXXX)"
repro_b="$(mktemp -d /tmp/wo-gr-s2-002-repro-b.XXXXXX)"
for target in "$repro_a/package" "$repro_b/package"; do
  PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/crazyflow-gradient-s2-002-matplotlib \
  .venv/bin/python examples/jax/mellinger_ki_z_sensitivity.py \
    --run-ki-z-identifiability-pilot --output-dir "$target"
  (cd "$target" && sha256sum --quiet -c SHA256SUMS)
done
diff -qr "$repro_a/package" "$repro_b/package"
diff -qr "$repro_a/package" artifacts/day23-ki-z-sensitivity
```

`GO_FOR_FUTURE_KI_Z_OPTIMIZATION` is not an optimization result and does not authorize a later
run by itself. Any technical gate failure must withhold output; weak scientific identifiability
must remain `NO_GO_KI_Z_NOT_YET_IDENTIFIABLE` without changing horizon, bias, perturbation or
thresholds.

## Unclipped two-stage allocation diagnostic

Day 24 is a diagnostic-local replay of the accepted Day-22 case. Do not run the Day-19 analyzer
or either F0.1 CLI: those commands would traverse protected Day-13 inputs. The diagnostic may read
only the checksum-verified Day-19 derivative/analyzer source and accepted Day-22 package. It must
not change shared source, controller behavior, bounds, gains, defaults or thresholds.

Create the five-file package only when the target directory does not exist:

```bash
cd /tmp/crazyflow-gradient-research-wo-gr-f0-003
PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-003 \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-003-matplotlib \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python \
  examples/jax/mellinger_unclipped_allocation_diagnostic.py \
  --diagnostic-only-unclipped-allocation \
  --output-dir artifacts/day24-unclipped-allocation
```

Expected compact output:

```text
diagnostic_status=DIAGNOSTIC_ONLY_SATURATION_PRESENT
normal_acceptance_gate=FAIL_ZERO_MOTOR_SATURATION_GATE
seed05_saturated_count=11
seed05_motor0_maximum_requested_pwm=74753.53125
seed05_motor0_maximum_upper_exceedance_pwm=9218.53125
test_opened=False
training_runs_started=0
```

Run the focused regressions and style checks with the repository-local environment:

```bash
cd /tmp/crazyflow-gradient-research-wo-gr-f0-003
PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-003 \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-003-matplotlib \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python -m pytest -q \
  -p no:cacheprovider \
  tests/unit/test_mellinger_unclipped_allocation_diagnostic.py \
  tests/unit/test_mellinger_saturation_diagnostic.py \
  tests/unit/test_mellinger_friday_evidence.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/control/test_mellinger.py \
  tests/unit/control/test_transform.py
PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-003 \
PYTHONDONTWRITEBYTECODE=1 \
RUFF_CACHE_DIR=/tmp/crazyflow-gradient-f0-003-ruff \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python -m ruff check \
  examples/jax/mellinger_unclipped_allocation_diagnostic.py \
  tests/unit/test_mellinger_unclipped_allocation_diagnostic.py
PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-003 \
PYTHONDONTWRITEBYTECODE=1 \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python -m ruff format --check \
  examples/jax/mellinger_unclipped_allocation_diagnostic.py \
  tests/unit/test_mellinger_unclipped_allocation_diagnostic.py
```

For byte-level reproduction, use two fresh targets from generator/test commit
`728e8b9f89e4f92c7fa4eefee9808a82904975c2`. Each target must pass 4/4 checksums, and all three
directory comparisons must be empty:

```bash
cd /tmp/crazyflow-gradient-research-wo-gr-f0-003
for target in \
  /tmp/crazyflow-gradient-f0-003-repro-a \
  /tmp/crazyflow-gradient-f0-003-repro-b; do
  test ! -e "$target"
  PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-003 \
  PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-003-matplotlib \
  /home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python \
    examples/jax/mellinger_unclipped_allocation_diagnostic.py \
    --diagnostic-only-unclipped-allocation --output-dir "$target"
  (cd "$target" && sha256sum --quiet -c SHA256SUMS)
done
diff -qr /tmp/crazyflow-gradient-f0-003-repro-a \
  /tmp/crazyflow-gradient-f0-003-repro-b
diff -qr /tmp/crazyflow-gradient-f0-003-repro-a \
  artifacts/day24-unclipped-allocation
diff -qr /tmp/crazyflow-gradient-f0-003-repro-b \
  artifacts/day24-unclipped-allocation
```

Also require Day-19/21/22/23/24 checksum checks, `git diff --check`, the exact eleven-path
allowlist, and unchanged shared source/default/dependency/lockfile diffs. Required result facts are
Default `0/400`; Seed 05 `11/400`, only motor 0 upper at indices 48–58 and `[0.48,0.59) s`;
Stage-A torque-axis clip `0`; Stage-A motor clip `11`; Stage-B additional clip `false`/`0`;
maximum request `74753.53125 PWM`, `0.13687989115715027 N`, `24716.875`; and maximum exceedance
`9218.53125 PWM`, `0.016879891976714134 N`, `1501.0977783203125`. Any failed identity,
nonfinite/undefined inverse, replacement case, partial package or relaxed threshold is
`WITHHELD`.

## Day-25 cf21B_500 robust baseline foundation and D-047 correction

This procedure implements `WO-GR-G2-003`, prospective D-045 and the sole D-047 correction cycle
`CORR-01`. The first independent review of result
`c4bbb57443836aecba52d72cb73a11ee4cbdb5dd` was `NOT ACCEPTED`; the correction does not
rewrite that event as a pass. It measures integral-boundary
contact in both fixed mass variants without using it as a stop/pass criterion; all nonintegral
gates remain hard. It does not revise the blocked/rejected G2.1 or G2.2 records, change a shared
controller/default, perform optimization, or open Day-10, Day-13 or Test data.

The retained numeric contract is exact: every class has the same positive-z Float32 `0.06 m/s`
component, with base displacement `0.06 * (time - 2 s)`, base velocity `0.06 m/s`, zero base
acceleration/jerk, and product derivatives through jerk under the C3 smoothstep7 2-s entry /
4-s plateau / 2-s exit envelope. It is added after class-scaled Fourier construction and before
the fixed center. References have 8-s parent support; the simulation is one continuous 6-s
rollout from the parent start; and the half-open 2-s scored window follows the 2/3/4-s warm-up
without reset. Finiteness, ground/floor and zero-thrust gates cover all 6 s. Integral contact,
torque/motor/Stage-B clipping, tracking, Loss v1, reserve and wrench are scored-window-only.

The correction rewrites exactly two commits from base
`a4e4136f8312851790555e3de6cdf23065252edd`. The first commit contains the four source/test
paths and has subject `research: add cf21b robust evaluator foundation`; its stable generator
identity is `2288ab2fe3b55f585fb9b14a8fe9d87c07eb6c1a`. The second contains the ten package and four
documentation paths and has subject `research: record cf21b robust foundation evidence`. No
third commit is permitted.

Run from the retained worker worktree with the live repository's existing `.venv` and only
`/tmp` caches:

```bash
cd /tmp/crazyflow-gradient-research-wo-gr-g2-001
PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-g2-001 \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-g2-001-corr-pytest-matplotlib \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python -m pytest -q \
  -p no:cacheprovider \
  tests/unit/test_mellinger_robust_evaluation.py \
  tests/unit/test_mellinger_cf21b_robust_foundation.py \
  tests/unit/test_mellinger_research.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_ki_z_sensitivity.py \
  tests/unit/test_mellinger_unclipped_allocation_diagnostic.py \
  tests/unit/control/test_mellinger.py \
  tests/unit/control/test_transform.py

PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-g2-001 \
PYTHONDONTWRITEBYTECODE=1 \
RUFF_CACHE_DIR=/tmp/crazyflow-gradient-g2-001-corr-ruff \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python -m ruff check \
  crazyflow/control/mellinger/research/robust_evaluation.py \
  examples/jax/mellinger_cf21b_robust_foundation.py \
  tests/unit/test_mellinger_robust_evaluation.py \
  tests/unit/test_mellinger_cf21b_robust_foundation.py

PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-g2-001 \
PYTHONDONTWRITEBYTECODE=1 \
RUFF_CACHE_DIR=/tmp/crazyflow-gradient-g2-001-corr-ruff \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python -m ruff format --check \
  crazyflow/control/mellinger/research/robust_evaluation.py \
  examples/jax/mellinger_cf21b_robust_foundation.py \
  tests/unit/test_mellinger_robust_evaluation.py \
  tests/unit/test_mellinger_cf21b_robust_foundation.py
```

The package retains the checksum-pinned pre-correction benchmark input. Its historical
measurement command was:

```bash
cd /tmp/crazyflow-gradient-research-wo-gr-g2-001
PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-g2-001 \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-g2-001-matplotlib \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python \
  examples/jax/mellinger_cf21b_robust_foundation.py \
  --measure-benchmark \
  --benchmark-output /tmp/gr-g2-001-benchmark-measurements.json \
  --world-counts 4 16 32 --steady-repeats 3 --child-timeout-seconds 300
sha256sum /tmp/gr-g2-001-benchmark-measurements.json
```

The executed raw-file SHA-256 is
`9ca5d2e07da2e41b1144117c744cd36a6a908e078bd4f7e21c06d44323b2e8d0`; its canonical
measurement-payload SHA-256 is
`282dc6e652a560152efac0f7dce62dd9d9bbc845fc4c05f5dda4fc48d2e564e5`. The expected rule-based
result on this CPU is 32 worlds, scoped only to the next CPU batch.

Separately, `CORR-01` runs one new non-package 4/16/32 fresh-process benchmark with unchanged
gates. This validates the corrected executable without replacing the package's pinned input:

```bash
cd /tmp/crazyflow-gradient-research-wo-gr-g2-001
test ! -e /tmp/gr-g2-001-corr-benchmark-measurements.json
PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-g2-001 \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-g2-001-corr-benchmark-matplotlib \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python \
  examples/jax/mellinger_cf21b_robust_foundation.py \
  --measure-benchmark \
  --benchmark-output /tmp/gr-g2-001-corr-benchmark-measurements.json \
  --world-counts 4 16 32 --steady-repeats 3 --child-timeout-seconds 300
sha256sum /tmp/gr-g2-001-corr-benchmark-measurements.json
```

The correction run records `PASS_4_16_32_BENCHMARK`, replicated-block consistency PASS, zero
swap, all resource/timeout guards PASS and recommendation 32. Its result-file SHA-256 is
`e1096e1d896ffeb784ffda3c4df0b939b7bf3291f185665e71babbd49c2d9b16`; its canonical raw-
measurement SHA-256 is `a91e0dfefaf008ca9c9b9b031319888855af87f1f3c8912f93b2ed5fe88b06aa`.

Create two fresh reproducibility packages and the final package from that one unchanged
benchmark input. Every target must be absent before its command; overwrite refusal is part of
the contract.

```bash
cd /tmp/crazyflow-gradient-research-wo-gr-g2-001
test ! -e /tmp/gr-g2-001-corr-repro-a
PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-g2-001 \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-g2-001-corr-matplotlib \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python \
  examples/jax/mellinger_cf21b_robust_foundation.py \
  --generate-package \
  --benchmark-input /tmp/gr-g2-001-benchmark-measurements.json \
  --output-dir /tmp/gr-g2-001-corr-repro-a/package

test ! -e /tmp/gr-g2-001-corr-repro-b
PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-g2-001 \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-g2-001-corr-matplotlib \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python \
  examples/jax/mellinger_cf21b_robust_foundation.py \
  --generate-package \
  --benchmark-input /tmp/gr-g2-001-benchmark-measurements.json \
  --output-dir /tmp/gr-g2-001-corr-repro-b/package

test ! -e artifacts/day25-cf21b-robust-foundation
PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-g2-001 \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-g2-001-corr-matplotlib \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python \
  examples/jax/mellinger_cf21b_robust_foundation.py \
  --generate-package \
  --benchmark-input /tmp/gr-g2-001-benchmark-measurements.json \
  --output-dir artifacts/day25-cf21b-robust-foundation
```

Verify all three package identities, historical evidence, and the final repository boundary:

```bash
cd /tmp/crazyflow-gradient-research-wo-gr-g2-001
(cd /tmp/gr-g2-001-corr-repro-a/package && sha256sum --quiet -c SHA256SUMS)
(cd /tmp/gr-g2-001-corr-repro-b/package && sha256sum --quiet -c SHA256SUMS)
diff -qr /tmp/gr-g2-001-corr-repro-a/package /tmp/gr-g2-001-corr-repro-b/package
diff -qr /tmp/gr-g2-001-corr-repro-a/package artifacts/day25-cf21b-robust-foundation
diff -qr /tmp/gr-g2-001-corr-repro-b/package artifacts/day25-cf21b-robust-foundation
(cd artifacts/day19-h100-analysis && sha256sum --quiet -c SHA256SUMS)
(cd artifacts/day20-stage2-loss-diagnostics && sha256sum --quiet -c SHA256SUMS)
(cd artifacts/day21-friday-evidence && sha256sum --quiet -c SHA256SUMS)
(cd artifacts/day22-saturation-diagnostic && sha256sum --quiet -c SHA256SUMS)
(cd artifacts/day23-ki-z-sensitivity && sha256sum --quiet -c SHA256SUMS)
(cd artifacts/day24-unclipped-allocation && sha256sum --quiet -c SHA256SUMS)
(cd artifacts/day25-cf21b-robust-foundation && sha256sum --quiet -c SHA256SUMS)
git diff --check a4e4136f8312851790555e3de6cdf23065252edd..HEAD
git diff --name-status a4e4136f8312851790555e3de6cdf23065252edd..HEAD
git status --short --branch
```

Each package generation automatically compares against the explicit old result
`c4bbb57443836aecba52d72cb73a11ee4cbdb5dd` and must record
`PASS_PRE_CORRECTION_C4_INVARIANCE`. Required unchanged pins are the numeric report projection
`fd1195f9d10c88a2d550090631f6b21d0d4f80fd47a2bbe09a52e7c9a0975a61` (5,465,162 paths;
644,397,249 bytes), parent projection
`bcb40e132f4b85e99f814f0a9db4d90244523b488697868ece8597d3654de82c` (16 records; 18,037
bytes), old report `31d4b468a31b140c1dfe14bd2f1bb3d72ecdf648d4b42094d707d37eb0e74afb`, old contract
`d34f08a9053e489bbf51c91de1791057ebb468bd562a34b6ee1b6976dbb466d6`, and benchmark
`9ca5d2e07da2e41b1144117c744cd36a6a908e078bd4f7e21c06d44323b2e8d0`.

After committing the 14 evidence/documentation paths, generate once more from the result commit
and require byte identity to the final package. This is the stable-generator-provenance gate:

```bash
cd /tmp/crazyflow-gradient-research-wo-gr-g2-001
test ! -e /tmp/gr-g2-001-corr-result-repro
PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-g2-001 \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-g2-001-corr-result-matplotlib \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python \
  examples/jax/mellinger_cf21b_robust_foundation.py \
  --generate-package \
  --benchmark-input /tmp/gr-g2-001-benchmark-measurements.json \
  --output-dir /tmp/gr-g2-001-corr-result-repro/package
(cd /tmp/gr-g2-001-corr-result-repro/package && sha256sum --quiet -c SHA256SUMS)
diff -qr /tmp/gr-g2-001-corr-result-repro/package \
  artifacts/day25-cf21b-robust-foundation
```

Required result: 32 complete paired rollouts; mass-only PyTree isolation; finite arrays,
objectives and gradients; complete carry and production/diagnostic identity; zero ground/floor,
zero thrust over all 6 s; zero scored-window Stage-A torque clip, Stage-A motor clip and Stage-B
additional clip; all report summaries and per-split/per-class Loss-v1/trajectory aggregates
recomputed from stored arrays; and the explicit negative physical-value-match finding.
The final diff must contain exactly the 18 Day-25 allowlist paths. Shared controller/API,
tracking, experiment, trajectory, `__init__.py`, config/TOML, default, dependency, lockfile,
Day-19–24 and protected-path diffs must remain empty. The live checkout status must remain
unchanged. A failed nonintegral gate, partial package, unpinned benchmark input, nonidentical
reproduction, old-c4 comparator failure, numeric projection change, third commit, or widened
claim is `BLOCKED/WITHHELD`.

## Day-26 G3.4 freeze package verification and reproduction

The accepted package is
`artifacts/day26-g3-identifiability-freeze-v2/`. It reports
`COMPLETE_G3_4_PARAMETER_ELIGIBILITY_AUDIT`, uses the fixed
`six-batches-of-four` execution mode, and records optimizer counts `0/0`. It does not authorize
G4; the prospective G4 resource state remains `WITHHELD_PENDING_REBENCHMARK`.

Verify the final package without rewriting it:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
package=artifacts/day26-g3-identifiability-freeze-v2
test -d "$package" && test ! -L "$package"
test "$(find "$package" -maxdepth 1 -type f | wc -l)" -eq 13
test "$(find "$package" -maxdepth 1 -type f -printf '%m\n' | LC_ALL=C sort -u)" = 600
test "$(wc -l < "$package/SHA256SUMS")" -eq 12
(cd "$package" && sha256sum --quiet -c SHA256SUMS)
(cd "$package" && find . -type f -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum) \
  | sha256sum
```

The last command must print exactly:

```text
57934a75f9fcbfec5480184405170fd87b6ff82b8c0cf942e1b55af82f87dbfe  -
```

The inventory is exactly 13 regular mode-`600` files totaling `31,722,155` bytes. The 12 payload
checksums are listed in [`ARTIFACT_MANIFEST.md`](ARTIFACT_MANIFEST.md#day-26-g34-parameter-eligibility-package).
Required semantic facts are:

- status `COMPLETE_G3_4_PARAMETER_ELIGIBILITY_AUDIT`;
- Single-GO `kp_xy`, `kp_z`, and `ki_z`;
- `kd_xy` is `DIAGNOSTIC_ONLY`;
- `kd_z`, `mass`, and `mass_thrust` are withheld for
  `WITHHELD_TECHNICAL_ROBUSTNESS` / `NONFINITE_NEAR_LIMIT_EFFECT_ROLLOUT`;
- joint proposal `kp_xy + kp_z` is for Hauptleitung review only;
- Loss recommendation `RETAIN_LOSS_V1_PLUS_EXTERNAL_GATES`;
- `G4` is `WITHHELD_PENDING_REBENCHMARK`, `g4_executed=false`, and `g5_executed=false`;
- optimizer initialization/update counts are exactly `0/0`.

Normal reproduction must use a clean detached checkout at the exact generator commit
`5c28db1aae6bee67f06a2764f535f5c38924495e`, not the later compatibility or artifact commit.
The output directory must be absent:

```bash
cd /tmp/gr-g3-004-foundation01-generator-reproduction/checkout
test "$(git rev-parse HEAD)" = 5c28db1aae6bee67f06a2764f535f5c38924495e
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test ! -e /tmp/gr-g3-004-foundation01-generator-reproduction/package
PYTHONDONTWRITEBYTECODE=1 \
SCIPY_ARRAY_API=1 \
JAX_PLATFORM_NAME=cpu \
JAX_ENABLE_X64=false \
JAX_ENABLE_COMPILATION_CACHE=false \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python \
  examples/jax/mellinger_g3_freeze_v2.py \
  --generate-package \
  --generator-commit 5c28db1aae6bee67f06a2764f535f5c38924495e \
  --execution-mode six-batches-of-four \
  --batch-timeout-seconds 900 \
  --total-timeout-seconds 2400 \
  --output-dir /tmp/gr-g3-004-foundation01-generator-reproduction/package
```

The reproduction must again produce 13 files, pass all 12 checksums, pass the strict semantic and
PNG validators, and compare byte-for-byte with the accepted package. The generator checkout is
then removed normally, without `--force`, only after the reproduction and detached-clean checks
pass.

CPU/JAX startup telemetry is outside the package and outside the scientific result. A
JAX-starting reproduction may classify stderr as
`ALLOWED_ENVIRONMENT_TELEMETRY_CPU_FALLBACK` only when all of the following hold: the process
otherwise exits zero; provenance independently proves backend `cpu`; stderr is exactly 111 bytes
with SHA-256
`61d6836bb62ae5de8df26847232e5d1848fa056fa22a93506765685f242cd82d`;
and the bytes are the already reviewed JAX NVIDIA/CUDA CPU-fallback notice. Do not filter, rewrite,
normalize, or copy that telemetry into the package. Empty stderr is also valid; every other
stderr byte, nonzero exit, cache/AOT/XLA/SIGILL marker, nonempty sentinel, package mismatch, or
optimizer count other than `0/0` is a hard stop.

The package is pure JAX Float32 CPU simulation evidence. It is not authorization to execute G4,
select or optimize a candidate, access Test, change a controller/default, or perform firmware,
hardware, safety, flight, or Sim2Real work.
