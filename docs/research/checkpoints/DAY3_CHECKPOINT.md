# Day-3 checkpoint

Date: 2026-07-24 (Europe/Berlin)  
Repository: `/home/noah3/bachelorarbeit/crazyflow-gradient-research`  
Branch: `research/differentiable-mellinger`  
Base/end HEAD: `32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed` (no Sprint-3 commit)  
Scientific source state:
`8559648a3c52ca262f8eb5b4160540cc8e138c42c85d748f13daefe996d268f7`

## Provenance transition and preflight

The user authorized `32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed` as the later clean Sprint-2
checkpoint. Its direct parent is the documented Day-2 execution HEAD
`9eac04ea28997e76b9bc8c6e97ca53ef93398d55`. The commit contains exactly the documented
Day-2 source, test, documentation, and twelve artifact files. The twelve artifacts are tracked;
their content was not changed.

Historical Day-2 execution metadata remains:

- HEAD `9eac04ea28997e76b9bc8c6e97ca53ef93398d55`;
- `source_dirty: true`;
- source state `a1d7f1dce6ed3b1aa9cbb7cc53abb4a3bcada379129dfbde1787c1a5d1564229`.

Sprint-3 JSONs instead record:

- `base_head: 32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed`;
- clean preflight worktree;
- the separate historical Day-2 execution block;
- source state `8559648a3c52ca262f8eb5b4160540cc8e138c42c85d748f13daefe996d268f7`.

Read-only preflight:

- branch and HEAD correct;
- direct parent exactly `9eac04ea...`;
- commit file set exact and free of unrelated files;
- worktree clean;
- existing regression `21 passed in 48.01s`;
- H200 Day-2 JSON retained `all_validations_passed: true`;
- Python `3.12.13`, pip `26.1.2`, and `No broken requirements found`;
- `git diff --check` passed.

## Dependency result

`.venv/bin/python -c "import optax; print(optax.__version__)"` returned `0.2.8`. The same exact
wheel and SHA-256 were already resolved in `pixi.lock` as a transitive Flax dependency. No package
was installed, and `pyproject.toml`, `pixi.lock`, environment metadata, and the local environment
were not changed.

## Implemented files

New:

- `crazyflow/control/mellinger/optimization.py`
- `examples/jax/mellinger_gain_optimization.py`
- `tests/unit/test_mellinger_gain_optimization.py`
- `docs/research/checkpoints/DAY3_CHECKPOINT.md`

Modified:

- `crazyflow/control/mellinger/__init__.py`
- `docs/user-guide/mellinger-gradient-rollout.md`
- `docs/research/PROJECT_STATE.md`
- `docs/research/RUNBOOK.md`
- `docs/research/ARTIFACT_MANIFEST.md`
- `docs/research/DECISIONS.md`

Unchanged:

- `crazyflow/control/mellinger/tracking.py`
- `examples/jax/mellinger_tracking.py`
- `examples/jax/mellinger_batch_diagnostics.py`
- all Day-1/Day-2 tests;
- all dependency/lock files;
- all checked-in controller/dynamics defaults;
- all Day-1/Day-2 artifacts.

## Optimization design

The main experiment used `N=2, M=1`, H200, 500 Hz simulation, 100 Hz control, seed `20260724`,
`optax.adam`, learning rate `1e-2`, and 50 updates. The smoke runs used the same design at H20 for
three updates. The unchanged four raw scalar leaves map through the existing logistic transform:
`kp_xy`, `kp_z`, `kd_xy`, and `kd_z`.

The scalar passed to `jax.value_and_grad(..., has_aux=True)` is explicitly
`per_case_loss[0]`, Figure-8/world-0 train loss. Only its gradient reaches Optax. Circle/world 1
and the equal combined mean are evaluated and recorded after every step but never influence the
gradient, Adam state, update, hyperparameters, or selection. The selected checkpoint is minimum
train loss with the earliest exact tie.

The objective remains pure JAX and functional. The temporal recurrence remains `jax.lax.scan`.
There are no host callbacks, differentiated NumPy conversions, native bridge calls,
rematerialization, or gradient checkpointing.

## Main optimization result

| State | Step | Train loss | Validation loss | Combined loss |
|---|---:|---:|---:|---:|
| Initial | 0 | `0.07310392707586288` | `0.024092912673950195` | `0.04859841987490654` |
| Final | 50 | `0.04376906901597977` | `0.01579861156642437` | `0.029783841222524643` |
| Best train | 50 | `0.04376906901597977` | `0.01579861156642437` | `0.029783841222524643` |

Step 0 differs from the Day-2 H200 train baseline by only `-3.7252903e-8` and passes
`rtol=1e-5`, `atol=1e-7`. All 50 parameter PyTrees, gradients, and Adam states were finite.

Initial raw gains in `(kp_xy, kp_z, kd_xy, kd_z)` order:

```text
(-0.9808291792869568, -0.2744370102882385,
 -1.3862943649291992, -0.5596157312393188)
```

Initial physical gains:

```text
(0.4000000059604645, 1.25, 0.20000000298023224, 0.5)
```

Final/best raw gains:

```text
(-1.411259651184082, 0.19198988378047943,
 -0.919587254524231, -0.11490653455257416)
```

Final/best physical gains:

```text
(0.31563901901245117, 1.5052711963653564,
 0.26378148794174194, 0.6184355020523071)
```

Selected physical distances to lower/upper limits:

| Gain | To lower | To upper |
|---|---:|---:|
| `kp_xy` | `0.2156390190` | `0.8843609810` |
| `kp_z` | `1.2052711964` | `0.9947288036` |
| `kd_xy` | `0.2137814879` | `0.5362185121` |
| `kd_z` | `0.5184355021` | `0.5815644979` |

Initial-to-selected loss changes:

| Objective | Difference | Relative |
|---|---:|---:|
| Train | `-0.029334858059883118` | `-40.1276091%` |
| Validation | `-0.008294301107525826` | `-34.4263113%` |
| Combined | `-0.018814578652381897` | `-38.7143835%` |

Validation improvement is an observed diagnostic result and was not a target or acceptance
direction.

## Gradient control at selected checkpoint

Raw leaves and L2 norms:

| Objective | `kp_xy` | `kp_z` | `kd_xy` | `kd_z` | Norm |
|---|---:|---:|---:|---:|---:|
| Train | `0.00706788665` | `-0.00632207049` | `-0.02975456789` | `-0.00199637772` | `0.03129286692` |
| Validation | `0.00019996922` | `-0.00524421595` | `-0.00637444854` | `-0.00151096261` | `0.00839394983` |
| Combined | `0.00363392709` | `-0.00578314299` | `-0.01806450821` | `-0.00175367005` | `0.01939206012` |

Train/validation gradient cosine: `0.865163266658783`.

Sprint-3 directions are fixed normalized raw vectors for `kd_xy`, `kd_z`, and
`(kp_xy + kp_z + kd_xy + kd_z) / 2`. Maximum relative central-difference errors at epsilon
`1e-2`:

- train `0.002328432397916913`;
- combined `0.0015936176059767604`.

The train norm exceeded `1e-8`, so the required normalized negative-gradient step was evaluated:
`0.043769072741270065 -> 0.04345794394612312`, difference `-0.00031112879514694214`.

Aggregation checks:

- batch combined minus separate `N=1` mean: `-1.862645149230957e-8`;
- combined-gradient maximum leaf difference: `2.2351741790771484e-8`;
- reversed-world loss difference: exactly `0`;
- reversed-world maximum gradient difference: `1.862645149230957e-9`;
- JIT/eager loss difference: `-5.587935447692871e-9`;
- JIT/eager maximum gradient difference: `1.862645149230957e-8`;
- JAXPR still contains `scan`.

At the selected point, both trajectories had exactly zero motor-saturation, nonpositive-thrust
gate, floor-clip, and nonfinite-state fractions. These diagnostic branches were not hidden or
used to loosen gradient thresholds.

## Robustness matrix

The matrix contains exactly 32 unique finite forward cells:

```text
2 parameter sets x 4 horizons x 2 trajectories x 2 mass conditions
```

Mass conditions:

- `audited_mismatch`: controller `0.029 kg`, dynamics `0.0319 kg`;
- `matched_controller_to_dynamics`: experiment-local controller `0.0319 kg`, unchanged dynamics
  `0.0319 kg`.

No optimization or differentiation was performed in the matrix. All cells had zero saturation,
gate, floor, and nonfinite fractions.

| Parameter set | Matrix mean | Matrix maximum | Worst cell |
|---|---:|---:|---|
| Baseline | `0.03194340391564765` | `0.07911745458841324` | H400 Figure-8 audited mismatch |
| Best train | `0.019883187764207833` | `0.05063283443450928` | H100 Figure-8 audited mismatch |

Mass-level means:

| Parameter set | Audited mismatch | Matched | Matched minus mismatch |
|---|---:|---:|---:|
| Baseline | `0.03735753348155413` | `0.026529274349741172` | `-0.010828259131812956` |
| Best train | `0.023419516684953123` | `0.01634685884346254` | `-0.007072657841490582` |

H200 losses:

| Parameters | Trajectory | Mismatch | Matched |
|---|---|---:|---:|
| Baseline | Figure-8 | `0.0731038972735405` | `0.05819285288453102` |
| Best train | Figure-8 | `0.04376910254359245` | `0.03429412469267845` |
| Baseline | Circle | `0.024092918261885643` | `0.010177765041589737` |
| Best train | Circle | `0.015798604115843773` | `0.006269511301070452` |

All 16 best-train cells happened to have lower loss than their corresponding baseline cells.
This was not required and is not evidence of broad robustness or causality.

## Profiling and H400 gate

All measurements used `time.perf_counter_ns` and `jax.block_until_ready`; plotting and JSON I/O
were outside timed regions.

- compile plus first optimization step: `15.743286041 s`;
- cached Optax step min/median/max:
  `0.150861597/0.153715902/0.156133985 s`;
- 50-update loop: `32.805345890 s`.

| Horizon | Cached forward min/median/max [s] | Cached forward/backward min/median/max [s] |
|---:|---|---|
| 20 | `0.002154341/0.002343864/0.002529971` | `0.018295410/0.021593010/0.022080168` |
| 100 | `0.003684601/0.004290380/0.004302392` | `0.081877362/0.084925729/0.086341175` |
| 200 | `0.006359332/0.006714594/0.007662749` | `0.168802769/0.175491063/0.183454022` |
| 400 | `0.012123366/0.013119240/0.014692449` | `0.321645154/0.328635742/0.331804723` |

H400 completed. Its cached forward/backward median divided by H200 was `1.8726636922815836`,
below the factor-10 later-sprint scaling indicator. No speedup is claimed, peak memory was not
measured, and no rematerialization was introduced.

## Smoke reproducibility

Both H20 runs selected step 3 and changed train loss from `0.0022911166306585073` to
`0.002285428112372756`. Both reported all validations true.

After removing only the six timing keys named in `RUNBOOK.md`:

- `optimization_result.json` scientific payloads were equal;
- `optimized_gains.json` files were byte-identical;
- `robustness_results.json` files were byte-identical;
- all three PNG pairs were byte-identical.

Plot hashes for both runs:

- `gain_history.png`:
  `e0f8a295b5b680dfc0720355d201a4cf88170da67b4aa81b5a87cb1a770035bc`;
- `optimization_history.png`:
  `f2ae04703199f7a2c0a5ac2794509e386e034626fc6d12acf4ecdc79bff082ce`;
- `robustness_matrix.png`:
  `5f1a46358a9b78070c0224ef7018217ddf11f655f9ce6f2b299078d0b46e4539`.

## All 18 artifact paths and SHA-256

Smoke run 1:

- `artifacts/day3-audit/smoke-run-1/optimization_result.json` —
  `4e186701e3afb2d3854fcfda05c4e20f9bd4820298435c3b8c66a891a634460f`
- `artifacts/day3-audit/smoke-run-1/optimized_gains.json` —
  `a6ff65bb8792645c74c25a02d2fe7814fb476554d66f5d474853d35e9e2e3631`
- `artifacts/day3-audit/smoke-run-1/robustness_results.json` —
  `7425ce71b5d6bf4d98d7551f171b1c5284de32e8eddc774dbc833a4ea2e1c59a`
- `artifacts/day3-audit/smoke-run-1/optimization_history.png` —
  `f2ae04703199f7a2c0a5ac2794509e386e034626fc6d12acf4ecdc79bff082ce`
- `artifacts/day3-audit/smoke-run-1/gain_history.png` —
  `e0f8a295b5b680dfc0720355d201a4cf88170da67b4aa81b5a87cb1a770035bc`
- `artifacts/day3-audit/smoke-run-1/robustness_matrix.png` —
  `5f1a46358a9b78070c0224ef7018217ddf11f655f9ce6f2b299078d0b46e4539`

Smoke run 2:

- `artifacts/day3-audit/smoke-run-2/optimization_result.json` —
  `47ebbb0eae2ca1c8eb7cf25013c0a1ec4e7a3ed265a4606c19211e358087ceb4`
- `artifacts/day3-audit/smoke-run-2/optimized_gains.json` —
  `a6ff65bb8792645c74c25a02d2fe7814fb476554d66f5d474853d35e9e2e3631`
- `artifacts/day3-audit/smoke-run-2/robustness_results.json` —
  `7425ce71b5d6bf4d98d7551f171b1c5284de32e8eddc774dbc833a4ea2e1c59a`
- `artifacts/day3-audit/smoke-run-2/optimization_history.png` —
  `f2ae04703199f7a2c0a5ac2794509e386e034626fc6d12acf4ecdc79bff082ce`
- `artifacts/day3-audit/smoke-run-2/gain_history.png` —
  `e0f8a295b5b680dfc0720355d201a4cf88170da67b4aa81b5a87cb1a770035bc`
- `artifacts/day3-audit/smoke-run-2/robustness_matrix.png` —
  `5f1a46358a9b78070c0224ef7018217ddf11f655f9ce6f2b299078d0b46e4539`

H200 main:

- `artifacts/day3-audit/optimize-h200/optimization_result.json` —
  `a6a2e2b9cf18b014e263e1651008404cafb024bb0faefdc631444fbaa1e72e75`
- `artifacts/day3-audit/optimize-h200/optimized_gains.json` —
  `a641783e3cf929b79d5c40e8f1cad7ed9e311fa3a96339dbc0312ec0d5ff4d5a`
- `artifacts/day3-audit/optimize-h200/robustness_results.json` —
  `acb5d9c013862d4a0e46155016af56d70b289fa3ad38053dce081805a7de9ade`
- `artifacts/day3-audit/optimize-h200/optimization_history.png` —
  `99888e47ad7dd88743b43638edf1bffffa6e5ddcb20984cc96361311bb2a06f5`
- `artifacts/day3-audit/optimize-h200/gain_history.png` —
  `6899c7130abb601df4d62fd07ce80abcaa7778c3393d1fbd53b9367bb814d10e`
- `artifacts/day3-audit/optimize-h200/robustness_matrix.png` —
  `bf54eb82a2816fdde69411dac07c8973b9c7858d0f0f2d8845ecfc58c820fdf6`

Each directory contains exactly its six required filenames.

## Acceptance result

Final prescribed quality command results:

- `.venv/bin/python -m pip check`: `No broken requirements found`;
- combined focused pytest: `36 passed in 170.92s`;
- Ruff lint: `All checks passed!`;
- Ruff format: `9 files already formatted`.

- [x] 21 Day-1/Day-2 regression tests preserved.
- [x] 15 new tests implemented and focused combined suite passed.
- [x] `pip check`, Ruff lint, Ruff format, and `git diff --check` passed.
- [x] H200 Step 0 reproduced Day-2 train baseline within fixed tolerances.
- [x] All 50 updates had finite losses, gradients, parameters, and Optax states.
- [x] Selected step is greater than zero and lowers train loss.
- [x] Validation did not affect gradients, updates, hyperparameters, or selection.
- [x] All physical gains remain strictly inside unchanged logistic bounds.
- [x] Selected-point gradient controls passed without changing epsilon or threshold.
- [x] Exactly 32 finite robustness cells completed.
- [x] H400 completed; scaling ratio did not trigger the factor-10 indicator.
- [x] Controller/dynamics defaults and all Day-1/Day-2 artifacts remained unchanged.
- [x] Two smoke runs are scientifically identical and all plots byte-identical.
- [x] Every output directory contains exactly six files.
- [x] Main result reports `all_validations_passed: true`.
- [x] No native bridge, BetaFlight, TinyMPC, hardware, noise, or disturbance feature was added.

## Scientific limitations

Two analytic trajectories, four horizons, and two controller-mass conditions are controlled
simulation evidence only. They do not establish broad generalization, disturbance rejection,
stochastic seed robustness, physical accuracy, firmware equivalence, hardware safety, or
sim-to-real performance. The selected gains are a research artifact, not a default configuration.
