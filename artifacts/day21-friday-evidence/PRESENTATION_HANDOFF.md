# Friday presentation handoff

## Fixed labels and boundary

Seed 05 at update 5000 is used only as the **provisional visualization candidate**, selected by the
smallest already existing selected fixed-Validation loss among the ten frozen H100 runs. It is
not a final candidate or flight candidate, and no gains were averaged. Test remained unopened.

| Figure | What it proves | What it does not prove | One-sentence slide title |
|---|---|---|---|
| `h100_loss_curves.png` | All ten recorded Train/fixed-Validation histories contain updates 1–5000 under the frozen Stage-1 protocol. | Convergence, Test performance, robustness or generalization. | All ten frozen H100 Train/Validation histories are recorded through update 5000. |
| `h100_runtime_and_reliability.png` | 10/10 runs completed without documented abort; per-process GNU wall time is available; accepted selected-Validation values show small numerical spread. | Calendar/GPU time or a robust success probability. | Existing runs completed technically, with tightly clustered selected Validation results. |
| `h100_selected_gain_evolution.png` | Checkpoint-selected physical Stage-1 gains at updates 100…5000 for each seed. | Continuous between-checkpoint behavior, convergence or a valid averaged controller. | Selected gain paths are visible per seed without gain averaging. |
| `h100_selected_end_gains.png` | The ten selected end-gain combinations and the visualization-only Seed-05 highlight. | Flight suitability or final candidate approval. | Seed 05 is highlighted only for visualization, not selected for flight. |
| `stage2_loss_contributions.png` | Six existing weighted loss contributions for Figure-8/Train and Circle/Validation at H20, defaults, zero updates. | Optimized-gain behavior or convergence. | Stage-2 makes all six existing loss contributions inspectable. |
| `stage2_loss_term_gain_gradients.png` | 48 local Jacobians with respect to four dimensionless sigmoid-bound Stage-1 variables in the same H20 smoke. | Physical-unit gradients, global sensitivity or optimized behavior. | Local loss×variable sensitivities are explicit, but remain an infrastructure smoke. |
| Default/candidate rollout | No rollout figure is included. | The attempted narrow comparison did not meet the unchanged zero-motor-saturation acceptance gate. | Rollout evidence is openly withheld; no replacement value is shown. |


## Missing rollout evidence

`rollout_trajectory.png` and `rollout_tracking_error.png` are intentionally absent. The narrow
comparison did not meet the unchanged zero-motor-saturation acceptance gate, so no rollout arrays,
metrics, plots or replacement values are included. A different candidate, case or threshold was
not substituted.


## Duration and reliability wording

- Total `9286.50` s is the sum of ten
  process wall times, not calendar duration, GPU time or parallel elapsed time.
- Technical reliability is exactly 10/10 complete/no documented abort under the frozen protocol.
- Numerical variation is the accepted distribution of selected fixed-Validation loss and is not
  a success probability.

## Stage-2 claim boundary

The copied plots are byte-identical accepted Day-20 evidence: six existing loss terms and 48 local
Jacobians with respect to four dimensionless internal sigmoid-bound variables; Figure-8/Train and
Circle/Validation; H20; default gains; zero optimizer updates.

## Global claim boundary

Simulation evidence only. No Test, H200/H400 study, new optimization, convergence proof,
generalization, robustness, controller superiority, firmware equivalence, hardware validation,
flight safety, `cf21B_500` applicability or Sim2Real transfer is established.
