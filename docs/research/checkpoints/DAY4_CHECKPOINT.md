# Day-4 checkpoint

Datum: 2026-07-26 (Europe/Berlin)  
Repository: `/home/noah3/bachelorarbeit/crazyflow-gradient-research`  
Branch: `research/differentiable-mellinger`  
Ausgangs-/End-HEAD: `32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed`  
Sprint-3-Source-State:
`8559648a3c52ca262f8eb5b4160540cc8e138c42c85d748f13daefe996d268f7`

## Sprintgrenze

Sprint 4 ist ein reiner Freeze-, Audit-, Dokumentations- und
Betreuer-Übergabesprint. Es wurde keine wissenschaftliche Funktionalität implementiert, keine
Python-Datei geändert, kein Test/Skript/Notebook/Config ergänzt, keine Abhängigkeit geändert und
kein Simulations- oder Optimierungslauf ausgeführt. Kein vorhandenes Ergebnisartefakt wurde
regeneriert oder verändert.

## Ausgangslage und Provenienz

Der read-only Preflight ergab die zulässige Ausgangslage 1:

```text
## research/differentiable-mellinger
 M crazyflow/control/mellinger/__init__.py
 M docs/research/ARTIFACT_MANIFEST.md
 M docs/research/DECISIONS.md
 M docs/research/PROJECT_STATE.md
 M docs/research/RUNBOOK.md
 M docs/user-guide/mellinger-gradient-rollout.md
?? artifacts/day3-audit/
?? crazyflow/control/mellinger/optimization.py
?? docs/research/checkpoints/DAY3_CHECKPOINT.md
?? examples/jax/mellinger_gain_optimization.py
?? tests/unit/test_mellinger_gain_optimization.py
```

HEAD war
`32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed`, Branch
`research/differentiable-mellinger`. Der wissenschaftliche Source-State wurde mit der in
[`RUNBOOK.md`](../RUNBOOK.md#day-3-optax-gain-optimization) dokumentierten Scope-/Hashmethode
read-only neu berechnet und stimmte exakt mit
`8559648a3c52ca262f8eb5b4160540cc8e138c42c85d748f13daefe996d268f7`
überein.

Es gibt keinen späteren Sprint-3-Checkpoint-Commit. Tag 3 bleibt ein bei HEAD `32cb8ffa`
ausgeführter uncommitted Source-/Test-/Dokumentations-/Artefaktstand. Historische Tag-2-Metadaten
bleiben unverändert.

## Vollständiger Auditumfang

Vollständig gelesen wurden:

- Root-`AGENTS.md`;
- alle drei verlangten Kern-/Optimierungsdateien und `mellinger/__init__.py`;
- alle drei JAX-Beispiele;
- alle drei fokussierten Testdateien;
- der bestehende User Guide;
- `PROJECT_STATE.md`, `RUNBOOK.md`, `ARTIFACT_MANIFEST.md`, `DECISIONS.md`;
- die drei vorhandenen Checkpoints;
- die H200-Hauptergebnis-JSONs der drei Sprints;
- alle drei JSON-Dateien unter `artifacts/day3-audit/optimize-h200/`.

Für die korrekte Erklärung von `SimData`, den zwei Scans, Controllerstufen und Rotoreinheiten
wurden zusätzlich die unmittelbar aufgerufenen Crazyflow-Implementierungen read-only inspiziert.

## Read-only-Verifikation

Einmalig ausgeführte vorgeschriebene Befehle:

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

Tatsächliche Ergebnisse:

- `pip check`: `No broken requirements found`;
- pytest: `36 passed in 118.67s (0:01:58)`;
- Ruff-Lint: `All checks passed!`;
- Ruff-Format: `10 files already formatted`.

Die Preflight-Umgebung war Python `3.12.13` und pip `26.1.2` aus der Repository-`.venv`. Der
nicht schreibbare Benutzer-Pip-Cache erzeugte nur die bereits bekannte Warnung. Alle Zahlen dieses
Abschnitts stammen aus den hier festgehaltenen, im
[`RUNBOOK.md`](../RUNBOOK.md#sprint-4-freeze-audit-and-handover-verification)
wiedergegebenen Befehlsausgaben.

## JSON-, Artefakt- und Plot-Audit

- Alle `20` vorhandenen JSON-Dateien unter `artifacts/` wurden erfolgreich geparst.
- Alle `14` Ergebnis-JSONs mit Feld `all_validations_passed` melden `true`.
- Tag-1-Figure-8-H200, Tag-1-Circle-H200, Tag-2-Batch-H200 und
  Tag-3-Optimierung-H200 melden jeweils alle Hauptvalidierungen bestanden.
- Die ausgewählten Tag-3-Gains, Initial-/Selected-Losses, Schrittwahl und Gradientennormen stimmen
  zwischen den Tag-3-JSONs, `PROJECT_STATE.md` und `DAY3_CHECKPOINT.md` überein.
- Alle `18` Tag-3-Dateien existieren.
- Alle `18` SHA-256-Werte stimmen exakt mit
  [`DAY3_CHECKPOINT.md`](DAY3_CHECKPOINT.md#all-18-artifact-paths-and-sha-256) überein.
- Jedes der drei Tag-3-Verzeichnisse enthält exakt seine sechs dokumentierten Dateien.
- Die drei H200-PNGs und die drei Smoke-Run-1-PNGs wurden visuell in Originalauflösung geprüft.
  Achsen, Datenreihen und Legenden sind lesbar; es gibt keine offensichtlich leere Reihe oder
  abgeschnittene Achse. Die drei Smoke-Run-2-PNGs haben jeweils denselben SHA-256 wie das
  entsprechende visuell geprüfte Smoke-Run-1-PNG und sind damit byteidentisch.

Die Pfad- und Hashwerte stehen vollständig in
[`DAY3_CHECKPOINT.md`](DAY3_CHECKPOINT.md#all-18-artifact-paths-and-sha-256); Sprint 4 hat keinen
Wert und keine Datei verändert.

## Wissenschaftliche Einordnung

- Optimiert wurden vier Gains des reinen JAX-Mellinger-Reglers.
- Train war ausschließlich Figure-8/H200 mit perfektem Simulationszustand.
- Circle war Validation und beeinflusste weder Gradient, Adam-Update noch Auswahl.
- Das beste getestete Train-Checkpoint lag beim letzten getesteten Schritt `50`.
- Seine Train-Gradientennorm war `0.0312928669154644`.
- Ein weiterer lokaler negativer Gradientenschritt senkte Train von
  `0.043769072741270065` auf `0.04345794394612312`.
- Deshalb sind Konvergenz und globale Güte nicht nachgewiesen.
- Die Validation-Änderung von `-34.4263113 %` ist ein positiver Einzelbefund für eine analytische
  Hold-out-Trajektorie.
- Die Matrix umfasst `32` Zellen aus zwei Trajektorien, vier Horizonten, zwei Parametersätzen und
  zwei Controller-Massenbedingungen.
- Peak Memory wurde nicht gemessen.
- Es existiert kein Noise-, Delay-, Disturbance-, Battery-, Aerodynamics-Variation-,
  Estimator-, Hardware-, Firmware- oder sim-to-real-Nachweis.

Quelle aller Ergebniszahlen:
[`optimization_result.json`](../../../artifacts/day3-audit/optimize-h200/optimization_result.json)
und
[`robustness_results.json`](../../../artifacts/day3-audit/optimize-h200/robustness_results.json).

## Dokumentationsbefunde

### Befund D4-A1: veraltete aktuelle Aussage „kein Optimierungsloop“

- Datei/Symbol:
  [`PROJECT_STATE.md`](../PROJECT_STATE.md), Abschnitt „Known limits and risks“.
- Evidenz: Tag 3 enthält den dokumentierten Train-only-Adam-Lauf in
  [`optimization_result.json`](../../../artifacts/day3-audit/optimize-h200/optimization_result.json).
- Auswirkung: Eine aktuelle Limitationsliste konnte den tatsächlichen Projektstand falsch
  darstellen.
- Sprint-4-Behandlung: aktuelle Aussage korrigiert; die historische Tag-2-Grenze und Entscheidung
  D-016 bleiben als historische Sprintgrenze unverändert.

### Befund D4-A2: Beschleunigungs-Sollwert fälschlich als ungenutzt beschrieben

- Datei/Symbol:
  `crazyflow/control/mellinger/control.py`,
  `MellingerStateData.cmd`-Dokumentation versus `state2attitude`.
- Evidenz: `state2attitude` liest `cmd[..., 6:9]` als `setpoint_acc` und verwendet es in
  `mass * (setpoint_acc - gravity_vec)`.
- Auswirkung: Eine fachliche Erklärung könnte den Feed-forward-Anteil weglassen.
- Sprint-4-Behandlung: in
  [`CODE_WALKTHROUGH.md`](../CODE_WALKTHROUGH.md#7-mellinger-stufen-und-konsumierte-setpoint-felder)
  korrekt erklärt; kein Code geändert.
- Vorgeschlagene Betreuerentscheidung: spätere, separat autorisierte Korrektur des
  Quelldocstrings.

### Befund D4-A3: behaupteter Integrator-Reset am nonpositive-thrust gate

- Datei/Symbol:
  [`mellinger-gradient-rollout.md`](../../user-guide/mellinger-gradient-rollout.md),
  Abschnitt „Non-smooth operations“, versus `attitude2force_torque`.
- Evidenz: Der Zweig setzt `torque_pwm` auf null, gibt aber das zuvor berechnete
  `r_int_error` zurück; ein ausdrücklicher Reset der Integratorblätter ist nicht vorhanden.
- Auswirkung: Der Gate-Zustand könnte in einer mündlichen Erklärung falsch beschrieben werden.
- Sprint-4-Behandlung: Befund und tatsächliches Verhalten in
  [`CODE_WALKTHROUGH.md`](../CODE_WALKTHROUGH.md#19-nichtglatte-operationen-und-gradientenauswirkungen)
  dokumentiert; kein Code geändert.
- Vorgeschlagene Betreuerentscheidung: vor einem Hardwarepfad gewünschte Firmwaresemantik und
  Dokumentation abgleichen.

### Befund D4-A4: Rotoreinheit im Transform-Docstring

- Datei/Symbol:
  `crazyflow/control/transform.py`, `motor_force2rotor_vel`.
- Evidenz: Der Rückgabedocstring nennt rad/s; First-Principles-Dynamik und die
  `rpm2thrust`-Kennlinie interpretieren das Feld als RPM und konvertieren es für physikalische
  Terme ausdrücklich nach rad/s.
- Auswirkung: Absolute Rotorgrenzen und Hardwareabbildung könnten mit falscher Einheit
  kommuniziert werden. Die normierten Aufwandsterme bleiben als Quotienten intern
  dimensionslos.
- Sprint-4-Behandlung: im Walkthrough als RPM erklärt und der Widerspruch markiert; kein Code
  geändert.
- Vorgeschlagene Betreuerentscheidung: Einheit vor Firmware-/Hardwareabbildung verbindlich
  bestätigen und später Quelldokumentation vereinheitlichen.

## Neue und aktualisierte Dokumente

Neu:

- `docs/research/CODE_WALKTHROUGH.md`
- `docs/research/RESULTS_SUMMARY.md`
- `docs/research/SUPERVISOR_MEETING_BRIEF.md`
- `docs/research/checkpoints/DAY4_CHECKPOINT.md`

Aktualisiert:

- `docs/research/RUNBOOK.md`
- `docs/research/PROJECT_STATE.md`
- `docs/research/ARTIFACT_MANIFEST.md`
- `docs/research/DECISIONS.md`

Sprint 4 hat keine neuen wissenschaftlichen Artefakte erzeugt.

## Abnahme

- [x] Keine Python-, Test-, Dependency-, Lock-, Default- oder Ergebnisartefaktdatei geändert.
- [x] Wissenschaftlicher Sprint-3-Source-State unverändert.
- [x] `pip check`, vorhandene fokussierte Tests sowie Ruff-Lint und -Format bestanden.
- [x] Alle vorhandenen JSONs parsebar; dokumentierte Hauptvalidierungen wahr.
- [x] Alle Tag-3-Dateien, exakte Verzeichnissätze und SHA-256 geprüft.
- [x] Tag-3-PNGs visuell beziehungsweise bei byteidentischem Replay inhaltlich abgedeckt.
- [x] Walkthrough erklärt Datenfluss, Shapes, Einheiten, Mathematik, Aufrufe und alle
  projektspezifischen Funktionen der relevanten Dateien.
- [x] Ergebniszahlen in der Ergebniszusammenfassung tragen konkrete Quellen.
- [x] Gesprächsbrief trennt Fakten, Grenzen, Hypothesen und Entscheidungen.
- [x] Keine Hardwarebereitschaft, Konvergenz, globale Güte, breite Generalisierung oder
  Controllerüberlegenheit behauptet.
- [x] Autoritative Pflichtdokumente aktualisiert.
- [x] Keine neuen wissenschaftlichen Ergebnisse oder Artefakte erzeugt.

## Abschließender Repository-Audit

Nach Fertigstellung aller Dokumente wurde der wissenschaftliche Source-State erneut mit der
dokumentierten Scope-/Hashmethode geprüft und stimmte weiterhin exakt mit
`8559648a3c52ca262f8eb5b4160540cc8e138c42c85d748f13daefe996d268f7`
überein. Alle `20` JSON-Dateien blieben parsebar, alle `14` vorhandenen Ergebnisflags blieben
`true`, alle `18` Tag-3-Hashes stimmten weiterhin, und alle drei Tag-3-Verzeichnisse enthielten
weiterhin exakt sechs Dateien. `git diff --check` bestand ohne Ausgabe.

Abschließender `git status --short --branch`:

```text
## research/differentiable-mellinger
 M crazyflow/control/mellinger/__init__.py
 M docs/research/ARTIFACT_MANIFEST.md
 M docs/research/DECISIONS.md
 M docs/research/PROJECT_STATE.md
 M docs/research/RUNBOOK.md
 M docs/user-guide/mellinger-gradient-rollout.md
?? artifacts/day3-audit/
?? crazyflow/control/mellinger/optimization.py
?? docs/research/CODE_WALKTHROUGH.md
?? docs/research/RESULTS_SUMMARY.md
?? docs/research/SUPERVISOR_MEETING_BRIEF.md
?? docs/research/checkpoints/DAY3_CHECKPOINT.md
?? docs/research/checkpoints/DAY4_CHECKPOINT.md
?? examples/jax/mellinger_gain_optimization.py
?? tests/unit/test_mellinger_gain_optimization.py
```

Gegenüber dem Preflight kamen ausschließlich die vier erlaubten neuen Sprint-4-Markdown-Dateien
hinzu; die vier verpflichtenden Forschungsdokumente wurden inhaltlich aktualisiert. Alle anderen
Statuszeilen gehören unverändert zum dokumentierten Sprint-3-Ausgangspunkt.
