# Sprint 12: eingefrorenes H100-Replikationsprotokoll

## Status und Grenze

Dieses Verzeichnis schreibt das spätere konfirmatorische H100-Verfahren vollständig vorab fest.
Sprint 12 autorisiert und startet **keinen** Forschungsrun. Der erfolgreiche Sprint-10-Langpilot
mit Seed `20260731` ist ausschließlich explorative Planungsevidenz und gehört weder zu den zehn
Replikationsseeds noch in deren Aggregation.

Das vor Seed 01 angenommene Sprint-12A-Amendment
`amendments/2026-08-01-local-ssd-persistence-v1.md` ersetzt ausschließlich die ursprüngliche
externe Persistenzpflicht durch lokale, prüfsummengeschützte Aufbewahrung auf der internen SSD.
Der Basis-Freeze bleibt in Commit `73868cfd84d25bec0c6612b31ee34f4e67acb307` historisch
nachvollziehbar. Alle wissenschaftlichen und statistischen Regeln bleiben unverändert.

Maschinenlesbare Quellen:

- `protocol.json`: oberster Freeze- und Freigabestatus;
- `seed_manifest.json`: Seederzeugung und zehn Seeds;
- `requirements.json`: Config-, Source-, Runtime- und wissenschaftliche Semantik;
- `run_inventory.json`: feste Reihenfolge, Configs, Pfade und Ressourcenplanung;
- `analysis_plan.json`: Auswahl, Endpunkte, Statistik, Fehler- und GO-/NO-GO-Regeln;
- `release_gate.json`: ausdrückliches NO-GO für Runs in Sprint 12 und Sprint 12A;
- `amendments/2026-08-01-local-ssd-persistence-v1.json`: maschinenlesbares aktives Amendment;
- `amendments/2026-08-01-local-ssd-persistence-v1.md`: menschenlesbarer Änderungsnachweis;
- `SHA256SUMS`: vollständiges Protokollinventar.

## Vorab deklarierte Seeds

Die Seeds werden nicht aus Pilotergebnissen ausgewählt. Für Index `NN` wird

```text
SHA256(
  "crazyflow-gradient-research|h100-confirmatory-replication|v1|" +
  "a269046fb0e70cb5ecc8b4c1783b87aecd866b7f|seed-index=NN|retry=K"
)
```

berechnet. Die ersten vier Digest-Bytes werden als unsigned Big-Endian gelesen und mit
`0x7fffffff` auf 31 Bit begrenzt. `K` wird nur bei 0, einer Kollision oder dem ausgeschlossenen
Pilotseed erhöht. Alle zehn Seeds benötigten `K=0`:

| Index | Root-Seed |
|---:|---:|
| 01 | 1432116264 |
| 02 | 366692846 |
| 03 | 235438753 |
| 04 | 1369406745 |
| 05 | 1081462774 |
| 06 | 1276309202 |
| 07 | 987511476 |
| 08 | 2125653507 |
| 09 | 31735934 |
| 10 | 986925065 |

Die unterschiedlichen Root-Seeds erzeugen über den vorhandenen
Split/Episode/World/Component-`fold_in`-Vertrag getrennte deterministische Train-Ströme. Die vier
Validation-Welten bleiben absichtlich bei allen Seeds identisch und werden aus demselben
versionierten Validation-Manifest gebaut.

## Eingefrorene wissenschaftliche Semantik

Jede der zehn Configs unterscheidet sich von der Sprint-10-Pilotconfig ausschließlich in
`run_id` und `root_seed`:

- CPU, `cf2x_L250`, First-Principles-Dynamik und `Control.state`;
- H100, 500 Hz Simulation, 100 Hz Control, ein Drone je Welt;
- vier Train-Welten mit neuer Episode pro Update;
- dieselben vier festen Validation-Welten pro Update;
- genau 5.000 Adam-Updates, Lernrate `0.001`, Checkpointintervall 100;
- `fixed_support_prefix_v2` und alle unveränderten Trajektorienparameter aus Sprint 10;
- ein gemeinsamer Stage-1-Gainvektor: `kp_xy`, `kp_z`, `kd_xy`, `kd_z`;
- Bounds: `[0.10,1.20]`, `[0.30,2.50]`, `[0.05,0.80]`, `[0.10,1.20]`;
- Controller-Masse `0.029 kg` bleibt unverändert;
- nominale Dynamikmasse `0.0319 kg`, pro Train-Episode/Welt ausschließlich ±`0.0002 kg`;
- Lossgewichte Position `1.0`, Geschwindigkeit `0.10`, Effort `0.001`, Smoothness `0.001`,
  Terminal `0.10`, Altitude `0.05`;
- Loss-Skalen Position `0.25`, Geschwindigkeit `1.0`, Altitude-Margin `0.15` und Softness `0.05`;
- gleiche arithmetische Mittelung über die vier Fälle jedes Splits.

Deaktiviert bleiben Action Delay, Wrench-Randomisierung, Sensorrauschen, UKF, alle Nicht-Stage-1-
Gains, H200, H400, Test, automatische Verlängerung, automatische Fortsetzung, Resume und parallele
Seed-Ausführung. Der Testpfad bleibt ausschließlich eine opake Config-Referenz und darf nicht
geöffnet, geparst, validiert, gebaut, simuliert oder ausgewertet werden.

Erforderlich sind Source-Fingerprint
`6b3fed4a445b82b2a1354115edc982a15e0d93cd4aff91fc8f51475e60f01c5e`, Runtime-Fingerprint
`e3c408d547653ebbb87107ca3c31b844c12acc43448378a5a48266306ee3d2a4`, CPU-Backend,
Python 3.12.13, JAX/JAXlib 0.10.1, NumPy 2.5.1, Optax 0.2.8 und `jax_enable_x64=false`.

## Exakte Auswahlregel

Kandidaten sind ausschließlich die 5.000 abgeschlossenen Post-Update-Zustände 1 bis 5.000.
Nach jedem Update wird die Loss auf den festen vier Validation-Welten bestimmt. Der Runner ersetzt
die Auswahl nur bei

```text
validation_loss < best_validation
```

und wählt damit das globale Validation-Minimum. Eine exakte Gleichheit ersetzt die bestehende
Auswahl nicht; der früheste exakte Tie gewinnt. Schritt 0 ist kein Kandidat. Test wird nicht
verwendet.

Die ältere Kurzform „earliest checkpoint“ bedeutet nicht, dass nur die Schritte 100, 200, ...
verglichen werden: Der Code vergleicht alle Updates. Jeder Intervallcheckpoint speichert
`selected_step` und `selected_raw_gains`, auch wenn der ausgewählte Schritt zwischen zwei
Checkpointdateien liegt.

## Primäres Ergebnis und Unsicherheit

Primär ist pro Seed `run_summary.selected.selected_validation_loss`. Über die zehn Seeds werden
in Manifestreihenfolge berichtet:

1. alle zehn Einzelwerte;
2. arithmetisches Mittel;
3. Stichprobenstandardabweichung mit `ddof=1`;
4. zweiseitiges 95%-Student-t-Intervall des Mittels mit neun Freiheitsgraden und
   `t(0.975,9)=2.2621571627409915`;
5. Median sowie Q1/Q3 mit `numpy.quantile(..., method="linear")`.

Unterstützend wird je Seed die relative Validation-Verbesserung
`1 - selected_validation_loss / validation_loss_at_step_1` berechnet und identisch aggregiert.

Sekundär und ausschließlich deskriptiv sind ausgewählter Schritt, Loss bei Schritt 5.000,
Train-Loss und Gradient-L2 am ausgewählten/finalen Schritt, Validation-minus-Train-Lücke,
ausgewählte/finale Gains und Bound-Abstände, die vorhandenen Losskomponenten, Tracking-/Technical-
Metriken sowie Walltime, Peak-RSS, Swap und Artefaktvolumen. Es werden keine nicht vom Runner
erzeugten Basisgrößen behauptet, keine sekundären p-Werte berechnet und keine Multiplizitätsaussage
gemacht.

## Fehlläufe, Wiederholungen und Ausschlüsse

- Ein Preflight-Mismatch oder eine lokale Kapazitäts-/Integritätsabweichung blockiert den Start
  und zählt nicht als Versuch.
- Es gibt keine Ersatzseeds und keine Imputation.
- Ein dokumentierter exogener Host-, Strom- oder Speicherfehler darf höchstens einmal mit
  identischem Seed und identischer Config vollständig ab Update 0 wiederholt werden.
- Resume ist auch bei einem exogenen Fehllauf ausgeschlossen.
- Nonfinite-Werte, Train-/Validation-Gatefehler, Trajektorienfehler, ein regulärer Timeout,
  Peak-RSS ≥12 GiB, Swap, Fingerprintabweichung oder Testzugriff werden nicht wiederholt.
- Jeder Fehllauf bleibt unverändert lokal erhalten, wird vor einer zulässigen Wiederholung
  vollständig gehasht und klassifiziert und darf nicht gelöscht, bewegt, umbenannt, komprimiert,
  verändert oder überschrieben werden.
- Ein zweiter oder nicht wiederholbarer Fehler macht den Seed ungültig, stoppt weitere Starts und
  setzt das Panel auf NO-GO.
- Nur der erfolgreiche Lauf desselben Seeds wird bei einer erlaubten exogenen Wiederholung
  aggregiert; der erste Versuch bleibt Audit-Evidenz.

## GO/NO-GO für einen späteren Horizontschritt

GO erlaubt nur die Vorbereitung eines späteren, wiederum vorab eingefrorenen H200/H400-Protokolls
und verlangt gemeinsam:

- 10/10 gültige vorab deklarierte Seeds ohne Ersatz;
- sämtliche Completion-, Checksum-, Finite-, Technical-, Fingerprint-, Ressourcen- und Test-Gates;
- positive relative Validation-Verbesserung bei 10/10 Seeds;
- mindestens 50% Verbesserung bei mindestens 8/10 Seeds;
- untere Grenze des eingefrorenen 95%-t-Intervalls der mittleren Verbesserung mindestens 50%;
- relative 95%-CI-Halbbreite des mittleren primären Loss höchstens 25%;
- finale Validation-Loss jedes Seeds höchstens 110% seines ausgewählten Minimums;
- normalisierter nächster Bound-Abstand jedes ausgewählten Gains mindestens 0.01;
- vollständige unveränderte lokale Rohdaten sowie erfolgreiche unmittelbare und vorgeschriebene
  erneute Prüfung sämtlicher SHA-256-Indizes;
- keinerlei ungeplante wissenschaftliche, statistische, Runtime- oder Ablaufänderung.

Jede verletzte Bedingung ist NO-GO. Dann werden Evidenz und Diagnose bewahrt, aber weder Gains
nachgetunt noch Seeds ersetzt oder ergänzt, Horizonte gestartet oder Test geöffnet.

## Sequenz, Zeit, Speicher und Persistenz

Ausführung erfolgt in aufsteigender Manifestreihenfolge, immer mit genau einem frischen CPU-
Prozess. Der nächste Seed darf erst starten, nachdem der vorherige Run vollständig geprüft und
sein vollständiger lokaler SHA-256-Index erneut erfolgreich geprüft wurde.

- beobachtete Planung pro Seed: 925,96 s = 15:25,96 und 2.201.956 KiB Peak-RSS;
- konservative Planung pro Seed: 2.289,55 s = 38:09,55;
- hart pro Seed: 7.200 s und 12 GiB Peak-RSS;
- zehn Seeds erwartet: 9.259,6 s = 2:34:19,60 reine Compute-Walltime;
- zehn Seeds konservativ: 22.895,45 s = 6:21:35,45;
- erwartete Rohdaten: 4.375.055.630 Byte = etwa 4,4 GB beziehungsweise 4,075 GiB;
- Faktor-zwei-Planung: 8.753.335.700 Byte;
- vor Seed 01: mindestens 20.000.000.000 Byte lokal frei;
- vor Seed 02 bis 10: mindestens 10.000.000.000 Byte plus 437.505.563 Byte je noch
  ausstehendem Seed einschließlich des zu startenden Seeds lokal frei;
- Peak-RSS wird wegen streng sequentieller Ausführung nicht mit zehn multipliziert.

Nach jedem erfolgreichen oder fehlgeschlagenen Versuch wird lokal ein vollständiger
`RUN_SHA256SUMS`-Index über alle anderen Run-Dateien erstellt und sofort vollständig mit
`sha256sum -c RUN_SHA256SUMS` geprüft. Vor dem nächsten Seed wird dieser Index erneut vollständig
geprüft. Jede fehlende Datei oder Abweichung sperrt alle späteren Seeds. Die Runs verbleiben
unverändert im Repository-Artefaktbereich und werden nicht in gewöhnliches Git aufgenommen.

Externe Kapazitätsprüfung, externe Kopie und externe Zielverifikation sind keine Gates mehr. Der
Nutzer akzeptiert ausdrücklich, dass der vollständige Ausfall oder Verlust der einzelnen
internen SSD alle Rohdaten zerstören kann. SHA-256 erkennt unbemerkte Veränderungen, ist aber
keine unabhängige Datensicherung.

## Sprint-13-Vorlage und Verifikation

Die Vorlage besitzt auch nach Sprint 12A absichtlich nur einen Dry-run-Modus:

```bash
artifacts/day12-h100-replication-protocol/sprint13_runbook_template.sh --dry-run 1
.venv/bin/python artifacts/day12-h100-replication-protocol/verify_protocol.py
(cd artifacts/day12-h100-replication-protocol && sha256sum -c SHA256SUMS)
```

Die generatorerzeugten JSON-Dateien lassen sich ohne Überschreiben des Freeze-Pakets in einem
leeren temporären Ziel bytegleich reproduzieren:

```bash
repro_dir="$(mktemp -d /tmp/day12-protocol-reproduction.XXXXXX)"
.venv/bin/python artifacts/day12-h100-replication-protocol/generate_protocol.py \
  --output-dir "$repro_dir"
diff -qr --no-dereference \
  --exclude=README.md \
  --exclude=SHA256SUMS \
  --exclude=generate_protocol.py \
  --exclude=2026-08-01-local-ssd-persistence-v1.md \
  --exclude=sprint13_runbook_template.sh \
  --exclude=verify_protocol.py \
  artifacts/day12-h100-replication-protocol "$repro_dir"
```

Der Generator verweigert ein nicht leeres Ziel. Das temporäre Reproduktionsverzeichnis enthält
nur Kopien des eingefrorenen Protokolls und keine Runartefakte.

Ein späterer Sprint 13 muss Branch, Protokollcommit, Source/Runtime, Speicher, Launcher und
Einzelprozessüberwachung erneut prüfen und ausdrücklich freigeben. Die Vorlage startet selbst
keinen Runner und enthält keine Seed-Schleife.
