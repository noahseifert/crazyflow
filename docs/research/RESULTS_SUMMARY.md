# Ergebniszusammenfassung

Status: 2026-08-02
Einordnung: formaler Gradient-Repositoryabschluss mit reproduzierter Zehn-Seed-H100-Synthese;
Testsplit geschlossen

## Aktuelles Abschlussfazit

Die zehn vorhandenen, vorab deklarierten H100-Seeds liefern positive Simulationsevidenz unter
der eingefrorenen H100-Konfiguration und der dokumentierten Validationauswahl. Jeder Seed besitzt
genau einen vollständigen 5.000-Update-Run ohne Resume- oder Ersatzseed. Bei allen zehn Seeds ist
Update 5.000 das früheste globale Validationminimum; sämtliche Validationkurven fallen über alle
4.999 Übergänge strikt.

Die mittlere relative Validationverbesserung von Update 1 zum ausgewählten Update beträgt
`74,7788986450 %`. Das gemäß eingefrorenem Analyseplan berechnete zweiseitige 95-%-Student-t-
Intervall des Mittels (`df=9`, `t=2,2621571627409915`, Stichproben-SD mit `ddof=1`) ist
`[74,7757723574 %; 74,7820249326 %]`. Quelle und Rechenweg sind
[`h100_replication_analysis.json`](../../artifacts/day19-h100-analysis/h100_replication_analysis.json)
und
[`analyze_h100_replication.py`](../../artifacts/day19-h100-analysis/analyze_h100_replication.py).

## Zehn-Seed-Ergebnisse

Die deterministische Regel betrachtet pro Seed alle abgeschlossenen Post-Update-Schritte 1 bis
5.000 auf den vier festen Validationwelten. Ein kleinerer Validation-Loss ersetzt die bisherige
Auswahl; bei exakter Gleichheit bleibt daher der frühere Schritt. Schritt 0 und Test sind keine
Kandidaten.

| Index | Root-Seed | Validation Schritt 1 | Auswahl-/Endschritt | Validation Auswahl/Ende | Relative Verbesserung |
|---:|---:|---:|---:|---:|---:|
| 01 | 1432116264 | 0,010010135360062122 | 5000 | 0,002524877665564418 | 74,776788 % |
| 02 | 366692846 | 0,010010110214352608 | 5000 | 0,002525355899706483 | 74,771947 % |
| 03 | 235438753 | 0,010010110214352608 | 5000 | 0,0025249135214835405 | 74,776366 % |
| 04 | 1369406745 | 0,010010110214352608 | 5000 | 0,0025246848817914724 | 74,778651 % |
| 05 | 1081462774 | 0,010010127909481525 | 5000 | 0,002523899544030428 | 74,786541 % |
| 06 | 1276309202 | 0,010010110214352608 | 5000 | 0,002524987794458866 | 74,775624 % |
| 07 | 987511476 | 0,0100101288408041 | 5000 | 0,0025246264412999153 | 74,779281 % |
| 08 | 2125653507 | 0,010010136291384697 | 5000 | 0,0025241717230528593 | 74,783843 % |
| 09 | 31735934 | 0,010010110214352608 | 5000 | 0,0025242639239877462 | 74,782856 % |
| 10 | 986925065 | 0,010010110214352608 | 5000 | 0,002524841111153364 | 74,777090 % |

Aggregierte ausgewählte Validation-Loss:

- Mittelwert: `0,0025246622506529095`;
- Stichproben-SD: `4,360755287449738 × 10⁻⁷`;
- 95-%-t-Intervall: `[0,002524350301011905; 0,002524974200293914]`;
- Median: `0,0025247629964724183`;
- lineares Q1/Q3: `0,0025243545533157885 / 0,00252490455750376`.

Aggregierte relative Validationverbesserung:

- Mittelwert: `74,7788986450 %`;
- Stichproben-SD: `0,0043702487451` Prozentpunkte;
- 95-%-t-Intervall: `[74,7757723574 %; 74,7820249326 %]`;
- Median: `74,7778701492 %`;
- lineares Q1/Q3: `74,7764718090 % / 74,7819621879 %`.

Der mittlere finale Train-Loss beträgt `0,00247700703330338`, die mittlere finale
Gradientennorm `0,00027522838208824394`. Im gesamten Panel sind die Maxima für
Motorsättigungs-, Bodenclip-, Zero-Thrust- und Nonfinite-Anteil jeweils `0,0`; der größte
beobachtete Positionsfehler ist `0,03437680006027222 m`. Der kleinste normalisierte Abstand
eines ausgewählten Gains zur nächsten eingefrorenen Grenze ist `0,08598490194840865`.

## Konfiguration, Provenienz und Testgrenze

Alle Runs verwenden CPU, H100, 500-Hz-Simulation, 100-Hz-Control, vier Train- und vier feste
Validationwelten, 5.000 Adam-Updates bei Lernrate `0,001`, `fixed_support_prefix_v2`, Stage-1-
Gains `kp_xy/kp_z/kd_xy/kd_z` und ausschließlich Massenrandomisierung `±0,0002 kg`. Die
Controllermasse bleibt `0,029 kg`, die nominale Dynamikmasse `0,0319 kg`. Delay, Wrench,
Messrauschen und UKF sind deaktiviert.

Die zehn Runprovenienzen referenzieren Branch `research/differentiable-mellinger`, HEAD
`ff85c8e9c0e73bcf97dfd5f2a747aa374a834f91`, den eingefrorenen Source-Fingerprint
`6b3fed4a445b82b2a1354115edc982a15e0d93cd4aff91fc8f51475e60f01c5e` sowie die dokumentierten
Runtime-, Gain-Registry- und Validation-Fingerprints. Die Synthese verifiziert 610 Primärdateien,
600/600 indexierte Dateihashes, 500/500 interne Checkpoint-Payload-Hashes und 100.000/100.000
erwartete Metrikzeilen.

Der Testsplit blieb im belegten Prozess geschlossen. Die Synthese öffnet den Testmanifestpfad
nicht, findet keine Testmetrik und verwendet Test weder für Auswahl noch Bericht. Der
Sicherungsnachweis der ungetrackten Primärdaten steht in
[`PRIMARY_DATA_ARCHIVE.md`](PRIMARY_DATA_ARCHIVE.md).

## Sprint-17- und Horizontgrenze

Die datenbasierten H100-Gates sind positiv. Die separate Ablaufkonformitätsbedingung ist dadurch
nicht automatisch erfüllt: Die historische Lead-Übergabe berichtet zwei fehlgeschlagene
Just-in-time-Wrapper/Preflights und eine regelwidrige Fortsetzung in Sprint 17; im Repository
fehlen Autorisierungs-ID, Abweichungsakte, Waiver und eine ausdrückliche spätere
Horizontfreigabe. Seed 09 und 10 bleiben technisch gültig und gehören ohne post-hoc Ausschluss in
das vorab deklarierte Panel. H200/H400 bleibt formal NO-GO beziehungsweise nicht freigegeben.
Das ist kein negatives H100-Ergebnis.

Belegt ist ausschließlich positive Simulationsevidenz unter der eingefrorenen H100-Konfiguration
und Validationauswahl. Nicht belegt sind Testleistung, H200/H400-Leistung, allgemeine
Überlegenheit, globale Optimalität, Firmware-/STM32-Äquivalenz, Hardware/HIL/Flug oder Sim2Real.

## Historischer Kurzbefund Tag 1 bis Sprint 5

Der Forschungsprototyp zeigt für den untersuchten reinen JAX-Pfad, dass ein dimensionsloser
Tracking-Loss durch analytische Sollwerte, Mellinger-Regler, First-Principles-Dynamik und
`lax.scan` bis zu vier rohen Gainvariablen differenziert werden kann. Die numerischen
Gradientenkontrollen, Batchrelationen und der festgelegte Train-only-Adam-Lauf bestanden. Der
ausgewählte Parametersatz ist das **beste getestete Train-Checkpoint des festgelegten
H200-Laufs** und ein **simulationsoptimierter Gain-Kandidat**. Er ist weder Hardwarefreigabe noch
Nachweis von Konvergenz, globaler Güte oder Controllerüberlegenheit. Quellen:
[`DAY1_CHECKPOINT.md`](checkpoints/DAY1_CHECKPOINT.md),
[`DAY2_CHECKPOINT.md`](checkpoints/DAY2_CHECKPOINT.md) und
[`DAY3_CHECKPOINT.md`](checkpoints/DAY3_CHECKPOINT.md).

## Tag 1: Ausgangslage und Einzelfallevidenz

Tag 1 führte Figure-8 und Circle getrennt mit perfekten Simulationszuständen aus. Für H200 betrug
der Figure-8-Loss `0.07310394197702408`, die Rohraum-Gradientennorm
`0.05602377653121948` und der maximale relative Richtungsfehler
`0.00015475136751774698`. Quelle:
[`figure8_result.json`](../../artifacts/day1-audit/figure8-h200/figure8_result.json).

Für Circle/H200 betrug der Loss `0.02409282885491848`, die Rohraum-Gradientennorm
`0.014793669804930687` und der maximale relative Richtungsfehler
`0.007185550406575203`. Quelle:
[`circle_result.json`](../../artifacts/day1-audit/circle-h200/circle_result.json).

Beide H200-Läufe meldeten `all_validations_passed: true`; die gleiche Quelle enthält die einzelnen
Validierungsflags. Diese Ergebnisse zeigen lokale Differenzierbarkeit und Reproduzierbarkeit für
die zwei ausgeführten analytischen Fälle, nicht physikalische Modelltreue.

## Tag 2: Batch- und Gradientenevidenz

Tag 2 stapelte Figure-8/Train und Circle/Validation auf der Weltachse. Die Befehle hatten
`(200,2,1,13)`, der Train-Loss betrug `0.07310396432876587`, der Validation-Loss
`0.024092847481369972` und das gleiche arithmetische Mittel
`0.048598404973745346`. Quelle:
[`batch_diagnostics.json`](../../artifacts/day2-audit/batch-h200/batch_diagnostics.json).

Die Rohraum-Gradientennormen waren `0.056023720651865005` für Train,
`0.014793653041124344` für Validation und `0.03447338938713074` für Combined. Der
Train-/Validation-Gradientenkosinus war `0.842276394367218`. Quelle:
[`batch_diagnostics.json`](../../artifacts/day2-audit/batch-h200/batch_diagnostics.json).

Die maximalen relativen Richtungsfehler betrugen `0.0002238699671579525` für Train und
`0.000441420212155208` für Combined. Ein normierter negativer Rohraum-Schritt der Länge `0.01`
änderte Train von `0.07310396432876587` auf `0.07254575937986374` und Combined von
`0.048598404973745346` auf `0.04825476557016373`. Quelle:
[`batch_diagnostics.json`](../../artifacts/day2-audit/batch-h200/batch_diagnostics.json).

Batch und getrennte Einzelfälle stimmten innerhalb der festgelegten Toleranzen überein; eine
Umkehr der Weltreihenfolge änderte Combined-Loss und -Gradient nicht. Die gesamte Tag-2-H200-
Validierung meldete `true`. Quelle:
[`batch_diagnostics.json`](../../artifacts/day2-audit/batch-h200/batch_diagnostics.json).

## Tag 3: festgelegter Train-only-Adam-Lauf

Der Hauptlauf verwendete H200, `50` Adam-Updates, Lernrate `0.01`, Seed `20260724`,
Simulationsfrequenz `500 Hz` und Regelfrequenz `100 Hz`. Nur Figure-8/Welt 0 war
differenziertes Updateziel; Circle/Welt 1 und Combined waren Auswertung. Quelle:
[`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json).

### Baseline- und ausgewählte Gains

| Gain | Baseline | Bestes getestetes Train-Checkpoint, Schritt 50 | Einheit | Quelle |
|---|---:|---:|---|---|
| `kp_xy` | `0.4000000059604645` | `0.31563901901245117` | N/m | [`optimized_gains.json`](../../artifacts/day3-audit/optimize-h200/optimized_gains.json) |
| `kp_z` | `1.25` | `1.5052711963653564` | N/m | gleiche Quelle |
| `kd_xy` | `0.20000000298023224` | `0.26378148794174194` | N·s/m | gleiche Quelle |
| `kd_z` | `0.5` | `0.6184355020523071` | N·s/m | gleiche Quelle |

Die ausgewählten rohen Werte waren `kp_xy=-1.411259651184082`,
`kp_z=0.19198988378047943`, `kd_xy=-0.919587254524231` und
`kd_z=-0.11490653455257416`. Quelle:
[`optimized_gains.json`](../../artifacts/day3-audit/optimize-h200/optimized_gains.json).

### Loss vor und nach dem Lauf

| Ziel | Schritt 0 | Ausgewählter Schritt 50 | Änderung | Relative Änderung | Quelle |
|---|---:|---:|---:|---:|---|
| Train/Figure-8 | `0.07310392707586288` | `0.04376906901597977` | `-0.029334858059883118` | `-40.1276091 %` | [`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json) |
| Validation/Circle | `0.024092912673950195` | `0.01579861156642437` | `-0.008294301107525826` | `-34.4263113 %` | gleiche Quelle |
| Combined | `0.04859841987490654` | `0.029783841222524643` | `-0.018814578652381897` | `-38.7143835 %` | gleiche Quelle |

Die Validation-Verbesserung von rund `34.4 %` ist positiv, stammt aber nur aus einer einzelnen
analytischen Hold-out-Trajektorie und war weder Gradientenziel noch Auswahlkriterium. Quelle:
[`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json).

Der beste getestete Punkt war der letzte gespeicherte Schritt `50`. Die Train-Gradientennorm war
dort noch `0.0312928669154644`. Ein zusätzlicher lokaler negativer Gradientenschritt änderte den
Train-Loss von `0.043769072741270065` auf `0.04345794394612312`. Damit sind weder Stationarität
noch Konvergenz nachgewiesen. Quelle:
[`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json).

### Gradientenkontrollen am ausgewählten Punkt

| Größe | Train | Validation | Combined | Quelle |
|---|---:|---:|---:|---|
| Rohraum-Gradientennorm | `0.0312928669154644` | `0.008393949829041958` | `0.01939206011593342` | [`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json) |
| Maximaler relativer Richtungsfehler | `0.002328432397916913` | nicht als Tag-3-Auswahlkontrolle ausgewertet | `0.0015936176059767604` | gleiche Quelle |

Der Train-/Validation-Gradientenkosinus war `0.865163266658783`. Batch/Separate,
Weltreihenfolge, JIT/eager und `scan`-JAXPR bestanden. Quelle:
[`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json).

## Robustheitsmatrix

Die Matrix enthält genau `32` Vorwärtszellen:
`2` Parametersätze × `4` Horizonte × `2` Trajektorien × `2` Massenbedingungen. Sie enthält keine
weiteren Updates. Quelle:
[`robustness_results.json`](../../artifacts/day3-audit/optimize-h200/robustness_results.json).

| Parametersatz | Matrixmittel | Matrixmaximum | Schlechteste Zelle | Quelle |
|---|---:|---:|---|---|
| Baseline | `0.03194340391564765` | `0.07911745458841324` | H400, Figure-8, auditierter Mismatch | [`robustness_results.json`](../../artifacts/day3-audit/optimize-h200/robustness_results.json) |
| Bestes getestetes Train-Checkpoint | `0.019883187764207833` | `0.05063283443450928` | H100, Figure-8, auditierter Mismatch | gleiche Quelle |

Die Massenbedingungen waren Controller/Dynamik `0.029/0.0319 kg` und experiment-lokal
`0.0319/0.0319 kg`. Quelle:
[`robustness_results.json`](../../artifacts/day3-audit/optimize-h200/robustness_results.json).
Das ist ein kontrollierter Zweipunktvergleich und keine Domain Randomization.

Alle Matrixzellen waren endlich. In den Zellen waren Motorsättigungs-, nonpositive-thrust-gate-,
Floor-Clip- und Nonfinite-Anteile jeweils `0.0`. Quelle:
[`robustness_results.json`](../../artifacts/day3-audit/optimize-h200/robustness_results.json).

## Profiling

| Horizont | Cached Forward Median [s] | Cached Forward/Backward Median [s] | Quelle |
|---:|---:|---:|---|
| H20 | `0.002343864` | `0.021593010` | [`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json) |
| H100 | `0.004290380` | `0.084925729` | gleiche Quelle |
| H200 | `0.006714594` | `0.175491063` | gleiche Quelle |
| H400 | `0.013119240` | `0.328635742` | gleiche Quelle |

Der H400/H200-Medianquotient der Forward/Backward-Auswertung war `1.8726636922815836`.
Compile plus erster Optimierungsschritt dauerte `15.743286041 s`, der gecachte Optax-Schritt im
Median `0.153715902 s` und die Schleife mit den dokumentierten Updates `32.805345890 s`. Quelle:
[`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json).
Die Werte sind synchronisierte Beobachtungen; Peak Memory wurde laut derselben Quelle nicht
gemessen, und es wird kein Speedup behauptet.

## Reproduzierbarkeit

Tag 3 wurde bei HEAD
`32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed` mit dem wissenschaftlichen Source-State
`8559648a3c52ca262f8eb5b4160540cc8e138c42c85d748f13daefe996d268f7`
ausgeführt. Quelle:
[`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json).

Die zwei H20-Replays besitzen nach Ausschluss der sechs ausdrücklich benannten Timingfelder
gleichen wissenschaftlichen Payload; die Gain-/Robustheits-JSONs und die drei Plotpaare sind
byteidentisch. Die achtzehn Tag-3-Dateien stimmen mit den SHA-256-Werten im Checkpoint überein.
Quellen: [`DAY3_CHECKPOINT.md`](checkpoints/DAY3_CHECKPOINT.md#smoke-reproducibility) und
[`DAY4_CHECKPOINT.md`](checkpoints/DAY4_CHECKPOINT.md#json--artefakt--und-plot-audit).

## Evidenz und Aussagegrenzen

| Aussage | Evidenz | Gültigkeitsbereich | Nicht daraus ableitbar |
|---|---|---|---|
| Der reine JAX-Trackingpfad liefert endliche, numerisch kontrollierte Gain-Gradienten. | Tag-1-/Tag-2-Richtungsableitungen und lokale Descent-Tests in [`figure8_result.json`](../../artifacts/day1-audit/figure8-h200/figure8_result.json) und [`batch_diagnostics.json`](../../artifacts/day2-audit/batch-h200/batch_diagnostics.json) | Die ausgeführten analytischen CPU-Simulationsfälle | Firmware-, Hardware- oder allgemeine physikalische Gradientengültigkeit |
| Weltachsen-Batching reproduziert getrennte Einzelfälle. | Batch/Separate-, Mittelwert- und Reihenfolgeprüfungen in [`batch_diagnostics.json`](../../artifacts/day2-audit/batch-h200/batch_diagnostics.json) | Zwei Welten, eine Drohne je Welt, gemeinsame Gains | Beliebige Weltzahlen, heterogene Modelle oder Multi-Drohnen-Interaktionen |
| Der festgelegte Train-only-Lauf senkte den Figure-8-Train-Loss an den getesteten Checkpoints. | Historie und Checkpointwahl in [`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json) | Ein Adam-Design, ein Seed, ein Train-Horizont, perfekte Zustände | Konvergenz, globale Güte oder Überlegenheit gegenüber anderen Verfahren |
| Circle wurde nicht für Gradient, Adam-Update oder Auswahl benutzt. | Split-/Objective- und Validationflags in [`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json) | Der festgelegte Zwei-Welt-Lauf | Unabhängiger echter Testdatensatz oder breite Generalisierung |
| Circle-Loss war am ausgewählten Checkpoint geringer als an der Baseline. | Initial-/Selected-Loss in [`optimized_gains.json`](../../artifacts/day3-audit/optimize-h200/optimized_gains.json) | Eine analytische Hold-out-Trajektorie | Erwartete Verbesserung auf anderen Trajektorien oder Hardware |
| Die feste Kreuzmatrix ist vollständig und endlich. | `cells`, `robustness_matrix_complete` in [`robustness_results.json`](../../artifacts/day3-audit/optimize-h200/robustness_results.json) | Zwei Trajektorien, vier Horizonte, zwei Massenbedingungen, zwei Parametersätze | Störungs-, Sensor-, Delay-, Batterie-, Aerodynamik- oder Estimatorrobustheit |
| Die Massenanpassung verändert Ergebnisse in der Simulation. | Massenbedingungen und Zellen in [`robustness_results.json`](../../artifacts/day3-audit/optimize-h200/robustness_results.json) | Zwei kontrollierte Controller-Massenwerte | Identifizierte reale Massenverteilung oder Domain Randomization |
| Die bestandenen Läufe aktivierten die diagnostizierten harten Zweige nicht. | Branchanteile in [`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json) und [`robustness_results.json`](../../artifacts/day3-audit/optimize-h200/robustness_results.json) | Nur die ausgeführten Läufe | Gültige Gradienten oder Stabilität bei Sättigung, Gate oder Bodenkontakt |
| Die Artefakte sind zum dokumentierten Source-State nachvollziehbar. | Provenienz und Hashes in [`DAY3_CHECKPOINT.md`](checkpoints/DAY3_CHECKPOINT.md) und [`DAY4_CHECKPOINT.md`](checkpoints/DAY4_CHECKPOINT.md) | Repository- und Artefaktstand des Audits | Reproduzierbarkeit auf beliebiger Hardware oder anderer Softwareumgebung |

## Ausdrücklich nicht nachgewiesen

- Keine Konvergenz: Der ausgewählte Punkt hatte noch die dokumentierte von null verschiedene
  Train-Gradientennorm, und der lokale Descent-Test senkte den Loss erneut. Quelle:
  [`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json).
- Keine globale Güte und kein Vergleich mit anderen Controllerfamilien.
- Keine breite Generalisierung: Circle ist nur eine analytische Hold-out-Trajektorie.
- Kein Noise-, Delay-, Disturbance-, Battery-, Aerodynamics-Variation- oder Estimatormodell.
- Keine Hardware-, Firmware-, sim-to-real- oder vollständige physikalische Validierung.
- Keine nachgewiesene Einheiten- oder Implementationsgleichheit der JAX-Gains mit
  Firmwareparametern.
- Kein dokumentierter Upload-, Arm-, Kill-, Begrenzungs- oder schrittweiser Flugtestprozess.
- Kein BetaFlight- oder TinyMPC-Vergleich.
- Keine Peak-Memory-Messung.
- Kein empirisch kalibriertes Noise- oder Domain-Randomization-Modell; ohne Flight-/Mocap-Daten
  wäre ein solches Modell derzeit eine unbelegte Annahme.

Die Grenzen sind im
[`DAY3_CHECKPOINT.md`](checkpoints/DAY3_CHECKPOINT.md#scientific-limitations) und im
[`SUPERVISOR_MEETING_BRIEF.md`](SUPERVISOR_MEETING_BRIEF.md) als Diskussionsgrundlage
festgehalten.

## Sprint 5: technische Infrastruktur- und Skalierungsergebnisse

Sprint 5 erzeugt keine neue wissenschaftliche Controllerbewertung. Der einzige Optimierungslauf
ist absichtlich eine winzige technische Prüfung mit zwei Trainings- und zwei festen
Validierungswelten, H20 und genau einem Stage-1-Adam-Update.

| Messgröße | Wert |
|---|---:|
| Train-Loss vor dem Update | `0.0005378891946747899` |
| Validation-Loss nach dem Update | `0.0005391469458118081` |
| gemeinsame Gradientennorm | `0.00010596758511383086` |
| externer Walltime | `17.11 s` |
| externer Peak-RSS | `1,116,596 KiB` |
| Swaps | `0` |
| Train-Compile | `8.204728028 s` |
| erste Train-Ausführung | `0.013780410 s` |
| Validation-Compile | `0.882622932 s` |
| Validation steady | ungefähr `0.0008–0.0010 s` |
| Testmetriken | keine |

Die sehr kleine technische Trajektorienamplitude ist für H20 nötig, um die analytischen
Beschleunigungs-/Jerkfilter einzuhalten. Deshalb sind absolute Losswerte und die Änderung nach
einem Schritt keine Güteaussage. Nachgewiesen ist nur: Sampleerzeugung, wahre Massenvariation,
Delay, Wrench, echter Mellinger-/Dynamikpfad, Ableitung, Optax, feste Validation, Auswahl,
Checkpoint und Artefaktschreiben funktionieren zusammen mit endlichen Werten.

Frische H20-Gradientenprozesse ergaben:

| Welten | Compile [s] | steady Median [s] | externer Walltime [s] | Peak-RSS [KiB] |
|---:|---:|---:|---:|---:|
| 1 | `6.396366` | `0.009443` | `13.35` | `1,024,200` |
| 2 | `7.139935` | `0.008859` | `14.87` | `1,036,532` |
| 4 | `7.218194` | `0.027623` | `15.06` | `1,045,444` |

Der geringe Peak-RSS-Anstieg bei festem H20 zeigt nur, dass der lokale 1→2→4-Smoke nicht
speicherlimitiert war. Der 4-Welten-steady-Wert ist etwa dreimal so groß wie bei zwei Welten; bei
nur fünf Wiederholungen und kleinen Laufzeiten ist daraus keine präzise Skalierungskurve
abzuleiten. H400 und 16 Welten wurden nicht gemessen.

Der alte Tag-3-Vier-Gain-Kandidat bleibt als eindeutige Baseline erhalten (`kp_xy=0.3156390`,
`kp_z=1.5052712`, `kd_xy=0.2637815`, `kd_z=0.6184355`). Weitere geplante Baselines sind
Crazyflow-Defaults, erweiterte Gains ohne Domain Randomization und erweiterte Gains mit Domain
Randomization. Letztere zwei existieren noch nicht als Ergebnis und dürfen nicht vorweggenommen
werden.

Seit Sprint 5 existieren zwar Delay- und Wrench-Fähigkeiten, aber keine empirisch kalibrierten
Modelle. Die alte Aussage „kein Modell“ gilt für Tag 1–3; die aktuelle präzise Grenze lautet:
konfigurierbare heuristische Simulationsmodelle implementiert, Workstation-Preset deaktiviert,
keine Realismus- oder Sim2Real-Evidenz.
