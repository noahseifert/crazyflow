# Gesprächsbrief für die Betreuerbesprechung

Status: 2026-07-26  
Zweck: Fakten, Grenzen, Hypothesen und notwendige Entscheidungen vor weiterer Implementierung

## 1. Technisch korrekte 90-Sekunden-Zusammenfassung

Ich habe einen reinen JAX-Forschungsprototyp aufgebaut, der analytische Figure-8- und
Circle-Sollwerte durch Crazyflows JAX-Mellinger-Regler und First-Principles-Dynamik rollt und den
Tracking-Loss bis zu vier Positions-/Geschwindigkeits-Gains differenziert. Die Zeitrekursion läuft
funktional über `lax.scan`; zentrale Richtungsableitungen und lokale negative Gradientenschritte
bestätigen die lokale numerische Plausibilität. Quelle:
[`DAY2_CHECKPOINT.md`](checkpoints/DAY2_CHECKPOINT.md).

Im festgelegten H200-Adam-Lauf wurde ausschließlich Figure-8/Welt 0 trainiert. Circle/Welt 1 war
Validation und beeinflusste weder Gradient noch Update noch Checkpointwahl. Das beste getestete
Train-Checkpoint war der letzte getestete Schritt `50`; der Train-Loss änderte sich von
`0.07310392707586288` auf `0.04376906901597977`. Circle änderte sich gleichzeitig von
`0.024092912673950195` auf `0.01579861156642437`, war aber nur ein einzelner analytischer
Hold-out-Fall. Quelle:
[`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json).

Die Train-Gradientennorm war am ausgewählten Punkt noch `0.0312928669154644`, und ein weiterer
lokaler Schritt senkte den Train-Loss erneut. Deshalb behaupte ich weder Konvergenz noch globale
Güte. Die Robustheitsmatrix umfasst nur zwei Trajektorien, vier Horizonte und zwei
Controller-Massenbedingungen. Es gibt keine Sensor-, Delay-, Störungs-, Estimator- oder
Hardwarevalidierung. Quelle:
[`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json)
und
[`robustness_results.json`](../../artifacts/day3-audit/optimize-h200/robustness_results.json).

Mein Vorschlag ist deshalb: Scope, Zielcontroller, Parameterabbildung und Hardware-/Sicherheitsweg
zuerst mit den Betreuern verbindlich festlegen. Bis dahin bleibt Feature-Freeze.

## 2. Was ich ursprünglich tun sollte

- Differenzierbare Trajektorienverfolgung mit Crazyflows reinem JAX-Mellinger-Regler und
  reinem JAX-Dynamikpfad untersuchen.
- Glatte Solltrajektorien funktional ausrollen.
- Einen interpretierbaren Tracking-Loss und seinen Gradient bezüglich ausgewählter Gains prüfen.
- Ergebnisse als Simulationsevidenz dokumentieren, ohne native Bridge im Gradientenpfad.

Der abgegrenzte Forschungszweck steht in [`PROJECT_STATE.md`](PROJECT_STATE.md#research-goal).

## 3. Was der Prototyp inzwischen zusätzlich kann

- Zwei analytische Fälle gemeinsam auf der Weltachse auswerten und Train/Validation numerisch
  trennen.
- Batchloss und -gradient gegen getrennte Einzelfälle kontrollieren.
- JIT/eager, feste Replays, Weltreihenfolge und zentrale Richtungsableitungen prüfen.
- Vier rohe Gains über eine glatte beschränkte Abbildung mit Train-only Adam aktualisieren.
- Alle getesteten Checkpoints speichern und rein nach Train-Loss auswählen.
- Baseline und bestes getestetes Train-Checkpoint in einer festen Kreuzmatrix vergleichen.
- Laufzeitbeobachtungen synchronisieren und Provenienz/Artefakthashes festhalten.

Evidenz: [`DAY2_CHECKPOINT.md`](checkpoints/DAY2_CHECKPOINT.md) und
[`DAY3_CHECKPOINT.md`](checkpoints/DAY3_CHECKPOINT.md).

## 4. Was noch nicht implementiert oder validiert ist

Fakten:

- keine Sensorgeräusche, Delays, Störungen, Batterie- oder Aerodynamikvariation;
- keine Domain Randomization;
- kein Estimator- oder UKF-in-the-loop;
- keine Flight-Log-/Mocap-Systemidentifikation;
- keine Hardware-, Firmware- oder sim-to-real-Validierung;
- keine native Bridge im Repository oder Gradientenpfad;
- keine BetaFlight- oder TinyMPC-Integration beziehungsweise Vergleichsstudie;
- keine Peak-Memory-Messung;
- kein dokumentierter Upload-, Arm-, Kill-, Begrenzungs- oder schrittweiser Flugtestprozess.

Quelle der Grenzen: [`DAY3_CHECKPOINT.md`](checkpoints/DAY3_CHECKPOINT.md#scientific-limitations)
und
[`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json).

Offene Hypothese: Ein empirisch aus Logs identifiziertes Delay- oder Noisemodell könnte die
Sim2Real-Aussage verbessern. Ohne Logauswertung wäre die Parametrisierung aber nicht kalibriert und
damit wissenschaftlich schwach.

## 5. Warum die Gains noch nicht direkt geflogen werden sollten

- Der ausgewählte Satz ist ein simulationsoptimierter Gain-Kandidat für ein einzelnes
  Controller-/Objective-Design, nicht für die reale Zielimplementierung freigegeben.
- Alle ausgeführten Läufe nutzten perfekte Simulationszustände; reale Schätzung, Latenzen und
  Messfehler fehlen.
- Die JAX-Gains sind nicht nachgewiesen einheiten- oder implementationsgleich zu
  Firmwareparametern.
- Der Controller verwendet im auditierten Zustand `0.029 kg`, die Dynamik `0.0319 kg`; der
  Zweipunktvergleich löst die reale Modellfrage nicht. Quelle:
  [`robustness_results.json`](../../artifacts/day3-audit/optimize-h200/robustness_results.json).
- Die bestandenen Läufe aktivierten Sättigung, nonpositive-thrust gate und Floor-Clipping jeweils
  mit Anteil `0.0`; Verhalten in sicherheitsrelevanten Grenzbereichen ist damit nicht validiert.
  Quelle:
  [`robustness_results.json`](../../artifacts/day3-audit/optimize-h200/robustness_results.json).
- Es gibt noch keinen bestätigten Labor- und Sicherheitsprozess.

## 6. Prioritäten der Betreuerentscheidungen

1. Verbindlichen Bachelorarbeitsumfang zwischen sauberem Simulationsabschluss, begrenzter
   Sim2Real-Erweiterung und voller Langfristvision festlegen.
2. Zielcontroller und tatsächliche Zielimplementierung auf der Crazyflie benennen.
3. Parameter, Einheiten und Abbildung zwischen JAX-Mellinger und Zielimplementierung bestätigen
   oder verwerfen.
4. Verfügbare Flight-/Mocap-Logs, Datenqualität und Systemidentifikationsziel klären.
5. Entscheiden, ob genau eine empirisch kalibrierte Sensing-Komponente Teil der Arbeit wird.
6. UKF-Schnittstelle und Bachelorrelevanz festlegen.
7. Train-, Validation- und echten Testtrajektoriensatz verbindlich definieren.
8. Lossgewichte, Gain-Grenzen, Massenbehandlung und Erfolgskriterien fachlich freigeben.
9. Labor-, Sicherheits- und schrittweisen Hardwareprozess benennen.
10. BetaFlight/TinyMPC erst nach Entscheidung über Mellinger-Hardwarevalidierung einordnen.

## 7. Konkrete Fragen an die Betreuer

- Was ist der verbindliche Bachelorarbeitsumfang?
- Welcher Controller soll tatsächlich auf der Crazyflie laufen?
- Sind die JAX-Mellinger-Gains direkt auf eine Firmwareimplementierung abbildbar?
- Welche Parameter und Einheiten besitzt die Zielimplementierung?
- Welche Flight- und Mocap-Logs sind verfügbar?
- Welches Sensor-, Delay- und Latenzmodell soll aus den Logs identifiziert werden?
- Welche Array-API-UKF-Implementierung und Schnittstelle existiert?
- Ist UKF-in-the-loop Bestandteil der Bachelorarbeit oder Ausblick?
- Welche Trajektorien bilden Train, Validation und echten Test?
- Wer bestätigt Loss-Gewichte, Gain-Grenzen und Sicherheitsgrenzen?
- Wie soll der Controller-/Dynamik-Massenunterschied behandelt werden?
- Welcher Labor- und Sicherheitsprozess gilt vor einem Flug?
- Soll zuerst Mellinger auf Hardware validiert werden, bevor BetaFlight/TinyMPC beginnen?
- Was bedeutet im Projekt messbar „bester Controller“?
- Welche Ergebnisse wären ausreichend für Bachelorarbeit, Folgeprojekt und Paper?

## 8. Drei mögliche Bachelorarbeitsumfänge

### Option A – Simulationsprototyp sauber abschließen

Differenzierbare Trajektorienverfolgung, validierte Gradienten, Batch und Gain-Optimierung
dokumentieren; Hardware und realistische Sensorik als Ausblick behandeln. Das ist der bereits am
besten belegte und risikoärmste Abschluss.

### Option B – Bachelor-realistische Sim2Real-Erweiterung

Zusätzlich Flight-Log-/Mocap-Analyse und genau eine empirisch kalibrierte Sensing-Komponente,
vorzugsweise Delay oder Noise; kein vollständiger UKF-/Hardware-/Multi-Controller-Stack. Die
Komponente muss aus verfügbaren Daten identifiziert und mit einem echten Testsplit geprüft werden.

### Option C – Vollständige Vision

System Identification, Noise, Delay, UKF-in-the-loop, Hardwareablation und
BetaFlight/TinyMPC-Vergleich. Das sind wahrscheinlich mehrere Forschungsphasen und eher
Master-/Paper-Umfang als ein einzelner Bachelorarbeitsumfang.

## 9. Empfohlene nächste Stufe

Empfohlen ist zunächst Option A verbindlich als sichere Abgabegrenze festzulegen. Falls belastbare
Flight-/Mocap-Logs, eine klare Zielcontroller-Schnittstelle und ausreichende Betreuungszeit
vorliegen, kann Option B als genau abgegrenzte Erweiterung ergänzt werden. Vor jeder weiteren
Implementierung sollten die Punkte Zielcontroller, Einheitenabbildung, Testsplit, empirische
Kalibrierung und Laborprozess schriftlich entschieden werden. Option C sollte als Forschungsroadmap
außerhalb des unmittelbaren Bachelor-Scopes behandelt werden.

## 10. Wahrscheinliche Betreuerfragen und ehrliche Kurzantworten

| Frage | Kurze Antwort |
|---|---|
| Ist der Gradient wirklich durch den Regler und die Dynamik gegangen? | Ja, im reinen JAX-Pfad; JAXPR-Scan, zentrale Differenzen und lokale Descent-Tests bestanden. Evidenz: [`optimization_result.json`](../../artifacts/day3-audit/optimize-h200/optimization_result.json). |
| Wurde Circle mittrainiert? | Nein. Circle war Aux-/Validation-Auswertung und beeinflusste weder Gradient, Adam-State noch Auswahl. |
| Ist der ausgewählte Punkt konvergiert? | Nein. Der letzte getestete Punkt hatte noch die dokumentierte Train-Gradientennorm, und ein weiterer lokaler Schritt senkte den Loss. |
| Sind die Gains für die reale Crazyflie geeignet? | Unbekannt. Firmwareabbildung, Sensorik, Latenz, Sicherheit und Hardwaretest fehlen. |
| Was bedeutet die Circle-Verbesserung? | Positives Ergebnis für genau eine analytische Hold-out-Trajektorie, keine breite Generalisierung. |
| Ist die Massenstudie Robustheitsnachweis? | Nur ein kontrollierter Zweipunktvergleich, keine Domain Randomization oder reale Massenidentifikation. |
| Wurden harte Grenzfälle getestet? | Die Diagnostik war vorhanden, aber die bestandenen Läufe aktivierten die entsprechenden Zweige nicht. |
| Warum nicht sofort Noise oder UKF ergänzen? | Ohne Logs, Zielinterface und Systemidentifikation wären Parameter und Erfolgskriterium nicht empirisch begründet. |
| Wurde ein anderer Controller geschlagen? | Nein. BetaFlight und TinyMPC wurden nicht integriert oder verglichen. |
| Was ist jetzt wissenschaftlich belastbar? | Lokale Differenzierbarkeit, Batchrelationen, Train-only-Updategrenze, endliche feste Matrix und dokumentierte Reproduzierbarkeit im definierten Simulationsscope. |
| Was wäre der nächste saubere Versuch? | Nach Betreuerentscheidung: echter Testsplit plus genau eine aus Logs identifizierte Komponente oder Abschluss als sauber dokumentierter Simulationsprototyp. |

## 11. Update nach Sprint 5

Nach der vom Nutzer berichteten mündlichen Betreuerentscheidung wurde die Simulationinfrastruktur
kontrolliert erweitert. Train, Validation und Test sind jetzt strukturell getrennt; Train kann
mehrere neu randomisierte Welten mit einem gemeinsamen Gain-Satz verwenden, Validation besitzt
ein festes Manifest, und Test ist über eine separate API eingefroren. Der Trainingsrunner erzeugt
keine Testmetrik.

Implementiert sind eine vorläufige wahre Massenvariation von ±`0.2 g`, exakt definiertes
Kontrollintervall-Delay, korrelierte äußere Kräfte/Drehmomente, eine benannte gestufte
Gain-Registry und geglättete zufällige Fouriertrajektorien. Delay und Wrench sind mangels
Messkalibrierung im vorbereiteten Workstation-Preset deaktiviert. Ein H20/2-Welten/1-Update-Smoke
lief technisch erfolgreich; er ist keine wissenschaftliche Optimierung.

Die wichtigste neue fachliche Entscheidung lautet nicht mehr „darf Domain Randomization überhaupt
existieren?“, sondern:

1. Welche Delay-, Kraft- und Drehmomentverteilungen dürfen wissenschaftlich verwendet werden und
   aus welchen Logs/Mocap-Zeitstempeln werden sie geschätzt?
2. Soll die ±`0.2 g`-Massenhalbbreite als reine Sensitivitätsvorgabe bleiben oder durch gemessene
   Fahrzeug-/Batteriemassen ersetzt werden?
3. Welche Gainstufe ist Gegenstand der Arbeit: nur nachgewiesenes `kp/kd`, zusätzlich Integrale,
   oder Attitude-/Yaw-Gains mit entsprechend erweitertem Objective?
4. Welche Yaw-/Attitude-Metrik und welches anregende Manöver sind für Stage 4 verbindlich?
5. Wie viele Seeds/Episoden und welche Difficulty-Tiers sind für Train/Validation/Test angemessen?
6. Wann wird der ausgewählte Checkpoint eingefroren und der echte Test genau einmal ausgewertet?
7. Ist der vorbereitete `16 × H400 × 200` Stage-1-Lauf angemessen, oder soll zuerst eine kleinere
   Horizon-/Welt-Skalierung auf der Workstation erfolgen?
8. Bleibt die Arbeit reine Simulation, oder wird ein späteres, separat freizugebendes
   Firmware-/Hardwarepaket verlangt?

Aktualisierte ehrliche Kurzantworten:

| Frage | Antwort |
|---|---|
| Ist Domain Randomization implementiert? | Ja, als seed-deterministische Simulationsinfrastruktur. Nur die Massenhalbbreite ist im Workstation-Preset aktiv; Delay/Wrench sind unkalibrierte Heuristiken. |
| Wurde Test ausgewertet? | Nein. Das Manifest wurde validiert und seine ID protokolliert; keine Testepisode wurde gebaut. |
| Werden alle Gains optimiert? | Nein. Vier sind Stage 1; weitere sind registriert und gestuft. `ki_m_xy` ist zurückgestellt, `kd_omega_z` strukturell ausgeschlossen. |
| Beweist der Smoke Robustheit? | Nein. Er beweist technischen Datenfluss, endlichen Gradienten, Update, Validation und Checkpointing bei winziger H20-Anregung. |
| Kann der Workstationlauf jetzt blind starten? | Nein. H400/16-Welten-Skalierung, Verteilungsfreigabe und wissenschaftliche Auswahlregeln müssen vorher bestätigt werden. |
