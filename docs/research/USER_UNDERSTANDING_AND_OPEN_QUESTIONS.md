# Nutzerverständnis und offene Fragen

Stand: 2026-07-31. Dieses Dokument ist der Einstieg für einen separaten Verständnis-Chat. Es
beschreibt den aktuellen Sprint-7-Stand; historische Zahlen und Herleitungen stehen im
`CODE_WALKTHROUGH.md`, exakte Befehle im `RUNBOOK.md` und überprüfte Ausführungsevidenz im
`checkpoints/DAY7_CHECKPOINT.md`.

## 1. Das System in einem Satz

Ein gemeinsamer Satz Mellinger-Reglergains wird durch mehrere voneinander unabhängige, zufällig
parametrisierte Crazyflow-Welten differenziert optimiert; feste Validation darf den Checkpoint
auswählen, der eingefrorene Testsplit darf keine Designentscheidung beeinflussen.

## 2. Was eine Episode enthält

Eine Episode ist kein versteckter Simulatorzustand, sondern ein vollständig realisiertes
`EpisodeBatch`:

- Referenzposition, -geschwindigkeit, -beschleunigung und Yaw für `T+1` Zeitpunkte;
- die daraus gebildeten `T` Mellinger-Sollwerte;
- eine konstante wahre Masse je Welt;
- einen konstanten ganzzahligen Action Delay je Welt;
- zeitabhängige äußere Kraft- und Drehmomentfolgen;
- den initialisierten vollständigen Crazyflow-Zustand.

Alle Arrays besitzen explizite Zeit-/Welt-/Drohnenachsen. Der Gainvektor besitzt keine Weltachse.

## 3. Warum es drei Simulationen gibt

Train darf sich ändern und nach jedem Update neue Episoden ziehen. Validation muss gleich bleiben,
damit Losswerte zwischen Checkpoints vergleichbar sind. Test muss bis zum Ende unangetastet
bleiben, sonst wäre er nur eine zweite Validation.

Darum reicht eine numerische Maske in einem gemeinsamen Batch nicht. Train und Validation werden
aus getrennten `Sim`-Instanzen gebaut. Der Trainingstyp kennt Test nicht. Nur ein ausdrücklich
aufgerufener Test-Builder kann später den Testpfad erzeugen.

## 4. Reproduzierbarkeit

Ein Root-Seed wird nicht fortlaufend „verbraucht“. Stattdessen definiert jede Zufallsgröße ihre
Koordinate:

```text
(Split, Episode, Welt, Komponente)
```

Komponenten sind Trajektorie, Masse, Delay und Wrench. Deshalb verändert ein neuer Massensampler
nicht stillschweigend die Trajektorienseeds. Validation/Test besitzen versionierte JSON-Manifeste
mit eigenen IDs und disjunkten Seeds.

Checkpointkompatibilität wird zusätzlich durch SHA-256-Fingerprints der vollständigen Config und
der Gain-Registry abgesichert.

## 5. Physik- und Modellsemantik

Die wahre Dynamikmasse des auditierten Modells ist ungefähr `31.9 g`, die Controllerannahme
ungefähr `29 g`. Diese Differenz wird nicht versteckt. Die wahre Masse wird je Welt um höchstens
`0.2 g` verändert; der Controllerwert bleibt fest.

Delay wird in Kontrollintervallen gezählt. Ein Delay von zwei bedeutet: bei Intervall `t` wird die
Vorgabe `max(t-2,0)` ausgeführt. Damit ist Delay null exakt identisch zur Baseline. Die erste
Vorgabe füllt die noch nicht vorhandene Vergangenheit.

Äußere Kraft und äußeres Drehmoment sind world-frame Eingaben in Newton beziehungsweise
Newtonmeter. Eine AR(1)-Folge macht sie zeitlich korreliert. Die vorhandene Dynamik transformiert
das world-frame Drehmoment intern in den Bodyframe.

Wichtig: Die Massenspanne ist eine vorläufige Engineeringvorgabe. Delay und Wrench sind nicht aus
Flugdaten identifiziert. Im Workstation-Preset bleiben sie deshalb aus.

## 6. Warum Fouriertrajektorien

Die alten Circle-/Figure-8-Trajektorien sind gute deterministische Baselines, aber keine
Trainingsverteilung. Der neue Generator mischt mehrere Sinusharmonische pro Achse. Eine glatte
Hülle zwingt Anfang und Ende auf Hover mit Geschwindigkeit, Beschleunigung und Jerk null.

Vor der Simulation werden Workspace, Höhen, Geschwindigkeit, Beschleunigung, Jerk, Yawrate,
Neigung und spezifische Kraft geprüft. Nach höchstens einer konfigurierten Zahl Versuche bricht
der Generator ab. Diese Grenzen schützen numerisch/plausibilisieren Simulationen; sie sind keine
Hardware-Sicherheitszertifizierung.

Sprint 7 unterscheidet jetzt zwei explizite Verteilungen. `legacy_normalized_time_v1` läuft die
gesamte Zufallsgeometrie immer in der Episodendauer ab. Darum wird dieselbe Geometrie bei H100
gegenüber H400 viermal schneller, sechzehnmal beschleunigungsreicher und vierundsechzigmal
jerkreicher. Diese historische Semantik bleibt reproduzierbar.

`fixed_support_prefix_v2` erzeugt stattdessen immer einen vollständigen viersekündigen Parent,
prüft dessen Grenzen und liefert H100/H200/H400 als Prefixe. Dadurch wählen alle Horizonte für
einen Seed dieselben Koeffizienten und denselben Rejection-Versuch. Der Preis ist bewusst: H100
und H200 enden nicht mehr automatisch im Hover. Das ist keine stille Korrektur, sondern eine neue
versionierte Aufgabenverteilung.

## 7. Welche Gains wirklich im Vektor sind

Stage 1 optimiert vier symmetrisch gruppierte Positions-/Dämpfungsgains: `kp_xy`, `kp_z`,
`kd_xy`, `kd_z`. Dafür existieren bereits Gradientenevidenz und die historischen Bounds.

Stage 2 ergänzt `ki_xy`, `ki_z`, benötigt aber längere oder bias-behaftete Episoden. Stage 3
ergänzt Roll/Pitch-`kR`, `kw` und `kd_omega`. Stage 4 registriert Yaw-`kR`, `kw`, `ki_m`, darf aber
erst mit Yaw-Anregung und Yaw-/Attitude-Metrik wissenschaftlich laufen.

`ki_m_xy` wird nicht automatisch von null weg optimiert, weil weder Initialisierung noch Grenzen
belegt sind. `kd_omega_z` ist ausgeschlossen: Der Controller setzt den zugehörigen Fehler im
ausgeführten Pfad explizit auf null.

## 8. Loss und Bewertung

Der skalare Loss behält die historischen Gewichte für Position, Geschwindigkeit, Motoraufwand,
Smoothness, terminale Position und weiche Höhenstrafe. Neue Werte sind nur Diagnostik:

- Positions-RMSE, p95 und Maximum;
- Geschwindigkeits-RMSE;
- terminaler Positions- und Geschwindigkeitsfehler;
- normalisierter Motoraufwand und Änderungsaufwand;
- Motor-Sättigung, Nullschub-Gate, Bodenclip und Nonfinite-Anteil;
- technische Erfolgs-/Fehlermarkierung je Episode.

Ein sinkender Loss in einem Ein-Schritt-Smoke ist kein wissenschaftlicher Erfolg. Interessant
wären erst festgelegte Baselines, mehrere Seeds, getrennte Validation und ein nach allen
Entscheidungen einmalig geöffneter Test.

## 9. Was der technische Smoke bewiesen hat

Bei zwei Train- und zwei Validationwelten, H20 und einem Adam-Update liefen Generator,
Randomisierung, Crazyflow-Regler, Dynamik, Gradient, Update, Validationauswahl, Metriken,
Provenienz und Checkpoint zusammen. Loss und Gradient waren endlich. Peak-RSS lag ungefähr bei
1.09 GiB, externe Walltime bei 17.11 s, ohne Swaps.

Nicht bewiesen wurden nützliche Gainverbesserung, Konvergenz, breite Generalisierung,
realitätsnahe Störungen, H400-Skalierung, Testleistung oder Hardwareübertragbarkeit.

## 10. Checkpoint und Resume mental modelliert

Adam besteht nicht nur aus Gains, sondern aus Schrittzähler sowie ersten und zweiten Momenten für
jeden Parameter. Ein valides Resume muss alles wiederherstellen. Der Checkpoint speichert deshalb
alle Optax-Leaves samt dtype/Shape, den aktuellen und ausgewählten Rohvektor, Schritt und nächste
Episode, Historie, Auswahlregel, Root-Seed, Manifest-IDs, Fingerprints und Provenienz.

Resume soll in einen neuen Ordner schreiben. Alte Artefakte werden nicht überschrieben. Eine
geänderte Config ist absichtlich kein Resume, sondern ein neuer Versuch.

Sprint 6 hat dieses Modell real statt nur per JSON-Roundtrip geprüft: zwei Updates am Stück sind
bitgenau identisch zu einem Update, geplantem Stopp und dem zweiten Update in einem neuen Prozess.
Verglichen wurden Rohwerte und transformierte Gains, alle Adam-Zustandsblätter, Zähler,
Validationauswahl, vollständige Train-/Validationhistorie, Seeds und Fingerprints. Bei identischem
CPU-Backend, Dtype, Source und Runtime gilt deshalb bewusst `rtol=0`, `atol=0`.

Der Runner öffnet das eingefrorene Testmanifest jetzt gar nicht mehr. Er speichert nur noch dessen
konfigurierten Pfad als undurchsichtige Referenz. Validation wird weiterhin geladen,
inhaltsgehasht und zur Auswahl verwendet. Das ist stärker als „Test geladen, aber nicht gebatcht“.

## 10a. Warum Sprint 6 trotzdem NO-GO ist

Resume-Readiness und Skalierungs-Readiness sind zwei verschiedene Fragen. Resume ist bestanden.
Der erste vorgeschriebene Skalierungspilot H100 × eine Welt scheitert aber schon vor JIT: Die für
H400 vorbereitete, unveränderte Workstation-Trajektorienverteilung erzeugt für H100 in 16
deterministischen Versuchen keine gültige Trajektorie.

Darum existieren für H100 keine Compile-, First-Execution- oder Steady-State-Zeiten. Der Prozess
brauchte 4.41 s, Peak-RSS war ungefähr 578 MiB, Swap null. Der Fehler ist also keine
Speicherknappheit, sondern eine noch ungeklärte Horizon-/Trajektorienverträglichkeit. H100 × 4,
H200 × 4 lokal sowie H100 × 4, H200 × 8 und H400 × 16 auf der Workstation wurden regelkonform
nicht gestartet.

Die Trajektorienamplituden, Limits oder Verteilung still zu ändern wäre kein technischer Fix,
sondern eine wissenschaftliche Designentscheidung. Bis diese getrennt entschieden und
dokumentiert ist, gilt NO-GO für inkrementelle Workstation-Piloten und erst recht für den
Hauptlauf.

## 10b. Was Sprint 7 gelöst hat – und was nicht

Der construction-only Audit fand über 32 feste Seeds Akzeptanzraten von 0% bei H20/H50/H100,
37,5% bei H200 und 100% bei H400. Selbst 2.048 H100-Kandidaten mit nur diagnostisch erhöhtem
Versuchslimit ergaben keinen gültigen Kandidaten. Damit war „einfach mehr Versuche“ keine
begründbare Lösung.

Die neue v2-Verteilung bestand zweimal dieselbe 128-Seed-Suite bei H100/H200/H400 vollständig.
H100/H200 waren exakte Prefixe, H400 war seedweise exakt Legacy-v1, verschiedene Seeds blieben
verschieden, und die vorher festgelegten Diversitätskriterien bestanden. Danach liefen H100×1,
H100×4 und H200×4 real durch JIT und Ausführung. Ein H100-Zwei-Update-Resume war bitgenau.

Das ist ein GO für einen späteren inkrementellen CPU-Workstation-Piloten mit eigener v2-Config.
Es ist kein GO für `workstation.json`, H400×16, den Hauptlauf, Test, GPU, Hardware oder Sim2Real.
Der aktuelle Research-Builder setzt `device="cpu"` ausdrücklich im Code; die lokale Umgebung hat
zusätzlich nur `cpu:0`.

## 11. Die vier Baselines, die später verglichen werden sollen

1. unveränderte Crazyflow-Defaults;
2. eindeutig dokumentierter Tag-3-Vier-Gain-Kandidat;
3. erweiterte Gainstufe ohne Domain Randomization;
4. dieselbe erweiterte Gainstufe mit freigegebener Domain Randomization.

Nur 1 und 2 besitzen bereits konkrete Werte. 3 und 4 sind geplante Experimente, keine Ergebnisse.

## 12. Fragen, die ich erklären können sollte

- Warum ist eine separate Test-API stärker als eine Maske?
- Warum ist der Gainvektor nicht `(N,K)`?
- Warum bleibt die Controller-Masse fest, wenn die Dynamikmasse variiert?
- Welche Zeiteinheit hat Delay, und was passiert bei `d=0` und am Episodenanfang?
- In welchem Koordinatensystem liegen Kraft und Drehmoment?
- Warum macht zeitliche Korrelation ein Modell nicht automatisch realistisch?
- Wie garantiert die Hülle Ruhe am Anfang und Ende?
- Was unterscheidet Bound, Positivitätsbedingung und Stabilitätsgarantie?
- Warum ist `kd_omega_z` wirkungslos?
- Warum darf Validation auswählen, Test aber nicht?
- Warum gehören Lossgewichte, Masse und Trajektorienparameter nicht zum Controller-Gainvektor?
- Was muss ein Optax-Checkpoint zusätzlich zu den Gains speichern?

## 13. Offene technische Fragen

- Welche inventarisierte CPU-Workstation, Laufzeit- und RSS-Grenze gilt für den nächsten Piloten?
- Wie skalieren die neue explizite v2-Config und exact Resume dort bei H100×4 und H200×8?
- Darf erst nach diesen Gates ein H400×16-Pilot vorbereitet werden?
- Ist `jax.checkpoint`/`remat` bei längeren Horizonten nötig?
- Soll ein echtes Resume Zwischenartefakte aus dem alten Lauf referenzieren oder in das neue
  Verzeichnis kopieren?
- Wie wird der wiederkehrende XLA-Cache-Hinweis zu CPU-Features vor Cross-Machine-Replay gelöst?
- Soll die derzeit falsche `(N,1)`-Dokumentation von `SimCore.rng_key` in einem separaten
  Crazyflow-Core-Sprint korrigiert werden?
- Soll ein späterer eigener Sprint die derzeit harte CPU-Codewahl konfigurierbar machen und eine
  getrennte GPU-Numerik-/Resume-/Speicherprüfung definieren?

## 14. Offene wissenschaftliche/Betreuerfragen

- Welche gemessene Massenverteilung gilt mit Batterie und Payload?
- Welche Logs liefern Command-, State-, Motor- und Mocap-Zeitstempel zur Delayidentifikation?
- Welche Kraft-/Drehmomentstörungen sind beobachtbar oder überhaupt identifizierbar?
- Welche Gainstufe und welches Objective sind der verbindliche Bachelor-Scope?
- Welche Train-/Validation-/Testanzahlen und Difficulty-Tiers sind ausreichend?
- Welche Yaw-/Attitude-Anregung ist erlaubt und fachlich sinnvoll?
- Welche Abbruch-, Nonfinite- und Sicherheitskriterien gelten für einen Workstation-Hauptlauf?
- Soll es später Firmware-/Hardwaretests geben; wenn ja, wer autorisiert Mapping und
  Sicherheitsprozess?

## 15. Empfohlene Lernreihenfolge im Verständnis-Chat

1. Shapes `T,N,M` und die zwei `lax.scan`-Ebenen.
2. Splitobjekte und Seedkoordinaten anhand eines Zwei-Welten-Beispiels.
3. Eine Trajektorie einschließlich analytischer Produktregel ableiten.
4. Einen GainSpec vom Rohwert bis in das Controllerdictionary verfolgen.
5. Ein EpisodeBatch vom Hostaufbau bis zum Loss verfolgen.
6. Einen Checkpoint öffnen und jeden Resume-relevanten Block erklären.
7. Erst danach einen wissenschaftlichen Hauptlauf und dessen Baselines entwerfen.
