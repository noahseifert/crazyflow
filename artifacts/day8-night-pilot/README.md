# Sprint 8: manueller CPU-Nachtpilot

## Freigabe und feste Semantik

**Entscheidung: GO für genau einen manuellen, technischen und explorativen CPU-Piloten mit 50
Optimizer-Updates.** Dies ist weder ein Hauptlauf noch eine Seedstudie oder wissenschaftliche
Testauswertung. Der Launcher startet keine Folgekonfiguration und endet spätestens nach 50
Updates oder sechs Stunden.

Die unveränderliche Config `pilot_config.json` verwendet H100, vier Train- und vier
Validation-Welten, genau Seed `20260731`, Stage 1 (`kp_xy`, `kp_z`, `kd_xy`, `kd_z`) und einen
gemeinsamen Gainvektor. Trajektorien verwenden `fixed_support_prefix_v2`. Ausschließlich die
wahre Dynamikmasse wird uniform um ±0,0002 kg variiert; die Controller-Massenannahme bleibt
unverändert. Action Delay, Wrench/Disturbance, Sensorrauschen und UKF sind aus. Lossgewichte,
Gain-Bounds und Manifeste bleiben unverändert. Der Runner behandelt den Testpfad nur als opaque
Referenz; der Testsplit wird nicht geöffnet, gebaut, simuliert oder ausgewertet.

## Gemessenes Gate und Nachtbudget

Der einzige frische H100×4-Prozess führte zwei Updates aus und bestand das Gate:

- Exit 0, alle Train-/Validation-Metriken und Gradienten endlich, alle technischen Gates wahr;
- zwei vollständige atomare Checkpoints mit gültiger Payload-Prüfsumme;
- Train-Compile 7,433 s, Validation-Compile 0,725 s;
- 13,098 s bis zum ersten vollständigen Update;
- 4,096 s zwischen erstem und zweitem vollständigen Update als operative Steady-State-Zeit;
- externe Walltime 19,24 s, Peak-RSS 1.198.476 KiB = 1,143 GiB, Swap 0.

Aus `13,098 s + (N-1) × 4,096 s + 2,046 s` ergeben sich rund 52,0/113,5/215,9 s für
10/25/50 Updates. Gewählt sind 50 Updates, weil sie innerhalb der harten Auswahl die meisten
Stabilitäts-, Trend- und Checkpointbeobachtungen liefern. Erwartet werden etwa 3,6 Minuten; selbst
ein transparenter Sicherheitsfaktor 2 ergibt nur etwa 7,2 Minuten. Das feste Timeout bleibt
trotzdem bei 21.600 s (sechs Stunden). Der Lauf darf und soll sofort enden, sobald Update 50
erreicht ist.

## Betriebsbedingungen vor dem Start

- Laptop ans Netzteil anschließen und ausreichende Kühlung sicherstellen.
- Windows- und WSL-Schlafmodus für die Laufzeit verhindern.
- WSL nicht mit `wsl --shutdown` beenden.
- Das verwendete Terminal beziehungsweise die tmux-Session nicht versehentlich beenden.
- Vom Repository-Root aus `sha256sum -c artifacts/day8-night-pilot/SHA256SUMS` ausführen.

## Exakter Startbefehl

`tmux` ist auf der vorbereiteten Maschine installiert und daher der empfohlene Startweg:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
tmux new-session -d -s crazyflow-s8-night 'cd /home/noah3/bachelorarbeit/crazyflow-gradient-research && exec artifacts/day8-night-pilot/launch_night_pilot.sh'
```

Der normale Vordergrundbefehl lautet:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
artifacts/day8-night-pilot/launch_night_pilot.sh
```

Jeder Start erzeugt atomar getrennte Runner-Ausgaben in einem neuen Ordner
`artifacts/day8-night-pilot/runs/<UTC>-<Git-HEAD>/runner-output`. Ein existierender Runordner wird
abgelehnt. `launch_manifest.txt` hält Git-HEAD, Config-SHA-256, kanonischen Configfingerprint,
Zeitbudget und exakten Befehl fest; `environment.txt` hält System-, CPU-, RAM-, Python-, JAX- und
Paketdaten fest. Runner-stdout und -stderr liegen vollständig in `stdout.log` und `stderr.log`.

## Statuskontrolle

Direkt nach dem tmux-Start zeigt dieser Befehl den vom Launcher ausgegebenen Runordner:

```bash
tmux capture-pane -pt crazyflow-s8-night -S -30
```

Danach `RUN_ROOT` mit dem dort ausgegebenen absoluten Pfad setzen, zum Beispiel:

```bash
RUN_ROOT=/home/noah3/bachelorarbeit/crazyflow-gradient-research/artifacts/day8-night-pilot/runs/<UTC>-<Git-HEAD>
tail -f "$RUN_ROOT/stdout.log" "$RUN_ROOT/stderr.log"
ps -o pid,ppid,pgid,etimes,rss,pcpu,pmem,args --forest -g "$(cat "$RUN_ROOT/process_group.pid")"
ls -1 "$RUN_ROOT/runner-output"/checkpoint-step-*.json 2>/dev/null | tail
```

Der Prozess läuft, solange die Prozessgruppe angezeigt wird. Jeder vollständige Update erzeugt
`checkpoint-step-XXXXXX.json`; `metrics.jsonl` enthält je Update genau eine Train- und eine
Validation-Zeile. Bei Erfolg endet die tmux-Session und `launcher_status.txt` enthält
`exit_status=0`; der Runner schreibt `run_summary.json` mit `status=success` und 50 abgeschlossenen
Updates.

## Kontrolliertes Beenden und Resume

Eine laufende Prozessgruppe wird mit SIGTERM kontrolliert beendet:

```bash
kill -TERM -- "-$(cat "$RUN_ROOT/process_group.pid")"
```

Ein gerade laufendes, noch nicht committetes JAX-Update wird dabei verworfen. Alle zuvor
vollständig geschriebenen Checkpoints bleiben erhalten. Es gibt bewusst keinen Signalcheckpoint
für partiellen Optimizerzustand. Resume verwendet den realen Runner, dieselbe fingerprintete
50-Update-Config und zwingend einen neuen Runordner:

```bash
LAST_CHECKPOINT="$(ls -1 "$RUN_ROOT/runner-output"/checkpoint-step-*.json | sort | tail -n 1)"
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
artifacts/day8-night-pilot/launch_night_pilot.sh --resume "$LAST_CHECKPOINT"
```

Ein Checkpoint nach Update 50 ist bereits vollständig und muss nicht resumed werden. Config-,
Source-, Runtime-, Registry-, Seed-, Validation- oder Checksum-Mismatch werden hart abgelehnt.

## Morgendliche Auswertung

Den letzten Runordner setzen und die rein lesende Prüfung ausführen:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
RUN_ROOT="$(ls -td artifacts/day8-night-pilot/runs/* | head -n 1)"
.venv/bin/python artifacts/day8-night-pilot/inspect_pilot.py "$RUN_ROOT"
sed -n '1,200p' "$RUN_ROOT/launch_manifest.txt"
sed -n '1,240p' "$RUN_ROOT/time.txt"
tail -n 40 "$RUN_ROOT/stdout.log" "$RUN_ROOT/stderr.log"
find "$RUN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum
```

`evaluation_status=PASS` verlangt 50 sequentielle, prüfsummenkorrekte Checkpoints, vollständige
Train-/Validation-Metriken, ausschließlich endliche Werte, bestandene technische Gates, keine
Testmetrik und einen erfolgreichen Summary. `RESUMABLE` bezeichnet einen integren letzten
Checkpoint nach einer Unterbrechung; der ausgegebene `resume_command` startet ihn in einem neuen
Runordner. `FAIL` darf nicht durch eine zweite Konfiguration, H200/H400, eine Seedschleife oder
Testauswertung umgangen werden.

In den Projektleitungs-Chat gehören anschließend genau:

1. die vollständige Ausgabe von `inspect_pilot.py`;
2. `launch_manifest.txt` und `time.txt`;
3. die letzten 40 Zeilen von `stdout.log` und `stderr.log`;
4. die erzeugte SHA-256-Liste;
5. bei `RESUMABLE` oder `FAIL` zusätzlich der exakte Abbruchzeitpunkt und ob der Resume-Befehl
   ausgeführt wurde.

Keine Testsplit-Datei und kein Testartefakt soll kopiert, geöffnet oder erzeugt werden.
