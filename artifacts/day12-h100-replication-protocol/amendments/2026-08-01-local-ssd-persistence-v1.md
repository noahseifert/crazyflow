# Sprint 12A: Persistenz-Amendment 2026-08-01, Version 1

Status: angenommen vor Seed 01

Wirksam ab: `2026-08-01T12:51:18Z`

Basisprotokoll: Commit `73868cfd84d25bec0c6612b31ee34f4e67acb307`

Maschinenlesbare Fassung: `2026-08-01-local-ssd-persistence-v1.json`

## Anlass und zeitliche Einordnung

Der Nutzer hat vor dem ersten konfirmatorischen Seed entschieden, keinen externen Datenträger und
keinen externen Forschungsspeicher zu verwenden. Alle Replikationsrohdaten dürfen dauerhaft auf
der internen 2-TB-SSD im Repository-Artefaktbereich verbleiben. Beim Amendment-Preflight waren
`1.006.592.950.272` Byte lokal frei.

Zum Wirksamkeitszeitpunkt waren null konfirmatorische Seeds gestartet und keine konfirmatorischen
Ergebnisse bekannt. Die Änderung ist daher nicht resultatsabhängig. Sie betrifft ausschließlich
Speicherort, Integritätsablauf und Ausfallsicherheit.

## Historische, hiermit ersetzte Regel

Das Basisprotokoll verlangte nach jedem Versuch eine bytegenaue Kopie von Roh-Run und SHA-256-
Index auf genehmigten, gesicherten externen Forschungsspeicher sowie die vollständige
Hashverifikation am Ziel vor dem nächsten Seed. Externe Kapazität, Kopie und Zielverifikation
waren Ausführungsgates. Der unveränderte historische Wortlaut und seine Begründung bleiben in
Commit `73868cfd84d25bec0c6612b31ee34f4e67acb307` nachvollziehbar.

## Neue verbindliche lokale Persistenzregel

1. Jeder erfolgreiche oder fehlgeschlagene Replikationsversuch verbleibt vollständig und
   unverändert unter `artifacts/day13-h100-replication/runs/` auf der internen SSD.
2. Roh-Runs werden nicht in gewöhnliches Git aufgenommen.
3. Nach jedem Versuch wird `RUN_SHA256SUMS` über sämtliche anderen Run-Dateien erzeugt und sofort
   lokal vollständig mit `sha256sum -c RUN_SHA256SUMS` geprüft.
4. Vor jedem späteren Seed wird der vollständige Index des vorherigen Seeds erneut lokal geprüft.
5. Eine fehlende Datei oder Hashabweichung sperrt den aktuellen und sämtliche späteren Seeds.
6. Während der gesamten konfirmatorischen Serie darf kein lokaler Run gelöscht, verschoben,
   umbenannt, komprimiert, verändert oder überschrieben werden.
7. Vor Seed 01 müssen mindestens `20.000.000.000` Byte lokal frei sein.
8. Vor Seed 02 bis 10 müssen mindestens `10.000.000.000` Byte plus
   `437.505.563 * verbleibende Seeds einschließlich des zu startenden Seeds` lokal frei sein.
9. Der vorhandene Sprint-10-Pilotrun bleibt vollständig unverändert.
10. Externe Kapazitätsprüfung, externe Kopie und externe Ziel-Hashverifikation sind keine Gates
    mehr.

## Bewusst akzeptierte Einschränkung

Der vollständige Ausfall oder Verlust der einzelnen internen SSD kann sämtliche dort
gespeicherten Replikationsrohdaten zerstören. Dieses Single-Drive-Risiko wird ausdrücklich
akzeptiert. SHA-256 erkennt unbemerkte Veränderung oder fehlende Dateien, ersetzt aber keine
unabhängige Datensicherung. Die lokale Regel verbessert Integritätsdetektion, nicht Redundanz oder
Katastrophenfestigkeit.

## Unveränderte Bestandteile

Unverändert bleiben alle zehn Seeds und Configdateien einschließlich ihrer Hashes, ihre Reihenfolge,
H100 mit vier Train- und vier festen Validation-Welten, exakt 5.000 Updates, Checkpointintervall
100, Gains, Bounds, Optimizer, Lernrate, Lossgewichte, Massenvariation, deaktivierte Komponenten,
Source-/Runtime-/Gain-/Validation-Fingerprints, Auswahlregel und Tie-Breaking, primäre und
sekundäre Endpunkte, Unsicherheitsdarstellung, wissenschaftliche GO-/NO-GO-Schwellen,
sequentielle Ausführung und der vollständig gesperrte Testsplit.

Sprint 12A startet keinen Run. Ein späterer Sprint 13 benötigt weiterhin einen neuen exakten
Preflight und eine gesonderte Autorisierung für genau Seed 01.
