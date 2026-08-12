# Primary-data archive evidence

Status date: 2026-08-02
Evidence scope: independent backup of the immutable Day-10 and H100 primary-data directories

## Archived source directories

The archive was created from these unchanged repository-local primary-data directories:

- `artifacts/day10-sparse-long-pilot/runs`
- `artifacts/day13-h100-replication`

The repository-local originals remain intentionally outside ordinary Git. Their current closure
inventory is 61 files / 437,505,563 bytes for Day 10 and 610 files / 4,375,130,360 bytes for the
ten H100 runs. Their repository-side hash indexes are verified independently by the closure
procedure in [`RUNBOOK.md`](RUNBOOK.md#gradient-repository-closure-reproduction).

## Verified archive receipt from 2 August 2026

| Field | Verified value |
|---|---|
| Archive name | `crazyflow-gradient-primary-data_2026-08-02.tar` |
| Archive format | uncompressed sparse TAR |
| Archive size | 4,813,772,800 bytes |
| SHA-256 | `8766b98f3a8a231f4bd377ecb9b4fc66ae0fa4f26d02fc3c117dbdb595f244ce` |
| Local staging path | `/home/noah3/backup-staging/crazyflow-gradient-primary-data_2026-08-02.tar` |
| Independent destination | Uni-OneDrive, `Bachelorarbeit/Archive/Gradient-Research/` |
| Destination check before cloud synchronization | exactly 4,813,772,800 bytes; `sha256sum -c`: `OK` |
| Cloud-client confirmation | OneDrive status “Aktuell”; TAR and `.sha256` present online; no `.uploading` file |
| Web-view size | 4,700,950 KB, equal to 4,813,772,800 bytes at 1,024 bytes per KiB |

The TAR structure was read completely and the local archive was checked successfully with
`sha256sum -c`. The local source archive and the local OneDrive destination copy were therefore
byte-verified against the same SHA-256 value before synchronization; the synchronized client
state and web view then established cloud presence.

## Exact evidence boundary

No post-synchronization cloud download followed by another SHA-256 check was performed. This
receipt therefore does **not** claim a fresh byte verification of a downloaded cloud object. It
also does not constitute:

- Git versioning of any primary-data file;
- a new semantic validation of every experiment;
- a new simulation, optimization, seed, or Test evaluation;
- proof that every future copy will remain available or unchanged.

At completion of the 2 August 2026 receipt, repository originals, staging archive, and OneDrive
copy were all reported present. The closure documentation records that dated evidence without
silently strengthening it into a later re-verification.
