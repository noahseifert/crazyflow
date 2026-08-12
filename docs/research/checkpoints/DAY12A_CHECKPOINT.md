# Day 12A checkpoint: pre-Seed-01 local persistence amendment

## Boundary and timing

- Branch: `research/differentiable-mellinger`
- Base protocol commit: `73868cfd84d25bec0c6612b31ee34f4e67acb307`
- Amendment ID: `2026-08-01-local-ssd-persistence-v1`
- Effective UTC: `2026-08-01T12:51:18Z`
- Confirmatory seeds/results before adoption: `0` / none
- Permitted untracked path: `artifacts/day10-sparse-long-pilot/runs/`
- No research run, seed, Resume, H200/H400, Test access, push, or amend

The user's pre-result decision replaces only external persistence/fault-tolerance gates. The
original rule and rationale remain auditable at the base commit. The Sprint-10 pilot remains
unchanged.

## Active local persistence rule

Every successful or failed replication attempt remains complete and immutable under
`artifacts/day13-h100-replication/runs/` on the user-declared internal 2-TB SSD and outside
ordinary Git. After each attempt, create `RUN_SHA256SUMS` over every other run file and verify it
immediately and completely with `sha256sum -c`. Before each later seed, verify the previous
seed's complete index again. Any missing file or mismatch blocks all later seeds.

During the confirmatory series, no run may be deleted, moved, renamed, compressed, modified, or
overwritten. External capacity, copy, destination verification, and receipt are no longer gates.

Capacity thresholds:

- Seed 01: at least `20,000,000,000` free local bytes;
- Seed `N=2..10`: at least `10,000,000,000 + 437,505,563 * (11-N)` free local bytes;
- adoption preflight: `1,006,592,950,272` free local bytes, Seed-01 gate PASS.

## Methodological invariants

The seed manifest and all ten configs are byte-identical to the base commit. The root seeds,
order, H100/four-Train/four-fixed-Validation/5,000-update design, checkpoint interval 100, gains,
bounds, Adam/learning rate, loss, mass variation, disabled components, source/runtime/gain/
Validation fingerprints, selection and earliest-exact-tie rule, endpoints, uncertainty,
scientific GO/NO-GO thresholds, sequential execution, retry classification, and Test lock remain
unchanged. The storage amendment is not result-dependent.

Pinned fingerprints:

- Source: `6b3fed4a445b82b2a1354115edc982a15e0d93cd4aff91fc8f51475e60f01c5e`
- Runtime: `e3c408d547653ebbb87107ca3c31b844c12acc43448378a5a48266306ee3d2a4`
- Gain registry: `2e8dbc409dc994f8376d132757ad8c3b0a8744805865c71340e405579c5f3d99`
- Validation: `11e53e460bc93a64835af03bd5a6d22a2e4fb3f7c9d187324d23e7a3deac0c20`

## Verification

```text
SHA256SUMS: 22/22 PASS
Protocol verifier: PASS
Generator-owned JSON replay: byte-identical
Base seed manifest/configs: byte-identical
Config count: 10
Source/runtime/gain/Validation fingerprints: unchanged
Runbook Bash syntax: PASS
Seed-01 dry-run and local capacity gate: PASS
External storage required: false
Accepted single-drive risk: true
Test manifest opened: false
Research runs started: 0
```

`SHA256SUMS` is 1,950 bytes and hashes to
`8c7702e84621a1df78328c53d95d336d0e2b0adf81b0f7d0304b7631356d2675`. The current protocol
contains 23 files totaling 125,699 bytes.

## Accepted limitation and next step

Complete loss or failure of the single internal SSD can destroy all raw replication evidence.
SHA-256 detects missing or modified data but supplies no independent backup. The user knowingly
accepts this scientific and operational limitation.

Sprint 12A authorizes no run. Subject to a new exact preflight and separate user authorization,
Sprint 13 is GO to start exactly Seed 01 only. No later seed, loop, Resume, H200/H400, or Test is
authorized by this checkpoint.
