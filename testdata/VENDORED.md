# Vendored catalog and registry

A pinned copy of the parts of [blis-catalog] and [blis-registry] that this repository's
tests read. Both trees live here (`testdata/catalog`, `testdata/registry`) and are the
default the test suite resolves against.

[blis-catalog]: https://github.com/inference-sim/blis-catalog
[blis-registry]: https://github.com/inference-sim/blis-registry

## Provenance

| tree | upstream | commit |
|---|---|---|
| `testdata/catalog` | blis-catalog | `747a2213e13030ae675f9872c3e2a76b6b770d50` (2026-10-07) |
| `testdata/registry` | blis-registry | `46d930c0fb96e603b76c507acaf14910d48c4ccb` (2026-10-07) |

Copied verbatim — no edits. Only what a kernel actually opens is here, which is why the
copy is ~800K rather than the 21M the two repositories occupy:

| path | what |
|---|---|
| `catalog/hardware/*.yaml` | chip descriptors |
| `catalog/networks/*.yaml` | inter-node fabrics |
| `catalog/devices/storage.yaml` | storage-device facts |
| `catalog/models/<name>/graph.yaml` | derived model graphs (32) |
| `registry/coefficients/*.yaml` | fitted coefficient sets |

Deliberately absent: each model's `config.json` and `model.yaml`. They are vendor
provenance that the graph is *derived from*, and nothing in this repository reads them —
they are also 8.4M of the catalog's 16M.

## Why vendored rather than read from a sibling checkout

The tests used to read absolute paths under one developer's home directory
(`/Users/sri/Documents/Projects/blis-catalog`) and SKIP when the read failed. That made
the suite unrunnable for anyone else and, worse, quiet about it: a skip reads as a pass.

It hid a real failure for the life of the `v0.0.0-20261005164647` pseudo-version pin. The
sibling catalog had already moved to blis-schemas v0.2.0 field names
(`read_bandwidth_mb_s`), which the pinned schema rejected, so **every** test that built a
kernel from a committed fixture skipped on `catalog unavailable` — 41 of them, 46 counting
subtests, with the suite still reporting `ok` — and
`cmd/score`, `cmd/shape` and `cmd/bandprobe` each exited 1 against the real catalog. A
pinned copy makes the suite hermetic and that class of drift a test failure instead of a
silent skip.

## Updating

Re-copy the five paths above from an upstream checkout, update the commit table, and run
`go test ./...`. A coefficient or chip rename is *supposed* to break the fixtures that
name it — that coupling is the point, and it is why the copy is pinned to a commit rather
than tracking a branch.

The suite still honours `-catalog` and `-registry` flags (and `BLIS_CATALOG` /
`BLIS_REGISTRY`), so a working copy can be scored against a live upstream checkout
without touching this one.
