## Summary

<!-- 1-3 sentences: what does this PR do and why? -->

**Head:** `<sha>`. Closes # / Addresses #

## Cross-repository dependencies

<!-- Every BLIS repository this PR pins or is consumed by, with the EXACT head SHA. Every SHA
here must equal its branch head at push time (regenerate this section, do not hand-edit).
Process: https://github.com/inference-sim/blis-schemas/blob/main/docs/contributing/cross-repo-changes.md -->

| Repository | PR | Pinned head / release | Where it is pinned |
|---|---|---|---|
| | | | |

- [ ] No `replace` in `go.mod`; every unreleased pin is marked `TEMPORARY` with its PR.
- [ ] Merge order stated (schemas, then catalog / registry / kernel, then inference-sim).
- [ ] `git merge-tree --write-tree origin/main HEAD` is clean.
- [ ] N/A: this PR changes nothing another repository reads.

## Tests

<!-- Commands run and their results. For each behaviour change: the test that fails without
it, and the mutation that showed it (what was reverted, which test failed). -->

```sh
```

## Held-out and gate evidence

<!-- Required when behaviour, a coefficient or pricing changes. The pre-registered gate (and
when it was written), the baseline build, before / after figures on the SAME points from a
script (quote the output path), seed means with pooled SE where latency is compared, and any
recorded deviation. Coefficient and pricing changes follow
https://github.com/inference-sim/blis-registry/blob/main/docs/fitting-principles.md -->

- [ ] N/A: no behaviour change.

## Data and names

- [ ] No third-party data committed; derived values carry source URL plus commit or sha256.
- [ ] Name scan clean: no internal project, customer or colleague names.

## Duplicate-PR check

<!-- The searches run (open PRs and issues in every affected repository) and why this PR does
not duplicate an existing one. -->

## AI assistance

<!-- State whether AI assistance was used and how. A human submitter has reviewed every
changed line and can defend the change end to end. -->
