# Documentation and releases

<p class="lede">These pages are built with Material for MkDocs and published to GitHub
Pages, one copy per release line, the same way as blis-schemas' documentation. The examples,
the figure, the list of divergences, the pins and the table of assumptions are generated from
the code or tested against it. The prose is kept right by review.</p>

## Preview locally

```sh
python3 -m venv .venv
.venv/bin/pip install -r docs/requirements.txt
.venv/bin/mkdocs serve
```

The site is then at <http://127.0.0.1:8000>. CI builds it with `mkdocs build --strict`, which
fails on a page missing from the navigation, a link to a missing page or heading, or a missing
included file.

## Where things live

| Path | Contents |
|---|---|
| `mkdocs.yml` | Site configuration and the navigation. A new page must be added here. |
| `docs/` | The pages, one directory per section. |
| `docs/figures/` | Drawn figures, as inline SVG that takes its colors from the page. |
| `docs/generated/` | Files written by `internal/docgen`. Never edit these by hand. |
| `docs/stylesheets/extra.css` | Typography, figure and chart styles. |
| `docs/requirements.txt` | The pinned Python packages the site is built with. |
| `example_test.go` | The Go shown on the pages, run and checked by `go test`. |
| `docs_test.go` | The tests that tie the pages to the code. |

A page includes a file with the snippets syntax, `--8<-- "path"`, or a marked section of one
with `--8<-- "path:name"`, where the path is relative to the repository root.

The pages use American spelling. They are written for engineers who have not read the code:
each opens with a paragraph that stands alone, defines a term before it uses it, and points to
the source for detail rather than reproducing it.

## What keeps the pages correct

All of these run in `go test ./...`, after `scripts/fetch-testdata.sh`.

**Generated files.** `go run ./internal/docgen` writes `docs/generated/`: the step figure,
priced by this kernel; the known divergences, read from the source comments; and the pins,
read from `go.mod`, `testdata/upstream.lock` and the constants in `new.go`.
`TestGeneratedDocsAreCurrent` fails while a committed file differs from what the generator
would write. A change to pricing therefore moves the figure, and the test says to regenerate
it.

**Go examples.** The Go on the pages is sections of `example_test.go`, each between a start
and an end marker comment that name it. Go runs each example and compares what it
prints with its `// Output:` comment, so a snippet cannot show a call that no longer compiles.
`TestDocOutputsMatchTheirExamples` checks that the output shown beside a snippet on a page is
that example's output.

**Tables tied to the code.** `TestDocAssumptionsMatchTheCode` checks the table of assumptions
on [Provenance, Evidence, Resolved](../reference/provenance.md) against the names the code
can record, in both directions, and `TestDocOptionalCoefficientsMatchTheCode` does the same for
the optional coefficients.

No test can check that the prose is right. When a change alters behavior, search these pages
for it and update them in the same pull request.

## Publishing

`.github/workflows/docs.yml` builds and publishes the site with
[mike](https://github.com/jimporter/mike), which keeps one directory per version on the
`gh-pages` branch.

| Event | What is published |
|---|---|
| A pull request touching the docs | Nothing. The site is built with `--strict`, and the job fails on any warning. |
| A push to `main` | Version `dev`, rebuilt from `main`. Until a release is published, the site root serves `dev`. |
| A release `vX.Y.Z` is published on GitHub | Version `X.Y`, rebuilt from the tag. If `X.Y` is the newest stable release line, the alias `latest` moves to it and the site root serves it. |
| A pre-release, or a tag not of the form `vX.Y.Z` | Nothing. |
| A manual run from the Actions tab | The version given, built from the chosen branch or tag. It becomes `latest` only if asked. |

Versions are release lines, not patches: `v0.1.0` and `v0.1.1` both publish to `0.1`, each
replacing the last.

### Making a release

1. Merge the changes to `main`, with `docs/generated/` regenerated.
2. Publish a release on GitHub with a new tag, `vX.Y.Z`.

The workflow then publishes `X.Y`.

A release created by a workflow using the default `GITHUB_TOKEN` starts no other workflow, so
it would not publish the docs. Release automation, if it is added, must publish with another
token or start this workflow itself.

### Backfilling or repairing a version

Run the **docs** workflow by hand from the Actions tab. Choose the branch or tag to build from,
give the release line as `MAJOR.MINOR`, and choose whether it becomes `latest`. A tag older than
the documentation has no `mkdocs.yml` and cannot be built; build such a line from a later
commit whose code it matches.

The repository's GitHub Pages setting must serve the `gh-pages` branch, from its root. The
site is then at <https://inference-sim.github.io/blis-latency-kernel/>.
