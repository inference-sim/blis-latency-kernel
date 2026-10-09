# blis-latency-kernel

A Go implementation of the BLIS inference cost model: given a model graph, a chip, a set of
calibrated coefficients and a deployment, it prices one forward pass over a batch.

It implements the `Kernel` interface from
[blis-schemas](https://github.com/inference-sim/blis-schemas) and reads its inputs from
[blis-catalog](https://github.com/inference-sim/blis-catalog) (declared hardware and model
facts) and [blis-registry](https://github.com/inference-sim/blis-registry) (calibrated
coefficients). It holds no data of its own.

The model it realizes is specified in `inference-sim/docs/perf-model`: thirteen primitives,
composed per layer as a max over resources and summed across layers, with the exceptions and
their measured evidence recorded there.

## What it is for

A discrete-event simulator calls `StepTime` once per simulated step, so the design splits
sharply between what is resolved once and what is computed per call. `New` does the layout
resolution, coefficient scoping, graph planning and all batch-independent arithmetic; a step is
then a walk over distinct layer *kinds* rather than layers, and costs a few hundred nanoseconds.
`StepTimeInto` is the allocation-free variant for callers that supply their own breakdown map.

Nothing on that path is a table lookup. The kernel evaluates closed-form laws, which is what
lets it price a part or a shape that was never measured — and also what bounds its accuracy
against an interpolating model on shapes that were.

The model it prices arrives as a `ModelGraph`: a DAG of nine cost primitives, defined in
[`blis-schemas`](https://github.com/inference-sim/blis-schemas) (`spec/model`) and
instantiated in [`blis-catalog`](https://github.com/inference-sim/blis-catalog) as
`models/<name>/graph.yaml`. This kernel consumes graphs; it neither defines nor stores
them, and its library code contains no model names — dispatch is on the primitives, so a
model's identity never reaches a branch. See
[The BLIS Repositories](https://github.com/inference-sim/inference-sim/blob/main/docs/concepts/blis-repositories.md).

## Layout

| Path | Owns |
|---|---|
| `kernel.go` | The interface methods and the step-time composition |
| `new.go` | `New`: resolution, coefficient lifting, precomputation |
| `internal/price` | The cost laws as pure functions: efficiency ramp, collective spans, KV bytes |
| `internal/resolve` | Scenario to layout and fabric; which conditional collectives survive; how the decode-context knobs resolve |
| `internal/harness` | What the scoring commands share |
| `cmd/score` | Absolute inter-token latency against published benchmark runs |
| `cmd/shape` | Concurrency-response shape against NVIDIA's AISimulate accuracy snapshot |
| `cmd/worked-table` | Generates the worked step-time table in the design document |

## Accuracy

`cmd/shape` is the standing evaluation: it scores the ratio of step times across concurrency at
a fixed deployment against NVIDIA's published snapshot, with both sides normalised to their own
lowest concurrency. The current figures, and the conditions under which they hold, are reported
by the command itself rather than restated here, because a number in a README goes stale and a
number a command prints cannot.

## Tests

Behavioural rather than structural: each asserts what a method does for a given input, not that
a symbol exists. Several carry a comment naming the defect they were written against, because a
test whose purpose is forgotten is the next test to be deleted.

```
go test ./...
go run ./cmd/shape -testdata testdata/aisimulate
```

Both run against the pinned copies under `testdata/` (see `testdata/VENDORED.md`), and
blis-schemas is an ordinary module dependency. To score against a live upstream checkout
instead, set `BLIS_CATALOG` / `BLIS_REGISTRY` for the tests, or pass `-catalog` / `-registry`
to the commands.

Where a law has an exact engine counterpart, the tests check it against that counterpart
transcribed from vLLM v0.31.0 -- the decode-context per-rank length, the prefill-context zigzag,
the KV page arithmetic, the rank layout every collective group is built from -- over grids or
random draws rather than at one point. Where a quantity has no closed form, they assert
metamorphic relations: inflate one primitive's coefficients and check which step times move,
change one width and check which terms follow.

## Sources of truth, pinned

Every claim this code makes about what the engine does is stated against one release, and every
pin is a commit, not a moving name:

| what | source | pin |
|---|---|---|
| engine behaviour | vLLM `v0.31.0` | `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| interface and document schemas | blis-schemas `v0.2.2` | `a0ba5d42ff753713f4dabc74a4ca38b374a91ef4` (`go.mod`) |
| coefficients | blis-registry `v0.1.1` | `f7519b12b3393851a3ad416e26c819dd951d4c0d` (`testdata/VENDORED.md`) |
| catalog | blis-catalog `0.2.1` | `28e82d4c249893ee25b1412a63d2347c5165b004` (`testdata/VENDORED.md`) |

A comment citing `vllm/<path>:<lines>` means those lines at `v0.31.0`; read them with
`git show v0.31.0:<path>` in a vLLM checkout. Engine behaviour this repository encodes itself
-- how decode-context parallelism combines, which layouts start, how a graph mode is
dispatched, what an unstated setting defaults to -- is v0.31.0's. Version-scoped facts the
kernel reads from a blis-schemas rules pack instead (which widths the custom all-reduce
supports, when sequence-parallel MoE engages, which DCP backend names a release accepts) are
that pack's, and the pack is chosen by the scenario's `engine_version`; only the v0.29 pack is
published today.

Where the kernel cannot know what the engine would choose -- a model's own DCP defaults, a
backend-preferred block size, a downgraded graph mode -- it does not guess silently. It either
refuses the deployment, or prices the engine's stock default and records that default in
`Provenance()` as an entry from set `blis-latency-kernel` with method `assumed`, which
`Evidence()` counts. Where v0.31.0's choice follows from the deployment, it is made as the
engine makes it and recorded the same way: an unstated block size on a DSA sparse-MLA model,
which is 64 rather than the stock 16 (`sparse_mla.go`). A deployment that states every setting carries no
such entry, with three exceptions: the cudagraph capture ceiling, which blis-schemas has no
field to state, so a capturing deployment always runs vLLM's default; a stated DCP backend
that could not be checked against the release, because the rules value carries no list of
accepted names; and a DSA model on Hopper whose cache selects FlashInfer's SM90 sparse MLA,
which needs a FlashInfer release the kernel cannot see installed. An optional coefficient
the deployment needs and the scenario's coefficient sets do not carry is recorded the same
way, naming what is priced without it.

Where the kernel knowingly prices something differently from v0.31.0 -- because correcting it
would move scored results fitted under the current structure, and is therefore a refit rather
than a fix -- the code says so at the site, with the engine lines it diverges from. Those
notes are headed KNOWN DIVERGENCE, KNOWN over-charge or COVERAGE LIMIT, so
`grep -rnE "KNOWN DIVERGENCE|KNOWN over-charge|COVERAGE LIMIT" --include='*.go' .` lists
them.

## Using it from a simulator

Everything a simulator needs is on `kernel.Kernel`:

- `Deployment()` is the pool as stated: its role and its engine settings, including the
  admission settings (`block_size`, `max_num_seqs`, `max_num_batched_tokens`) a scheduler
  sizes itself from.
- `Resolved()` is what resolution settled: the tensor-, data- and expert-parallel widths
  (`TensorParallel()`, `DataParallel()`, `ExpertParallel()` apply the floor of one), the
  all-reduce backend, and every request the layout overrode, with the reason.

The two context-parallel widths are the one exception: `kernel.Resolution` does not carry them
yet, so `DecodeContextParallelWidth()` and `PrefillContextParallelWidth()` sit on `*Kernel`
until it does.

Deployment identity -- the model, the chip, the expert geometry -- is not re-exported. A
harness that configures another backend for the same deployment takes the documents from
`OpenInputs` and builds the kernel with `New(in)`, so it holds the very chip and graph the
kernel priced.

## Dependencies run one way

This module depends on `blis-schemas` and nothing else of substance. It does NOT depend on the
simulator: the simulator consumes this kernel, so an import in this direction would close a cycle
between the two repositories. The command that compares this kernel against the simulator's
earlier roofline and trained-physics models therefore lives in the simulator, as
`cmd/blisbaseline`, where the dependency runs the correct way.

## Licence

Apache 2.0, matching the other BLIS repositories.
