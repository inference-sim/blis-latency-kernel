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
| `internal/resolve` | Scenario to layout and fabric; which conditional collectives survive |
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

Both need `blis-catalog`, `blis-registry` and `blis-schemas` checked out alongside this
repository; the commands take their paths as flags.

## Dependencies run one way

This module depends on `blis-schemas` and nothing else of substance. It does NOT depend on the
simulator: the simulator consumes this kernel, so an import in this direction would close a cycle
between the two repositories. The command that compares this kernel against the simulator'''s
earlier roofline and trained-physics models therefore lives in the simulator, as
`cmd/blisbaseline`, where the dependency runs the correct way.

## Licence

Apache 2.0, matching the other BLIS repositories.
