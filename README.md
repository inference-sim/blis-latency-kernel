# blis-latency-kernel

The cost model of [BLIS](https://github.com/inference-sim). Given a model, a GPU, a set of
coefficients for that GPU and a deployment, it answers the two questions a serving simulator asks:
how long one forward pass over a batch of requests takes, and how much GPU memory the
deployment holds.

**Documentation: <https://inference-sim.github.io/blis-latency-kernel/>**

It is a Go library implementing the `Kernel` interface from
[blis-schemas](https://github.com/inference-sim/blis-schemas). It reads the model and the
hardware from [blis-catalog](https://github.com/inference-sim/blis-catalog) and the coefficients
from [blis-registry](https://github.com/inference-sim/blis-registry), and holds no data of its
own. [inference-sim](https://github.com/inference-sim/inference-sim) calls it once per simulated
step in pull request [#1851](https://github.com/inference-sim/inference-sim/pull/1851). It
prices the engine behavior of vLLM v0.31.0.

## Use

```sh
go get github.com/inference-sim/blis-latency-kernel@latest
```

```go
k, err := latencykernel.Open("my-deployment.yaml", latencykernel.Repos{
	Scenarios: "scenarios", Catalog: "blis-catalog", Registry: "blis-registry",
})
if err != nil {
	panic(err)
}
e := k.StepTime(batch) // e.Overlap, e.NoOverlap, e.Bottleneck, e.PerResource
```

[Price a batch from Go](https://inference-sim.github.io/blis-latency-kernel/latest/guides/go/)
walks through it with a real deployment, and
[Pricing a step](https://inference-sim.github.io/blis-latency-kernel/latest/concepts/step/)
explains what the estimate means.

## Develop

```sh
scripts/fetch-testdata.sh   # the catalog and registry, at the commits testdata/upstream.lock pins
go test ./...
```

The tests are behavioral: they check what the kernel does for an input, most often against a
transcription of the vLLM source or an exact relation the answer must satisfy.
`testdata/README.md` describes the test inputs, including the measurement corpora the scoring
commands use, which are third-party data and not distributed here.
[Contributing](https://inference-sim.github.io/blis-latency-kernel/latest/contributing/)
describes the layout of the repository and the conventions a change is reviewed against.

## License

Apache 2.0, matching the other BLIS repositories.
