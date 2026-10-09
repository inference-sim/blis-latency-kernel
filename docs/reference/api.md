# API

<p class="lede">The package's exported names, what each is for, and what this kernel does
for each method of the interface it implements. The full Go documentation is on
<a href="https://pkg.go.dev/github.com/inference-sim/blis-latency-kernel">pkg.go.dev</a>;
the interface itself is defined and documented in
<a href="https://pkg.go.dev/github.com/inference-sim/blis-schemas/kernel">blis-schemas</a>.</p>

```go
import latencykernel "github.com/inference-sim/blis-latency-kernel"
```

## Building a kernel

| Name | What it does |
|---|---|
| `Open(scenario, repos)` | Loads a scenario file and everything it names, and builds a kernel for its first pool. |
| `OpenPool(scenario, repos, i)` | The same for pool `i` of a disaggregated deployment. |
| `OpenInputs(scenario, repos, i)` | Loads the documents without building a kernel, for a caller that needs them too. |
| `New(inputs)` | Builds a kernel from documents already in memory. `Open` and `OpenPool` end here. |
| `Inputs` | The documents a kernel is built from: scenario, deployment, pool index, model graph, chip, fabric, storage devices, coefficient sets and engine rules. |
| `Repos` | The three roots a scenario's names resolve against: scenarios, catalog, registry. |
| `LoadBundle(path)` | Reads a scenario and its deployment from one file of two YAML documents. |

[Price a batch from Go](../guides/go.md) shows each in use.

## Kernel methods

`*Kernel` implements `kernel.Kernel` from blis-schemas. Every method is a pure function of
its arguments and safe to call from many goroutines at once.

| Method | What this kernel does |
|---|---|
| `StepTime(batch)` | Prices one forward pass. [Pricing a step](../concepts/step.md). |
| `FixedBytes()` | Memory per rank before any request. [Memory](../concepts/memory.md). |
| `SequenceVariableBytes(tokens)` | One request's KV cache at a length, in whole pages. |
| `SequenceFixedBytes()` | One request's recurrent state, where it does not grow with length. |
| `TierTime(tier, dir, bytes, depth)` | A transfer to or from an offload tier: the larger of the device's latency and the time to move the bytes at the slower of the device and the host link, that rate shared among the transfers queued. An unknown tier prices as the largest representable duration. |
| `PDTransferTime(tokens, from, to)` | Moving one request's KV cache between pools: the whole request's cache, which decode-context parallelism does not divide, over NVLink within a node or an NVLink rack, and over the fabric otherwise. |
| `AdmissionOverhead(tokens)` | Host time before a request can be scheduled: a fixed part per request and a part per prompt token. |
| `OutputTokenOverhead()` | Host time per emitted token. |
| `CompletionOverhead()` | Host time when a request finishes. |
| `Provenance()` | Every coefficient in scope and every assumption, each with its method. [Provenance](provenance.md#provenance). |
| `Resolved()` | The configuration after resolution, with every override. [Resolved](provenance.md#resolved). |
| `Deployment()` | The pool as the document stated it. A deep copy: writing to it changes nothing in the kernel. |

And four that are not on the interface:

| Method | What it does |
|---|---|
| `Evidence()` | How many of the values in `Provenance` rest on evidence, of how many, and which were assumed. [Evidence](provenance.md#evidence). |
| `DecodeContextParallelWidth()` | How many ranks shard the decode KV cache. |
| `PrefillContextParallelWidth()` | How many ranks split a prefill. |
| `StepTimeInto(batch, per)` | `StepTime`, writing the per-resource breakdown into a map the caller reuses instead of allocating one. |

The two context-parallel widths are methods here only until `kernel.Resolution` in
blis-schemas carries them.

## Constants

| Name | Value |
|---|---|
| `EngineBehaviourVersion` | The vLLM release whose behavior the kernel prices: `0.31.0`. |
| `EngineBehaviourCommit` | That release's commit. |
| `KernelAssumptionSet` | The set name on `Provenance` entries the kernel supplied itself: `blis-latency-kernel`. |
