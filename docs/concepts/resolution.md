# Refuse, assume, override

<p class="lede">A deployment states what its author asked for; the engine decides what
actually runs. Before pricing anything, the kernel works out what the engine would do. Where
it cannot price that faithfully it refuses; where it has to fill in a setting the deployment
left out, it assumes the engine's default and says so; where the engine would run something
other than what was asked, it prices what would run and records the difference.</p>

All three happen once, in `New`. A kernel that exists has already been through them, and
reports their results through three methods: `Provenance`, `Evidence` and `Resolved`.

## Refuse

`New` returns an error, and no kernel, when it should not price a deployment at all. There
are four reasons:

- **The documents are malformed.** A scenario file is decoded strictly, so a misspelled key
  is an error, and the kernel runs blis-schemas' field-level validation on every document it
  is given. A missing field or a value out of range is the author's to fix, whatever version
  of the engine is meant.
- **The engine would not start it.** vLLM v0.31.0 refuses some layouts at startup, for
  example prefill-context parallelism on a model whose attention is not latent. Pricing such
  a deployment would describe something that cannot run.
- **The kernel cannot price it truthfully.** Pipeline parallelism, for example, is not
  modeled, and pricing the whole model as one stage would misstate every term.
- **A coefficient it needs is missing.** A rate or a floor the price depends on has no entry
  in the coefficient sets the scenario names. Pricing that term at zero would give a step
  that looks plausible and is wrong by whatever the term contributes.

[What New refuses](../reference/refusals.md) lists every case.

## Assume

Many settings may be left out of a deployment, and the engine fills them in. The kernel fills
them in the same way, from vLLM v0.31.0's defaults, and records each one in `Provenance` as an
entry from the set `blis-latency-kernel` with the method `assumed`. The block size, the CUDA
graph mode and the engine's token budget per step are examples.

A deployment that states every setting carries almost no such entries. The exceptions are
listed on [Provenance, Evidence, Resolved](../reference/provenance.md). The one that
matters most is the version of vLLM: the kernel prices v0.31.0's behavior whatever version a
scenario declares, and records `engine_version` as an assumption when the two differ. Every
scenario in this repository declares v0.29.0, the only release blis-schemas publishes rules
for, so every one of them records it. Three facts do follow the declared version, because
they are read from those rules; [Sources of truth](../reference/sources.md#two-sources-of-engine-behavior)
names them.

Some coefficients are optional: the kernel can price without them, by leaving out the term
they set or, for the experts' routing imbalance, by taking the experts as evenly loaded. A
host overhead is an example. When one the deployment needs is missing, that too
is recorded as an assumption, naming what was priced without it, so a missing coefficient
lowers the prediction's evidence rather than silently lowering its time.

## Override

Some settings the engine changes even when they are stated. Asked to use its custom
all-reduce at a width that kernel does not support, vLLM falls back to NCCL. Given an fp8 KV
cache on a sparse-attention model on Hopper, it stores the cache in a packed format that is
larger. The kernel prices what would run, and `Resolved` reports each change as an override:
the field, what was requested, what was resolved, and why.

## The `Evidence` summary

`Provenance` lists every registry value in scope for the deployment, whether or not this
deployment's step uses it, followed by the kernel's own assumptions. `Evidence` counts them,
and counts as evidence a value whose method is `measured`, `literature` or `vendor_spec`. For
the deployment on [Pricing a step](step.md):

```go
--8<-- "example_test.go:evidence"
```

```text
110 of 149 values rest on evidence; 38 were assumed
  assumed       38
  measured      93
  not_charged    1
  vendor_spec   17
the kernel assumed max_cudagraph_capture_size
the kernel assumed engine_version
```

Most of the 38 assumed values are not the kernel's: they are registry coefficients whose
method is recorded as `assumed`. Twenty-four are collective coefficients for 16-GPU groups,
which this eight-GPU deployment never forms; most of the rest are host overheads and CUDA
graph memory. The one
`not_charged` value is a cost the registry records as deliberately zero. The registry's
documentation says how each was obtained. Only the last two lines are assumptions the kernel
made itself.

## Why it is done this way

A cost model that quietly substitutes a default, or prices a deployment that would not run,
produces a number that looks as trustworthy as any other and is not. Every one of the three
outcomes exists so that such a number cannot leave the kernel unmarked: either there is no
number, or the number says what it rests on.
