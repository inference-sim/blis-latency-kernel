# Memory

<p class="lede">A GPU's memory holds two kinds of thing. Some is taken before any request
arrives: the model's weights, scratch space, captured CUDA graphs, communication buffers.
The rest is the KV cache, which grows with every request's context. The kernel reports the
first as one breakdown per GPU, and the second as a function of a request's length, so a
caller can work out how many requests fit.</p>

Everything here is per rank. A *rank* is one GPU's process in the deployment; a *pool* is the
set of GPUs that run one engine configuration, and every rank in it holds about the same.

## What is held before any request

`FixedBytes` returns five figures, kept separate because a caller deciding whether to add GPUs
or to raise the memory it gives the engine needs to know which one dominates.

Weights
:   Each rank holds its share of the model's parameters. The linear layers are at the width
    they are served in; the embedding table stays at the checkpoint's width, because the
    engine does not quantize it. Dense layers are divided across the tensor-parallel group. Experts are divided by expert
    parallelism into whole experts per rank; without it, every rank holds a slice of every
    expert. When the experts do not divide evenly, some ranks hold one more than the rest,
    and the kernel reports the fuller rank, because its memory runs out first.

Activations
:   Scratch space for the largest step the engine schedules:

    number of buffers × token budget per step × activation width × 2 bytes,

    with a floor beneath it. The number of buffers comes from NVIDIA's memory model, which
    states it separately for dense and mixture-of-experts models and by tensor-parallel
    width. The token budget is the engine's `max_num_batched_tokens`. Activations stay
    16-bit even when the weights are served in fp8 or fp4, because quantized matrix
    multiplies return their results at the input's width.

CUDA graphs
:   The memory a captured graph holds, for the capture mode the deployment runs in.

EPLB replicas
:   The redundant copies of popular experts that expert-parallel load balancing (EPLB)
    adds, sized from the experts' weights. They are moved out of the weights figure rather
    than added to it, so the total counts them once. Zero when it is off.

Communication buffers
:   What the collective library reserves for the communicator, plus the engine's own
    workspace.

The weights and the replicas are computed from the model graph. The other magnitudes come
from the registry: the activation and CUDA graph figures from its `cost-model-memory` set,
the communicator and workspace from `cost-model-primitives`. A kernel lists each registry
value in scope, with how it was obtained, in
[`Provenance`](../reference/provenance.md). For the
deployment on [Pricing a step](step.md):

```go
--8<-- "example_test.go:memory"
```

```text
weights       78.1 GiB
activations    1.1 GiB
CUDA graphs    0.8 GiB
comm buffers   3.9 GiB
EPLB           0.0 GiB
total         83.9 GiB per rank
a 4,096-token request: 137 MiB of KV cache per rank
```

## What each request adds

`SequenceVariableBytes(tokens)` is the KV cache one request holds at a given length. Each
layer that keeps a cache stores, for every token,

\[
n_\text{tensors} \cdot \max\!\left(1, \left\lfloor \frac{n_\text{kv}}{\text{tp}}
\right\rfloor\right) \cdot d_\text{head} \cdot \text{bytes per element}
\]

where \(n_\text{tensors}\) is two for ordinary attention, a key and a value, and one for
latent attention (MLA), which caches a single compressed vector per token. Four details
change the answer by large factors:

- **Heads are not split.** Tensor parallelism divides the KV heads among the ranks, but a
  head is never split: a model with 8 KV heads holds one per rank at tp 8, and still one at
  tp 16. A latent cache has a single head, so no tensor-parallel width shrinks it.
- **The cache's data type is its own.** An fp8 cache holds one byte per element even for a
  model served in bf16. Some packed formats state their own size: the sparse-attention
  `fp8_ds_mla` layout takes 656 bytes per token per layer, scales included.
- **Memory comes in pages.** The cache is allocated in blocks of `block_size` tokens, so a
  request's length is rounded up to a whole number of blocks before it is charged.
- **Decode-context parallelism shards tokens.** It spreads a request's tokens across its
  group, so each rank holds a share of them. Prefill-context parallelism does not: every rank
  keeps the whole cache.

The example above is a latent cache: 576 one-byte elements per token in each of 61 layers is
35,136 bytes per token, and a 4,096-token request holds 137 MiB on every rank.

## What each request holds regardless of length

`SequenceFixedBytes` is for models with recurrent layers (Mamba-2, gated delta networks, Kimi
delta attention), whose per-request state does not grow with the context. How much of it a
request holds depends on the engine's recurrent cache mode:

| Mode | State a request holds | Counted in |
|---|---|---|
| `none` | One page, plus one per speculative token | `SequenceFixedBytes` |
| `align` | Two pages, plus one per speculative token | `SequenceFixedBytes` |
| `all` | One page per block of its context, plus one per speculative token | `SequenceVariableBytes` |

vLLM uses `align` whenever prefix caching is on. For a model with no recurrent layers,
`SequenceFixedBytes` is zero.

## In the source

The breakdown is composed in `computeFixedBytes` and its magnitudes resolved in `liftMemory`,
both in [`new.go`](https://github.com/inference-sim/blis-latency-kernel/blob/main/new.go). The
KV and recurrent-state laws are in
[`internal/price/memory.go`](https://github.com/inference-sim/blis-latency-kernel/blob/main/internal/price/memory.go).
