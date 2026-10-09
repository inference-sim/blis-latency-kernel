# Provenance, Evidence, Resolved

<p class="lede">Three methods say what a kernel rests on. <code>Provenance</code> lists
every value in scope and where each came from. <code>Evidence</code> counts how many rest on
evidence. <code>Resolved</code> reports the configuration it priced, including every request
the engine would not honor. This page lists what each can contain.</p>

## Provenance

Each entry is a `kernel.CoefficientOrigin`: a name, the set it came from, the method that
produced it, and its scope. Most entries are registry coefficients: every entry of the
scenario's sets whose scope covers this chip and model, whether or not this deployment's
steps use it. Their set is the registry set that supplied them, such as
`cost-model-primitives`. The registry's
documentation describes each set and method.

Entries the kernel supplied itself carry the set `blis-latency-kernel` (the constant
`KernelAssumptionSet`) and the method `assumed`. Their scope holds the value used and the
reason. They are of two kinds.

### Engine settings the kernel filled in

<!-- assumptions -->

| Name | Recorded when |
|---|---|
| `block_size` | The deployment states no block size. 64 for a DSA sparse-attention model, whose indexer accepts only 64; otherwise vLLM's default of 16. |
| `cudagraph_mode` | No CUDA graph mode is stated. vLLM's default, `FULL_AND_PIECEWISE`. |
| `dcp_comm_backend` | Decode-context parallelism is on and no combine backend is stated: the stock `ag_rs`. Also when one is stated but the engine rules give no list to check it against. |
| `engine_version` | The scenario declares a vLLM release other than the one whose behavior the kernel prices (v0.31.0). |
| `mamba_cache_mode` | A model with recurrent layers states no recurrent cache mode: `align` with prefix caching on, `none` with it off. |
| `max_cudagraph_capture_size` | Any CUDA graph is captured. The largest batch a graph covers cannot be stated in a deployment, so it is always vLLM's default. |
| `max_num_batched_tokens` | No token budget per step is stated. vLLM's default for the chip's memory size. |
| `max_num_seqs` | No limit on concurrent requests is stated. vLLM's default for the chip's memory size. |
| `mla_context_chunk` | Decode-context parallelism is on for a latent-attention model and no context length is stated, so the chunk in which a prefill gathers its cached context is taken at its largest. |
| `sparse_mla_backend` | A DSA sparse-attention model on Hopper selects FlashInfer's sparse MLA backend, which needs a FlashInfer release the kernel cannot see installed. |

### Optional coefficients that were missing

When a coefficient the kernel can price without is absent from the scenario's sets, and the
deployment needs it, the entry is named for the coefficient, its value is `absent`, and its
reason says what was priced without it. These are the coefficients it can apply to; a
per-kind name ends in the kind it covers, such as `_mla` or `_kda`, and is needed only where
the model has a layer of that kind.

| Coefficient | Without it |
|---|---|
| `attention_decode_floor`, `attention_decode_rate` | No measured decode attention: the KV read is charged at memory bandwidth and the per-layer floor is not charged |
| `attention_decode_rate_<kind>` | The chip-wide attention pair is used for that kind |
| `attention_prefill_floor`, `attention_prefill_work_scale` | Prefill attention is charged on the plain efficiency ramp, with no floor |
| `recurrent_decode_floor_<kind>`, `recurrent_decode_rate_<kind>` | The recurrent mixer's decode term is not charged |
| `moe_routing_imbalance_median` | Experts are priced as evenly loaded |
| `host_link_bandwidth` | An offload transfer is not capped at the host link's rate |
| `host_admission_per_token`, `host_admission_per_request`, `host_output_token`, `host_completion`, `host_launch_eager_per_layer`, `host_launch_per_kernel`, `host_replay_graph_per_step` | That host cost is charged at zero |

## Evidence

`Evidence()` returns three things: how many entries of `Provenance` rest on evidence (method
`measured`, `literature` or `vendor_spec`), how many there are in all, and the names of those
whose method is `assumed`. The last includes the kernel's own entries above and any registry
coefficient recorded as assumed. Entries with the methods `copied` and `not_charged` are in
the total and in neither of the other two.
[Refuse, assume, override](../concepts/resolution.md#the-evidence-summary) shows a worked
example.

## Resolved

`Resolved()` returns a `kernel.Resolution`: the tensor-, data- and expert-parallel widths the
layout resolved to, the all-reduce backend that will run, whether asynchronous scheduling,
cascade attention and sequence-parallel MoE are on, and a list of overrides. Each
override names the field, what was requested, what was resolved, and why. The kernel records
these:

| Field | When |
|---|---|
| `allreduce_backend` | The custom all-reduce was requested at a tensor-parallel width it does not support, or across nodes on a chip without multi-node NVLink. NCCL runs instead. |
| `cache_dtype` | A sparse-attention model's KV cache is stored in a packed format its backend chooses, such as `fp8_ds_mla` for a stated `fp8` on Hopper. |
| `cp_kv_cache_interleave_size` | Decode-context parallelism is on, the interleave is unstated, the block size is above one, and a NIXL connector is configured, so vLLM sets the interleave to the block size. |
| `mamba_cache_mode` | A hybrid model states `none` with prefix caching on, which vLLM runs as `align`. |

The two context-parallel widths are not yet fields of `kernel.Resolution`. Until blis-schemas
adds them, read them from `DecodeContextParallelWidth()` and `PrefillContextParallelWidth()`.
