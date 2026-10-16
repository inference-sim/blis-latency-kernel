package latencykernel

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
	"github.com/inference-sim/blis-schemas/spec/model"
)

// deepEPA2A is one DeepEP backend's measured MoE dispatch + combine price: floor plus
// bytes over rate, on the same tokens x hidden x 2 x top_k basis the routed all-to-all is
// charged on (blis-registry moe_a2a_floor_* / moe_a2a_rate_*, fitted from vLLM's own
// DeepEP prepare/finalize calls).
type deepEPA2A struct {
	variant string
	floor   float64 // seconds
	rate    float64 // bytes per second
}

// seconds prices one MoE layer's dispatch and combine.
func (d *deepEPA2A) seconds(bytes float64) float64 {
	return d.floor + bytes/d.rate
}

// deepEPVariant names the measured DeepEP family the resolved backend runs, or "" for a
// backend that is not DeepEP. DeepEP v2 picks its path once, from enforce_eager, not per
// step: use_cudagraph = not enforce_eager (vllm/model_executor/layers/fused_moe/
// all2all_utils.py:286-298 at v0.31.0). The deployment states no enforce_eager, so a
// resolved cudagraph mode of NONE is read as it.
func (k *Kernel) deepEPVariant() string {
	switch k.rules.EffectiveAll2AllBackend(k.pool.Engine.All2AllBackend) {
	case "deepep_high_throughput":
		return "deepep_high_throughput"
	case "deepep_low_latency":
		return "deepep_low_latency"
	case "deepep_v2":
		if k.graphOther == graphModeEager && k.graphDecode == graphModeEager {
			return "deepep_v2_eager"
		}
		return "deepep_v2_graph"
	}
	return ""
}

// liftDeepEP resolves the DeepEP price when one runs. Where the registry has none for this
// part and width, or the expert group leaves the node (the measurement is one node), the
// generic routed all-to-all stands and that is recorded rather than silent
// (blis-latency-kernel#57).
func (k *Kernel) liftDeepEP(c *resolve.Coefficients) {
	k.deepEP = nil
	if !k.routedAll2All() {
		return
	}
	variant := k.deepEPVariant()
	if variant == "" {
		return
	}
	key := collKey{Op: model.OpAll2All, Group: price.GroupExpert}
	width, _ := k.groupSize(key.Group)
	chip := strings.ReplaceAll(k.chip.Name, "-", "_")
	stem := fmt.Sprintf("%s_%drank_%s", variant, width, chip)
	floorName, rateName := "moe_a2a_floor_"+stem, "moe_a2a_rate_"+stem
	if k.crossesNodes(key) || !c.Has(floorName) || !c.Has(rateName) {
		why := "no " + floorName + " in the scenario's coefficient sets"
		if k.crossesNodes(key) {
			why = "the expert group spans nodes and the DeepEP measurement is one node"
		}
		k.assume("moe_a2a_"+variant, "generic routed all-to-all",
			why+", so the dispatch is priced as NCCL's all-to-all over top_k bytes")
		return
	}
	k.deepEP = &deepEPA2A{
		variant: variant,
		floor:   c.ValueOr(floorName, 0) * 1e-6,
		rate:    c.ValueOr(rateName, 0) * 1e6,
	}
	if variant == "deepep_v2_eager" || variant == "deepep_v2_graph" {
		k.assume("deepep_v2_mode", variant,
			"DeepEP v2's path follows enforce_eager (all2all_utils.py:286-298), which the "+
				"deployment cannot state; a resolved cudagraph_mode of NONE is read as it")
	}
}

// deepEPBufferMB is VLLM_DEEPEP_BUFFER_SIZE_MB's default (vllm/envs.py:1990-1992 at
// v0.31.0). The deployment cannot state the variable, so the default is priced.
const deepEPBufferMB = 1024

// deepEPBufferBytes is the HBM one rank's DeepEP buffers hold. vLLM builds them while the
// MoE kernels are made at weight loading (process_weights_after_loading ->
// maybe_make_prepare_finalize -> all2all_manager.get_handle), after the memory snapshot,
// so determine_available_memory counts them in non_kv_cache_memory (vllm/utils/
// mem_utils.py:300-321; vllm/v1/worker/gpu_worker.py:673-676): they come out of the KV
// cache. Per backend, at v0.31.0 (vllm/distributed/device_communicators/all2all.py):
//
//	high_throughput  num_nvl_bytes = buffer MB (:224), plus the same again as RDMA bytes
//	                 when the group spans nodes (:228-233)
//	low_latency      num_nvl_bytes = buffer MB (:306) plus DeepEP's
//	                 get_low_latency_rdma_size_hint(max_num_batched_tokens, hidden, ep, E)
//	                 (:308-313; FusedMoEConfig.max_num_tokens is max_num_batched_tokens,
//	                 fused_moe/layer.py:272,359)
//	v2               ElasticBuffer.calculate_buffer_size(max_num_batched_tokens, hidden,
//	                 top_k), DeepEP d4f41e4e93 csrc/elastic/buffer.hpp:622-655
func (k *Kernel) deepEPBufferBytes(g *model.Graph) int64 {
	if !k.routedAll2All() {
		return 0
	}
	backend := k.rules.EffectiveAll2AllBackend(k.pool.Engine.All2AllBackend)
	if !strings.HasPrefix(backend, "deepep_") {
		return 0
	}
	nvl := int64(deepEPBufferMB) << 20
	ep, _ := k.groupSize(price.GroupExpert)
	topK := 0
	for _, l := range k.plan.Layers {
		topK = max(topK, l.TopK)
	}
	hidden, tokens := g.Global.HiddenSize, k.batchedTokens
	switch backend {
	case "deepep_high_throughput":
		if k.crossesNodes(collKey{Op: model.OpAll2All, Group: price.GroupExpert}) {
			return 2 * nvl
		}
		return nvl
	case "deepep_low_latency":
		return nvl + lowLatencyRDMABytes(tokens, deepEPLLHidden(hidden), ep, k.totalExperts)
	case "deepep_v2":
		return elasticBufferBytes(tokens, deepEPXferHidden(hidden), topK, ep)
	}
	return 0
}

// deepEPLLHidden rounds up to a hidden size DeepEP's low-latency kernels are compiled for
// (DeepEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES, prepare_finalize/deepep_ll.py:57-79).
func deepEPLLHidden(hidden int) int {
	for _, h := range []int{2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192} {
		if h >= hidden {
			return h
		}
	}
	return hidden
}

// deepEPXferHidden rounds a BF16 row up to DeepEP's 512-byte copy atom
// (maybe_roundup_layer_hidden_size, prepare_finalize/deepep_v2.py:98-105).
func deepEPXferHidden(hidden int) int {
	return alignUp(hidden*2, 512) / 2
}

func alignUp(n, a int) int { return (n + a - 1) / a * a }

// lowLatencyRDMABytes is DeepEP's get_low_latency_rdma_size_hint: the LowLatencyLayout's
// two send, two receive and two signalling buffers (DeepEP d4f41e4e93
// csrc/legacy/config.hpp:131-188), aligned to its 128-byte buffer alignment.
func lowLatencyRDMABytes(tokens, hidden, ranks, experts int) int64 {
	scales := int64(hidden / 128)
	t, h, e := int64(tokens), int64(hidden), int64(experts)
	dispatchMsg := 16 + max64(h*2, h+scales*4)
	combineMsg := scales*4 + h*2
	send := max64(t*dispatchMsg, e*t*combineMsg)
	recv := max64(e*t*dispatchMsg, e*t*combineMsg)
	signal := int64(alignUp(int(e*4), 128))
	total := 2*send + 2*recv + 2*signal
	return (total + 128) / 128 * 128
}

// elasticBufferBytes is DeepEP v2's calculate_buffer_size for one node with NVLink
// scale-up (direct dispatch and combine, no hybrid mode, single reduction): the larger of
// a ranks x tokens dispatch layout and a top_k x tokens combine layout, 2 MiB aligned
// (DeepEP d4f41e4e93 csrc/elastic/buffer.hpp:556-655; TokenLayout, deep_ep/include/deep_ep/
// common/layout.cuh:179-208, 32-byte TMA alignment). BF16 dispatch, as vLLM runs an
// unquantized-activation MoE. Its workspace is excluded, as DeepEP's figure excludes it.
func elasticBufferBytes(tokens, hidden, topK, ranks int) int64 {
	a := func(n int) int64 { return int64(alignUp(n, 32)) }
	dispatchToken := a(hidden*2) + a(topK*8+(1+topK)*4)
	combineToken := a(hidden*2) + a(topK*8)
	dispatch := int64(ranks) * int64(tokens) * dispatchToken
	combine := int64(topK) * int64(tokens) * combineToken
	const align = 2 << 20
	return (max64(dispatch, combine) + align - 1) / align * align
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
