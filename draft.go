package latencykernel

import (
	"fmt"
	"strconv"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// The speculator: its draft passes, priced per step, and its weights and caches, resident
// while the deployment speculates.
//
// An earlier form never read the graph's speculator. A deployment drafting tokens was
// priced at the target's verify width alone, so its draft passes were free and its draft
// weights were not resident: about 7.9 GB across the model for DeepSeek-V4.1-Flash's three
// DSpark layers and its context layer (3.9 GB a rank at TP8),
// and a whole decoder layer per MTP model. The drafting mode decides how the stack runs
// (blis-schemas model.Drafting):
//
//	autoregressive   num_spec_tokens passes of one position per request, each feeding
//	                 the next, as MTP and EAGLE drafters run
//	parallel_block   ONE pass of num_spec_tokens positions per request, DSpark's "each
//	                 request emits exactly N = num_speculative_tokens query tokens"
//	                 (vllm/v1/worker/gpu/spec_decode/dspark/speculator.py at v0.31.0)
//
// Around each pass run the draft's own embedding and head, over the same rows; a
// parallel-block drafter's sequential head runs once per draft position over one row per
// request; and its context stack runs once per step over every token the target
// schedules, prefill tokens included.

// liftDraft resolves the speculator against the deployment: whether it runs, and its
// layers' caches. inferred is the per-token width of a layer whose cache the graph leaves
// to inference.
func (k *Kernel) liftDraft(c *resolve.Coefficients, inferred func(price.PlannedLayer) float64) error {
	d := k.plan.Draft
	spec := k.pool.Engine.Speculative
	switch {
	case k.numSpecTokens() == 0:
		return nil
	case d == nil:
		k.assume("speculator", "not priced",
			"the deployment speculates and the model graph states no speculator, so its draft "+
				"passes cost nothing and its weights are not resident")
		return nil
	}
	k.speculating = true
	for _, l := range k.plan.Layers {
		if l.AttnKind == model.AttentionSparseMLA && l.AttnQHeads > 0 {
			k.assume("sparse_read_per_query", "1 + num_spec_tokens",
				"a verify step's sparse attention is charged one window-and-top-k read per "+
					"query token, as vLLM gathers them (vllm/models/deepseek_v41/nvidia/"+
					"flashmla.py:200-203); the registry's sparse rate was fitted on single-token "+
					"decode, and how much of that read a request's tokens share in cache was "+
					"never measured")
			break
		}
	}
	if spec.Method != "" && d.Method != "" && spec.Method != d.Method {
		k.assume("speculator", d.Method,
			"the deployment names method "+spec.Method+" and the graph states a "+d.Method+
				" speculator, which is the one priced")
	}
	if n := k.numSpecTokens(); d.NumSpec > 0 && n != d.NumSpec {
		// The graph sizes what depends on the draft length -- a parallel-block drafter's
		// block, a compressor ring's entries (blis-schemas model.Cache.Entries) -- at its
		// own num_spec_tokens.
		k.assume("num_spec_tokens", strconv.Itoa(n),
			"the deployment drafts "+strconv.Itoa(n)+" tokens a step and the graph's "+
				"speculator states "+strconv.Itoa(d.NumSpec)+"; the passes are priced at the "+
				"deployment's count, and what the graph sizes at its own count is taken as "+
				"stated")
	}
	lift := func(layers []price.PlannedLayer) []layerKV {
		kv := make([]layerKV, len(layers))
		for i, l := range layers {
			if l.AttnQHeads <= 0 {
				continue
			}
			per := inferred(l)
			kv[i].inferred = per
			for _, t := range k.cacheTerms(layerCaches(l), l.Count) {
				t.draft = true
				k.caches = append(k.caches, t)
			}
			if len(l.AttnCaches) > 0 {
				kv[i].stated = true
				continue
			}
			kv[i].perToken = per
			k.draftKVBytesPerToken += per * float64(l.Count)
		}
		return kv
	}
	k.draftLayerKV = lift(d.Layers)
	k.draftContextKV = lift(d.Context)
	return nil
}

// priceDraft prices one step's draft work, adding each resource's time to per, and returns
// the stage time and the kernels it launches.
func (k *Kernel) priceDraft(b kernel.Batch, s *stepShape,
	perResource *[numResources]float64) (float64, int) {
	d := k.plan.Draft
	n := k.numSpecTokens()

	// The requests that sample this step draft: every decode, and a prefill whose chunk
	// completes its prompt. A chunk that leaves prompt to go samples nothing.
	var drafting []kernel.ReqShape
	for _, r := range b.Reqs {
		ctx := r.Computed + r.Scheduled
		if ctx <= 0 || (r.PromptLen > 0 && ctx < r.PromptLen) {
			continue
		}
		drafting = append(drafting, kernel.ReqShape{Computed: ctx, PromptLen: ctx})
	}

	var overlap float64
	kernels := 0
	if len(d.Context) > 0 {
		overlap += k.priceLayers(d.Context, nil, k.draftContextKV, s, perResource)
		kernels += d.ContextKernels
	}
	if len(drafting) == 0 {
		return overlap, kernels
	}

	passes, positions := n, 1
	if d.Drafting == model.DraftingParallelBlock {
		passes, positions = 1, n
	}
	for i := range drafting {
		drafting[i].Scheduled = positions
	}
	pass := kernel.Batch{Reqs: drafting, DecodeThreshold: positions, SMBudget: b.SMBudget}
	ps := k.shapeOf(pass)
	one := []layerKV{{}}
	var per [numResources]float64
	t := k.priceLayers(d.Layers, nil, k.draftLayerKV, &ps, &per)
	for _, part := range []price.PlannedLayer{d.Embedding, d.Head} {
		if part.Count > 0 {
			t += k.priceLayers([]price.PlannedLayer{part}, nil, one, &ps, &per)
		}
	}
	// An autoregressive drafter's passes differ only in reading one more token of context
	// each, so one pass is priced and charged passes times.
	overlap += float64(passes) * t
	for r := range per {
		perResource[r] += float64(passes) * per[r]
	}
	kernels += passes * d.Kernels

	if d.SequentialHead.Count > 0 {
		for i := range drafting {
			drafting[i].Scheduled = 1
		}
		seq := kernel.Batch{Reqs: drafting, DecodeThreshold: 1, SMBudget: b.SMBudget}
		ss := k.shapeOf(seq)
		var per [numResources]float64
		t := k.priceLayers([]price.PlannedLayer{d.SequentialHead}, nil, one, &ss, &per)
		overlap += float64(n) * t
		for r := range per {
			perResource[r] += float64(n) * per[r]
		}
		kernels += n * d.SequentialKernels
	}
	return overlap, kernels
}

// liftStepTerms resolves the step terms construction can settle once: recurrent prefill
// forms, per-role attention forms, the host gather rate and where engram tables live, the
// FlashMLA head padding, and the regions a model breaks out of its own capture.
func (k *Kernel) liftStepTerms(c *resolve.Coefficients, g *model.Graph) error {
	k.resolveEngram(k.rules)
	// Every layer a step can price: the target's, its embedding and head, and, while the
	// deployment speculates, the drafter's. A term needed only by a draft layer is needed.
	layers := append([]price.PlannedLayer{k.plan.Head, k.plan.Embedding}, k.plan.Layers...)
	if d := k.plan.Draft; d != nil && k.speculating {
		layers = append(append(append(layers, d.Layers...), d.Context...), d.Embedding,
			d.Head, d.SequentialHead)
	}
	usedRec := map[model.RecurrentKind]bool{}
	hostGather, engram, maxLenExtent := false, false, false
	for _, l := range layers {
		if l.RecurrentKind != "" {
			usedRec[l.RecurrentKind] = true
		}
		for _, gt := range l.Gathers {
			hostGather = hostGather || k.gatherResidency(gt) == model.ResidencyHost
			engram = engram || gt.Control == model.ResidencyEngramCPUOffload
		}
		for _, cw := range l.Context {
			maxLenExtent = maxLenExtent || cw.MaxModelLenExtent
		}
	}
	if engram {
		k.assumeEngram()
	}
	if maxLenExtent && k.maxModelLen <= 0 {
		k.assume("max_model_len", "context",
			"a decode kernel the graph says runs over a buffer padded to max_model_len is "+
				"charged over each request's context instead: neither the deployment nor the "+
				"graph states the bound")
	}

	k.recurrentPrefill = map[model.RecurrentKind]floorRate{}
	for kind, suffix := range map[model.RecurrentKind]string{
		model.RecurrentMamba2: "mamba2",
		model.RecurrentKDA:    "kda",
		model.RecurrentGDN:    "gdn",
	} {
		if !usedRec[kind] {
			continue
		}
		rate := c.ValueOr("recurrent_prefill_rate_"+suffix, 0)
		if rate > 0 {
			floor := k.optional(c, "recurrent_prefill_floor_"+suffix, true,
				"the "+suffix+" mixer's prefill floor is not charged")
			k.recurrentPrefill[kind] = floorRate{
				floor: time.Duration(floor * float64(time.Microsecond)), rate: rate * 1e6,
			}
			continue
		}
		k.assume("recurrent_prefill_rate_"+suffix, "chunked-scan FLOPs",
			"no measured prefill form for the "+suffix+" mixer, so its prefill tokens are "+
				"priced as the chunked scan the engine runs for them, from its FLOPs on the "+
				"BF16 ramp, and not at the decode scan's rate, which is a different kernel")
	}
	if len(usedRec) > 0 {
		bf16 := price.Format{DType: model.DTypeBF16}
		if err := k.addFit(c, bf16, false); err != nil {
			return fmt.Errorf("a recurrent prefill is priced on the BF16 ramp: %w", err)
		}
		k.prefillFit = k.fit(bf16, false)
	}

	k.attentionByRole = map[string]floorRate{}
	for _, l := range layers {
		for _, a := range l.Attentions {
			if a.HoldsKV() || a.Role == "" {
				continue
			}
			if _, done := k.attentionByRole[a.Role]; done {
				continue
			}
			rate := k.optional(c, "attention_decode_rate_"+a.Role, true,
				"no measured decode form for the "+a.Role+" kernel, so it is priced with "+
					"its attention kind's pair")
			if rate <= 0 {
				// Recorded once: the absence above is the record.
				k.attentionByRole[a.Role] = floorRate{}
				continue
			}
			floor := k.attentionFloor
			if f := k.optional(c, "attention_decode_floor_"+a.Role, true,
				"the "+a.Role+" kernel inherits the part-wide attention floor"); f > 0 {
				floor = time.Duration(f * float64(time.Microsecond))
			}
			k.attentionByRole[a.Role] = floorRate{floor: floor, rate: rate * 1e6}
		}
	}
	for role, fr := range k.attentionByRole {
		if fr.rate <= 0 {
			delete(k.attentionByRole, role)
		}
	}

	k.hostGatherBytesPerSecond = c.ValueOr("host_gather_rate_uva", 0) * 1e6
	if k.hostGatherBytesPerSecond <= 0 {
		k.hostGatherBytesPerSecond = k.hostBytesPerSecond
		if hostGather && k.hostGatherBytesPerSecond <= 0 {
			return fmt.Errorf("a table is read from host memory and the scenario's sets " +
				"carry neither host_gather_rate_uva nor host_link_bandwidth, so its rows " +
				"have no rate to cross the host link at")
		}
		if hostGather {
			k.assume("host_gather_rate_uva", "host_link_bandwidth",
				"no measured rate for a gather from pinned host memory, so a host-resident "+
					"table's rows are read at the host link's rate")
		}
	}

	// FlashMLA serves the family on SM90 and SM100 alike -- V4's FlashMLA class everywhere
	// but SM120 (vllm/models/deepseek_v4/nvidia/model.py:1170-1177), and V4.1's SM100 mega
	// attention subclasses it, its cost scaling "with *padded* heads -- always 64"
	// (deepseek_v41/nvidia/flash_mla_mega_attn.py:76-80, 143) -- so both pad. SM120 runs
	// FlashInfer instead.
	k.padFlashMLA = deepseekV4Family(g) && k.layout.TP > 1 && k.family != 12

	// The eager regions of a breakable capture: one per attention layer -- each layer's
	// indexer and attention are one @eager_break_during_capture region
	// (vllm/models/deepseek_v41/attention.py:766-792, 947-971; deepseek_v4/attention.py:538,
	// 733 at v0.31.0) -- and, on V4.1 with its tables in host memory, the Engram prefetch's
	// start and finish (nvidia/engram.py:464-492).
	// Not under dual-batch overlap: vLLM leaves the breakable wrapper off when it micro-batches
	// (vllm/v1/worker/gpu_model_runner.py:5396-5407).
	dbo := k.pool.Engine.DBO != nil && k.pool.Engine.DBO.Enabled
	if a := g.Global.Architecture; !dbo && (a == rules.ArchDeepseekV41 || a == rules.ArchDeepseekV4) {
		for _, l := range k.plan.Layers {
			if l.AttnQHeads > 0 {
				k.captureBreaks += l.Count
			}
			for _, gt := range l.Gathers {
				if gt.Control == model.ResidencyEngramCPUOffload && k.engramOnHost {
					// The prefetch's start and its finish.
					k.captureBreaks += 2 * l.Count
				}
			}
		}
	}
	return nil
}

// assumeEngram records how an engram table's placement was settled where the deployment did
// not settle it.
func (k *Kernel) assumeEngram() {
	ec := k.pool.Engine.EngramConfig
	if ec == nil || ec.CPUOffload == nil {
		where := "HBM"
		if k.engramOnHost {
			where = "host memory"
		}
		k.assume("engram_cpu_offload", strconv.FormatBool(k.engramOnHost),
			"unstated, so the release's EngramConfig.cpu_offload default places the Engram "+
				"tables in "+where+" (vllm/config/engram.py)")
	}
	if (ec == nil || ec.DPSharedMemory == nil) && k.layout.DP > 1 {
		k.assume("engram_dp_shared_memory", strconv.Itoa(k.engramShards)+" shards",
			"unstated, so it resolves as the engine resolves it: on when cpu_offload is on, "+
				"use_thp off, dp above one and elastic EP off (resolve_dp_shared_memory, "+
				"vllm/config/engram.py:92-97); the kernel cannot see "+
				"whether /dev/shm holds the tables, where the engine falls back to splitting them")
	}
}

// resolveEngram settles where an engram_cpu_offload table lives and how many ways its heads
// split (blis-schemas deployment.EngramConfig). cpu_offload unstated is the release's
// default; tables in host memory are shared by a node's data-parallel replicas where
// dp_shared_memory resolves on (the engine's default when cpu_offload is on, use_thp is off
// and dp exceeds one), and otherwise split across them too, tp x local-dp ways.
func (k *Kernel) resolveEngram(r resolve.EngineRules) {
	k.engramOnHost = true
	if pack, ok := r.(*rules.Pack); ok && pack != nil {
		k.engramOnHost = pack.EngramCPUOffloadDefault
	}
	ec := k.pool.Engine.EngramConfig
	if ec != nil && ec.CPUOffload != nil {
		k.engramOnHost = *ec.CPUOffload
	}
	dp := max(k.layout.DP, 1)
	shared := k.engramOnHost && dp > 1
	if ec != nil && ec.UseTHP != nil && *ec.UseTHP {
		shared = false
	}
	if ec != nil && ec.DPSharedMemory != nil {
		shared = *ec.DPSharedMemory
	}
	local := k.pool.Parallel.DPLocal
	if local <= 0 {
		local = dp
	}
	k.engramShards = max(k.layout.TP, 1)
	if !shared && local > 1 {
		k.engramShards *= local
	}
}

// gatherResidency is where a table resolves to.
func (k *Kernel) gatherResidency(g price.PlannedGather) model.Residency {
	if k.gatherOnHost(g) {
		return model.ResidencyHost
	}
	return model.ResidencyDevice
}
