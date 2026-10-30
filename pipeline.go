package latencykernel

import (
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// Pipeline parallelism: a pp > 1 pool runs the model as pp stages on disjoint ranks, each
// stage tp ranks wide holding a contiguous share of the decoder layers, and hands each
// stage's output to the next over whatever link joins them.
//
// What a stage holds is the engine's split. vLLM gives each of pp ranks
// num_hidden_layers // pp layers and spreads the remainder over all but the last stage,
// counting from the second-to-last backwards (get_pp_indices,
// vllm/distributed/utils.py:126-169@5df63a72); the embedding is on the first stage and the
// final norm, LM head and sampling on the last (PPMissingLayer elsewhere,
// vllm/model_executor/models/utils.py:895-919@5df63a72). The drafter runs on the last
// stage, which holds the sampled tokens.
//
// What crosses a boundary is the model's intermediate tensors, sent from one stage's TP
// rank to the same rank of the next with each rank sending only its 1/tp slice
// (send_tensor_dict with all_gather_group=tp, vllm/distributed/parallel_state.py and
// gpu_worker.py:1306-1360@5df63a72): per token, the graph's
// pipeline_boundary_bytes_per_token (one hidden-width activation when unstated) over tp
// ranks in parallel. A boundary inside one node crosses NVLink; one between nodes the
// fabric's NIC, at its effective rate.
//
// How the stages compose into what a simulator sees is the contract's (blis-schemas
// kernel.StepEstimate.Stages, engine.Pipelined): each stage is priced as a step of its own
// over its layers, and the step is their sum. A stage's per-step host cost is its own --
// every stage is a separate worker process launching its own kernels -- and the scheduler's
// is the first stage's, where the engine core runs.
//
// What is NOT priced: the engine's PP-specific rank placement of a TP group spanning nodes
// (a TP group stays within its stage), and a stage-level split of DP or context
// parallelism, which vLLM does not combine with PP for these models.

// pipelineStage is one stage's share of the model.
type pipelineStage struct {
	// layers are the plan's layer kinds with each count the stage's share; a kind the stage
	// holds none of is absent.
	layers []price.PlannedLayer
	fits   [][]gemmFit
	kv     []layerKV
	// first and last say whether the stage holds the embedding, or the head and drafter.
	first, last bool
	// kernels is the stage's own launch count: its layers' and, on the end stages, the
	// embedding's and head's.
	kernels, prefillOnlyKernels int
	// totalLayers is how many decoder layers the stage holds.
	totalLayers int
	// sendSeconds prices one token's boundary send to the next stage (zero on the last).
	sendBytesPerToken float64
	crossesNode       bool
}

// ppIndices is get_pp_indices (vllm/distributed/utils.py:126-169@5df63a72) without
// VLLM_PP_LAYER_PARTITION, which a deployment cannot state: each stage's [start, end).
func ppIndices(layers, pp int) [][2]int {
	parts := make([]int, pp)
	for i := range parts {
		parts[i] = layers / pp
	}
	if rem := layers % pp; rem > 0 {
		for i := 2; i < rem+2; i++ {
			parts[pp-i]++
		}
	}
	out := make([][2]int, pp)
	start := 0
	for i, n := range parts {
		out[i] = [2]int{start, start + n}
		start += n
	}
	return out
}

// liftPipeline splits the plan into stages, once. Called after lift, so every layer kind's
// fits and KV are resolved.
func (k *Kernel) liftPipeline(g *model.Graph) error {
	pp := max(k.layout.PP, 1)
	if pp <= 1 {
		return nil
	}
	// A cache a graph states is sized per layer kind over the whole stack (cacheTerms), not
	// per stage, so a pipeline split of it would charge each rank every stage's cache.
	// Refused rather than overstated; no model with stated caches serves under PP here.
	if len(k.caches) > 0 {
		return fmt.Errorf("pipeline parallelism (pp %d) on a model whose graph states its "+
			"caches is not priced: the stated caches are sized over the whole stack, not "+
			"per stage", pp)
	}
	seq := g.Stack.Expand()
	if len(seq) < pp {
		return fmt.Errorf("pipeline parallelism (pp %d) over %d decoder layers leaves a stage "+
			"with none, which vLLM refuses", pp, len(seq))
	}
	boundary := float64(g.Global.PipelineBoundaryBytesPerToken)
	if boundary <= 0 {
		boundary = float64(g.Global.HiddenSize) * price.ActivationBytes
	}
	// The stage's ranks are pp-major consecutive blocks of tp (rank = pp_rank x tp + tp_rank
	// within a DP rank, parallel_state.py:2045-2060@5df63a72), so two stages share a node when
	// both blocks fit in one.
	gpn := k.layout.GPUsPerNode
	stageNode := func(s int) int {
		if gpn <= 0 {
			return 0
		}
		return s * max(k.layout.TP, 1) / gpn
	}
	k.stages = make([]pipelineStage, pp)
	for s, span := range ppIndices(len(seq), pp) {
		counts := map[string]int{}
		for _, id := range seq[span[0]:span[1]] {
			counts[id]++
		}
		st := &k.stages[s]
		st.first, st.last = s == 0, s == pp-1
		st.totalLayers = span[1] - span[0]
		for i, l := range k.plan.Layers {
			n := counts[l.ID]
			if n == 0 {
				continue
			}
			l.Count = n
			st.layers = append(st.layers, l)
			st.fits = append(st.fits, k.layerFits[i])
			st.kv = append(st.kv, k.layerKV[i])
			per := l.Kernels
			st.kernels += per * n
			st.prefillOnlyKernels += l.PrefillOnlyKernels * n
		}
		if st.first && k.plan.Embedding.Count > 0 {
			st.kernels += k.plan.Embedding.Kernels
			st.prefillOnlyKernels += k.plan.Embedding.PrefillOnlyKernels
		}
		if st.last {
			st.kernels += k.plan.Head.Kernels
			st.prefillOnlyKernels += k.plan.Head.PrefillOnlyKernels
		} else {
			st.sendBytesPerToken = boundary / float64(max(k.layout.TP, 1))
			st.crossesNode = gpn > 0 && stageNode(s) != stageNode(s+1)
		}
	}
	k.liftPipelineMemory()
	k.assume("pipeline_stages", strconv.Itoa(pp),
		"each stage is priced as its own step over its share of the layers (get_pp_indices, "+
			"vllm/distributed/utils.py:126-169@5df63a72), the embedding on the first, the head "+
			"and drafter on the last, plus its boundary send; the stage's per-rank memory is "+
			"the most loaded stage's")
	return nil
}

// pipelineSendSeconds prices one stage's boundary send for a step of tokens: the slice
// each TP rank sends to its peer in the next stage, all tp in parallel over their own
// links, floor plus bytes over the link's rate. The floor is the point-to-point
// collective floor the registry measures (the all-to-all's, a pair exchange).
func (k *Kernel) pipelineSendSeconds(st *pipelineStage, tokens float64) float64 {
	if st.sendBytesPerToken <= 0 || tokens <= 0 {
		return 0
	}
	rate := k.nvlinkBytesPerSecond
	if st.crossesNode && (!k.fabric.RackIsOneDomain || !k.sameRack(st)) {
		rate = k.nicBytesPerSecond
	}
	floor := k.collectiveFloors[collKey{Op: model.OpAll2All, Group: price.GroupExpert}].Seconds()
	return floor + tokens*st.sendBytesPerToken/rate
}

// sameRack reports whether a stage and the next sit in one NVLink rack domain.
func (k *Kernel) sameRack(st *pipelineStage) bool {
	rack := k.layout.GPUsPerRack
	if rack <= 0 {
		rack = k.chip.GPUsPerRack
	}
	if rack <= 0 {
		return false
	}
	pp := max(k.layout.PP, 1)
	return pp*max(k.layout.TP, 1) <= rack
}

// stagedStepTime prices a pipeline-parallel step: each stage over its layers, with its own
// host cost and boundary send, and the step as their sum (StepEstimate.Stages).
func (k *Kernel) stagedStepTime(b kernel.Batch,
	into map[kernel.Resource]time.Duration) kernel.StepEstimate {
	s := k.shapeOf(b)
	var total [numResources]float64
	stages := make([]kernel.StageEstimate, len(k.stages))
	var overlap, noOverlap time.Duration
	for i := range k.stages {
		st := &k.stages[i]
		var per [numResources]float64
		o := k.priceLayers(st.layers, st.fits, st.kv, &s, &per)
		if st.first && k.plan.Embedding.Count > 0 {
			o += k.priceLayers([]price.PlannedLayer{k.plan.Embedding}, nil, []layerKV{{}}, &s, &per)
		}
		draftKernels := 0
		if st.last {
			o += k.priceHead(k.plan.Head, &s, &per)
			if k.speculating {
				var draft float64
				draft, draftKernels = k.priceDraft(b, &s, &per)
				o += draft
			}
		}
		send := k.pipelineSendSeconds(st, s.tokensF)
		if st.crossesNode {
			per[idxNIC] += send
		} else {
			per[idxNVLink] += send
		}
		o += send
		host := k.stageHost(b, st, draftKernels)
		per[idxHost] = host.Seconds()
		stageNo, _ := summarize(&per, nil)
		if st.first {
			if sched := k.schedulerHost(b, stageNo); sched > 0 {
				host += sched
				per[idxHost] = host.Seconds()
				stageNo, _ = summarize(&per, nil)
			}
		}
		stageOv := seconds(o) + host
		if stageOv > stageNo {
			stageOv = stageNo
		}
		stages[i] = kernel.StageEstimate{Overlap: stageOv, NoOverlap: stageNo}
		overlap += stageOv
		noOverlap += stageNo
		for r := range total {
			total[r] += per[r]
		}
	}
	per := into
	if per == nil {
		per = make(map[kernel.Resource]time.Duration, numResources)
	}
	_, bottleneck := summarize(&total, per)
	return kernel.StepEstimate{Overlap: overlap, NoOverlap: noOverlap, Bottleneck: bottleneck,
		PerResource: per, Stages: stages}
}

// stageHost is one stage's launch-and-replay cost, which every stage's worker pays for its
// own kernels: the target's host form (targetHost, launchSeconds) over the stage's launches.
// A piecewise-captured stage replays one segment per layer it holds; the drafter, on the
// last stage, is charged as the single-stage step charges it.
func (k *Kernel) stageHost(b kernel.Batch, st *pipelineStage, draftKernels int) time.Duration {
	mode := k.graphOther
	if len(b.Reqs) == 0 || b.UniformDecode(k.uniformDecodeWidth) {
		mode = k.graphDecode
	}
	kernels := st.kernels
	if !stepPrefills(b) {
		kernels -= st.prefillOnlyKernels
	}
	ownGraphs := draftKernels > 0 && k.draftHasOwnGraphs()
	modeKernels := kernels
	if !ownGraphs {
		modeKernels += draftKernels
	}
	replay := func(m graphMode) time.Duration {
		switch m {
		case graphModeFull:
			return k.replayPerStep
		case graphModePiecewise:
			if k.captureBreaks > 0 {
				return time.Duration(float64(k.replayPerStep) * float64(1+k.captureBreaks))
			}
			return time.Duration(float64(k.replayPerStep) * float64(max(st.totalLayers, 1)))
		}
		return time.Duration(float64(k.launchEager) * float64(modeKernels))
	}
	host := replay(mode)
	if b.PaddedTokens() > k.captureTokens && mode != graphModeEager {
		if eager := replay(graphModeEager); eager > host {
			host = eager
		}
	}
	if ownGraphs {
		host += k.draftHost(b, draftKernels)
	}
	return host + seconds(k.launchPerKernel.Seconds()*float64(kernels+draftKernels))
}

// stageResidentBytes is the most loaded stage's weights, which is what a rank of a
// pipeline-parallel pool holds: every rank sizes its KV pool from its own free memory and
// the engine keeps the smallest pool across workers (get_kv_cache_configs step 5,
// vllm/v1/core/kv_cache_utils.py:2530-2560@5df63a72).
func (k *Kernel) stageResidentBytes() float64 {
	worst := 0.0
	for i := range k.stages {
		st := &k.stages[i]
		var b float64
		for _, l := range st.layers {
			b += float64(l.Count) * k.residentBytes(l)
		}
		if st.first && k.plan.Embedding.Count > 0 {
			b += k.residentBytes(k.plan.Embedding)
		}
		if st.last {
			b += k.residentBytes(k.plan.Head)
			b += k.draftResidentBytes()
		}
		worst = math.Max(worst, b)
	}
	return worst
}

// liftPipelineMemory sets what one rank of a pipeline-parallel pool holds: its stage's
// share of the KV cache rather than the whole model's. A rank caches only its own layers'
// KV, every worker sizes its pool from its own free memory, and the engine keeps the
// smallest pool (get_kv_cache_configs, kv_cache_utils.py:2530-2660@5df63a72), so the
// binding rank is the stage that holds the most KV per token. The drafter's caches are the
// last stage's. A P/D handoff still moves every stage's KV, so pdTransferBytes keeps the
// whole model's figure.
func (k *Kernel) liftPipelineMemory() {
	worst, worstStage := -1.0, 0
	for i := range k.stages {
		st := &k.stages[i]
		var per float64
		for j, l := range st.layers {
			per += st.kv[j].perToken * float64(l.Count)
		}
		if st.last && k.speculating {
			per += k.draftKVBytesPerToken
		}
		if per > worst {
			worst, worstStage = per, i
		}
	}
	k.ppKVBytesPerToken, k.ppKVStage = worst, worstStage
}
