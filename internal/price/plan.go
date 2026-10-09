package price

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/spec/model"
)

// A Plan is a model graph collapsed for repeated evaluation.
//
// Step time is the hot path: a discrete-event simulator calls it once per simulated step,
// which is millions of times over a run. So the graph is flattened once, at construction,
// into the smallest form that still prices correctly — and after that a step costs
// arithmetic over a short slice with no map lookups, no string comparisons and no
// allocation.
//
// Three things make that possible.
//
// The stack collapses to a per-kind layer COUNT rather than a layer list. A 108-layer
// hybrid has three distinct kinds, so a step walks three entries and multiplies, not 108.
// Across the twenty-four models in the catalog that is 1436 layers against 38 distinct
// kinds: a 37x reduction in per-step work, and it is why the graph's compression is a
// storage choice rather than a speed one.
//
// Conditional nodes are resolved out. Whether a collective runs depends on the layout,
// which is fixed at construction, so a plan contains only the nodes that will actually
// execute. No condition is evaluated per step.
//
// Shape-dependent work is separated from shape-independent work. Weight bytes and expert
// counts do not vary with the batch, so they are summed once into scalars. Only the terms
// that depend on token counts are recomputed.
type Plan struct {
	// Layers holds one entry per distinct layer kind, with how many times it appears.
	Layers []PlannedLayer
	// Head is the final norm and language-model head, priced once per step rather than
	// per layer.
	Head PlannedLayer
	// TotalLayers is the expanded layer count, kept for reporting rather than for pricing.
	TotalLayers int

	// TotalKernels is the launch count for one step: every layer's kernels times its
	// multiplicity, plus the head's. The per-kernel dispatch cost multiplies it, and that
	// product dominates a single-request decode step.
	TotalKernels int
}

// PlannedGEMM is one dense GEMM's shape and the per-token work it contributes, so a
// shape-aware efficiency can be weighted by the work each shape carries.
type PlannedGEMM struct {
	N, K          int
	FLOPsPerToken float64
	// ShardN is true when tensor parallelism splits the OUTPUT width rather than the
	// reduction, which is what decides the shape a rank actually runs.
	//
	// Derived from the graph rather than assumed, because the two cases shard opposite
	// dimensions and a shape-aware efficiency reads both. A column-parallel GEMM takes
	// the full hidden state and splits its output, so its K equals hidden_size; a
	// row-parallel GEMM consumes a sharded activation and reduces to the full hidden
	// state, so its N equals hidden_size. On minimax-m3 at hidden 6144 that classifies
	// qkv_proj (k=6144) and mlp_gate_up (k=6144) as column-parallel and o_proj (n=6144)
	// and mlp_down (n=6144) as row-parallel, which is what those projections are.
	//
	// Sharding the wrong axis is not a small error: mlp_gate_up is the widest GEMM in
	// the layer, and dividing its K by tp would price a 6144-deep reduction as 768-deep.
	ShardN bool
}

// PlannedLayer is one layer kind with its multiplicity and its pre-summed work.
type PlannedLayer struct {
	ID    string
	Count int

	// DenseFLOPsPerToken is the projection and dense-MLP work for one token, summed over
	// every GEMM in the layer. A step multiplies it by the token count.
	DenseFLOPsPerToken float64
	// DenseGEMMs is every dense GEMM's shape, kept alongside the FLOPs sum because a
	// GEMM's efficiency depends on its own n and k and not only on the step's token
	// count. Measured on AISimulate's vLLM fp8 sweep for H200 at a fixed m of 1024, the
	// median efficiency rises from 0.007 at k=32 to 0.476 at k=51200 -- so a ramp in m
	// alone misprices a narrow GEMM by orders of magnitude, and tensor parallelism is
	// what makes projections narrow. The pre-summed scalar above cannot express that,
	// which is why the shapes survive planning.
	//
	// Sharding is NOT applied here: it depends on tp, which planning does not know.
	// The consumer divides N or K as the parallelism dictates.
	DenseGEMMs []PlannedGEMM
	// DenseWeightBytes is the parameter bytes those GEMMs read, independent of batch.
	DenseWeightBytes float64

	// ExpertFLOPsPerTokenPerExpert is the grouped-GEMM work for one token routed to one
	// expert. A step multiplies by tokens and by top_k.
	ExpertFLOPsPerTokenPerExpert float64
	// ExpertWeightBytesPerExpert is one expert's parameter bytes. At decode every local
	// expert is touched, so a step multiplies by the local expert count.
	ExpertWeightBytesPerExpert float64
	// SharedExpertFLOPsPerToken is dense work every token pays, outside the routing.
	SharedExpertFLOPsPerToken float64
	SharedExpertWeightBytes   float64
	TopK                      int

	// Attention, when the layer has it. Zero heads means it does not.
	//
	// These name the layer's PRIMARY attention: the one that reads the KV cache the
	// engine sizes. A layer may run more than one attention kernel -- a block-sparse
	// layer scores which blocks to read before reading them -- and every one of them is
	// in Attentions. The singular fields are the primary entry, kept because the KV
	// geometry and the prefill term are properties of that one attention rather than of
	// the set.
	AttnQHeads, AttnKVHeads, AttnHeadDim int
	AttnKind                             model.AttentionKind
	AttnWindow                           int

	// AttnIndexTopK and AttnCompressRatio bound a sparse-MLA layer's KV read.
	//
	// They are carried because for this kind the byte count is NOT the context. The
	// kernel reads min(context, topk) tokens at full resolution plus, where the
	// architecture compresses the remainder rather than discarding it,
	// (context - topk) / ratio more. A layer selecting by a sliding window states a
	// window and no topk, in which case the window IS the selected count.
	//
	// Without these the only available byte count is the full context, which on
	// DeepSeek-V4-Pro overstates a 1M-token decode read by 128x. Measured on h200, such a
	// step costs 30.6us against 13.5us at 16K context; the full-context reading cannot
	// produce that curve at any rate below datasheet peak.
	AttnIndexTopK     int
	AttnCompressRatio int

	// AttnKVLoRARank is a latent layer's compressed KV width, zero for any other kind.
	// It is carried for the decode-context combine, whose OUTPUT crosses at this width
	// while its query crosses at the full head width: MLADCPManager is built with
	// query_head_dim = kv_lora_rank + qk_rope_head_dim and output_head_dim = kv_lora_rank
	// (vllm/model_executor/layers/attention/mla_attention.py:697-708 at v0.31.0).
	AttnKVLoRARank int

	// Attentions is every attention kernel this layer launches, in graph order.
	//
	// A layer with one attention has one entry and the singular fields repeat it. The
	// distinction matters because a secondary attention is NOT a smaller copy of the
	// primary: a block-sparse layer's indexer scans the whole context to rank blocks
	// while the attention it feeds reads only the chosen ones, so the two have different
	// head geometry AND different bounds. Collapsing them charges the step once and
	// drops the other -- on MiniMax-M3 at 8K context the dropped indexer read is 0.94x
	// the attention read it selects for, and 15x at 131K.
	Attentions []PlannedAttention

	// Recurrent state, when the layer has it.
	RecurrentKind       model.RecurrentKind
	RecurrentStateBytes float64

	// ElementwiseBytesPerToken is normalization and activation traffic per token.
	ElementwiseBytesPerToken float64

	// Collectives that survived resolution, with the bytes each moves per token.
	Collectives []PlannedCollective

	// Kernels is how many kernel launches this layer makes.
	//
	// Counted from the graph's surviving nodes rather than assumed, because the launch
	// cost is per kernel and a step's total is the count times a per-launch constant. It
	// is a lower bound: the graph records the primitives, and a real layer also launches
	// the epilogues, the router, and the permute and unpermute around a grouped GEMM that
	// the primitives imply. So the count is the primitive count, and the per-launch
	// constant it multiplies is calibrated against measurement with that in mind.
	Kernels int
}

// PlannedAttention is one attention kernel a layer launches.
//
// Window is the per-token read bound in tokens, zero when the read tracks context. A
// secondary attention with no window is the expensive case rather than the cheap one:
// it is the kernel that scans everything.
type PlannedAttention struct {
	QHeads, KVHeads, HeadDim int
	Kind                     model.AttentionKind
	Window                   int
	// CompressRatio is how many tokens share one cached state, zero or one where every
	// token has its own. A pooled indexer caches one state per index_kpool tokens, so
	// its scan covers the span's STATES rather than its tokens. Orthogonal to Window:
	// the window bounds how far back the scan looks, the ratio how many states that
	// span holds.
	CompressRatio int
	// Role is the graph's label for what this kernel does, e.g. "block_index_scores".
	// Empty on a layer's primary attention.
	Role string
}

// HoldsKV reports whether this attention reads the KV cache the engine sizes.
//
// A block-index scorer keeps its own narrow cache, sized by the model's own index
// projections rather than by num_key_value_heads, so it must not contribute to the
// geometry that sizes pages. Decided by role rather than by head count because a
// small head count is not what makes a cache separate.
func (a PlannedAttention) HoldsKV() bool { return a.Role == "" }

// GroupAxis names the parallelism group a collective runs across.
//
// It exists because the primitive alone does not identify a collective's cost. A
// tensor-parallel all-gather and a decode-context-parallel one are the same OP at two
// different widths, and a collective's floor grows with its group: over the nine parts
// this registry carries at both 4 and 8 ranks, the 8-rank floor is 1.35x to 1.59x the
// 4-rank figure for an all-gather and 1.35x to 1.60x for a reduce-scatter, reaching
// 1.02x to 1.89x for an all-reduce. So a consumer that keyed coefficients by op alone
// would price one of two groups against the other's width. The group is what picks the
// width, so it travels with the collective rather than being inferred from the op.
//
// The growth is not uniform across primitives, which is itself the reason to carry the
// axis rather than a single correction factor: an all-to-all's floor barely moves with
// width (1.00x to 1.16x in fp16), since a shuffle's setup does not grow the way a ring's
// does.
type GroupAxis uint8

const (
	// GroupTP is the tensor-parallel group: every ring-shaped collective a model graph
	// emits reduces or gathers across it.
	GroupTP GroupAxis = iota
	// GroupExpert is the expert-parallel group, which a routed MoE dispatch spans.
	GroupExpert
	// GroupDCP is the decode-context-parallel group. Nothing in a model graph names it:
	// DCP is a layout choice, so its collectives are added by the pricer rather than
	// planned, and this axis exists to give them their own width.
	GroupDCP
	// GroupPCP is the prefill-context-parallel group, which gathers the KV a split
	// prefill wrote. Separate from GroupDCP because the two widths are independent and
	// the engine builds them as separate process groups.
	GroupPCP
)

// PlannedCollective is one surviving collective and what it moves.
type PlannedCollective struct {
	Op model.Op
	// Group is the parallelism axis this collective spans, which selects its width.
	Group GroupAxis
	// BytesPerToken is the payload one token contributes. For a routed dispatch this is
	// multiplied by top_k at pricing time; for a ring-shaped one it is not, which is the
	// distinction the backend decides.
	BytesPerToken float64
	RoutedByTopK  bool
}

// Emitter decides whether a conditional node survives. The resolver implements it; the
// plan takes it as a parameter so this package does not depend on the resolver.
type Emitter interface {
	Emits(model.EmitCondition) bool
	Recognizes(model.EmitCondition) bool
}

// BuildPlan flattens a graph under a layout.
//
// Any condition the emitter does not recognize is an error rather than a dropped node:
// silently omitting a collective would remove a cost with nothing reporting it, and the
// resulting step time would look plausible.
// ActivationBytes is the width an activation crosses a collective or a normalization at:
// the model's compute dtype, bf16, whatever format the weights are served in. vLLM's
// quantized linears quantize their input transiently and return out_dtype=x.dtype
// (vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:147,153 at v0.31.0), so the
// hidden state between projections, the reductions over it and the norms that read it are
// 16-bit on an fp8, int8 or fp4 deployment as on a bf16 one. The MoE dispatch is priced at
// the same width: the default allgather_reducescatter backend moves hidden states
// unquantized (vllm/config/parallel.py:202); a backend that quantizes its dispatch moves
// fewer bytes, which this does not see.
//
// An earlier form sized all three at the served weight width, so an fp8 deployment's
// collectives and norms were charged half their bytes.
const ActivationBytes = 2.0

func BuildPlan(g *model.Graph, em Emitter, dtypeBytes, stateDtypeBytes float64) (*Plan, error) {
	if g == nil {
		return nil, fmt.Errorf("no graph to plan")
	}
	counts := map[string]int{}
	order := []string{}
	for _, id := range g.Stack.Expand() {
		if _, seen := counts[id]; !seen {
			order = append(order, id)
		}
		counts[id]++
	}

	byID := make(map[string]model.LayerKind, len(g.LayerKinds))
	for _, lk := range g.LayerKinds {
		byID[lk.ID] = lk
	}

	p := &Plan{TotalLayers: g.Stack.Layers()}
	for _, id := range order {
		lk, ok := byID[id]
		if !ok {
			return nil, fmt.Errorf("the stack names layer kind %q, which the graph does not declare", id)
		}
		pl, err := planLayer(lk, em, dtypeBytes, stateDtypeBytes, g.Global.HiddenSize)
		if err != nil {
			return nil, fmt.Errorf("layer kind %q: %w", id, err)
		}
		pl.ID = id
		pl.Count = counts[id]
		p.TotalKernels += pl.Kernels * pl.Count
		p.Layers = append(p.Layers, pl)
	}

	head, err := planNodes(g.Head, em, dtypeBytes, stateDtypeBytes, g.Global.HiddenSize)
	if err != nil {
		return nil, fmt.Errorf("head: %w", err)
	}
	head.ID = "head"
	head.Count = 1
	p.Head = head
	p.TotalKernels += head.Kernels
	return p, nil
}

func planLayer(lk model.LayerKind, em Emitter, dtypeBytes, stateBytes float64,
	hidden int) (PlannedLayer, error) {
	return planNodes(lk.Nodes, em, dtypeBytes, stateBytes, hidden)
}

// kernelsPerPrimitive is how many launches one graph primitive implies.
//
// The graph records primitives, not kernels. A GEMM is one kernel; a grouped GEMM is
// several, because the tokens must be permuted into expert order, the experts run, and the
// result unpermuted. Attention on a paged cache launches its own reshape alongside the
// attention kernel itself.
//
// These multipliers are the one place a launch count is asserted rather than counted, and
// they are conservative: a real implementation may fuse some of them. Under-counting makes
// the launch term optimistic, which the calibrated per-launch constant then partly absorbs
// — so the split between count and constant is less certain than their product.
var kernelsPerPrimitive = map[model.Op]int{
	model.OpGEMM:            1,
	model.OpGroupedGEMM:     4, // permute, group offsets, the grouped GEMM, unpermute
	model.OpAttention:       2, // the attention kernel and its cache reshape
	model.OpElementwise:     1,
	model.OpRecurrentUpdate: 2,
	model.OpAllReduce:       1,
	model.OpAllGather:       1,
	model.OpReduceScatter:   1,
	model.OpAll2All:         2, // dispatch and combine are separate launches
}

// weightBytes is the width this node's parameters are stored at: its own override where
// it states one, the checkpoint's otherwise. Mixed-precision MoE is the case that needs
// it -- DeepSeek-V4-Pro stores routed experts at fp4 beside fp8 everywhere else, and
// pricing them at the global width doubles 720 GiB of experts to 1,441 GiB.
func weightBytes(n model.Node, global float64) float64 {
	if n.WeightDType != "" {
		if b := n.WeightDType.Bytes(); b > 0 {
			return b
		}
	}
	return global
}

func planNodes(nodes []model.Node, em Emitter, dtypeBytes, stateBytes float64,
	hidden int) (PlannedLayer, error) {
	var pl PlannedLayer
	// fusable is the index in DenseGEMMs of the GEMM emitted by the IMMEDIATELY preceding
	// node, or -1 when the previous node was anything else. Fusion needs adjacency in the
	// node sequence, not in DenseGEMMs: minimax-m3's layer-0 has qkv_proj and mlp_gate_up
	// both column-parallel on the same K with the attention op between them, and vLLM
	// cannot fuse across that.
	fusable := -1
	for _, n := range nodes {
		if n.Emit != model.EmitAlways {
			if !em.Recognizes(n.Emit) {
				return pl, fmt.Errorf(
					"node %s carries condition %q, which this resolver cannot evaluate; "+
						"pricing it would either drop a cost or invent one",
					n.Op, n.Emit)
			}
			if !em.Emits(n.Emit) {
				continue
			}
		}
		// Every surviving node launches at least one kernel. Counted here, once, so a
		// conditional node that was resolved away costs no launch either.
		if launches, ok := kernelsPerPrimitive[n.Op]; ok {
			pl.Kernels += launches
		} else {
			pl.Kernels++
		}
		// Any node that is not a dense GEMM breaks a fusion chain: the engine fuses
		// projections of one hidden state, and an op between them consumes that state.
		// Set before the switch so every other case resets it without having to say so.
		wasFusable := fusable
		fusable = -1

		switch n.Op {
		case model.OpGEMM:
			// Two FLOPs per multiply-accumulate, per output element, per token.
			f := 2 * float64(n.N) * float64(n.K)
			pl.DenseFLOPsPerToken += f
			// hidden == 0 means the caller did not supply it; then neither test fires and
			// the consumer shards K, which is the pre-existing behaviour.
			g := PlannedGEMM{
				N: n.N, K: n.K, FLOPsPerToken: f,
				ShardN: hidden > 0 && n.K == hidden && n.N != hidden,
			}
			// Fuse into the previous GEMM when the engine would. vLLM concatenates
			// column-parallel projections that share an input ALONG THE OUTPUT
			// DIMENSION -- QKVParallelLinear and MergedColumnParallelLinear, both in
			// vllm/model_executor/layers/linear.py -- so three projections of the same
			// hidden state are one matmul with the output widths summed, not three
			// matmuls.
			//
			// The condition is CONSECUTIVE, column-parallel, and the same K. Consecutive
			// matters: minimax-m3's layer-1 has qkv_proj (n=9216) immediately followed by
			// index_qk_proj (n=640) on the same k=6144, which vLLM fuses; its layer-0 has
			// qkv_proj and mlp_gate_up both column-parallel on k=6144 but separated by
			// the attention op, which it cannot.
			//
			// This matters because efficiency is strongly non-linear in output width. On
			// AISimulate's vLLM fp8 sweep for H200 at 256 rows, measured efficiency is
			// 0.0008 at n=16 and 0.0138 at n=256, so a 640-wide projection priced as its
			// own kernel costs far more than the same work folded into a 9216-wide one --
			// and sharded over 8 ranks that 640 becomes 80.
			merged := false
			if wasFusable >= 0 {
				prev := &pl.DenseGEMMs[wasFusable]
				if prev.ShardN && g.ShardN && prev.K == g.K {
					prev.N += g.N
					prev.FLOPsPerToken += g.FLOPsPerToken
					fusable = wasFusable
					merged = true
				}
			}
			if !merged {
				pl.DenseGEMMs = append(pl.DenseGEMMs, g)
				fusable = len(pl.DenseGEMMs) - 1
			}
			// Weight bytes are per node whether or not the launches fuse: fusing changes
			// how many kernels read the parameters, not how many parameters there are.
			pl.DenseWeightBytes += float64(n.N) * float64(n.K) * weightBytes(n, dtypeBytes)
		case model.OpGroupedGEMM:
			// A gated expert holds three matrices; the node's N is the inner width.
			perExpert := 3 * float64(n.N) * float64(n.K)
			pl.ExpertFLOPsPerTokenPerExpert += 2 * perExpert
			// Only the WEIGHT terms take a node's dtype override: activations and
			// collectives cross at the served width whatever the weights are stored as.
			ew := weightBytes(n, dtypeBytes)
			pl.ExpertWeightBytesPerExpert += perExpert * ew
			pl.TopK = n.TopK
			if n.SharedExperts > 0 {
				inner := n.N
				if n.SharedIntermediateSize > 0 {
					inner = n.SharedIntermediateSize
				}
				shared := 3 * float64(inner) * float64(n.K) * float64(n.SharedExperts)
				pl.SharedExpertFLOPsPerToken += 2 * shared
				pl.SharedExpertWeightBytes += shared * ew
			}
		case model.OpAttention:
			pl.Attentions = append(pl.Attentions, PlannedAttention{
				QHeads: n.NumQHeads, KVHeads: n.NumKVHeads, HeadDim: n.HeadDim,
				Kind: n.AttentionKind, Window: n.Window, Role: n.Role,
				CompressRatio: n.CompressRatio,
			})
			// The primary is the first attention that reads the engine's KV cache, not
			// the last node seen. An earlier assignment here overwrote, so on a layer
			// whose indexer precedes its attention the primary was whichever came last.
			if pl.AttnQHeads == 0 && n.Role == "" {
				pl.AttnQHeads, pl.AttnKVHeads, pl.AttnHeadDim =
					n.NumQHeads, n.NumKVHeads, n.HeadDim
				pl.AttnKind, pl.AttnWindow = n.AttentionKind, n.Window
				pl.AttnIndexTopK, pl.AttnCompressRatio = n.IndexTopK, n.CompressRatio
				pl.AttnKVLoRARank = n.KVLoRARank
			}
		case model.OpRecurrentUpdate:
			pl.RecurrentKind = n.RecurrentKind
			pl.RecurrentStateBytes += recurrentBytes(n, stateBytes)
		case model.OpElementwise:
			b := float64(n.BytesPerToken)
			if b == 0 {
				// A normalization reads and writes the hidden state, at the activation
				// width.
				b = 2 * float64(hidden) * ActivationBytes
			}
			pl.ElementwiseBytesPerToken += b
		case model.OpAllReduce, model.OpAllGather, model.OpReduceScatter:
			// The hidden state a projection produced, at the activation width.
			pl.Collectives = append(pl.Collectives, PlannedCollective{
				Op: n.Op, Group: GroupTP, BytesPerToken: float64(hidden) * ActivationBytes,
			})
		case model.OpAll2All:
			pl.Collectives = append(pl.Collectives, PlannedCollective{
				Op: n.Op, Group: GroupExpert, BytesPerToken: float64(hidden) * ActivationBytes,
				// Whether top_k multiplies this is a backend property, set by the caller
				// that knows the resolved backend.
				RoutedByTopK: true,
			})
		}
	}
	return pl, nil
}

func recurrentBytes(n model.Node, stateBytes float64) float64 {
	switch n.RecurrentKind {
	case model.RecurrentMamba2:
		// A convolutional state and a temporal state, sized independently.
		convDim := n.IntermediateSize + 2*n.NumGroups*n.StateSize
		conv := float64(convDim) * float64(maxInt(n.ConvKernel-1, 1))
		temporal := float64(n.NumHeads) * float64(n.HeadDim) * float64(n.StateSize)
		if n.HeadDim == 0 {
			// Where the graph states no per-head width, the state is square in its size.
			temporal = float64(n.NumHeads) * float64(n.StateSize) * float64(n.StateSize)
		}
		return (conv + temporal) * stateBytes
	case model.RecurrentKDA, model.RecurrentGDN:
		// A matrix-valued state per head.
		return float64(n.NumHeads) * float64(n.StateSize) * float64(n.StateSize) * stateBytes
	}
	return 0
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
