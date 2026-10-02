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

// PlannedLayer is one layer kind with its multiplicity and its pre-summed work.
type PlannedLayer struct {
	ID    string
	Count int

	// DenseFLOPsPerToken is the projection and dense-MLP work for one token, summed over
	// every GEMM in the layer. A step multiplies it by the token count.
	DenseFLOPsPerToken float64
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

// PlannedCollective is one surviving collective and what it moves.
type PlannedCollective struct {
	Op model.Op
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

func planNodes(nodes []model.Node, em Emitter, dtypeBytes, stateBytes float64,
	hidden int) (PlannedLayer, error) {
	var pl PlannedLayer
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
		switch n.Op {
		case model.OpGEMM:
			// Two FLOPs per multiply-accumulate, per output element, per token.
			pl.DenseFLOPsPerToken += 2 * float64(n.N) * float64(n.K)
			pl.DenseWeightBytes += float64(n.N) * float64(n.K) * dtypeBytes
		case model.OpGroupedGEMM:
			// A gated expert holds three matrices; the node's N is the inner width.
			perExpert := 3 * float64(n.N) * float64(n.K)
			pl.ExpertFLOPsPerTokenPerExpert += 2 * perExpert
			pl.ExpertWeightBytesPerExpert += perExpert * dtypeBytes
			pl.TopK = n.TopK
			if n.SharedExperts > 0 {
				inner := n.N
				if n.SharedIntermediateSize > 0 {
					inner = n.SharedIntermediateSize
				}
				shared := 3 * float64(inner) * float64(n.K) * float64(n.SharedExperts)
				pl.SharedExpertFLOPsPerToken += 2 * shared
				pl.SharedExpertWeightBytes += shared * dtypeBytes
			}
		case model.OpAttention:
			pl.Attentions = append(pl.Attentions, PlannedAttention{
				QHeads: n.NumQHeads, KVHeads: n.NumKVHeads, HeadDim: n.HeadDim,
				Kind: n.AttentionKind, Window: n.Window, Role: n.Role,
			})
			// The primary is the first attention that reads the engine's KV cache, not
			// the last node seen. An earlier assignment here overwrote, so on a layer
			// whose indexer precedes its attention the primary was whichever came last.
			if pl.AttnQHeads == 0 && n.Role == "" {
				pl.AttnQHeads, pl.AttnKVHeads, pl.AttnHeadDim =
					n.NumQHeads, n.NumKVHeads, n.HeadDim
				pl.AttnKind, pl.AttnWindow = n.AttentionKind, n.Window
			}
		case model.OpRecurrentUpdate:
			pl.RecurrentKind = n.RecurrentKind
			pl.RecurrentStateBytes += recurrentBytes(n, stateBytes)
		case model.OpElementwise:
			b := float64(n.BytesPerToken)
			if b == 0 {
				// A normalization reads and writes the hidden state.
				b = 2 * float64(hidden) * dtypeBytes
			}
			pl.ElementwiseBytesPerToken += b
		case model.OpAllReduce, model.OpAllGather, model.OpReduceScatter:
			pl.Collectives = append(pl.Collectives, PlannedCollective{
				Op: n.Op, BytesPerToken: float64(hidden) * dtypeBytes,
			})
		case model.OpAll2All:
			pl.Collectives = append(pl.Collectives, PlannedCollective{
				Op: n.Op, BytesPerToken: float64(hidden) * dtypeBytes,
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
