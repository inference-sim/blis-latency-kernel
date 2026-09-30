package resolve

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"
)

// Layout is one pool's resolved parallelism and the engine decisions that follow from it.
//
// Resolution is where a requested setting becomes the setting that will run. A scenario
// states what a deployment asked for; the layout may not permit it. Recording both, and
// the reason, is what lets a prediction be read without re-deriving this logic.
type Layout struct {
	TP, PP, DP  int
	PCP, DCP    int
	ExpertWidth int

	GPUsPerNode int
	GPUsPerRack int
	// NodesSpanned is how many nodes the widest collective group covers. It selects
	// which coefficients apply and whether a cross-node penalty is charged at all.
	NodesSpanned int

	// SequenceParallelMoE is whether the engine makes the MoE input sequence-parallel.
	// When it does, a layer's tensor-parallel reduction is replaced by a reduce-scatter
	// and all-gather pair, so the two are alternatives rather than both running.
	SequenceParallelMoE bool

	// CustomAllReduce is whether the SM-consuming reduction kernel will run. A request
	// for it is declined when the group spans nodes without multi-node NVLink, or when
	// the rank count is one the kernel does not support — in which case the cost moves
	// from SMs to the NIC.
	CustomAllReduce bool

	// Overrides records each request the layout could not honour.
	Overrides []Override
}

// Override is one declined request, with the reason.
type Override struct {
	Field     string
	Requested string
	Resolved  string
	Reason    string
}

// Fabric is the hardware facts a layout needs. They come from the catalog rather than from
// an engine rules pack, because whether a rack is one NVLink domain is a property of the
// part and not of the software running on it.
type Fabric struct {
	// RackIsOneDomain is true where multi-node NVLink spans the rack, so a group inside
	// it crosses no boundary a collective must pay for.
	RackIsOneDomain bool
	// IntraNodeBwGBps and InterNodeBwGBps give the ratio a cross-node collective is
	// scaled by. The ratio belongs to the (chip, fabric) pairing rather than to either
	// alone, which is why both arrive together.
	IntraNodeBwGBps float64
	InterNodeBwGBps float64
}

// Ratio returns the intra-to-inter bandwidth ratio a collective's span is scaled by. One
// when no fabric is stated, which is the single-node case: nothing crosses, so nothing is
// scaled.
func (f Fabric) Ratio() float64 {
	if f.InterNodeBwGBps <= 0 || f.IntraNodeBwGBps <= 0 {
		return 1
	}
	return f.IntraNodeBwGBps / f.InterNodeBwGBps
}

// ResolveLayout derives a pool's layout from the scenario, the fabric it runs on, and the
// engine rules for its declared version.
func ResolveLayout(s *scenario.Scenario, pool scenario.Pool, fab Fabric,
	rules EngineRules) (Layout, error) {
	pl := pool.Parallel
	if pl.TP < 1 || pl.DP < 1 {
		return Layout{}, fmt.Errorf("tensor- and data-parallel widths must be at least 1")
	}
	l := Layout{
		TP: pl.TP, PP: pl.PP, DP: pl.DP, PCP: pl.PCP, DCP: pl.DCP,
		ExpertWidth: pl.ExpertParallelWidth(),
		GPUsPerNode: s.Cluster.GPUsPerNode,
		GPUsPerRack: s.Cluster.GPUsPerRack,
	}

	// The widest group decides how many nodes a collective spans. Expert parallelism is
	// usually the widest; tensor parallelism spans nodes only where it exceeds a node.
	widest := l.TP
	if l.ExpertWidth > widest {
		widest = l.ExpertWidth
	}
	if l.GPUsPerNode > 0 {
		l.NodesSpanned = (widest + l.GPUsPerNode - 1) / l.GPUsPerNode
	} else {
		l.NodesSpanned = 1
	}
	// Multi-node NVLink makes a rack one domain, so a group inside it spans no node
	// boundary a collective has to pay for.
	if l.GPUsPerRack > 0 && widest <= l.GPUsPerRack && fab.RackIsOneDomain {
		l.NodesSpanned = 1
	}

	l.SequenceParallelMoE = rules.SequenceParallelMoE(
		pool.Engine.All2AllBackend, pl.EnableExpertParallel, pl.TP, pl.DP)
	l.CustomAllReduce, l.Overrides = resolveCustomAllReduce(pool, l, fab, rules)
	return l, nil
}

// resolveCustomAllReduce decides whether the SM-consuming reduction kernel runs, and
// records why when a request for it is declined.
func resolveCustomAllReduce(pool scenario.Pool, l Layout, fab Fabric,
	rules EngineRules) (bool, []Override) {
	e := pool.Engine
	requested := true
	if e.DisableCustomAllReduce != nil {
		requested = !*e.DisableCustomAllReduce
	}
	if e.AllReduceBackend == "nccl" {
		requested = false
	}
	if !requested {
		return false, nil
	}

	var overrides []Override
	if !rules.CustomAllReduceSupportsWidth(l.TP) {
		return false, append(overrides, Override{
			Field: "allreduce_backend", Requested: "custom", Resolved: "nccl",
			Reason: fmt.Sprintf(
				"the kernel supports only certain rank counts and %d is not one, so the "+
					"reduction moves from SMs to the NIC", l.TP),
		})
	}
	// Crossing a node boundary disqualifies the kernel unless every rank has multi-node
	// NVLink. The catalog records whether a part does, through the rack tier.
	if l.GPUsPerNode > 0 && l.TP > l.GPUsPerNode && !fab.RackIsOneDomain {
		return false, append(overrides, Override{
			Field: "allreduce_backend", Requested: "custom", Resolved: "nccl",
			Reason: fmt.Sprintf(
				"a tensor-parallel width of %d spans more than the %d GPUs in a node and "+
					"the part has no multi-node NVLink, so the reduction moves from SMs to the NIC",
				l.TP, l.GPUsPerNode),
		})
	}
	return true, overrides
}

// Emits reports whether a conditional node is part of the graph under this layout.
//
// This is the whole of the condition evaluation, and it is a switch rather than an
// expression evaluator. The graph names which question to ask; the layout answers it. A
// condition the graph could state and this function could not answer is impossible,
// because the schema's enum and this switch are the same closed set.
func (l Layout) Emits(c model.EmitCondition) bool {
	switch c {
	case model.EmitAlways:
		return true
	case model.EmitTensorParallel:
		return l.TP > 1
	case model.EmitExpertParallel:
		return l.ExpertWidth > 1
	case model.EmitTensorParallelUnlessSequenceParallelMoE:
		return l.TP > 1 && !l.SequenceParallelMoE
	}
	// An unrecognized condition is not treated as false. Silently dropping a node would
	// remove a cost with nothing reporting it; a caller sees the graph failed validation.
	return false
}

// Recognizes reports whether Emits can answer a condition, so a resolver can reject a
// graph rather than price it with a node quietly missing.
func Recognizes(c model.EmitCondition) bool {
	switch c {
	case model.EmitAlways, model.EmitTensorParallel, model.EmitExpertParallel,
		model.EmitTensorParallelUnlessSequenceParallelMoE:
		return true
	}
	return false
}
