package price

import (
	"testing"

	"github.com/inference-sim/blis-schemas/spec/model"
)

// emitAll answers every condition affirmatively, so a plan built with it contains every
// node. Used where a test is about planning rather than about resolution.
type emitAll struct{}

func (emitAll) Emits(model.EmitCondition) bool { return true }
func (emitAll) Recognizes(c model.EmitCondition) bool {
	return c.Valid()
}

// emitNone drops every conditional node, which is the single-rank layout.
type emitNone struct{}

func (emitNone) Emits(c model.EmitCondition) bool { return c == model.EmitAlways }
func (emitNone) Recognizes(c model.EmitCondition) bool {
	return c.Valid()
}

// emitUnknown recognizes nothing, standing in for a resolver that meets a condition it
// cannot evaluate.
type emitUnknown struct{}

func (emitUnknown) Emits(model.EmitCondition) bool      { return false }
func (emitUnknown) Recognizes(model.EmitCondition) bool { return false }

// hybridGraph mirrors the widest model in the catalog: 108 layers over three kinds, which
// is the case that shows why a plan aggregates by kind.
func hybridGraph() *model.Graph {
	const hidden = 8192
	attn := model.LayerKind{ID: "attention", Nodes: []model.Node{
		{Op: model.OpElementwise},
		{Op: model.OpGEMM, N: 10240, K: hidden},
		{Op: model.OpAttention, AttentionKind: model.AttentionGQA,
			NumQHeads: 64, NumKVHeads: 2, HeadDim: 128},
		{Op: model.OpGEMM, N: hidden, K: hidden},
		{Op: model.OpAllReduce, Emit: model.EmitTensorParallel},
	}}
	moe := model.LayerKind{ID: "moe", Nodes: []model.Node{
		{Op: model.OpElementwise},
		{Op: model.OpGroupedGEMM, N: 5120, K: hidden, Experts: 512, TopK: 22,
			SharedExperts: 1, SharedIntermediateSize: 10240},
		{Op: model.OpAll2All, Emit: model.EmitExpertParallel},
	}}
	mamba := model.LayerKind{ID: "mamba", Nodes: []model.Node{
		{Op: model.OpElementwise},
		{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentMamba2,
			NumHeads: 256, HeadDim: 64, StateSize: 128, NumGroups: 8, ConvKernel: 4,
			IntermediateSize: 16384},
		{Op: model.OpAllReduce, Emit: model.EmitTensorParallel},
	}}
	seq := make([]string, 0, 108)
	for len(seq) < 108 {
		seq = append(seq, "mamba", "moe")
		if len(seq) < 108 && len(seq)%14 == 0 {
			seq = append(seq, "attention")
		}
	}
	return &model.Graph{
		Kind: "ModelGraph", Name: "hybrid",
		Global:     model.GlobalShape{HiddenSize: hidden, VocabSize: 131072, WeightDType: model.DTypeBF16},
		LayerKinds: []model.LayerKind{mamba, moe, attn},
		Stack:      model.Stack{Prologue: seq[:108]},
		Head: []model.Node{
			{Op: model.OpElementwise},
			{Op: model.OpGEMM, N: 131072, K: hidden},
		},
	}
}

// TestPlanAggregatesByKindNotByLayer is the property the hot path depends on. A plan's
// size must follow the number of distinct layer kinds, not the layer count, or step time
// scales with model depth for no reason.
func TestPlanAggregatesByKindNotByLayer(t *testing.T) {
	g := hybridGraph()
	p, err := BuildPlan(g, emitAll{}, 2, 4)
	if err != nil {
		t.Fatalf("BuildPlan: %v", err)
	}
	if p.TotalLayers != 108 {
		t.Errorf("TotalLayers = %d, want 108", p.TotalLayers)
	}
	if len(p.Layers) != 3 {
		t.Fatalf("the plan holds %d entries for a 3-kind stack; step time would scale "+
			"with depth", len(p.Layers))
	}
	// The counts must sum to the expanded stack, or work has been lost or duplicated.
	sum := 0
	for _, l := range p.Layers {
		sum += l.Count
	}
	if sum != 108 {
		t.Errorf("layer counts sum to %d, want 108", sum)
	}
}

// TestPlanIsIndependentOfHowTheStackIsSpelled: a literal sequence and an equivalent
// pattern-and-repeat must produce the same plan. The compression in the catalog is a
// storage choice, and a plan that differed between the two forms would make it a
// behavioural one.
func TestPlanIsIndependentOfHowTheStackIsSpelled(t *testing.T) {
	literal := hybridGraph()
	compressed := hybridGraph()
	// Respell the same sequence as a pattern where it happens to be periodic.
	uniform := make([]string, 0, 60)
	for i := 0; i < 60; i++ {
		uniform = append(uniform, "moe")
	}
	literal.Stack = model.Stack{Prologue: uniform}
	compressed.Stack = model.Stack{Pattern: []string{"moe"}, Repeat: 60}

	a, err := BuildPlan(literal, emitAll{}, 2, 4)
	if err != nil {
		t.Fatalf("literal: %v", err)
	}
	b, err := BuildPlan(compressed, emitAll{}, 2, 4)
	if err != nil {
		t.Fatalf("compressed: %v", err)
	}
	if len(a.Layers) != len(b.Layers) || a.TotalLayers != b.TotalLayers {
		t.Fatalf("the two spellings planned differently: %d/%d against %d/%d",
			len(a.Layers), a.TotalLayers, len(b.Layers), b.TotalLayers)
	}
	for i := range a.Layers {
		if a.Layers[i].Count != b.Layers[i].Count ||
			a.Layers[i].DenseFLOPsPerToken != b.Layers[i].DenseFLOPsPerToken {
			t.Errorf("layer %d differs between spellings", i)
		}
	}
}

// TestConditionalNodesAreResolvedAtPlanTime: a plan holds only nodes that will run, so no
// condition is evaluated per step.
func TestConditionalNodesAreResolvedAtPlanTime(t *testing.T) {
	g := hybridGraph()
	with, err := BuildPlan(g, emitAll{}, 2, 4)
	if err != nil {
		t.Fatalf("with collectives: %v", err)
	}
	without, err := BuildPlan(g, emitNone{}, 2, 4)
	if err != nil {
		t.Fatalf("without collectives: %v", err)
	}
	countCollectives := func(p *Plan) int {
		n := 0
		for _, l := range p.Layers {
			n += len(l.Collectives)
		}
		return n
	}
	if countCollectives(with) == 0 {
		t.Fatal("a tensor- and expert-parallel layout planned no collectives")
	}
	if got := countCollectives(without); got != 0 {
		t.Errorf("a single-rank layout planned %d collectives; they should be resolved "+
			"out rather than evaluated per step", got)
	}
	// The non-collective work must be identical: a layout changes which collectives run,
	// not what a matmul costs.
	for i := range with.Layers {
		if with.Layers[i].DenseFLOPsPerToken != without.Layers[i].DenseFLOPsPerToken {
			t.Errorf("layer %d: dense work changed with the layout", i)
		}
	}
}

// TestUnknownConditionIsAnErrorNotADroppedNode: the failure mode worth preventing. A plan
// that quietly skipped a node it could not evaluate would return a step time that looks
// plausible and omits a cost.
func TestUnknownConditionIsAnErrorNotADroppedNode(t *testing.T) {
	g := hybridGraph()
	if _, err := BuildPlan(g, emitUnknown{}, 2, 4); err == nil {
		t.Fatal("a condition the resolver cannot evaluate was silently dropped")
	}
}

func TestPlanRejectsAStackNamingAnUndeclaredKind(t *testing.T) {
	g := hybridGraph()
	g.Stack = model.Stack{Pattern: []string{"nonexistent"}, Repeat: 4}
	if _, err := BuildPlan(g, emitAll{}, 2, 4); err == nil {
		t.Fatal("a stack naming an undeclared layer kind was accepted")
	}
}

// BenchmarkBuildPlan measures construction, which happens once per deployment. It is
// allowed to be slow; the step path is not.
func BenchmarkBuildPlan(b *testing.B) {
	g := hybridGraph()
	for i := 0; i < b.N; i++ {
		if _, err := BuildPlan(g, emitAll{}, 2, 4); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWalkPlan measures the per-step traversal: what a step time pays to visit every
// planned layer. It must not scale with the 108-layer depth, and it must not allocate.
func BenchmarkWalkPlan(b *testing.B) {
	g := hybridGraph()
	p, err := BuildPlan(g, emitAll{}, 2, 4)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	var flops float64
	for i := 0; i < b.N; i++ {
		flops = 0
		for _, l := range p.Layers {
			flops += float64(l.Count) * (l.DenseFLOPsPerToken +
				float64(l.TopK)*l.ExpertFLOPsPerTokenPerExpert +
				l.SharedExpertFLOPsPerToken)
		}
	}
	if flops == 0 {
		b.Fatal("the traversal computed nothing")
	}
}
