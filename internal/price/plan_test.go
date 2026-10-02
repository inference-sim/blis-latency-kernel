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

// blockSparseGraph mirrors MiniMax-M3's sparse layer: an indexer that scores which blocks
// to read, then the bounded attention that reads them. The indexer comes FIRST, which is
// the order it runs in and the order that used to decide the primary by accident.
func blockSparseGraph() *model.Graph {
	const hidden = 6144
	sparse := model.LayerKind{ID: "sparse", Nodes: []model.Node{
		{Op: model.OpElementwise},
		{Op: model.OpGEMM, N: 9216, K: hidden},
		{Op: model.OpGEMM, N: 640, K: hidden, Role: "index_qk_proj"},
		{Op: model.OpAttention, Role: "block_index_scores",
			AttentionKind: model.AttentionGQA,
			NumQHeads:     4, NumKVHeads: 1, HeadDim: 128},
		{Op: model.OpAttention, AttentionKind: model.AttentionSWA,
			NumQHeads: 64, NumKVHeads: 4, HeadDim: 128, Window: 2176},
		{Op: model.OpGEMM, N: hidden, K: 8192},
	}}
	return &model.Graph{
		Global:     model.GlobalShape{HiddenSize: hidden, VocabSize: 200064},
		LayerKinds: []model.LayerKind{sparse},
		Stack:      model.Stack{Pattern: []string{"sparse"}, Repeat: 57},
	}
}

func TestPlanKeepsEveryAttentionKernelALayerLaunches(t *testing.T) {
	// A layer that launches two attention kernels must report two. An earlier plan
	// assigned the attention fields per node, so the second overwrote the first and the
	// step paid for one of them.
	p, err := BuildPlan(blockSparseGraph(), emitNone{}, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	l := p.Layers[0]
	if got := len(l.Attentions); got != 2 {
		t.Fatalf("layer launches 2 attention kernels, plan has %d", got)
	}
}

func TestPrimaryAttentionIsTheOneHoldingKVWhicheverOrderTheyAppearIn(t *testing.T) {
	// Asserted in BOTH orders on purpose. With the indexer first, a plan that takes the
	// last node seen gets the right answer by luck, so that arrangement alone cannot
	// tell a role-aware choice from the overwrite this replaced. With the indexer last,
	// only a role-aware choice still reports the attention that holds KV.
	for _, tc := range []struct {
		name       string
		indexFirst bool
	}{{"indexer first", true}, {"indexer last", false}} {
		t.Run(tc.name, func(t *testing.T) {
			g := blockSparseGraph()
			nodes := g.LayerKinds[0].Nodes
			var idx, main int
			for i, n := range nodes {
				if n.Op != model.OpAttention {
					continue
				}
				if n.Role == "" {
					main = i
				} else {
					idx = i
				}
			}
			if (idx < main) != tc.indexFirst {
				nodes[idx], nodes[main] = nodes[main], nodes[idx]
			}
			p, err := BuildPlan(g, emitNone{}, 2, 4)
			if err != nil {
				t.Fatal(err)
			}
			l := p.Layers[0]
			if l.AttnQHeads != 64 || l.AttnKVHeads != 4 {
				t.Fatalf("primary is %dq/%dkv, want 64q/4kv (the indexer is 4q/1kv)",
					l.AttnQHeads, l.AttnKVHeads)
			}
			if l.AttnKind != model.AttentionSWA || l.AttnWindow != 2176 {
				t.Fatalf("primary should carry the bounded read, got kind=%s window=%d",
					l.AttnKind, l.AttnWindow)
			}
		})
	}
}

func TestIndexerIsNotCountedAsHoldingKV(t *testing.T) {
	// The scorer keeps its own narrow cache, sized by the model's index projections
	// rather than by num_key_value_heads. Counting it as KV-holding doubled MiniMax-M3's
	// KV-holding layer count and let its geometry set the engine's page size.
	p, err := BuildPlan(blockSparseGraph(), emitNone{}, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	var holders int
	for _, a := range p.Layers[0].Attentions {
		if a.HoldsKV() {
			holders++
		}
	}
	if holders != 1 {
		t.Fatalf("%d attention kernels claim to hold the engine's KV, want 1", holders)
	}
}

func TestSingleAttentionLayerStillReportsItAsPrimary(t *testing.T) {
	// The common case must be unchanged: one attention, reported in both places.
	p, err := BuildPlan(hybridGraph(), emitNone{}, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range p.Layers {
		if len(l.Attentions) == 0 {
			continue
		}
		if len(l.Attentions) != 1 {
			t.Fatalf("layer has %d attentions, want 1", len(l.Attentions))
		}
		a := l.Attentions[0]
		if a.QHeads != l.AttnQHeads || a.KVHeads != l.AttnKVHeads ||
			a.HeadDim != l.AttnHeadDim || a.Kind != l.AttnKind {
			t.Fatalf("the sole attention and the primary fields disagree")
		}
	}
}

// mixedPrecisionMoE mirrors DeepSeek-V4-Pro: routed experts stored at fp4 beside an fp8
// checkpoint. The node-level override is the only way to express that, and getting it
// wrong doubles the expert weights, which is what made the kernel refuse a deployment
// InferenceX actually ran on 8 GPUs.
func mixedPrecisionMoE(expertDType model.DType) *model.Graph {
	const hidden = 7168
	nodes := []model.Node{
		{Op: model.OpGEMM, Role: "qkv_proj", N: 66560, K: hidden},
		{Op: model.OpGroupedGEMM, Role: "experts", N: 3072, K: hidden,
			Experts: 384, TopK: 6, WeightDType: expertDType},
	}
	return &model.Graph{
		Global:     model.GlobalShape{HiddenSize: hidden, VocabSize: 129280},
		LayerKinds: []model.LayerKind{{ID: "moe", Nodes: nodes}},
		Stack:      model.Stack{Pattern: []string{"moe"}, Repeat: 61},
	}
}

func TestANodeDTypeOverridesTheCheckpointWidthForWeightsOnly(t *testing.T) {
	// fp8 globally (2 bytes passed as the checkpoint width here), experts at fp4.
	const global = 1.0
	base, err := BuildPlan(mixedPrecisionMoE(""), emitNone{}, global, 4)
	if err != nil {
		t.Fatal(err)
	}
	over, err := BuildPlan(mixedPrecisionMoE(model.DTypeNVFP4), emitNone{}, global, 4)
	if err != nil {
		t.Fatal(err)
	}
	b, o := base.Layers[0], over.Layers[0]

	// The expert weights halve, because nvfp4 is half of fp8.
	got, want := o.ExpertWeightBytesPerExpert, b.ExpertWeightBytesPerExpert/2
	if got != want {
		t.Errorf("expert weight bytes = %.0f, want %.0f (half the fp8 figure)", got, want)
	}
	// The DENSE weights do not, because the override is on the expert node alone.
	// Asserted against the ABSOLUTE figure rather than only against the other graph: both
	// graphs share the dense node, so a bug that applied a wrong width to it in BOTH would
	// cancel in a comparison and pass. 66560 * 7168 parameters at the 1-byte global width.
	const wantDense = 66560.0 * 7168 * 1
	for name, l := range map[string]PlannedLayer{"without an override": b, "with one": o} {
		if l.DenseWeightBytes != wantDense {
			t.Errorf("%s: dense weight bytes = %.0f, want %.0f at the global width",
				name, l.DenseWeightBytes, wantDense)
		}
	}
	// FLOPs count arithmetic, not bytes, so a storage width cannot change them.
	if o.ExpertFLOPsPerTokenPerExpert != b.ExpertFLOPsPerTokenPerExpert {
		t.Errorf("expert FLOPs moved with a storage width: %.0f vs %.0f",
			o.ExpertFLOPsPerTokenPerExpert, b.ExpertFLOPsPerTokenPerExpert)
	}
}

func TestAnAbsentNodeDTypeLeavesEveryFigureUnchanged(t *testing.T) {
	// The common case must be untouched: every committed graph states no node dtype, so a
	// plan built without one has to match what it produced before the field existed.
	p, err := BuildPlan(hybridGraph(), emitNone{}, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range p.Layers {
		for _, n := range []float64{l.DenseWeightBytes, l.ExpertWeightBytesPerExpert} {
			if n < 0 {
				t.Fatalf("negative weight bytes: %f", n)
			}
		}
	}
	// hybridGraph's expert node is N=5120 over K=8192, three matrices per gated expert,
	// at the 2 bytes per parameter this plan was built with.
	var found bool
	for _, l := range p.Layers {
		if l.ExpertWeightBytesPerExpert > 0 {
			found = true
			if want := 3.0 * 5120 * 8192 * 2; l.ExpertWeightBytesPerExpert != want {
				t.Errorf("expert bytes = %.0f, want %.0f", l.ExpertWeightBytesPerExpert, want)
			}
		}
	}
	if !found {
		t.Fatal("no expert layer in the plan")
	}
}
