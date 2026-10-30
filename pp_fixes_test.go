package latencykernel

import (
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// A piecewise-captured stage replays its own layers' segments: one per layer plus one per
// sparse-MLA indexer it holds, plus one. Two stages of one model, one holding the sparse
// layers, replay different counts, and neither its bare layer count.
func TestAPiecewiseStageReplaysItsOwnSegments(t *testing.T) {
	k := &Kernel{graphDecode: graphModePiecewise, graphOther: graphModePiecewise,
		replayPerStep: time.Microsecond, captureTokens: 512}
	dense := price.PlannedLayer{Count: 4, AttnKind: model.AttentionGQA, AttnQHeads: 8,
		Attentions: []price.PlannedAttention{{}}}
	sparse := price.PlannedLayer{Count: 3, AttnKind: model.AttentionSparseMLA, AttnQHeads: 8,
		Attentions: []price.PlannedAttention{{}, {Role: "block_index_scores"}}}
	b := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 1, Computed: 9, PromptLen: 10}}}
	for _, c := range []struct {
		what string
		st   pipelineStage
		want time.Duration
	}{
		{"dense stage", pipelineStage{layers: []price.PlannedLayer{dense}, totalLayers: 4}, 5 * time.Microsecond},
		{"sparse stage", pipelineStage{layers: []price.PlannedLayer{sparse}, totalLayers: 3}, 7 * time.Microsecond},
	} {
		if got := k.stageHost(b, &c.st, 0); got != c.want {
			t.Errorf("%s: host %v, want %v", c.what, got, c.want)
		}
	}
	// Under a breakable capture the stage replays one piece per break it holds, plus one,
	// even where the whole model's count is larger.
	k.breakableCapture, k.captureBreaks = true, 99
	st := pipelineStage{layers: []price.PlannedLayer{sparse}, totalLayers: 3}
	if got := k.stageHost(b, &st, 0); got != 4*time.Microsecond {
		t.Errorf("breakable stage: host %v, want 4us", got)
	}
}

// The binding pipeline rank is the one whose own budget less its own fixed bytes over its own
// KV a token holds the fewest tokens, and FixedBytes and SequenceVariableBytes are that one
// rank's (get_kv_cache_configs step 5, vllm/v1/core/kv_cache_utils.py:2660-2680@5df63a72). Here
// the weight-heaviest stage is not the KV-heaviest: pairing the one's weights with the other's
// KV, as the kernel did, describes no worker and understates the pool.
func TestTheBindingStageIsOneWorkersPool(t *testing.T) {
	const gb = 1e9
	k := &Kernel{chip: hardware.Chip{MemoryGiB: 80}, graph: &model.Graph{}, blockSize: 16,
		stages: []pipelineStage{
			{memory: stageMemory{weights: 60 * gb, kvPerToken: 1000}},
			{memory: stageMemory{weights: 40 * gb, kvPerToken: 3000}},
		}}
	k.bindStage()
	k.fixed = k.computeFixedBytes(k.graph)
	if k.ppKVStage != 1 || k.FixedBytes().Weights != 40*gb || k.SequenceVariableBytes(16) != 16*3000 {
		t.Errorf("binding stage %d with %d weight bytes and %d KV bytes a block; vLLM's smallest pool is "+
			"stage 1's, 40 GB and 48000 B", k.ppKVStage, k.FixedBytes().Weights, k.SequenceVariableBytes(16))
	}
	if sm := k.PipelineStageMemory(); sm[0].Fixed.Weights != 60*gb || sm[0].PerBlockBytes != 16000 || sm[0].Binding || !sm[1].Binding {
		t.Errorf("PipelineStageMemory %+v", sm)
	}
	// The other way round: the KV-heaviest stage has the room. 80 GiB x 0.92 is 79.0 GB, so
	// stage 0 leaves 3.0 GB for 3,000 B tokens (1.0M) and stage 1 39.0 GB for 3,100 B (12.6M).
	k.stages[0].memory = stageMemory{weights: 76 * gb, kvPerToken: 3000}
	k.stages[1].memory = stageMemory{weights: 40 * gb, kvPerToken: 3100}
	k.bindStage()
	if k.ppKVStage != 0 {
		t.Errorf("binding stage %d; stage 0's pool is the smaller, 1.0M tokens against 12.6M", k.ppKVStage)
	}
}

// A tensor-parallel width that does not divide the KV heads of an attention that cannot
// replicate a head is refused, in the drafter's layers too; one that can replicate is priced.
func TestAnUnsplittableKVHeadCountIsRefused(t *testing.T) {
	g := func(notReplicated bool) *model.Graph {
		return &model.Graph{LayerKinds: []model.LayerKind{{ID: "a", Nodes: []model.Node{
			{Op: model.OpAttention, AttentionKind: model.AttentionGQA, NumQHeads: 128,
				NumKVHeads: 16, HeadDim: 64, KVHeadsNotReplicated: notReplicated}}}}}
	}
	for _, c := range []struct {
		tp      int
		fused   bool
		refused bool
	}{{8, true, false}, {16, true, false}, {32, true, true}, {12, true, true}, {32, false, false}} {
		err := refuseUnsplittableKVHeads(g(c.fused), c.tp)
		if (err != nil) != c.refused {
			t.Errorf("tp %d fused %v: %v, want refused %v", c.tp, c.fused, err, c.refused)
		}
	}
	draft := g(false)
	draft.Speculator = &model.Speculator{Head: []model.Node{{Op: model.OpGEMM}}}
	draft.LayerKinds = append(draft.LayerKinds, model.LayerKind{ID: "d", Nodes: []model.Node{
		{Op: model.OpAttention, NumKVHeads: 16, KVHeadsNotReplicated: true}}})
	if refuseUnsplittableKVHeads(draft, 32) == nil {
		t.Error("a drafter layer's unsplittable heads at tp 32 were priced")
	}
}
